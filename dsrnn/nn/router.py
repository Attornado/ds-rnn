"""Input-dependent channel router."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from dsrnn.nn.functional import BackwardRule, BudgetRounding, budget_from_sparsity, topk_mask

_MLP_ACTIVATIONS = {
    "identity": nn.Identity,
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
}


def _activation(name: str) -> nn.Module:
    """Resolve an activation name to a module.

    Args:
        name: Activation name.

    Returns:
        The activation module.
    """
    try:
        return _MLP_ACTIVATIONS[name]()
    except KeyError as error:
        raise ValueError(f"Unsupported router activation '{name}'.") from error


class ScaledSigmoid(nn.Module):
    """Scaled sigmoid with learnable sharpness and threshold.

    Computes sigmoid(tau * (x - mu)). tau is stored in log-space to keep it
    positive during optimization.
    """

    def __init__(self,
                 tau_init: float = 1.0,
                 mu_init: float = 0.0,
                 learnable: bool = True):
        """Initialize the scaled sigmoid.

        Args:
            tau_init: Initial sharpness. Must be positive.
            mu_init: Initial threshold.
            learnable: Whether tau and mu are learnable parameters.
        """
        super().__init__()
        if tau_init <= 0:
            raise ValueError(f"tau_init must be positive, got {tau_init}.")
        self.tau_init = tau_init
        self.mu_init = mu_init
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)), requires_grad=learnable)
        self.mu = nn.Parameter(torch.tensor(mu_init), requires_grad=learnable)

    @property
    def tau(self) -> torch.Tensor:
        """Return the current sharpness."""
        return self.log_tau.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the scaled sigmoid.

        Args:
            x: Input tensor.

        Returns:
            Tensor of the same shape with values in (0, 1).
        """
        return torch.sigmoid(self.tau * (x - self.mu))

    def reset_parameters(self) -> None:
        """Reset tau and mu to their initial values."""
        with torch.no_grad():
            self.log_tau.fill_(math.log(self.tau_init))
            self.mu.fill_(self.mu_init)


@dataclass
class RouterConfig:
    """Configuration for Router.

    Args:
        num_channels: Number of output channels to mask.
        input_dim: Dimensionality of the raw router input.
        context_dim: Optional low-dimensional context size. If None, the raw
            input is used as context.
        n_layers: 1 for a linear router, >1 for an MLP.
        hidden_dim: MLP hidden width. Defaults to the backbone input width.
        activation: Activation for the MLP hidden layers.
        sparsity: Target sparsity in [0, 1).
        k: Optional explicit active-channel budget. Overrides sparsity.
        budget_rounding: Rounding mode for the fractional budget.
        backward: Backward rule for the top-k selection.
        tau_init: Initial sigmoid sharpness.
        mu_init: Initial sigmoid threshold.
        learnable_tau_mu: Whether tau and mu are learnable.
        bias: Whether linear layers use biases.
    """
    num_channels: int
    input_dim: int
    context_dim: Optional[int] = None
    n_layers: int = 1
    hidden_dim: Optional[int] = None
    activation: str = "relu"
    sparsity: float = 0.9
    k: Optional[int] = None
    budget_rounding: BudgetRounding = "floor"
    backward: BackwardRule = "ste"
    tau_init: float = 1.0
    mu_init: float = 0.0
    learnable_tau_mu: bool = True
    bias: bool = True

    def __post_init__(self) -> None:
        if self.num_channels < 1:
            raise ValueError(f"num_channels must be positive, got {self.num_channels}.")
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if self.context_dim is not None and self.context_dim < 1:
            raise ValueError(f"context_dim must be positive, got {self.context_dim}.")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}.")
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}.")
        if self.k is not None and not 1 <= self.k <= self.num_channels:
            raise ValueError(f"k must be in [1, {self.num_channels}], got {self.k}.")
        if self.backward not in ("ste", "selected_only"):
            raise ValueError(f"Unknown backward rule '{self.backward}'.")


@dataclass
class RouterOutput:
    """Output of a Router forward pass.

    Args:
        mask: Channel mask of shape (B, N). Zero outside the top-k positions.
        idx: Selected channel indices of shape (B, k).
        scores: Post-sigmoid scores of shape (B, N).
        logits: Pre-sigmoid logits of shape (B, N).
    """
    mask: torch.Tensor
    idx: torch.Tensor
    scores: torch.Tensor
    logits: torch.Tensor


class Router(nn.Module):
    """Map a context vector to a sparse channel mask."""

    def __init__(self, config: RouterConfig):
        """Initialize the router.

        Args:
            config: Router configuration.
        """
        super().__init__()
        self.config = config

        if config.context_dim is None:
            self.context_proj = nn.Identity()
            backbone_input_dim = config.input_dim
        else:
            self.context_proj = nn.Linear(config.input_dim, config.context_dim, bias=config.bias)
            backbone_input_dim = config.context_dim

        if config.n_layers == 1:
            self.backbone = nn.Linear(backbone_input_dim, config.num_channels, bias=config.bias)
        else:
            hidden_dim = config.hidden_dim or backbone_input_dim
            layers = []
            in_features = backbone_input_dim
            for _ in range(config.n_layers - 1):
                layers.extend((nn.Linear(in_features, hidden_dim, bias=config.bias),
                               _activation(config.activation)))
                in_features = hidden_dim
            layers.append(nn.Linear(in_features, config.num_channels, bias=config.bias))
            self.backbone = nn.Sequential(*layers)

        self.scaled_sigmoid = ScaledSigmoid(tau_init=config.tau_init,
                                            mu_init=config.mu_init,
                                            learnable=config.learnable_tau_mu)
        self.k = config.k or budget_from_sparsity(config.num_channels,
                                                  config.sparsity,
                                                  rounding=config.budget_rounding)

    @property
    def routing_enabled(self) -> bool:
        """Return whether the router sparsifies (False means dense mask)."""
        return self.k < self.config.num_channels

    def reset_parameters(self) -> None:
        """Reset projection, backbone, and sigmoid parameters."""
        if isinstance(self.context_proj, nn.Linear):
            self.context_proj.reset_parameters()
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.scaled_sigmoid.reset_parameters()

    def forward(self,
                x: torch.Tensor,
                backward: Optional[BackwardRule] = None) -> RouterOutput:
        """Compute the channel mask for a batch of inputs.

        Args:
            x: Router input of shape (B, input_dim).
            backward: Optional override of the configured backward rule.

        Returns:
            A RouterOutput with mask, selected indices, scores, and logits.
        """
        context = self.context_proj(x)
        logits = self.backbone(context)
        scores = self.scaled_sigmoid(logits)

        if self.routing_enabled:
            mask, idx = topk_mask(scores, self.k, backward or self.config.backward)
        else:
            mask = torch.ones_like(scores)
            idx = torch.arange(self.config.num_channels, device=scores.device)
            idx = idx.unsqueeze(0).expand(scores.size(0), -1)

        return RouterOutput(mask=mask, idx=idx, scores=scores, logits=logits)
