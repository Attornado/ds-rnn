"""Base class for dynamic-sparse recurrent cells."""

from dataclasses import replace
from typing import Optional, Tuple

import torch
from torch import nn

from dsrnn.nn.functional import BackwardRule, Implementation, masked_matmul
from dsrnn.nn.router import Router, RouterConfig, RouterOutput

# (mask, idx) pairs for the input and hidden channels
MaskPair = Tuple[torch.Tensor, torch.Tensor]
RoutingOverride = Tuple[MaskPair, MaskPair]


class DSRNNCell(nn.Module):
    """Base class for dynamic-sparse recurrent cells.

    Holds the input-channel and hidden-channel routers and computes the
    per-timestep channel masks. Subclasses implement the gated state update.
    """

    supports_precompute: bool = True

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 router_config: RouterConfig,
                 implementation: Implementation = "gathered",
                 share_context_projection: bool = False):
        """Initialize the cell routers.

        Args:
            input_size: Number of input channels.
            hidden_size: Number of hidden channels.
            router_config: Configuration for the hidden-channel router. The
                input-channel router is derived from it by setting
                num_channels=input_size.
            implementation: Masked matmul implementation, "gathered" or "dense".
            share_context_projection: If True, the cell owns a single context
                projection shared by both routers. Requires
                router_config.context_dim.
        """
        super().__init__()
        if router_config.input_dim != input_size:
            raise ValueError(f"router_config.input_dim ({router_config.input_dim}) must equal "
                             f"input_size ({input_size}).")
        if router_config.num_channels != hidden_size:
            raise ValueError(f"router_config.num_channels ({router_config.num_channels}) must equal "
                             f"hidden_size ({hidden_size}).")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.implementation = implementation
        self.share_context_projection = share_context_projection
        self._backward: BackwardRule = router_config.backward

        if share_context_projection:
            if router_config.context_dim is None:
                raise ValueError("share_context_projection requires router_config.context_dim.")
            self.context_proj: Optional[nn.Module] = nn.Linear(input_size, router_config.context_dim)
            base_config = replace(router_config, input_dim=router_config.context_dim, context_dim=None)
        else:
            self.context_proj = None
            base_config = router_config

        self.hidden_router = Router(base_config)
        self.input_router = Router(replace(base_config, num_channels=input_size))

        self.last_input_routing: Optional[RouterOutput] = None
        self.last_hidden_routing: Optional[RouterOutput] = None

    def compute_routing(self, x: torch.Tensor) -> Tuple[RouterOutput, RouterOutput]:
        """Compute input and hidden channel masks for the current input.

        Args:
            x: Current cell input of shape (B, input_size).

        Returns:
            The (input_routing, hidden_routing) pair.
        """
        context = self.context_proj(x) if self.context_proj is not None else x
        self.last_input_routing = self.input_router(context)
        self.last_hidden_routing = self.hidden_router(context)
        return self.last_input_routing, self.last_hidden_routing

    def compute_routing_batched(self, x: torch.Tensor) -> Tuple[RouterOutput, RouterOutput]:
        """Compute routing for a flattened batch of inputs.

        Args:
            x: Flattened inputs of shape (N, input_size), e.g. N = B * T.

        Returns:
            The (input_routing, hidden_routing) pair over all N inputs.
        """
        context = self.context_proj(x) if self.context_proj is not None else x
        input_routing = self.input_router(context)
        hidden_routing = self.hidden_router(context)
        return input_routing, hidden_routing

    def _resolve_masks(self,
                       x: torch.Tensor,
                       routing: Optional[RoutingOverride]) -> RoutingOverride:
        """Return ((mask_x, idx_x), (mask_h, idx_h)) for this step.

        Computes routing from x when routing is None (storing the RouterOutputs
        for analysis); otherwise returns the given precomputed masks without
        storing them.

        Args:
            x: Current cell input of shape (B, input_size).
            routing: Optional precomputed ((mask_x, idx_x), (mask_h, idx_h)).

        Returns:
            The ((mask_x, idx_x), (mask_h, idx_h)) pair.
        """
        if routing is None:
            input_routing, hidden_routing = self.compute_routing(x)
            return (input_routing.mask, input_routing.idx), (hidden_routing.mask, hidden_routing.idx)
        return routing

    def masked_linear(self,
                      x: torch.Tensor,
                      mask: torch.Tensor,
                      idx: torch.Tensor,
                      weight: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply a masked linear map with the configured implementation.

        Args:
            x: Input activations of shape (B, I).
            mask: Channel mask of shape (B, I).
            idx: Active channel indices of shape (B, k).
            weight: Weight matrix of shape (O, I).
            bias: Optional bias of shape (O,).

        Returns:
            Output tensor of shape (B, O).
        """
        return masked_matmul(x, mask, weight, idx=idx, bias=bias,
                             implementation=self.implementation, backward=self._backward)

    def reset_parameters(self) -> None:
        """Reset the routers and the shared context projection, if present."""
        self.input_router.reset_parameters()
        self.hidden_router.reset_parameters()
        if self.context_proj is not None:
            self.context_proj.reset_parameters()

    def extra_repr(self) -> str:
        return (f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"implementation='{self.implementation}', "
                f"share_context_projection={self.share_context_projection}")
