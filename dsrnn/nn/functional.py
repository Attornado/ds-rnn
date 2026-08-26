"""Functional primitives for dynamic-sparse channel masking."""

import math
from typing import Literal, Optional, Tuple
import torch


BackwardRule = Literal["ste", "selected_only"]
Implementation = Literal["gathered", "dense"]
BudgetRounding = Literal["floor", "ceil", "round"]


def budget_from_sparsity(num_channels: int,
                         sparsity: float,
                         rounding: BudgetRounding = "floor") -> int:
    """
    Compute the active-channel budget for a target sparsity.

    Args:
        num_channels: Total number of channels.
        sparsity: Target sparsity in [0, 1).
        rounding: Rounding mode for the fractional budget. "floor" matches the
            original implementation; "ceil" matches the paper equation.

    Returns:
        Number of active channels, clamped to [1, num_channels].
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be positive, got {num_channels}.")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}.")

    value = (1.0 - sparsity) * num_channels
    if rounding == "floor":
        k = math.floor(value)
    elif rounding == "ceil":
        k = math.ceil(value)
    elif rounding == "round":
        k = round(value)
    else:
        raise ValueError(f"Unknown rounding mode '{rounding}'.")
    return min(num_channels, max(1, k))


def topk_mask(scores: torch.Tensor,
              k: int,
              backward: BackwardRule = "ste") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a weighted top-k mask from continuous scores.

    The mask is zero outside the top-k positions and holds the score values at
    the selected ones. With backward="ste", the Jacobian w.r.t. scores is the
    identity; with "selected_only", gradient flows only to selected scores.

    Args:
        scores: Continuous scores of shape (B, N).
        k: Number of selected channels per example.
        backward: Backward rule for the discrete selection.

    Returns:
        A (mask, idx) pair with shapes (B, N) and (B, k).
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B, N), got {tuple(scores.shape)}.")
    if not 1 <= k <= scores.size(-1):
        raise ValueError(f"k must be in [1, {scores.size(-1)}], got {k}.")
    if backward not in ("ste", "selected_only"):
        raise ValueError(f"Unknown backward rule '{backward}'.")

    idx = scores.topk(k, dim=-1).indices
    hard = torch.zeros_like(scores)
    hard.scatter_(-1, idx, scores.gather(-1, idx))

    if backward == "ste":
        return hard.detach() - scores.detach() + scores, idx
    return hard, idx


def gathered_matmul(x: torch.Tensor,
                    mask: torch.Tensor,
                    weight: torch.Tensor,
                    idx: torch.Tensor,
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute a gathered masked matmul with standard autograd. Gradient reaches only the gathered (selected) channels.
    For the "ste" backward rule, use gathered_recurrent_matmul instead.

    Args:
        x: Input activations of shape (B, H).
        mask: Channel mask of shape (B, H).
        weight: Shared weight matrix of shape (O, H).
        idx: Active channel indices of shape (B, k).
        bias: Optional bias of shape (O,).

    Returns:
        Output tensor of shape (B, O).
    """
    x_active = x.gather(1, idx) * mask.gather(1, idx)
    weight_active = (weight.unsqueeze(0)
                     .expand(x.size(0), -1, -1)
                     .gather(2, idx.unsqueeze(1).expand(-1, weight.size(0), -1)))
    output = torch.bmm(weight_active, x_active.unsqueeze(-1)).squeeze(-1)
    return output if bias is None else output + bias


def gathered_active_matmul(values: torch.Tensor,
                           idx: torch.Tensor,
                           weight: torch.Tensor,
                           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply a gathered linear map to already-masked active values.

    Args:
        values: Active inputs of shape (B, k).
        idx: Source-channel indices of shape (B, k).
        weight: Weight matrix of shape (O, I).
        bias: Optional bias of shape (O,).

    Returns:
        Output tensor of shape (B, O).
    """
    batch_size, k = values.shape
    out_features = weight.size(0)
    weight_active = (weight.unsqueeze(0)
                     .expand(batch_size, out_features, weight.size(1))
                     .gather(2, idx.unsqueeze(1).expand(-1, out_features, k)))
    output = torch.bmm(weight_active, values.unsqueeze(-1)).squeeze(-1)
    return output if bias is None else output + bias


class GatheredRecurrentMatmul(torch.autograd.Function):
    """
    Gathered column matmul with dense-equivalent mask gradients. The backward pass contracts the upstream gradient with the full weight
    matrix, so every mask entry receives gradient even though only the selected columns were gathered. Required for the
    "ste" backward rule.
    """

    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                mask: torch.Tensor,
                weight: torch.Tensor,
                idx: torch.Tensor) -> torch.Tensor:
        """Compute out = W_active @ (mask * x)_active.

        Args:
            ctx: Autograd context.
            x: Input activations of shape (B, H).
            mask: Channel mask of shape (B, H).
            weight: Shared weight matrix of shape (O, H).
            idx: Active channel indices of shape (B, k).

        Returns:
            Output tensor of shape (B, O).
        """
        output = gathered_matmul(x, mask, weight, idx)
        ctx.save_for_backward(x, mask, weight)
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor):
        """
        Return gradients for x, mask, and weight.

        Args:
            ctx: Autograd context with saved (x, mask, weight).
            grad_output: Upstream gradient of shape (B, O).

        Returns:
            Gradients for (x, mask, weight, idx), with None for idx.
        """
        x, mask, weight = ctx.saved_tensors
        gamma = grad_output @ weight
        grad_x = gamma * mask
        grad_mask = gamma * x
        grad_weight = grad_output.T @ (mask * x)
        return grad_x, grad_mask, grad_weight, None


def gathered_recurrent_matmul(x: torch.Tensor,
                              mask: torch.Tensor,
                              weight: torch.Tensor,
                              idx: torch.Tensor,
                              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute a gathered masked matmul with dense-equivalent mask gradients.

    Args:
        x: Input activations of shape (B, H).
        mask: Channel mask of shape (B, H).
        weight: Shared weight matrix of shape (O, H).
        idx: Active channel indices of shape (B, k).
        bias: Optional bias of shape (O,).

    Returns:
        Output tensor of shape (B, O).
    """
    output = GatheredRecurrentMatmul.apply(x, mask, weight, idx)
    return output if bias is None else output + bias


def dense_masked_matmul(x: torch.Tensor,
                        mask: torch.Tensor,
                        weight: torch.Tensor,
                        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute a masked matmul without gathering.

    Args:
        x: Input activations of shape (B, H).
        mask: Channel mask of shape (B, H).
        weight: Shared weight matrix of shape (O, H).
        bias: Optional bias of shape (O,).

    Returns:
        Output tensor of shape (B, O).
    """
    output = (mask * x) @ weight.T
    return output if bias is None else output + bias


def masked_matmul(x: torch.Tensor,
                  mask: torch.Tensor,
                  weight: torch.Tensor,
                  idx: Optional[torch.Tensor] = None,
                  bias: Optional[torch.Tensor] = None,
                  implementation: Implementation = "gathered",
                  backward: BackwardRule = "ste") -> torch.Tensor:
    """
    Dispatch between gathered and dense masked matmul implementations.

    Args:
        x: Input activations of shape (B, H).
        mask: Channel mask of shape (B, H).
        weight: Shared weight matrix of shape (O, H).
        idx: Active channel indices of shape (B, k). Required for "gathered".
        bias: Optional bias of shape (O,).
        implementation: "gathered" or "dense".
        backward: Backward rule. For "gathered", "ste" uses the custom
            autograd Function and "selected_only" uses the plain gathered path.

    Returns:
        Output tensor of shape (B, O).
    """
    if implementation == "dense":
        return dense_masked_matmul(x, mask, weight, bias=bias)
    if implementation == "gathered":
        if idx is None:
            raise ValueError("idx is required for the gathered implementation.")
        if backward == "ste":
            return gathered_recurrent_matmul(x, mask, weight, idx, bias=bias)
        return gathered_matmul(x, mask, weight, idx, bias=bias)
    raise ValueError(f"Unknown implementation '{implementation}'.")
