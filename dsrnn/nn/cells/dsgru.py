"""Dynamic-sparse GRU cell."""

from typing import Optional, Tuple

import torch
from torch import nn

from dsrnn.nn.cells.base import DSRNNCell, RoutingOverride
from dsrnn.nn.functional import Implementation
from dsrnn.nn.router import RouterConfig


class DSGRUCell(DSRNNCell):
    """GRU cell with input-dependent sparse gating."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 router_config: RouterConfig,
                 implementation: Implementation = "gathered",
                 share_context_projection: bool = False,
                 bias: bool = True):
        """Initialize the cell.

        Args:
            input_size: Number of input channels.
            hidden_size: Number of hidden channels.
            router_config: Configuration for the hidden-channel router.
            implementation: Masked matmul implementation, "gathered" or "dense".
            share_context_projection: If True, share one context projection
                between the two routers.
            bias: Whether gate pre-activations use biases.
        """
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         router_config=router_config,
                         implementation=implementation,
                         share_context_projection=share_context_projection)
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset cell weights and router parameters."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        if self.bias_ih is not None:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
        super().reset_parameters()

    def forward(self,
                x: torch.Tensor,
                hx: Optional[torch.Tensor] = None,
                routing: Optional[RoutingOverride] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one cell step.

        Args:
            x: Current input of shape (B, input_size).
            hx: Optional hidden state of shape (B, hidden_size).
            routing: Optional precomputed ((mask_x, idx_x), (mask_h, idx_h)).
                If None, routing is computed from x.

        Returns:
            A (h_new, h_new) pair.
        """
        batch_size = x.size(0)
        h = x.new_zeros(batch_size, self.hidden_size) if hx is None else hx

        (mask_x, idx_x), (mask_h, idx_h) = self._resolve_masks(x, routing)

        # All three gates share the input mask: one fused input-side matmul.
        i2h = self.masked_linear(x, mask_x, idx_x, self.weight_ih, bias=self.bias_ih)
        i2h_r, i2h_z, i2h_n = i2h.chunk(3, dim=-1)

        # Reset and update gates share the hidden mask: one fused matmul.
        w_hh_rz = self.weight_hh[:2 * self.hidden_size]
        bias_hh_rz = None if self.bias_hh is None else self.bias_hh[:2 * self.hidden_size]
        h2h_rz = self.masked_linear(h, mask_h, idx_h, w_hh_rz, bias=bias_hh_rz)
        h2h_r, h2h_z = h2h_rz.chunk(2, dim=-1)

        r = torch.sigmoid(i2h_r + h2h_r)
        z = torch.sigmoid(i2h_z + h2h_z)

        # Candidate gate: the reset gate is fused into the hidden mask.
        w_hh_n = self.weight_hh[2 * self.hidden_size:]
        bias_hh_n = None if self.bias_hh is None else self.bias_hh[2 * self.hidden_size:]
        h2h_n = self.masked_linear(h, mask_h * r, idx_h, w_hh_n, bias=bias_hh_n)
        n = torch.tanh(i2h_n + h2h_n)

        h_new = (1.0 - z) * n + z * h
        return h_new, h_new
