"""Dynamic-sparse GRU cell."""

from typing import Optional, Tuple

import torch
from torch import nn

from dsrnn.nn.cells.base import DSRNNCell, RoutingOverride
from dsrnn.nn.functional import Implementation, gathered_active_matmul
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

    @staticmethod
    def _update(gates_x: torch.Tensor,
                h2h_n: torch.Tensor,
                h: torch.Tensor,
                r: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """Apply the GRU state update from computed gates."""
        i2h_n = gates_x.chunk(3, dim=-1)[2]
        n = torch.tanh(i2h_n + h2h_n)
        return (1.0 - z) * n + z * h

    def forward_active(self,
                       x_active: torch.Tensor,
                       idx_x: torch.Tensor,
                       h: torch.Tensor,
                       mask_h: torch.Tensor,
                       idx_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one selected-only step from pre-gathered input values.

        Args:
            x_active: Already-masked input values of shape (B, k_x).
            idx_x: Input-channel indices of shape (B, k_x).
            h: Hidden state of shape (B, H).
            mask_h: Hidden mask of shape (B, H).
            idx_h: Hidden-channel indices of shape (B, k_h).

        Returns:
            A (h_new, h_new) pair.
        """
        gates_x = gathered_active_matmul(x_active, idx_x, self.weight_ih, self.bias_ih)
        i2h_r, i2h_z, _ = gates_x.chunk(3, dim=-1)

        h_active = torch.gather(h, dim=-1, index=idx_h)
        h_active = h_active * torch.gather(mask_h, dim=-1, index=idx_h)
        w_hh_rz = self.weight_hh[:2 * self.hidden_size]
        bias_hh_rz = None if self.bias_hh is None else self.bias_hh[:2 * self.hidden_size]
        h2h_rz = gathered_active_matmul(h_active, idx_h, w_hh_rz, bias_hh_rz)
        h2h_r, h2h_z = h2h_rz.chunk(2, dim=-1)

        r = torch.sigmoid(i2h_r + h2h_r)
        z = torch.sigmoid(i2h_z + h2h_z)

        w_hh_n = self.weight_hh[2 * self.hidden_size:]
        bias_hh_n = None if self.bias_hh is None else self.bias_hh[2 * self.hidden_size:]
        h2h_n = gathered_active_matmul(h_active * r, idx_h, w_hh_n, bias_hh_n)
        h_new = self._update(gates_x, h2h_n, h, r, z)
        return h_new, h_new

    def forward(self,
                x: torch.Tensor,
                hx: Optional[torch.Tensor] = None,
                routing: Optional[RoutingOverride] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one generic cell step."""
        batch_size = x.size(0)
        h = x.new_zeros(batch_size, self.hidden_size) if hx is None else hx

        (mask_x, idx_x), (mask_h, idx_h) = self._resolve_masks(x, routing)
        i2h = self.masked_linear(x, mask_x, idx_x, self.weight_ih, bias=self.bias_ih)
        i2h_r, i2h_z, _ = i2h.chunk(3, dim=-1)

        w_hh_rz = self.weight_hh[:2 * self.hidden_size]
        bias_hh_rz = None if self.bias_hh is None else self.bias_hh[:2 * self.hidden_size]
        h2h_rz = self.masked_linear(h, mask_h, idx_h, w_hh_rz, bias=bias_hh_rz)
        h2h_r, h2h_z = h2h_rz.chunk(2, dim=-1)

        r = torch.sigmoid(i2h_r + h2h_r)
        z = torch.sigmoid(i2h_z + h2h_z)

        w_hh_n = self.weight_hh[2 * self.hidden_size:]
        bias_hh_n = None if self.bias_hh is None else self.bias_hh[2 * self.hidden_size:]
        h2h_n = self.masked_linear(h, mask_h * r, idx_h, w_hh_n, bias=bias_hh_n)
        h_new = self._update(i2h, h2h_n, h, r, z)
        return h_new, h_new
