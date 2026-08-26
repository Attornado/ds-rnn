"""Dynamic-sparse LSTM cell."""
from typing import Optional, Tuple
import torch
from torch import nn
from dsrnn.nn.cells.base import DSRNNCell, RoutingOverride
from dsrnn.nn.functional import Implementation, gathered_active_matmul
from dsrnn.nn.router import RouterConfig


class DSLSTMCell(DSRNNCell):
    """LSTM cell with input-dependent sparse gating."""

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
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
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
    def _update(gates: torch.Tensor,
                c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LSTM gate nonlinearities and update the state."""
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def forward_active(self,
                       x_active: torch.Tensor,
                       idx_x: torch.Tensor,
                       h: torch.Tensor,
                       c: torch.Tensor,
                       mask_h: torch.Tensor,
                       idx_h: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one selected-only step from pre-gathered input values.

        Args:
            x_active: Already-masked input values of shape (B, k_x).
            idx_x: Input-channel indices of shape (B, k_x).
            h: Hidden state of shape (B, H).
            c: Cell state of shape (B, H).
            mask_h: Hidden mask of shape (B, H).
            idx_h: Hidden-channel indices of shape (B, k_h).

        Returns:
            A (h_new, (h_new, c_new)) pair.
        """
        h_active = torch.gather(h, dim=-1, index=idx_h)
        h_active = h_active * torch.gather(mask_h, dim=-1, index=idx_h)
        i2h = gathered_active_matmul(x_active, idx_x, self.weight_ih, self.bias_ih)
        h2h = gathered_active_matmul(h_active, idx_h, self.weight_hh, self.bias_hh)
        h_new, c_new = self._update(i2h + h2h, c)
        return h_new, (h_new, c_new)

    def forward(self,
                x: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                routing: Optional[RoutingOverride] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one generic cell step."""
        batch_size = x.size(0)
        if hx is None:
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
        else:
            h, c = hx

        (mask_x, idx_x), (mask_h, idx_h) = self._resolve_masks(x, routing)
        i2h = self.masked_linear(x, mask_x, idx_x, self.weight_ih, bias=self.bias_ih)
        h2h = self.masked_linear(h, mask_h, idx_h, self.weight_hh, bias=self.bias_hh)
        h_new, c_new = self._update(i2h + h2h, c)
        return h_new, (h_new, c_new)
