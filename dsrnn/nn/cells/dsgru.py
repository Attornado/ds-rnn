"""Dynamic-sparse GRU cell."""

from typing import Optional, Tuple

import torch
from torch import nn

from dsrnn.nn.cells.base import DSRNNCell
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
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one cell step.

        Args:
            x: Current input of shape (B, input_size).
            hx: Optional hidden state of shape (B, hidden_size).

        Returns:
            A (h_new, h_new) pair.
        """
        batch_size = x.size(0)
        h = x.new_zeros(batch_size, self.hidden_size) if hx is None else hx

        input_routing, hidden_routing = self.compute_routing(x)
        w_ih = self.weight_ih.view(3, self.hidden_size, self.input_size)
        w_hh = self.weight_hh.view(3, self.hidden_size, self.hidden_size)

        gates = []
        for i in range(2):
            bias_ih = None if self.bias_ih is None else self.bias_ih[i * self.hidden_size:(i + 1) * self.hidden_size]
            bias_hh = None if self.bias_hh is None else self.bias_hh[i * self.hidden_size:(i + 1) * self.hidden_size]
            i2h = self.masked_linear(x, input_routing.mask, input_routing.idx, w_ih[i], bias=bias_ih)
            h2h = self.masked_linear(h, hidden_routing.mask, hidden_routing.idx, w_hh[i], bias=bias_hh)
            gates.append(i2h + h2h)

        r = torch.sigmoid(gates[0])
        z = torch.sigmoid(gates[1])

        # Candidate gate: the reset gate is fused into the hidden mask.
        bias_ih_n = None if self.bias_ih is None else self.bias_ih[2 * self.hidden_size:]
        bias_hh_n = None if self.bias_hh is None else self.bias_hh[2 * self.hidden_size:]
        n_i2h = self.masked_linear(x, input_routing.mask, input_routing.idx, w_ih[2], bias=bias_ih_n)
        n_h2h = self.masked_linear(h, hidden_routing.mask * r, hidden_routing.idx, w_hh[2], bias=bias_hh_n)
        n = torch.tanh(n_i2h + n_h2h)

        h_new = (1.0 - z) * n + z * h
        return h_new, h_new
