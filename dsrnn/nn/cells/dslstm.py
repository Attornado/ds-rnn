"""Dynamic-sparse LSTM cell."""

from typing import Optional, Tuple

import torch
from torch import nn

from dsrnn.nn.cells.base import DSRNNCell
from dsrnn.nn.functional import Implementation
from dsrnn.nn.router import RouterConfig, RouterOutput


class DSLSTMCell(DSRNNCell):
    """LSTM cell with input-dependent sparse gating."""

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

    def forward(self,
                x: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                routing: Optional[Tuple[RouterOutput, RouterOutput]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one cell step.

        Args:
            x: Current input of shape (B, input_size).
            hx: Optional (h, c) state pair, each of shape (B, hidden_size).
            routing: Optional precomputed (input_routing, hidden_routing) pair.
                If None, routing is computed from x.

        Returns:
            A (h_new, (h_new, c_new)) pair.
        """
        batch_size = x.size(0)
        if hx is None:
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
        else:
            h, c = hx

        input_routing, hidden_routing = self._resolve_routing(x, routing)

        # All four gates share the input and hidden masks, so the input- and
        # hidden-side transforms are each fused into a single masked matmul.
        i2h = self.masked_linear(x, input_routing.mask, input_routing.idx, self.weight_ih,
                                 bias=self.bias_ih)
        h2h = self.masked_linear(h, hidden_routing.mask, hidden_routing.idx, self.weight_hh,
                                 bias=self.bias_hh)

        i, f, g, o = (i2h + h2h).chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)
