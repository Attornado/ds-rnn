"""Dense reference cell for independent recurrent READ and WRITE masks."""

from dataclasses import replace
from decimal import Decimal
from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from dsrnn.nn.router import Router, RouterConfig, RouterOutput


class DualReadWriteRNNCell(nn.Module):
    """Tanh recurrent cell with independent read and write masks.

    Args:
        input_size: Number of input features.
        hidden_size: Number of recurrent-state features.
        bias: Whether to include a candidate-state bias.
        read_router_config: Configuration for the read router. If omitted,
            ``read_mask`` must be supplied to :meth:`forward`.
        write_router_config: Configuration for the write router. If omitted,
            ``write_mask`` must be supplied to :meth:`forward`.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 *,
                 read_router_config: Optional[RouterConfig] = None,
                 write_router_config: Optional[RouterConfig] = None):
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        if (read_router_config is None) != (write_router_config is None):
            raise ValueError("Provide both read_router_config and write_router_config, or neither.")
        self.read_router = None
        self.write_router = None
        for name, config in (("read_router", read_router_config),
                             ("write_router", write_router_config)):
            if config is not None:
                if config.input_dim != input_size or config.num_channels != hidden_size:
                    raise ValueError(f"{name} config must have input_dim={input_size} "
                                     f"and num_channels={hidden_size}.")
                k = config.k
                if k is None:
                    k = max(1, ceil((Decimal(1) - Decimal(str(config.sparsity))) * hidden_size))
                setattr(self, name, Router(replace(config, k=k, binary=True, budget_rounding="ceil")))
        self.last_read_routing: Optional[RouterOutput] = None
        self.last_write_routing: Optional[RouterOutput] = None
        self.read_mask: Optional[torch.Tensor] = None
        self.write_mask: Optional[torch.Tensor] = None
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the cell and router parameters."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        for router in (self.read_router, self.write_router):
            if router is not None:
                router.reset_parameters()

    def compute_routing(self, x: torch.Tensor) -> Tuple[RouterOutput, RouterOutput]:
        """Compute the read and write routing outputs.

        Args:
            x: Input tensor of shape ``(batch_size, input_size)``.

        Returns:
            The read and write router outputs.

        Raises:
            ValueError: If the cell has no router configurations.
        """
        if self.read_router is None or self.write_router is None:
            raise ValueError("Router configurations are required to compute routing.")
        return self.read_router(x), self.write_router(x)

    def forward(self,
                x: torch.Tensor,
                hx: Optional[torch.Tensor] = None,
                *,
                read_mask: Optional[torch.Tensor] = None,
                write_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute one recurrent step.

        Args:
            x: Input tensor of shape ``(batch_size, input_size)``.
            hx: Previous state of shape ``(batch_size, hidden_size)``. Defaults
                to zeros.
            read_mask: Read mask of shape ``(batch_size, hidden_size)``. Uses
                the read router when omitted.
            write_mask: Write mask of shape ``(batch_size, hidden_size)``. Uses
                the write router when omitted.

        Returns:
            A pair containing the output and new hidden state.

        Raises:
            ValueError: If an input has an invalid shape or a required mask is
                unavailable.
        """
        if x.ndim != 2 or x.size(1) != self.input_size:
            raise ValueError(f"x must have shape (B, {self.input_size}).")
        state_shape = (x.size(0), self.hidden_size)
        h = x.new_zeros(state_shape) if hx is None else hx
        self.last_read_routing = None
        self.last_write_routing = None
        if read_mask is None:
            if self.read_router is None:
                raise ValueError("read_mask is required without read_router_config.")
            self.last_read_routing = self.read_router(x)
            read_mask = self.last_read_routing.mask
        if write_mask is None:
            if self.write_router is None:
                raise ValueError("write_mask is required without write_router_config.")
            self.last_write_routing = self.write_router(x)
            write_mask = self.last_write_routing.mask
        for name, tensor in (("hx", h), ("read_mask", read_mask),
                             ("write_mask", write_mask)):
            if tensor.shape != state_shape:
                raise ValueError(f"{name} must have shape {state_shape}.")
        self.read_mask = read_mask
        self.write_mask = write_mask

        h_read = read_mask * h
        candidate = torch.tanh(F.linear(h_read, self.weight_hh)
                               + F.linear(x, self.weight_ih, self.bias))
        candidate = torch.where((write_mask == 0) & ~torch.isfinite(candidate),
                                torch.zeros_like(candidate), candidate)
        h_new = (1.0 - write_mask) * h + write_mask * candidate
        return h_new, h_new
