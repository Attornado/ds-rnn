"""GRU cell."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class GRUCell(nn.Module):
    """GRU cell with the same initialization as the dynamic-sparse cells."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True):
        """Initialize the cell.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.
            bias: Whether gate pre-activations use biases.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
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
        """Reset weights: xavier for input weights, orthogonal for recurrent."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        if self.bias_ih is not None:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)

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

        w_ih = self.weight_ih.view(3, self.hidden_size, self.input_size)
        w_hh = self.weight_hh.view(3, self.hidden_size, self.hidden_size)

        gates = []
        for i in range(2):
            bias_ih = None if self.bias_ih is None else self.bias_ih[i * self.hidden_size:(i + 1) * self.hidden_size]
            bias_hh = None if self.bias_hh is None else self.bias_hh[i * self.hidden_size:(i + 1) * self.hidden_size]
            gates.append(F.linear(x, w_ih[i], bias_ih) + F.linear(h, w_hh[i], bias_hh))

        r = torch.sigmoid(gates[0])
        z = torch.sigmoid(gates[1])

        bias_ih_n = None if self.bias_ih is None else self.bias_ih[2 * self.hidden_size:]
        bias_hh_n = None if self.bias_hh is None else self.bias_hh[2 * self.hidden_size:]
        n = torch.tanh(F.linear(x, w_ih[2], bias_ih_n) + F.linear(r * h, w_hh[2], bias_hh_n))

        h_new = (1.0 - z) * n + z * h
        return h_new, h_new
