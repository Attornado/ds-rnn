"""Specialized sequence-level GRU."""
from typing import Optional, Tuple
import torch
from torch import nn
from dsrnn.nn.cells.gru import GRUCell


class GRU(nn.Module):
    """
    Single-layer GRU running the recurrent cell over a sequence.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 output_size: Optional[int] = None,
                 bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
        self.proj = nn.Identity() if output_size is None else nn.Linear(hidden_size, output_size)

    def forward(self,
                x: torch.Tensor,
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the GRU over a sequence."""
        if self.batch_first:
            x = x.transpose(0, 1)
        timesteps, batch_size, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size) if hx is None else hx

        outputs = []
        for t in range(timesteps):
            h, _ = self.cell(x[t], h)
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        output = self.proj(output)
        return output, h
