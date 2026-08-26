"""Dense LTC sequence model wrapping ncps.torch.LTC."""

from typing import Optional, Tuple

import torch
from ncps.torch import LTC as _NcpsLTC
from ncps.wirings import FullyConnected
from torch import nn


class LTC(nn.Module):
    """Dense LTC with a fully-connected wiring and an optional output head."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 output_size: Optional[int] = None,
                 **ltc_kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        wiring = FullyConnected(hidden_size)
        self.rnn = _NcpsLTC(input_size=input_size, units=wiring,
                            return_sequences=True, batch_first=batch_first,
                            **ltc_kwargs)
        self.proj = nn.Identity() if output_size is None else nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the LTC over a sequence. Returns (outputs, final_state)."""
        outputs, state = self.rnn(x, hx)
        return self.proj(outputs), state
