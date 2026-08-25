"""Specialized sequence-level dynamic-sparse LTC."""

from typing import Optional, Tuple

import torch
from torch import nn

from dsrnn.nn.cells.dsltc import DSLTCCell
from dsrnn.nn.functional import Implementation
from dsrnn.nn.router import RouterConfig


class DSLTC(nn.Module):
    """Single-layer DS-LTC optimized for sequence-level routing.

    Routing is computed over the full sequence with one batched call; the
    recurrent loop applies the precomputed masks per timestep.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 router_config: RouterConfig,
                 implementation: Implementation = "dense",
                 batch_first: bool = True,
                 output_size: Optional[int] = None,
                 share_context_projection: bool = False,
                 elapsed_time: float = 1.0,
                 **cell_kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.elapsed_time = elapsed_time
        self.cell = DSLTCCell(input_size=input_size,
                              hidden_size=hidden_size,
                              router_config=router_config,
                              implementation=implementation,
                              share_context_projection=share_context_projection,
                              **cell_kwargs)
        self.proj = nn.Identity() if output_size is None else nn.Linear(hidden_size, output_size)

    def _route_sequence(self, x: torch.Tensor):
        """Compute masks and indices for all timesteps at once."""
        timesteps, batch_size, _ = x.shape
        flat_x = x.reshape(timesteps * batch_size, self.input_size)
        input_routing, hidden_routing = self.cell.compute_routing_batched(flat_x)
        return (input_routing.mask.view(timesteps, batch_size, -1),
                input_routing.idx.view(timesteps, batch_size, -1),
                hidden_routing.mask.view(timesteps, batch_size, -1),
                hidden_routing.idx.view(timesteps, batch_size, -1))

    def forward(self,
                x: torch.Tensor,
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the DS-LTC over a sequence."""
        if self.batch_first:
            x = x.transpose(0, 1)
        timesteps, batch_size, _ = x.shape
        mask_x, idx_x, mask_h, idx_h = self._route_sequence(x)

        state = x.new_zeros(batch_size, self.cell.state_size) if hx is None else hx
        outputs = []
        for t in range(timesteps):
            _, state = self.cell(x[t], state,
                                 elapsed_time=self.elapsed_time,
                                 routing=((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t])))
            outputs.append(state)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        output = self.proj(output)
        return output, state
