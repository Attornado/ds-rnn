"""Specialized sequence-level dynamic-sparse LSTM."""
from typing import Optional, Tuple
import torch
from torch import nn
from dsrnn.nn.cells.dslstm import DSLSTMCell
from dsrnn.nn.functional import Implementation
from dsrnn.nn.router import RouterConfig


class DSLSTM(nn.Module):
    """
    Single-layer DS-LSTM optimized for sequence-level routing. Routing and input-channel gathering are computed over the
    full sequence before entering the recurrent loop.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 router_config: RouterConfig,
                 implementation: Implementation = "gathered",
                 batch_first: bool = True,
                 output_size: Optional[int] = None,
                 share_context_projection: bool = False,
                 bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = DSLSTMCell(input_size=input_size,
                               hidden_size=hidden_size,
                               router_config=router_config,
                               implementation=implementation,
                               share_context_projection=share_context_projection,
                               bias=bias)
        self.proj = nn.Identity() if output_size is None else nn.Linear(hidden_size, output_size)

    def _route_sequence(self, x: torch.Tensor):
        """Compute masks and indices for all timesteps at once."""
        timesteps, batch_size, _ = x.shape
        flat_x = x.reshape(timesteps * batch_size, self.input_size)
        input_routing, hidden_routing = self.cell.compute_routing_batched(flat_x)
        return (input_routing.mask.view(timesteps, batch_size, self.input_size),
                input_routing.idx.view(timesteps, batch_size, -1),
                hidden_routing.mask.view(timesteps, batch_size, self.hidden_size),
                hidden_routing.idx.view(timesteps, batch_size, -1))

    def forward(self,
                x: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the DS-LSTM over a sequence."""
        if self.batch_first:
            x = x.transpose(0, 1)
        timesteps, batch_size, _ = x.shape
        mask_x, idx_x, mask_h, idx_h = self._route_sequence(x)

        x_active = None
        if self.cell.implementation == "gathered":
            x_active = torch.gather(x, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)

        if hx is None:
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
        else:
            h, c = hx

        outputs = []
        for t in range(timesteps):
            if self.cell.implementation == "gathered":
                h, (h, c) = self.cell.forward_active(x_active[t], idx_x[t], h, c, mask_h[t], idx_h[t])
            else:
                h, (h, c) = self.cell(x[t], (h, c),
                                      routing=((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t])))
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        output = self.proj(output)
        return output, (h, c)
