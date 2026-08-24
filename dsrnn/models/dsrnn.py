"""Multi-layer RNN model parametrized by cell type."""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn

from dsrnn.nn.cells.dsgru import DSGRUCell
from dsrnn.nn.cells.dslstm import DSLSTMCell
from dsrnn.nn.cells.dsltc import DSLTCCell
from dsrnn.nn.cells.gru import GRUCell
from dsrnn.nn.cells.lstm import LSTMCell
from dsrnn.nn.functional import Implementation
from dsrnn.nn.router import RouterConfig, RouterOutput

# name -> (cell class, whether the constructor takes a router_config)
_CELL_TYPES: Dict[str, Tuple[Type[nn.Module], bool]] = {
    "dslstm": (DSLSTMCell, True),
    "dsgru": (DSGRUCell, True),
    "dsltc": (DSLTCCell, True),
    "lstm": (LSTMCell, False),
    "gru": (GRUCell, False),
}


def register_cell_type(name: str,
                       cell_cls: Type[nn.Module],
                       dynamic_sparse: bool = True) -> None:
    """Register a cell type for the DSRNN model.

    Args:
        name: Cell type name.
        cell_cls: Cell class. Its forward must map (x_t, state) to
            (output_t, new_state).
        dynamic_sparse: Whether the cell constructor takes a router_config.
    """
    _CELL_TYPES[name] = (cell_cls, dynamic_sparse)


def _slice_routing(routing: RouterOutput,
                   t: int,
                   seq_len: int,
                   batch_size: int) -> RouterOutput:
    """Select one timestep from a sequence-batched RouterOutput."""
    return RouterOutput(
        mask=routing.mask.view(seq_len, batch_size, -1)[t],
        idx=routing.idx.view(seq_len, batch_size, -1)[t],
        scores=routing.scores.view(seq_len, batch_size, -1)[t],
        logits=routing.logits.view(seq_len, batch_size, -1)[t],
    )


class DSRNN(nn.Module):
    """Multi-layer RNN model parametrized by cell type.

    Stacks cells of the given type with inter-layer dropout and an optional
    output projection. Dense baselines (cell_type="lstm"/"gru") and
    dynamic-sparse cells share the same wrapper, so dense-vs-DS is a
    configuration choice.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_type: str = "dslstm",
                 router_config: Optional[RouterConfig] = None,
                 num_layers: int = 1,
                 implementation: Implementation = "gathered",
                 share_context_projection: bool = False,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 output_size: Optional[int] = None,
                 precompute_routing: bool = False,
                 **cell_kwargs):
        """Initialize the model.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units per layer.
            cell_type: Cell type name (see register_cell_type).
            router_config: Router configuration. Required for dynamic-sparse
                cell types; num_channels must equal hidden_size and input_dim
                must equal input_size (adjusted per layer automatically).
            num_layers: Number of stacked layers.
            implementation: Masked matmul implementation for dynamic-sparse
                cells, "gathered" or "dense". DSLTCCell currently requires
                "dense".
            share_context_projection: If True, each cell shares one context
                projection between its two routers.
            dropout: Dropout probability between layers.
            batch_first: If True, input and output are (B, T, *).
            output_size: Optional output projection size. If None, the model
            returns hidden states.
            precompute_routing: If True, compute the channel masks for the
                whole sequence with one batched router call per layer instead
                of once per timestep. Only affects dynamic-sparse cells.
            **cell_kwargs: Additional keyword arguments for the cell
                constructors.
        """
        super().__init__()
        if cell_type not in _CELL_TYPES:
            raise ValueError(f"Unknown cell_type '{cell_type}'. Available: {sorted(_CELL_TYPES)}.")
        cell_cls, is_ds = _CELL_TYPES[cell_type]
        if is_ds and router_config is None:
            raise ValueError(f"router_config is required for cell_type '{cell_type}'.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.precompute_routing = precompute_routing

        cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            if is_ds:
                layer_config = replace(router_config, input_dim=layer_input_size)
                cell = cell_cls(input_size=layer_input_size,
                                hidden_size=hidden_size,
                                router_config=layer_config,
                                implementation=implementation,
                                share_context_projection=share_context_projection,
                                **cell_kwargs)
            else:
                cell = cell_cls(input_size=layer_input_size,
                                hidden_size=hidden_size,
                                **cell_kwargs)
            cells.append(cell)
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.proj = nn.Identity() if output_size is None else nn.Linear(hidden_size, output_size)

    def forward(self,
                x: torch.Tensor,
                hx: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """Run the model over a sequence.

        Args:
            x: Input sequence of shape (B, T, input_size) if batch_first,
                else (T, B, input_size).
            hx: Optional list of per-layer initial states. If None, states are
                zero-initialized by the cells.

        Returns:
            A (output, final_states) pair. output has shape (B, T, hidden_size)
            or (B, T, output_size) if batch_first, else (T, B, *).
            final_states is a list of per-layer final states.
        """
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size = x.size(0), x.size(1)

        states = [None] * self.num_layers if hx is None else list(hx)
        layer_input = x
        final_states = []
        for layer, cell in enumerate(self.cells):
            state = states[layer]
            outputs = []

            if self.precompute_routing and getattr(cell, "supports_precompute", False):
                flat = layer_input.reshape(seq_len * batch_size, layer_input.size(-1))
                routing = cell.compute_routing_batched(flat)
                for t in range(seq_len):
                    routing_t = (_slice_routing(routing[0], t, seq_len, batch_size),
                                 _slice_routing(routing[1], t, seq_len, batch_size))
                    out, state = cell(layer_input[t], state, routing=routing_t)
                    outputs.append(out)
            else:
                for t in range(seq_len):
                    out, state = cell(layer_input[t], state)
                    outputs.append(out)

            layer_output = torch.stack(outputs, dim=0)
            final_states.append(state)
            if self.dropout is not None and layer < self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            layer_input = layer_output

        output = layer_output
        if self.batch_first:
            output = output.transpose(0, 1)
        output = self.proj(output)
        return output, final_states
