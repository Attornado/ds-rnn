# Gathered dual READ/WRITE cell

`dsrnn.nn.cells.dual_read_write_sparse.SparseDualReadWriteRNNCell` is an
opt-in counterpart to `DualReadWriteRNNCell`. It inherits the same constructor,
initialization and independent routers. Parameter names match, so a reference
checkpoint can be loaded with `sparse.load_state_dict(reference.state_dict())`.
The reference implementation and existing experiment defaults are unchanged.

```python
from dsrnn.nn.cells.dual_read_write_sparse import SparseDualReadWriteRNNCell
from dsrnn.nn.router import RouterConfig

cell = SparseDualReadWriteRNNCell(
    input_size=16, hidden_size=64,
    read_router_config=RouterConfig(64, 16, k=8),
    write_router_config=RouterConfig(64, 16, k=4),
)
# x: (B, 16); previous: (B, 64)
# output, state = cell(x, previous)
```

## Executed forward operations

For batch size B, input width D and hidden width H:

| Tensor | Shape |
|---|---|
| READ indices | B x k_read |
| WRITE indices | B x k_write |
| Gathered previous state | B x k_read |
| Directly indexed recurrent weights | B x k_write x k_read |
| Gathered input weights | B x k_write x D |
| Candidate values | B x k_write |
| Output state | B x H |

Advanced indexing gathers both recurrent axes directly. It does not first
materialize a B x H x H tensor. Only written candidate values are evaluated.
Scatter replacement copies unwritten coordinates exactly, including when a
candidate for an unwritten coordinate would have been non-finite.

Recurrent forward arithmetic is O(B*k_write*k_read), plus input projection
O(B*k_write*D). Router execution, gather/scatter and full-state storage remain.
Full budgets naturally remove the corresponding arithmetic savings.

## Backward and surrogate costs

Fixed masks and selected-only routing use ordinary autograd over the gathered
operations. Only selected mask entries participate in that graph. Active
recurrent-weight outer products are accumulated into the shared dense H x H
parameter gradient; overlapping indices across examples are summed.

Full STE router derivatives require more than the forward active block:

- READ-mask derivatives at unselected coordinates require all columns of the
  selected WRITE rows: O(B*k_write*H) arithmetic/storage.
- WRITE-mask derivatives at unselected coordinates require their candidate
  values. Backward recomputes only the missing rows, using selected READ
  columns: O(B*(H-k_write)*k_read) recurrent work and
  O(B*(H-k_write)*D) input work/storage.
- Full mask derivatives are returned only for sides configured with STE. If
  one side uses selected-only routing, its selected derivatives reuse the
  active block and no unselected candidates are computed for that side.

These extra computations are skipped for each side unless it uses STE. There
is no full dense recurrent fallback, but dense weight-gradient allocation still
costs O(H^2). No training FLOP reduction or wall-clock speedup follows from the
forward arithmetic estimate alone.

## Supported contract

- Learned routers use the same independent hard budgets and STE/selected-only
  semantics as the reference; analysis attributes expose the actual masks.
- Explicit mask overrides must be binary, with equal counts within each batch.
  Indices may differ per example. Zero and full override budgets are supported.
  Fractional masks and unequal per-example budgets are rejected, with no dense
  fallback. Override validation can synchronize the device.
- Inputs, state, masks and parameters must share device and floating dtype.
  Autocast is not supported. Double backward is unavailable through the custom
  STE path; fixed and selected-only execution uses ordinary autograd.
- Written non-finite candidates propagate. Unwritten coordinates are copied.
  Gradients through non-finite computations remain undefined, as in the
  reference.

`tests/test_dual_read_write_sparse.py` compares sequences and first derivatives
against the reference, checks fixed-mask finite differences and audits executed
matrix shapes in forward and backward. CUDA cases are available for a machine
with CUDA; they must be run before claiming GPU equivalence.

The CogScale runner can use this path with
`--recurrent_cell rnn --implementation sparse`; `reference` remains the default.
GRU currently supports only the reference implementation.
