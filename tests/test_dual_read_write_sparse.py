"""Numerical and operation-shape audits for the gathered dual recurrence."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from dsrnn.nn.cells.dual_read_write import DualReadWriteRNNCell
from dsrnn.nn.cells.dual_read_write_sparse import (
    SparseDualReadWriteRNNCell,
    _GatheredDualStep,
)
from dsrnn.nn.router import RouterConfig


def cells(hidden, read_k, write_k, backward, dtype, device, bias=True, projected=False):
    kwargs = dict(context_dim=2, n_layers=2, activation="tanh") if projected else {}
    configs = dict(read_router_config=RouterConfig(hidden, 3, k=read_k, backward=backward, **kwargs),
                   write_router_config=RouterConfig(hidden, 3, k=write_k, backward=backward, **kwargs))
    reference = DualReadWriteRNNCell(3, hidden, bias=bias, **configs).to(device=device, dtype=dtype)
    sparse = SparseDualReadWriteRNNCell(3, hidden, bias=bias, **configs).to(device=device, dtype=dtype)
    sparse.load_state_dict(reference.state_dict(), strict=True)
    return reference, sparse


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("hidden,kr,kw", [(4, 1, 1), (7, 2, 3), (7, 5, 2), (4, 4, 4)])
@pytest.mark.parametrize("backward", ["ste", "selected_only"])
def test_routed_sequence_equivalence(device, dtype, batch, hidden, kr, kw, backward):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(101)
    reference, sparse = cells(hidden, kr, kw, backward, dtype, device,
                              bias=kr != 5, projected=kr == 2)
    data = torch.randn(8, batch, 3, dtype=dtype, device=device)
    initial = torch.randn(batch, hidden, dtype=dtype, device=device)
    upstream = torch.randn(8, batch, hidden, dtype=dtype, device=device)
    runs = []
    for cell in (reference, sparse):
        x = data.clone().requires_grad_()
        h0 = initial.clone().requires_grad_()
        h, states, routes = h0, [], []
        for xt in x:
            previous = h
            h, state = cell(xt, h)
            assert h is state
            assert torch.equal(h[cell.write_mask == 0], previous[cell.write_mask == 0])
            h.retain_grad()
            states.append(h)
            pair = (cell.last_read_routing, cell.last_write_routing)
            for route, k in zip(pair, (kr, kw)):
                assert torch.equal(route.mask.sum(1), h.new_full((batch,), k))
                route.scores.retain_grad()
            routes.append(pair)
        output = torch.stack(states)
        (output * upstream).sum().backward()
        runs.append((x, h0, output, states, routes))
    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (1e-10, 1e-9)
    a, b = runs
    for actual, expected in ((b[2], a[2]), (b[0].grad, a[0].grad), (b[1].grad, a[1].grad)):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for left, right in zip(a[3], b[3]):
        torch.testing.assert_close(left.grad, right.grad, atol=atol, rtol=rtol)
    for ref_pair, sparse_pair in zip(a[4], b[4]):
        for ref_route, sparse_route in zip(ref_pair, sparse_pair):
            assert torch.equal(ref_route.mask, sparse_route.mask)
            assert torch.equal(ref_route.idx, sparse_route.idx)
            torch.testing.assert_close(ref_route.scores.grad, sparse_route.scores.grad, atol=atol, rtol=rtol)
    for (name, p), (_, q) in zip(reference.named_parameters(), sparse.named_parameters()):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(q.grad, p.grad, atol=atol, rtol=rtol, msg=name)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("kr,kw", [(0, 0), (0, 2), (2, 0), (1, 1), (2, 2), (4, 4)])
def test_override_equivalence_and_mask_gradients(batch, kr, kw):
    torch.manual_seed(102)
    reference = DualReadWriteRNNCell(3, 4).double()
    sparse = SparseDualReadWriteRNNCell(3, 4).double()
    sparse.load_state_dict(reference.state_dict())
    x = torch.randn(batch, 3, dtype=torch.double)
    h = torch.randn(batch, 4, dtype=torch.double)
    read = torch.zeros_like(h)
    write = torch.zeros_like(h)
    # At kr=kw=2 each row has all four (R,W) combinations; rows differ.
    for row in range(batch):
        read[row, (torch.arange(kr) + row) % 4] = 1
        write[row, (torch.arange(kw) + row + 1) % 4] = 1
    upstream = torch.randn_like(h)
    runs = []
    for cell in (reference, sparse):
        args = [t.clone().requires_grad_() for t in (x, h, read, write)]
        output, _ = cell(args[0], args[1], read_mask=args[2], write_mask=args[3])
        output.backward(upstream)
        runs.append((output, args))
        assert torch.equal(output[write == 0], h[write == 0])
        assert cell.last_read_routing is cell.last_write_routing is None
    torch.testing.assert_close(runs[0][0], runs[1][0])
    for a, b in zip(runs[0][1], runs[1][1]):
        torch.testing.assert_close(a.grad, b.grad)
    for a, b in zip(reference.parameters(), sparse.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


def test_fixed_mask_gradcheck():
    """Check fixed-mask gradients with finite differences."""
    torch.manual_seed(103)
    x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    h = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    wh = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
    wx = torch.randn(4, 2, dtype=torch.double, requires_grad=True)
    bias = torch.randn(4, dtype=torch.double, requires_grad=True)
    ri = torch.tensor([[0, 2], [1, 3]])
    wi = torch.tensor([[1], [0]])
    r = torch.zeros_like(h).scatter(1, ri, 1.)
    w = torch.zeros_like(h).scatter(1, wi, 1.)
    assert torch.autograd.gradcheck(
        lambda x, h, wh, wx, bias: _GatheredDualStep.apply(
            x, h, r, w, wh, wx, bias, ri, wi, True, True),
        (x, h, wh, wx, bias))


class ShapeAudit(TorchDispatchMode):
    """Record matrix-product operands and advanced-indexing result shapes."""

    def __init__(self):
        super().__init__()
        self.products = []
        self.indexed_shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default, torch.ops.aten.addmm.default):
            self.products.append((str(func), tuple(tuple(a.shape) for a in args if isinstance(a, torch.Tensor))))
        if func == torch.ops.aten.index.Tensor:
            self.indexed_shapes.append(tuple(result.shape))
        return result


def test_fixed_and_selected_only_use_standard_autograd(monkeypatch):
    """The custom STE function must not run when full mask gradients are unnecessary."""
    def fail_custom_path(*args, **kwargs):
        raise AssertionError("custom STE path was used")

    monkeypatch.setattr(_GatheredDualStep, "apply", fail_custom_path)

    fixed = SparseDualReadWriteRNNCell(3, 6).double()
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    h = torch.randn(2, 6, dtype=torch.double, requires_grad=True)
    read = torch.tensor([[1., 0., 1., 0., 0., 0.]] * 2, dtype=torch.double)
    write = torch.tensor([[0., 1., 0., 1., 0., 0.]] * 2, dtype=torch.double)
    fixed(x, h, read_mask=read, write_mask=write)[0].sum().backward()

    config = RouterConfig(6, 3, k=2, backward="selected_only")
    routed = SparseDualReadWriteRNNCell(
        3, 6, read_router_config=config, write_router_config=config).double()
    routed(torch.randn(2, 3, dtype=torch.double, requires_grad=True))[0].sum().backward()


def test_mixed_ste_only_recomputes_the_ste_side():
    """Selected-only WRITE gradients must not trigger missing-row candidates."""
    batch, hidden, kr, kw = 2, 11, 2, 3
    cell = SparseDualReadWriteRNNCell(
        3, hidden,
        read_router_config=RouterConfig(hidden, 3, k=kr, backward="ste"),
        write_router_config=RouterConfig(hidden, 3, k=kw, backward="selected_only"),
    ).double()
    output, _ = cell(torch.randn(batch, 3, dtype=torch.double, requires_grad=True),
                     torch.randn(batch, hidden, dtype=torch.double, requires_grad=True))
    with ShapeAudit() as audit:
        output.sum().backward()
    assert (batch, hidden - kw, kr) not in audit.indexed_shapes
    assert (batch, kw, hidden) in audit.indexed_shapes


def test_no_hidden_dense_recurrent_product():
    """Check that learned-mask execution avoids dense recurrent products."""
    batch, hidden, kr, kw = 2, 11, 2, 3
    _, cell = cells(hidden, kr, kw, "ste", torch.double, "cpu")
    x = torch.randn(batch, 3, dtype=torch.double, requires_grad=True)
    h = torch.randn(batch, hidden, dtype=torch.double, requires_grad=True)
    with ShapeAudit() as forward:
        output, _ = cell(x, h)
    assert (batch, kw, kr) in forward.indexed_shapes
    assert ("aten.bmm.default", ((batch, kw, kr), (batch, kr, 1))) in forward.products
    # Only written candidate rows in forward, for both recurrent and input terms.
    assert [shapes for op, shapes in forward.products if op == "aten.bmm.default"] == [
        ((batch, kw, kr), (batch, kr, 1)), ((batch, kw, 3), (batch, 3, 1))]
    with ShapeAudit() as backward:
        output.sum().backward()
    assert (batch, hidden - kw, kr) in backward.indexed_shapes
    for audit in (forward, backward):
        assert (batch, hidden, hidden) not in audit.indexed_shapes
        for _, shapes in audit.products:
            assert (hidden, hidden) not in shapes
            assert (batch, hidden, hidden) not in shapes


def test_fixed_masks_skip_surrogate_recomputation():
    cell = SparseDualReadWriteRNNCell(3, 11).double()
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    h = torch.randn(2, 11, dtype=torch.double, requires_grad=True)
    read, write = torch.zeros_like(h), torch.zeros_like(h)
    read[:, :2] = 1
    write[:, 2:5] = 1
    output, _ = cell(x, h, read_mask=read, write_mask=write)
    with ShapeAudit() as audit:
        output.sum().backward()
    assert (2, 8, 2) not in audit.indexed_shapes
    assert (2, 3, 11) not in audit.indexed_shapes


def test_zero_initial_state_and_partial_override():
    reference, sparse = cells(6, 2, 3, "ste", torch.double, "cpu")
    x = torch.randn(3, 3, dtype=torch.double)
    for kwargs in ({}, {"write_mask": torch.zeros(3, 6, dtype=torch.double)},
                   {"read_mask": torch.ones(3, 6, dtype=torch.double)}):
        a, _ = reference(x, **kwargs)
        b, _ = sparse(x, **kwargs)
        torch.testing.assert_close(a, b)
        assert sparse.read_mask.shape == sparse.write_mask.shape == (3, 6)


def test_nonfinite_unwritten_candidate_is_never_evaluated():
    cell = SparseDualReadWriteRNNCell(2, 2)
    with torch.no_grad():
        cell.weight_hh.fill_(3e38)
        cell.weight_ih.fill_(-3e38)
    h = torch.ones(2, 2)
    write = torch.tensor([[0., 1.], [1., 0.]])
    out, _ = cell(torch.ones(2, 2), h, read_mask=torch.ones_like(h), write_mask=write)
    assert torch.equal(out[write == 0], h[write == 0])
    assert torch.isnan(out[write == 1]).all()


def test_invalid_overrides_fail_without_dense_fallback():
    cell = SparseDualReadWriteRNNCell(3, 4)
    x, h = torch.ones(2, 3), torch.ones(2, 4)
    for read, message in ((h * .5, "binary"),
                          (torch.tensor([[1., 0., 0., 0.], [1., 1., 0., 0.]]), "equal"),
                          (h[0], "shape")):
        with pytest.raises(ValueError, match=message):
            cell(x, h, read_mask=read, write_mask=h)
    with pytest.raises(ValueError, match="autocast"):
        with torch.autocast("cpu"):
            cell(x, h, read_mask=h, write_mask=h)
