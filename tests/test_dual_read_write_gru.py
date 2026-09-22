"""Semantic, equivalence, and gradient tests for the dual READ/WRITE GRU."""

import pytest
import torch

from dsrnn.nn.cells.dual_read_write_gru import (
    DualReadWriteGRUCell, SparseDualReadWriteGRUCell, _GatheredGRUStep,
)
from dsrnn.nn.cells.gru import GRUCell
from dsrnn.nn.router import RouterConfig


def _reference_step(cell, x, h, read, write):
    hidden = cell.hidden_size
    h_read = read * h
    w_ih = cell.weight_ih.view(3, hidden, cell.input_size)
    w_hh = cell.weight_hh.view(3, hidden, hidden)
    bias_ih = ([0, 0, 0] if cell.bias_ih is None
               else cell.bias_ih.view(3, hidden))
    bias_hh = ([0, 0, 0] if cell.bias_hh is None
               else cell.bias_hh.view(3, hidden))
    reset = torch.sigmoid(x @ w_ih[0].T + bias_ih[0]
                          + h_read @ w_hh[0].T + bias_hh[0])
    update = torch.sigmoid(x @ w_ih[1].T + bias_ih[1]
                           + h_read @ w_hh[1].T + bias_hh[1])
    content = torch.tanh(x @ w_ih[2].T + bias_ih[2]
                         + (reset * h_read) @ w_hh[2].T + bias_hh[2])
    proposal = (1 - update) * content + update * h_read
    return (1 - write) * h + write * proposal


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_reference_equation(batch_size, bias):
    torch.manual_seed(101)
    cell = DualReadWriteGRUCell(2, 4, bias=bias).double()
    x = torch.randn(batch_size, 2, dtype=torch.double)
    h = torch.randn(batch_size, 4, dtype=torch.double)
    read, write = torch.rand_like(h), torch.rand_like(h)
    out, state = cell(x, h, read_mask=read, write_mask=write)
    torch.testing.assert_close(out, _reference_step(cell, x, h, read, write))
    assert out is state


@pytest.mark.parametrize("bias", [False, True])
def test_dense_masks_match_existing_gru(bias):
    torch.manual_seed(102)
    dense = GRUCell(3, 5, bias=bias).double()
    dual = DualReadWriteGRUCell(3, 5, bias=bias).double()
    dual.load_state_dict(dense.state_dict())
    x_dense = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    h_dense = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
    x_dual = x_dense.detach().clone().requires_grad_()
    h_dual = h_dense.detach().clone().requires_grad_()
    expected, _ = dense(x_dense, h_dense)
    actual, _ = dual(x_dual, h_dual, read_mask=torch.ones_like(h_dual),
                     write_mask=torch.ones_like(h_dual))
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(x_dual.grad, x_dense.grad)
    torch.testing.assert_close(h_dual.grad, h_dense.grad)
    for dense_parameter, dual_parameter in zip(dense.parameters(), dual.parameters()):
        torch.testing.assert_close(dual_parameter.grad, dense_parameter.grad)


def test_all_four_states_read_isolation_and_exact_preservation():
    torch.manual_seed(103)
    cell = DualReadWriteGRUCell(2, 4).double()
    x = torch.randn(2, 2, dtype=torch.double)
    h = torch.randn(2, 4, dtype=torch.double)
    read = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]], dtype=torch.double)
    write = torch.tensor([[0., 0., 1., 1.], [1., 1., 0., 0.]], dtype=torch.double)
    out, _ = cell(x, h, read_mask=read, write_mask=write)
    assert torch.equal(out[write == 0], h[write == 0])

    changed_unread = h + 10 * (1 - read)
    unread_out, _ = cell(x, changed_unread, read_mask=read, write_mask=write)
    assert torch.equal(unread_out[write == 1], out[write == 1])

    changed_protected_read = h + ((read == 1) & (write == 0)).to(h)
    protected_out, _ = cell(x, changed_protected_read,
                            read_mask=read, write_mask=write)
    assert torch.count_nonzero(protected_out[write == 1] != out[write == 1]) > 0


def test_zero_write_preserves_state_when_proposal_is_nonfinite():
    cell = DualReadWriteGRUCell(2, 2)
    with torch.no_grad():
        cell.weight_ih.fill_(-3e38)
        cell.weight_hh.fill_(3e38)
    x = torch.ones(2, 2)
    h = torch.ones(2, 2)
    write = torch.tensor([[0., 0.], [0., 1.]])
    out, _ = cell(x, h, read_mask=torch.ones_like(h), write_mask=write)
    assert torch.equal(out[write == 0], h[write == 0])
    assert torch.isnan(out[write == 1]).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_routed_multistep_forward_backward(device, batch_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(104)
    config = RouterConfig(6, 3, k=3, backward="ste")
    cell = DualReadWriteGRUCell(
        3, 6, read_router_config=config, write_router_config=config,
    ).to(device=device, dtype=torch.double)
    inputs = torch.randn(4, batch_size, 3, device=device, dtype=torch.double,
                         requires_grad=True)
    initial = torch.randn(batch_size, 6, device=device, dtype=torch.double,
                          requires_grad=True)
    h = initial
    outputs = []
    for x in inputs:
        previous = h
        h, state = cell(x, h)
        assert h is state and h.shape == (batch_size, 6)
        assert torch.equal(h[cell.write_mask == 0], previous[cell.write_mask == 0])
        assert torch.all(cell.read_mask.sum(-1) == 3)
        assert torch.all(cell.write_mask.sum(-1) == 3)
        outputs.append(h)
    torch.stack(outputs).square().mean().backward()
    for tensor in (inputs, initial, *cell.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    for router in (cell.read_router, cell.write_router):
        assert torch.count_nonzero(router.backbone.weight.grad) > 0


@pytest.mark.parametrize("name", ["hx", "read_mask", "write_mask"])
def test_rejects_broadcast_state_or_masks(name):
    cell = DualReadWriteGRUCell(2, 4)
    kwargs = {"hx": torch.zeros(3, 4), "read_mask": torch.ones(3, 4),
              "write_mask": torch.ones(3, 4)}
    kwargs[name] = torch.ones(4)
    with pytest.raises(ValueError, match=name):
        cell(torch.zeros(3, 2), **kwargs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("kr,kw,bias", [(1, 1, True), (2, 3, False), (4, 2, True), (5, 5, True)])
@pytest.mark.parametrize("read_rule,write_rule", [
    ("ste", "ste"), ("selected_only", "selected_only"),
    ("ste", "selected_only"), ("selected_only", "ste")])
def test_sparse_gru_routed_sequence_equivalence(device, dtype, batch, kr, kw, bias, read_rule, write_rule):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(112)
    configs = dict(read_router_config=RouterConfig(5, 3, k=kr, backward=read_rule,
                                                   context_dim=2, n_layers=2),
                   write_router_config=RouterConfig(5, 3, k=kw, backward=write_rule,
                                                    context_dim=2, n_layers=2))
    ref = DualReadWriteGRUCell(3, 5, bias=bias, **configs).to(device=device, dtype=dtype)
    sparse = SparseDualReadWriteGRUCell(3, 5, bias=bias, **configs).to(device=device, dtype=dtype)
    sparse.load_state_dict(ref.state_dict(), strict=True)
    data = torch.randn(4, batch, 3, dtype=dtype, device=device)
    initial = torch.randn(batch, 5, dtype=dtype, device=device)
    upstream = torch.randn(4, batch, 5, dtype=dtype, device=device)
    runs = []
    for cell in (ref, sparse):
        x = data.clone().requires_grad_()
        h0 = initial.clone().requires_grad_()
        h, states, routes = h0, [], []
        for xt in x:
            previous = h
            h, output = cell(xt, h)
            assert h is output
            assert torch.equal(h[cell.write_mask == 0], previous[cell.write_mask == 0])
            states.append(h)
            for route, budget in ((cell.last_read_routing, kr), (cell.last_write_routing, kw)):
                assert torch.all(route.mask.sum(-1) == budget)
                route.scores.retain_grad()
                routes.append(route)
        stacked = torch.stack(states)
        (stacked * upstream).sum().backward()
        runs.append((x, h0, stacked, routes))
    tol = dict(atol=2e-5, rtol=2e-4) if dtype == torch.float32 else dict(atol=1e-10, rtol=1e-9)
    a, b = runs
    for actual, expected in ((b[0].grad, a[0].grad), (b[1].grad, a[1].grad), (b[2], a[2])):
        torch.testing.assert_close(actual, expected, **tol)
    for left, right in zip(a[3], b[3]):
        assert torch.equal(left.mask, right.mask)
        torch.testing.assert_close(right.scores.grad, left.scores.grad, **tol)
    for (name, left), (_, right) in zip(ref.named_parameters(), sparse.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(right.grad, left.grad, **tol, msg=name)
    for router in (sparse.read_router, sparse.write_router):
        assert torch.count_nonzero(router.backbone[0].weight.grad) > 0


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("kr,kw", [(0, 0), (0, 2), (2, 0), (1, 1), (2, 2), (4, 4)])
@pytest.mark.parametrize("rule", [None, "ste", "selected_only"])
def test_sparse_gru_external_masks_and_empty_budgets(batch, kr, kw, rule):
    torch.manual_seed(113)
    ref = DualReadWriteGRUCell(3, 4).double()
    sparse = SparseDualReadWriteGRUCell(3, 4).double()
    sparse.load_state_dict(ref.state_dict())
    x, h = torch.randn(batch, 3, dtype=torch.double), torch.randn(batch, 4, dtype=torch.double)
    read, write = torch.zeros_like(h), torch.zeros_like(h)
    for row in range(batch):
        read[row, (torch.arange(kr) + row) % 4] = 1
        write[row, (torch.arange(kw) + row + 1) % 4] = 1
    upstream = torch.randn_like(h)
    runs = []
    for cell in (ref, sparse):
        args = [t.clone().requires_grad_() for t in (x, h, read, write)]
        kwargs = dict(read_backward=rule, write_backward=rule) if cell is sparse else {}
        output, _ = cell(args[0], args[1], read_mask=args[2], write_mask=args[3], **kwargs)
        output.backward(upstream)
        assert torch.equal(output[write == 0], h[write == 0])
        runs.append((output, args))
    torch.testing.assert_close(runs[0][0], runs[1][0])
    for idx, (a, b) in enumerate(zip(runs[0][1], runs[1][1])):
        expected = a.grad
        if idx >= 2 and rule == "selected_only":
            expected = expected * (read if idx == 2 else write)
        torch.testing.assert_close(b.grad, expected)
    for a, b in zip(ref.parameters(), sparse.parameters()):
        torch.testing.assert_close(b.grad, a.grad)


@pytest.mark.parametrize("bias", [True, False])
def test_sparse_gru_fixed_mask_gradcheck(bias):
    torch.manual_seed(114)
    x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    h = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    wh = torch.randn(12, 4, dtype=torch.double, requires_grad=True)
    wx = torch.randn(12, 2, dtype=torch.double, requires_grad=True)
    bi = torch.randn(12, dtype=torch.double, requires_grad=True)
    bh = torch.randn(12, dtype=torch.double, requires_grad=True)
    ri, wi = torch.tensor([[0, 2], [1, 3]]), torch.tensor([[1, 2], [0, 1]])
    read, write = torch.zeros_like(h).scatter(1, ri, 1.), torch.zeros_like(h).scatter(1, wi, 1.)
    if bias:
        fn = lambda x, h, wh, wx, bi, bh: _GatheredGRUStep.apply(
            x, h, read, write, wh, wx, bi, bh, ri, wi, False, False)
        args = (x, h, wh, wx, bi, bh)
    else:
        fn = lambda x, h, wh, wx: _GatheredGRUStep.apply(
            x, h, read, write, wh, wx, None, None, ri, wi, False, False)
        args = (x, h, wh, wx)
    assert torch.autograd.gradcheck(fn, args)


def test_sparse_gru_preserves_unwritten_state_with_nonfinite_proposals():
    cell = SparseDualReadWriteGRUCell(2, 4)
    with torch.no_grad():
        cell.weight_ih.fill_(-3e38)
        cell.weight_hh.fill_(3e38)
    h = torch.ones(2, 4)
    write = torch.tensor([[0., 0., 1., 1.], [1., 1., 0., 0.]])
    result, _ = cell(torch.ones(2, 2), h, read_mask=torch.ones_like(h), write_mask=write)
    assert torch.equal(result[write == 0], h[write == 0])
    assert torch.isnan(result[write == 1]).all()


@pytest.mark.parametrize("rule", ["ste", "selected_only"])
def test_sparse_gru_matrix_shapes(rule):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Audit(TorchDispatchMode):
        def __init__(self):
            self.products = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default,
                        torch.ops.aten.addmm.default):
                self.products.append(tuple(tuple(x.shape) for x in args if isinstance(x, torch.Tensor)))
            return func(*args, **(kwargs or {}))

    h = torch.randn(2, 11, dtype=torch.double, requires_grad=True)
    read = torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]] * 2,
                        dtype=torch.double, requires_grad=True)
    write = torch.tensor([[0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.]] * 2,
                         dtype=torch.double, requires_grad=True)
    cell = SparseDualReadWriteGRUCell(3, 11).double()
    with Audit() as forward:
        result, _ = cell(torch.randn(2, 3, dtype=torch.double), h,
                         read_mask=read, write_mask=write, read_backward=rule, write_backward=rule)
    assert forward.products == [
        ((2, 2, 2), (2, 2, 1)), ((2, 2, 3), (2, 3, 1)),
        ((2, 3, 2), (2, 2, 1)), ((2, 3, 3), (2, 3, 1)),
        ((2, 3, 2), (2, 2, 1)), ((2, 3, 3), (2, 3, 1))]
    with Audit() as backward:
        result.sum().backward()
    for audit in (forward, backward):
        for shapes in audit.products:
            assert not any(shape[-2:] == (11, 11) for shape in shapes)
    if rule == "selected_only":
        assert not any(11 in shape for shapes in backward.products for shape in shapes)
