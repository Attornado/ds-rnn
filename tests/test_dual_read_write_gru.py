"""Semantic, equivalence, and gradient tests for the dual READ/WRITE GRU."""

import pytest
import torch

from dsrnn.nn.cells.dual_read_write_gru import DualReadWriteGRUCell
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
