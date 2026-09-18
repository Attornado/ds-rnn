"""Semantic and gradient tests for the dense dual READ/WRITE reference."""

import pytest
import torch

from dsrnn.nn.cells.dual_read_write import DualReadWriteRNNCell


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_reference_equation(batch_size, bias):
    torch.manual_seed(10)
    cell = DualReadWriteRNNCell(2, 4, bias=bias).double()
    x = torch.randn(batch_size, 2, dtype=torch.double)
    h = torch.randn(batch_size, 4, dtype=torch.double)
    read = torch.rand_like(h)
    write = torch.rand_like(h)
    candidate = torch.tanh((read * h) @ cell.weight_hh.T
                           + x @ cell.weight_ih.T
                           + (cell.bias if bias else 0))
    out, state = cell(x, h, read_mask=read, write_mask=write)
    torch.testing.assert_close(out, (1 - write) * h + write * candidate)
    assert out is state
    assert out.shape == (batch_size, 4)


def test_all_four_states_and_read_isolation():
    cell = DualReadWriteRNNCell(2, 4).double()
    with torch.no_grad():
        cell.weight_ih.zero_()
        cell.weight_hh.fill_(0.1)
    h = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]], dtype=torch.double)
    read = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]], dtype=torch.double)
    write = torch.tensor([[0., 0., 1., 1.], [1., 1., 0., 0.]], dtype=torch.double)
    x = torch.zeros(2, 2, dtype=torch.double)
    h_before = h.clone()
    out, _ = cell(x, h, read_mask=read, write_mask=write)
    candidate = torch.tanh(0.1 * (read * h).sum(-1, keepdim=True)).expand_as(h)
    assert torch.equal(out[write == 0], h[write == 0])
    torch.testing.assert_close(out[write == 1], candidate[write == 1])
    assert torch.equal(h, h_before)

    # Unread state cannot affect any written candidate, including its own.
    changed = h + 10 * (1 - read)
    out_changed, _ = cell(x, changed, read_mask=read, write_mask=write)
    assert torch.equal(out_changed[write == 1], out[write == 1])

    # Protected but readable state must still influence other written units.
    changed = h + ((read == 1) & (write == 0)).to(h)
    out_changed, _ = cell(x, changed, read_mask=read, write_mask=write)
    assert torch.all(out_changed[write == 1] != out[write == 1])


@pytest.mark.parametrize("read_value", [0., 1.])
@pytest.mark.parametrize("write_value", [0., 1.])
def test_uniform_mask_edges(read_value, write_value):
    torch.manual_seed(11)
    cell = DualReadWriteRNNCell(2, 4).double()
    x = torch.randn(3, 2, dtype=torch.double)
    h = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    out, _ = cell(x, h, read_mask=torch.full_like(h, read_value),
                  write_mask=torch.full_like(h, write_value))
    if write_value == 0:
        assert torch.equal(out, h)
        out.sum().backward()
        assert torch.equal(h.grad, torch.ones_like(h))
        for parameter in cell.parameters():
            assert torch.count_nonzero(parameter.grad) == 0
    else:
        expected = torch.tanh(x @ cell.weight_ih.T + cell.bias
                              + read_value * (h @ cell.weight_hh.T))
        torch.testing.assert_close(out, expected)


def test_zero_initial_state():
    cell = DualReadWriteRNNCell(2, 4).double()
    x = torch.ones(3, 2, dtype=torch.double)
    read = torch.zeros(3, 4, dtype=torch.double)
    write = torch.ones_like(read)
    out, _ = cell(x, read_mask=read, write_mask=write)
    expected, _ = cell(x, torch.zeros_like(read), read_mask=read, write_mask=write)
    assert torch.equal(out, expected)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_nonfinite_candidate_preserves_zero_writes():
    """Overflow must not corrupt protected state; active writes propagate NaN."""
    cell = DualReadWriteRNNCell(2, 2)
    with torch.no_grad():
        cell.weight_hh.fill_(3e38)
        cell.weight_ih.fill_(-3e38)
    x = torch.ones(2, 2)
    h = torch.ones(2, 2)
    write = torch.tensor([[0., 0.], [0., 1.]])
    out, _ = cell(x, h, read_mask=torch.ones_like(h), write_mask=write)
    assert torch.equal(out[write == 0], h[write == 0])
    assert torch.isnan(out[write == 1]).all()


def test_reference_gradients():
    """Check analytical gradients, including mask derivatives at zero writes."""
    torch.manual_seed(12)
    cell = DualReadWriteRNNCell(2, 4).double()
    x = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
    h = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    read = torch.tensor([[0., 1., .3, 0.]] * 3, dtype=torch.double, requires_grad=True)
    write = torch.tensor([[1., 0., .7, 0.]] * 3, dtype=torch.double, requires_grad=True)
    upstream = torch.randn_like(h)
    out, _ = cell(x, h, read_mask=read, write_mask=write)
    out.backward(upstream)
    with torch.no_grad():
        candidate = torch.tanh((read * h) @ cell.weight_hh.T
                               + x @ cell.weight_ih.T + cell.bias)
        delta = upstream * write * (1 - candidate.square())
        gamma = delta @ cell.weight_hh
        torch.testing.assert_close(cell.weight_hh.grad, delta.T @ (read * h))
        torch.testing.assert_close(cell.weight_ih.grad, delta.T @ x)
        torch.testing.assert_close(cell.bias.grad, delta.sum(0))
        torch.testing.assert_close(h.grad, upstream * (1 - write) + gamma * read)
        torch.testing.assert_close(x.grad, delta @ cell.weight_ih)
        torch.testing.assert_close(read.grad, gamma * h)
        torch.testing.assert_close(write.grad, upstream * (candidate - h))
        assert torch.count_nonzero(read.grad) > 0
        assert torch.count_nonzero(write.grad[write == 0]) > 0


def test_hidden_jacobian():
    torch.manual_seed(13)
    cell = DualReadWriteRNNCell(2, 4).double()
    x = torch.randn(1, 2, dtype=torch.double)
    h = torch.randn(4, dtype=torch.double)
    read = torch.tensor([[0., 1., 0., 1.]], dtype=torch.double)
    write = torch.tensor([[0., 0., 1., 1.]], dtype=torch.double)
    jacobian = torch.autograd.functional.jacobian(
        lambda previous: cell(x, previous[None], read_mask=read, write_mask=write)[0][0], h)
    candidate = torch.tanh((read * h) @ cell.weight_hh.T + x @ cell.weight_ih.T + cell.bias)[0]
    expected = (torch.diag(1 - write[0])
                + torch.diag(write[0] * (1 - candidate.square()))
                @ cell.weight_hh @ torch.diag(read[0]))
    torch.testing.assert_close(jacobian, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_multistep_forward_backward_smoke(device, batch_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(14)
    cell = DualReadWriteRNNCell(2, 4).to(device)
    x = torch.randn(5, batch_size, 2, device=device, requires_grad=True)
    initial = torch.randn(batch_size, 4, device=device, requires_grad=True)
    read = torch.rand(5, batch_size, 4, device=device, requires_grad=True)
    write = torch.rand(5, batch_size, 4, device=device)
    write[:, :, 0] = 0  # Protect this coordinate throughout the sequence.
    write.requires_grad_()
    h = initial
    expected = initial.detach().clone()
    outputs = []
    for t in range(5):
        h, state = cell(x[t], h, read_mask=read[t], write_mask=write[t])
        with torch.no_grad():
            candidate = torch.tanh((read[t] * expected) @ cell.weight_hh.T
                                   + x[t] @ cell.weight_ih.T + cell.bias)
            expected = (1 - write[t]) * expected + write[t] * candidate
        torch.testing.assert_close(h, expected)
        assert h is state
        assert h.shape == (batch_size, 4)
        assert h.device == x.device
        assert torch.equal(h[:, 0], initial[:, 0])
        outputs.append(h)
    torch.stack(outputs).square().mean().backward()
    for tensor in (x, initial, read, write, *cell.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


@pytest.mark.parametrize("name", ["hx", "read_mask", "write_mask"])
def test_rejects_broadcast_state_or_masks(name):
    cell = DualReadWriteRNNCell(2, 4)
    kwargs = {"hx": torch.zeros(3, 4), "read_mask": torch.ones(3, 4),
              "write_mask": torch.ones(3, 4)}
    kwargs[name] = torch.ones(4)
    with pytest.raises(ValueError, match=name):
        cell(torch.zeros(3, 2), **kwargs)
