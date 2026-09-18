"""Independent hard READ/WRITE routing and legacy-default regressions."""

import pytest
import torch
from copy import deepcopy

from dsrnn.nn.cells.dual_read_write import DualReadWriteRNNCell
from dsrnn.nn.functional import topk_mask
from dsrnn.nn.router import Router, RouterConfig


def make_cell(backward="ste", k_read=2, k_write=3):
    return DualReadWriteRNNCell(
        3, 6,
        read_router_config=RouterConfig(6, 3, k=k_read, backward=backward),
        write_router_config=RouterConfig(6, 3, k=k_write, backward=backward))


@pytest.mark.parametrize("backward", ["ste", "selected_only"])
@pytest.mark.parametrize("k", [1, 3, 6])
def test_binary_topk_surrogate(backward, k):
    scores = torch.tensor([[.1, .4, .3, .9, .7, .6], [.8, .2, .6, .1, .3, .5]],
                          dtype=torch.double, requires_grad=True)
    mask, idx = topk_mask(scores, k, backward, binary=True)
    expected = torch.zeros_like(scores).scatter(1, idx, 1.)
    assert torch.equal(mask, expected)
    assert torch.equal(mask.sum(1), torch.full((2,), float(k), dtype=mask.dtype))
    upstream = torch.arange(1, 13, dtype=torch.double).reshape(2, 6)
    (mask * upstream).sum().backward()
    torch.testing.assert_close(scores.grad, upstream if backward == "ste" else upstream * expected)


@pytest.mark.parametrize("backward", ["ste", "selected_only"])
def test_legacy_router_defaults_unchanged(backward):
    config = RouterConfig(6, 3, sparsity=.6, backward=backward)
    router = Router(config).double()
    assert config.binary is False
    assert config.budget_rounding == "floor"
    assert router.k == 2
    result = router(torch.ones(4, 3, dtype=torch.double))
    expected, idx = topk_mask(result.scores, 2, backward)
    assert torch.equal(result.mask, expected)
    assert torch.equal(result.idx, idx)
    dense = Router(RouterConfig(6, 3, k=6, backward=backward))
    dense_result = dense(torch.ones(4, 3))
    assert torch.equal(dense_result.mask, torch.ones(4, 6))
    assert not dense_result.mask.requires_grad


def test_independent_fractional_budgets_and_config_copy():
    read_config = RouterConfig(6, 3, sparsity=.6)
    write_config = RouterConfig(6, 3, sparsity=.4)
    cell = DualReadWriteRNNCell(3, 6, read_router_config=read_config,
                               write_router_config=write_config)
    read, write = cell.compute_routing(torch.ones(4, 3))
    assert torch.equal(read.mask.sum(1), torch.full((4,), 3.))
    assert torch.equal(write.mask.sum(1), torch.full((4,), 4.))
    assert read_config.budget_rounding == write_config.budget_rounding == "floor"
    assert not read_config.binary and not write_config.binary
    assert set(map(id, cell.read_router.parameters())).isdisjoint(
        map(id, cell.write_router.parameters()))


def test_distinct_input_dependent_masks_and_actual_analysis():
    cell = make_cell(k_read=1, k_write=1).double()
    with torch.no_grad():
        for router in (cell.read_router, cell.write_router):
            router.backbone.weight.zero_()
            router.backbone.bias.zero_()
        cell.read_router.backbone.weight[0, 0] = 1
        cell.read_router.backbone.weight[1, 0] = -1
        cell.write_router.backbone.weight[2, 0] = 1
        cell.write_router.backbone.weight[3, 0] = -1
    x = torch.tensor([[1., 0., 0.], [-1., 0., 0.]], dtype=torch.double)
    h = torch.ones(2, 6, dtype=torch.double)
    out, _ = cell(x, h)
    read, write = cell.last_read_routing, cell.last_write_routing
    assert read.idx.tolist() == [[0], [1]]
    assert write.idx.tolist() == [[2], [3]]
    for routing, mask in ((read, cell.read_mask), (write, cell.write_mask)):
        assert routing.mask is mask
        assert routing.logits.shape == routing.scores.shape == (2, 6)
        assert torch.equal(mask, torch.zeros_like(mask).scatter(1, routing.idx, 1.))
    expected, _ = cell(x, h, read_mask=read.mask, write_mask=write.mask)
    assert torch.equal(out, expected)
    assert torch.equal(out[write.mask == 0], h[write.mask == 0])
    assert cell.last_read_routing is None and cell.last_write_routing is None
    cell(x, h * 7)
    assert torch.equal(cell.read_mask, read.mask)
    assert torch.equal(cell.write_mask, write.mask)
    cell(x, h, write_mask=torch.zeros_like(h))
    assert cell.last_read_routing is not None
    assert cell.last_write_routing is None


@pytest.mark.parametrize("backward", ["ste", "selected_only"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_routed_multistep_gradients(backward, batch_size, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this test process")
    torch.manual_seed(25)
    cell = make_cell(backward).to(device=device, dtype=torch.double)
    x = torch.randn(3, batch_size, 3, device=device, dtype=torch.double, requires_grad=True)
    h = torch.randn(batch_size, 6, device=device, dtype=torch.double, requires_grad=True)
    outputs, routing_history = [], []
    for t in range(3):
        previous = h
        h, _ = cell(x[t], h)
        assert h.shape == (batch_size, 6) and h.device == x.device
        assert torch.equal(h[cell.write_mask == 0], previous[cell.write_mask == 0])
        for result, k in ((cell.last_read_routing, 2), (cell.last_write_routing, 3)):
            assert torch.equal(result.mask.sum(1), h.new_full((batch_size,), k))
            assert result.idx.shape == (batch_size, k)
            result.scores.retain_grad()
            routing_history.append(result)
        outputs.append(h)
    torch.stack(outputs).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    for router in (cell.read_router, cell.write_router):
        for parameter in router.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert router.backbone.weight.grad.abs().sum() > 0
    for result in routing_history:
        selected = result.mask.bool()
        assert result.scores.grad[selected].abs().sum() > 0
        if backward == "selected_only":
            assert torch.count_nonzero(result.scores.grad[~selected]) == 0
        else:
            assert result.scores.grad[~selected].abs().sum() > 0


def test_projected_mlp_routers_and_full_budget():
    config = RouterConfig(6, 3, context_dim=2, n_layers=2, k=6)
    cell = DualReadWriteRNNCell(3, 6, read_router_config=config, write_router_config=config)
    read, write = cell.compute_routing(torch.ones(4, 3))
    assert torch.equal(read.mask, torch.ones(4, 6))
    assert torch.equal(write.mask, torch.ones(4, 6))
    (read.mask.sum() + write.mask.sum()).backward()
    for router in (cell.read_router, cell.write_router):
        assert router.context_proj.weight.grad is not None
        assert torch.isfinite(router.context_proj.weight.grad).all()


def test_router_configuration_errors():
    config = RouterConfig(6, 3)
    with pytest.raises(ValueError, match="both"):
        DualReadWriteRNNCell(3, 6, read_router_config=config)
    with pytest.raises(ValueError, match="input_dim"):
        DualReadWriteRNNCell(3, 6, read_router_config=config,
                             write_router_config=RouterConfig(6, 2))
    cell = DualReadWriteRNNCell(3, 6)
    with pytest.raises(ValueError, match="read_mask"):
        cell(torch.ones(4, 3))


@pytest.mark.parametrize("hidden,sparsity,expected", [(10, .7, 3), (100, .7, 30),
                                                     (10, .699999, 4)])
def test_integer_boundary_budgets(hidden, sparsity, expected):
    config = RouterConfig(hidden, 3, sparsity=sparsity)
    cell = DualReadWriteRNNCell(3, hidden, read_router_config=config,
                               write_router_config=config)
    for result in cell.compute_routing(torch.ones(2, 3)):
        assert torch.equal(result.mask.sum(1), torch.full((2,), float(expected)))
        assert result.idx.shape == (2, expected)
    assert config.k is None and config.budget_rounding == "floor"


@pytest.mark.parametrize("backward", ["ste", "selected_only"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_composed_router_gradients(backward, batch_size):
    """Check router and temporal-state gradients against analytical values."""
    torch.manual_seed(72)
    cell = make_cell(backward).double()
    x = torch.randn(4, batch_size, 3, dtype=torch.double, requires_grad=True)
    initial = torch.randn(batch_size, 6, dtype=torch.double, requires_grad=True)
    upstream = torch.randn(4, batch_size, 6, dtype=torch.double)
    h = initial
    outputs = []
    for xt in x:
        h, _ = cell(xt, h)
        outputs.append(h)
    (torch.stack(outputs) * upstream).sum().backward()

    with torch.no_grad():
        expected = {name: torch.zeros_like(p) for name, p in cell.named_parameters()}
        expected_x = torch.zeros_like(x)
        history = []
        h = initial.detach()
        for xt in x:
            routing = []
            for router in (cell.read_router, cell.write_router):
                logits = xt @ router.backbone.weight.T + router.backbone.bias
                tau = router.scaled_sigmoid.log_tau.exp()
                scores = torch.sigmoid(tau * (logits - router.scaled_sigmoid.mu))
                mask = torch.zeros_like(scores).scatter(1, scores.argsort(descending=True)[:, :router.k], 1.)
                routing.append((mask, scores, logits, tau))
            read, write = routing[0][0], routing[1][0]
            candidate = torch.tanh((read * h) @ cell.weight_hh.T + xt @ cell.weight_ih.T + cell.bias)
            history.append((h, candidate, routing))
            h = (1 - write) * h + write * candidate
        torch.testing.assert_close(h, outputs[-1])
        carry = torch.zeros_like(h)
        for t in reversed(range(len(x))):
            h, candidate, routing = history[t]
            read, write = routing[0][0], routing[1][0]
            g = upstream[t] + carry
            delta = g * write * (1 - candidate.square())
            gamma = delta @ cell.weight_hh
            expected["weight_hh"] += delta.T @ (read * h)
            expected["weight_ih"] += delta.T @ x[t]
            expected["bias"] += delta.sum(0)
            expected_x[t] = delta @ cell.weight_ih
            carry = g * (1 - write) + gamma * read
            for name, router, route, mask_grad in zip(
                    ("read_router", "write_router"), (cell.read_router, cell.write_router),
                    routing, (gamma * h, g * (candidate - h))):
                mask, scores, logits, tau = route
                ds = mask_grad if backward == "ste" else mask_grad * mask
                da = ds * scores * (1 - scores)
                dz = da * tau
                expected[name + ".backbone.weight"] += dz.T @ x[t]
                expected[name + ".backbone.bias"] += dz.sum(0)
                expected[name + ".scaled_sigmoid.log_tau"] += (dz * (logits - router.scaled_sigmoid.mu)).sum()
                expected[name + ".scaled_sigmoid.mu"] -= dz.sum()
                expected_x[t] += dz @ router.backbone.weight
        torch.testing.assert_close(x.grad, expected_x)
        torch.testing.assert_close(initial.grad, carry)
        for name, parameter in cell.named_parameters():
            torch.testing.assert_close(parameter.grad, expected[name], msg=name)


@pytest.mark.parametrize("backward", ["ste", "selected_only"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_cpu_gpu_equivalence(backward, batch_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for cross-device equivalence")
    torch.manual_seed(73)
    cpu = make_cell(backward).double()
    gpu = deepcopy(cpu).cuda()
    x_cpu = torch.randn(5, batch_size, 3, dtype=torch.double, requires_grad=True)
    h_cpu = torch.randn(batch_size, 6, dtype=torch.double, requires_grad=True)
    x_gpu = x_cpu.detach().cuda().requires_grad_()
    h_gpu = h_cpu.detach().cuda().requires_grad_()
    upstream = torch.randn(5, batch_size, 6, dtype=torch.double)
    results = []
    for cell, x, initial in ((cpu, x_cpu, h_cpu), (gpu, x_gpu, h_gpu)):
        h = initial
        outputs, masks = [], []
        for xt in x:
            h, _ = cell(xt, h)
            outputs.append(h)
            masks.append(torch.stack((cell.read_mask, cell.write_mask)))
        outputs = torch.stack(outputs)
        (outputs * upstream.to(x.device)).sum().backward()
        results.append((outputs.detach().cpu(), torch.stack(masks).detach().cpu()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-9, atol=1e-11)
    assert torch.equal(results[0][1], results[1][1])
    for a, b in [(x_cpu, x_gpu), (h_cpu, h_gpu), *zip(cpu.parameters(), gpu.parameters())]:
        torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=1e-9, atol=1e-11)
