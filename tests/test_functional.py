"""Tests for dsrnn.nn.functional."""

import torch

from dsrnn.nn.functional import (GatheredRecurrentMatmul, budget_from_sparsity, dense_masked_matmul,
                                 gathered_recurrent_matmul, masked_matmul, topk_mask)


def make_inputs(B=2,
                H=6,
                O=6,
                k=3,
                seed=0):
    """Create random inputs for the masked matmul tests.

    Args:
        B: Batch size.
        H: Input size.
        O: Output size.
        k: Number of selected channels.
        seed: Random seed.

    Returns:
        A (x, mask, weight, idx) tuple of double-precision tensors.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(B, H, generator=generator, dtype=torch.double, requires_grad=True)
    weight = torch.randn(O, H, generator=generator, dtype=torch.double, requires_grad=True)
    scores = torch.rand(B, H, generator=generator, dtype=torch.double)
    idx = scores.topk(k, dim=-1).indices
    mask = torch.zeros(B, H, dtype=torch.double)
    mask.scatter_(1, idx, scores.gather(1, idx))
    mask.requires_grad_(True)
    return x, mask, weight, idx


def dense_reference(x, mask, weight):
    """Compute the dense masked matmul reference."""
    return (mask * x) @ weight.T


def test_forward_matches_dense():
    """Gathered forward matches the dense reference."""
    x, mask, weight, idx = make_inputs()
    output = GatheredRecurrentMatmul.apply(x, mask, weight, idx)
    assert torch.allclose(output, dense_reference(x, mask, weight), atol=1e-10)


def test_backward_matches_dense_autograd():
    """Gathered backward matches dense autograd for x, mask, and weight."""
    x, mask, weight, idx = make_inputs()
    x_ref = x.detach().clone().requires_grad_()
    mask_ref = mask.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    grad_output = torch.randn(x.size(0), weight.size(0), dtype=torch.double)

    GatheredRecurrentMatmul.apply(x, mask, weight, idx).backward(grad_output)
    dense_reference(x_ref, mask_ref, weight_ref).backward(grad_output)

    assert torch.allclose(x.grad, x_ref.grad, atol=1e-9)
    assert torch.allclose(mask.grad, mask_ref.grad, atol=1e-9)
    assert torch.allclose(weight.grad, weight_ref.grad, atol=1e-9)


def test_unselected_mask_entries_get_gradient():
    """Unselected mask entries receive gradient through the custom backward."""
    x, mask, weight, idx = make_inputs(H=8, k=3)
    GatheredRecurrentMatmul.apply(x, mask, weight, idx).sum().backward()
    unselected = torch.ones_like(mask, dtype=torch.bool).scatter_(1, idx, False)
    assert (mask.grad[unselected] != 0).any()


def test_gradcheck_x_and_weight():
    """Gradcheck passes for x and weight with a fixed mask."""
    x, mask, weight, idx = make_inputs(B=1, H=5, O=4, k=2)
    torch.autograd.gradcheck(
        lambda x_, weight_: GatheredRecurrentMatmul.apply(x_, mask.detach(), weight_, idx),
        (x, weight),
        eps=1e-6,
        atol=1e-4,
    )


def test_ste_reaches_all_scores():
    """STE backward gives gradient to every router score."""
    scores = torch.rand(4, 10, dtype=torch.double, requires_grad=True)
    mask, idx = topk_mask(scores, k=3, backward="ste")
    expected = torch.zeros_like(scores).scatter(1, idx, scores.detach().gather(1, idx))
    assert torch.allclose(mask.detach(), expected)
    mask.sum().backward()
    assert (scores.grad != 0).all()


def test_selected_only_reaches_selected_scores():
    """Selected-only backward gives gradient only to selected scores."""
    scores = torch.rand(4, 10, dtype=torch.double, requires_grad=True)
    mask, idx = topk_mask(scores, k=3, backward="selected_only")
    mask.sum().backward()
    selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)
    assert (scores.grad[selected] != 0).all()
    assert (scores.grad[~selected] == 0).all()


def test_gathered_and_dense_match_for_both_rules():
    """Gathered and dense paths produce identical gradients for both rules."""
    for backward in ("ste", "selected_only"):
        torch.manual_seed(0)
        batch, hidden, output, k = 3, 8, 5, 2
        grad_output = torch.randn(batch, output, dtype=torch.double)

        x_g = torch.randn(batch, hidden, dtype=torch.double, requires_grad=True)
        weight_g = torch.randn(output, hidden, dtype=torch.double, requires_grad=True)
        scores_g = torch.rand(batch, hidden, dtype=torch.double, requires_grad=True)
        mask_g, idx_g = topk_mask(scores_g, k, backward=backward)
        gathered_recurrent_matmul(x_g, mask_g, weight_g, idx_g).backward(grad_output)

        x_d = x_g.detach().clone().requires_grad_()
        weight_d = weight_g.detach().clone().requires_grad_()
        scores_d = scores_g.detach().clone().requires_grad_()
        mask_d, _ = topk_mask(scores_d, k, backward=backward)
        dense_masked_matmul(x_d, mask_d, weight_d).backward(grad_output)

        assert torch.allclose(x_g.grad, x_d.grad, atol=1e-9)
        assert torch.allclose(weight_g.grad, weight_d.grad, atol=1e-9)
        assert torch.allclose(scores_g.grad, scores_d.grad, atol=1e-9)


def test_masked_matmul_dispatch():
    """The dispatcher produces identical gathered and dense outputs."""
    x, mask, weight, idx = make_inputs()
    gathered = masked_matmul(x, mask, weight, idx=idx, implementation="gathered")
    dense = masked_matmul(x, mask, weight, implementation="dense")
    assert torch.allclose(gathered, dense, atol=1e-10)

    bias = torch.randn(weight.size(0), dtype=torch.double)
    output = masked_matmul(x, mask, weight, idx=idx, bias=bias, implementation="gathered")
    assert torch.allclose(output, gathered + bias, atol=1e-10)


def test_budget_rounding():
    """Budget rounding modes behave as documented."""
    assert budget_from_sparsity(64, 0.9, "floor") == 6
    assert budget_from_sparsity(64, 0.9, "ceil") == 7
    assert budget_from_sparsity(64, 0.9, "round") == 6
    assert budget_from_sparsity(10, 0.0, "floor") == 10
    assert budget_from_sparsity(10, 0.95, "floor") == 1


def main():
    """Run all tests."""
    test_forward_matches_dense()
    test_backward_matches_dense_autograd()
    test_unselected_mask_entries_get_gradient()
    test_gradcheck_x_and_weight()
    test_ste_reaches_all_scores()
    test_selected_only_reaches_selected_scores()
    test_gathered_and_dense_match_for_both_rules()
    test_masked_matmul_dispatch()
    test_budget_rounding()


if __name__ == "__main__":
    main()
