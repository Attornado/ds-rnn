"""Equivalence tests between generic and specialized DS cell paths."""

import torch

from dsrnn.nn.cells.dsgru import DSGRUCell
from dsrnn.nn.cells.dslstm import DSLSTMCell
from dsrnn.nn.cells.dsltc import DSLTCCell
from dsrnn.nn.router import RouterConfig


def _config(num_channels, input_dim, backward="selected_only", sparsity=0.9, seed=0):
    torch.manual_seed(seed)
    return RouterConfig(num_channels=num_channels,
                        input_dim=input_dim,
                        context_dim=4,
                        sparsity=sparsity,
                        backward=backward)


def test_dslstm_active_matches_generic():
    """forward_active matches the generic gathered path, forward and backward."""
    torch.manual_seed(0)
    B, T, D, H = 4, 7, 8, 16
    cell = DSLSTMCell(input_size=D,
                      hidden_size=H,
                      router_config=_config(H, D),
                      implementation="gathered")
    x = torch.randn(B, T, D, dtype=torch.double)
    cell = cell.double()

    flat = x.reshape(B * T, D)
    input_routing, hidden_routing = cell.compute_routing_batched(flat)
    mask_x = input_routing.mask.view(T, B, D)
    idx_x = input_routing.idx.view(T, B, -1)
    mask_h = hidden_routing.mask.view(T, B, H)
    idx_h = hidden_routing.idx.view(T, B, -1)

    h = x.new_zeros(B, H)
    c = x.new_zeros(B, H)
    outputs_generic = []
    for t in range(T):
        h, (h, c) = cell(x[:, t], (h, c),
                         routing=((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t])))
        outputs_generic.append(h)
    out_generic = torch.stack(outputs_generic, dim=1)

    x_active = torch.gather(x, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)
    h = x.new_zeros(B, H)
    c = x.new_zeros(B, H)
    outputs_active = []
    for t in range(T):
        h, (h, c) = cell.forward_active(x_active[t], idx_x[t], h, c, mask_h[t], idx_h[t])
        outputs_active.append(h)
    out_active = torch.stack(outputs_active, dim=1)

    assert torch.allclose(out_generic, out_active, atol=1e-10)


def test_dslstm_active_backward_matches_generic():
    """Gradients of forward_active match the generic gathered selected-only path."""
    torch.manual_seed(0)
    B, T, D, H = 4, 5, 8, 16
    cell = DSLSTMCell(input_size=D,
                      hidden_size=H,
                      router_config=_config(H, D),
                      implementation="gathered").double()
    x = torch.randn(B, T, D, dtype=torch.double, requires_grad=True)

    flat = x.reshape(B * T, D)
    input_routing, hidden_routing = cell.compute_routing_batched(flat)
    mask_x = input_routing.mask.view(T, B, D)
    idx_x = input_routing.idx.view(T, B, -1)
    mask_h = hidden_routing.mask.view(T, B, H)
    idx_h = hidden_routing.idx.view(T, B, -1)

    grad_out = torch.randn(B, H, dtype=torch.double)

    h = x.new_zeros(B, H)
    c = x.new_zeros(B, H)
    for t in range(T):
        h, (h, c) = cell(x[:, t], (h, c),
                         routing=((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t])))
    grads_generic = torch.autograd.grad(h.sum() + (grad_out * h).sum(), cell.parameters(),
                                        retain_graph=False, allow_unused=False)
    x.grad = None

    x_active = torch.gather(x, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)
    h = x.new_zeros(B, H)
    c = x.new_zeros(B, H)
    for t in range(T):
        h, (h, c) = cell.forward_active(x_active[t], idx_x[t], h, c, mask_h[t], idx_h[t])
    grads_active = torch.autograd.grad(h.sum() + (grad_out * h).sum(), cell.parameters(),
                                       retain_graph=False, allow_unused=False)

    for g_generic, g_active in zip(grads_generic, grads_active):
        assert torch.allclose(g_generic, g_active, atol=1e-9)


def test_dsgru_active_matches_generic():
    """DSGRU forward_active matches the generic gathered path."""
    torch.manual_seed(0)
    B, T, D, H = 4, 7, 8, 16
    cell = DSGRUCell(input_size=D,
                     hidden_size=H,
                     router_config=_config(H, D),
                     implementation="gathered").double()
    x = torch.randn(B, T, D, dtype=torch.double)

    flat = x.reshape(B * T, D)
    input_routing, hidden_routing = cell.compute_routing_batched(flat)
    mask_x = input_routing.mask.view(T, B, D)
    idx_x = input_routing.idx.view(T, B, -1)
    mask_h = hidden_routing.mask.view(T, B, H)
    idx_h = hidden_routing.idx.view(T, B, -1)

    h = x.new_zeros(B, H)
    outputs_generic = []
    for t in range(T):
        h, _ = cell(x[:, t], h, routing=((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t])))
        outputs_generic.append(h)
    out_generic = torch.stack(outputs_generic, dim=1)

    x_active = torch.gather(x, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)
    h = x.new_zeros(B, H)
    outputs_active = []
    for t in range(T):
        h, _ = cell.forward_active(x_active[t], idx_x[t], h, mask_h[t], idx_h[t])
        outputs_active.append(h)
    out_active = torch.stack(outputs_active, dim=1)

    assert torch.allclose(out_generic, out_active, atol=1e-10)


def test_dsltc_gathered_matches_dense():
    """Gathered DSLTC matches the dense path on identical masks."""
    torch.manual_seed(0)
    B, T, D, H = 4, 5, 6, 16
    config = _config(H, D)
    cell_dense = DSLTCCell(input_size=D,
                           hidden_size=H,
                           router_config=config,
                           implementation="dense").double()
    cell_gathered = DSLTCCell(input_size=D,
                              hidden_size=H,
                              router_config=_config(H, D, seed=0),
                              implementation="gathered").double()
    cell_gathered.load_state_dict(cell_dense.state_dict())

    x = torch.randn(B, T, D, dtype=torch.double)
    flat = x.reshape(B * T, D)
    input_routing, hidden_routing = cell_dense.compute_routing_batched(flat)
    mask_x = input_routing.mask.view(T, B, -1)
    idx_x = input_routing.idx.view(T, B, -1)
    mask_h = hidden_routing.mask.view(T, B, -1)
    idx_h = hidden_routing.idx.view(T, B, -1)

    state_dense = x.new_zeros(B, H)
    state_gathered = x.new_zeros(B, H)
    for t in range(T):
        routing_t = ((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t]))
        _, state_dense = cell_dense(x[:, t], state_dense, routing=routing_t)
        _, state_gathered = cell_gathered(x[:, t], state_gathered, routing=routing_t)

    assert torch.allclose(state_dense, state_gathered, atol=1e-9)


def test_dsltc_gathered_mask_receives_gradient():
    """Selected hidden channels receive gradient in the gathered LTC path."""
    torch.manual_seed(0)
    B, D, H = 4, 6, 16
    cell = DSLTCCell(input_size=D,
                     hidden_size=H,
                     router_config=_config(H, D),
                     implementation="gathered").double()
    x = torch.randn(B, D, dtype=torch.double)
    state = x.new_zeros(B, H)

    _, state = cell(x, state)
    state.sum().backward()
    router_grads = [p.grad for p in cell.hidden_router.parameters() if p.grad is not None]
    assert len(router_grads) > 0
    assert any(torch.any(g != 0) for g in router_grads)


def main():
    test_dslstm_active_matches_generic()
    test_dslstm_active_backward_matches_generic()
    test_dsgru_active_matches_generic()
    test_dsltc_gathered_matches_dense()
    test_dsltc_gathered_mask_receives_gradient()
    print("All model equivalence tests passed.")


if __name__ == "__main__":
    main()
