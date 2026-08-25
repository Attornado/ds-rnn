"""Equivalence tests between generic cell forward and forward_active."""

import torch

from dsrnn.models.dsgru_model import DSGRU
from dsrnn.models.dslstm_model import DSLSTM
from dsrnn.nn.cells.dsgru import DSGRUCell
from dsrnn.nn.cells.dslstm import DSLSTMCell
from dsrnn.nn.router import RouterConfig

B, T, D, H = 4, 7, 8, 16
SPARSITY = 0.9
ATOL = 1e-9


def make_cell(cell_cls):
    torch.manual_seed(0)
    router_config = RouterConfig(num_channels=H, input_dim=D, context_dim=4,
                                 sparsity=SPARSITY, backward="selected_only")
    return cell_cls(input_size=D, hidden_size=H, router_config=router_config,
                    implementation="gathered")


def route_sequence(cell, x):
    """Route a time-major input, returning time-major masks and indices."""
    flat = x.reshape(T * B, D)
    input_routing, hidden_routing = cell.compute_routing_batched(flat)
    return (input_routing.mask.view(T, B, D),
            input_routing.idx.view(T, B, -1),
            hidden_routing.mask.view(T, B, H),
            hidden_routing.idx.view(T, B, -1))


def test_dslstm_active_matches_generic():
    cell = make_cell(DSLSTMCell)
    torch.manual_seed(1)
    x = torch.randn(B, T, D, dtype=torch.double).transpose(0, 1)
    x = x.detach().requires_grad_(True)
    cell = cell.double()
    mask_x, idx_x, mask_h, idx_h = route_sequence(cell, x)
    x_active = torch.gather(x, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)

    h_g, c_g = x.new_zeros(B, H), x.new_zeros(B, H)
    h_a, c_a = h_g.clone(), c_g.clone()
    for t in range(T):
        h_g, (h_g, c_g) = cell(x[t], (h_g, c_g),
                               routing=((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t])))
        h_a, (h_a, c_a) = cell.forward_active(x_active[t], idx_x[t], h_a, c_a, mask_h[t], idx_h[t])

    assert torch.allclose(h_g, h_a, atol=ATOL)
    assert torch.allclose(c_g, c_a, atol=ATOL)


def test_dsgru_active_matches_generic():
    cell = make_cell(DSGRUCell)
    torch.manual_seed(1)
    x = torch.randn(B, T, D, dtype=torch.double).transpose(0, 1)
    x = x.detach().requires_grad_(True)
    cell = cell.double()
    mask_x, idx_x, mask_h, idx_h = route_sequence(cell, x)
    x_active = torch.gather(x, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)

    h_g, h_a = x.new_zeros(B, H), x.new_zeros(B, H)
    for t in range(T):
        h_g, _ = cell(x[t], h_g, routing=((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t])))
        h_a, _ = cell.forward_active(x_active[t], idx_x[t], h_a, mask_h[t], idx_h[t])

    assert torch.allclose(h_g, h_a, atol=ATOL)


def test_dslstm_active_backward_matches_generic():
    cell = make_cell(DSLSTMCell).double()
    torch.manual_seed(1)
    x = torch.randn(B, T, D, dtype=torch.double).transpose(0, 1).detach().requires_grad_(True)

    # Generic path.
    mask_x, idx_x, mask_h, idx_h = route_sequence(cell, x)
    h, c = x.new_zeros(B, H), x.new_zeros(B, H)
    out_g = []
    for t in range(T):
        h, (h, c) = cell(x[t], (h, c), routing=((mask_x[t], idx_x[t]), (mask_h[t], idx_h[t])))
        out_g.append(h)
    loss_g = torch.stack(out_g, dim=0).sum()
    loss_g.backward()
    x_grad_g = x.grad.clone()
    w_grad_g = cell.weight_hh.grad.clone()

    # Active path on fresh graph.
    x2 = x.detach().clone().requires_grad_(True)
    cell.zero_grad(set_to_none=True)
    mask_x, idx_x, mask_h, idx_h = route_sequence(cell, x2)
    x_active = torch.gather(x2, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)
    h, c = x2.new_zeros(B, H), x2.new_zeros(B, H)
    out_a = []
    for t in range(T):
        h, (h, c) = cell.forward_active(x_active[t], idx_x[t], h, c, mask_h[t], idx_h[t])
        out_a.append(h)
    loss_a = torch.stack(out_a, dim=0).sum()
    loss_a.backward()

    assert torch.allclose(loss_g, loss_a, atol=ATOL)
    assert torch.allclose(x_grad_g, x2.grad, atol=ATOL)
    assert torch.allclose(w_grad_g, cell.weight_hh.grad, atol=ATOL)


def test_sequence_models_match_cell_loop():
    for model_cls, cell_cls in ((DSLSTM, DSLSTMCell), (DSGRU, DSGRUCell)):
        cell = make_cell(cell_cls).double()
        torch.manual_seed(1)
        model = model_cls(input_size=D, hidden_size=H, router_config=cell.hidden_router.config,
                          implementation="gathered").double()
        model.cell.load_state_dict(cell.state_dict())

        torch.manual_seed(2)
        x = torch.randn(B, T, D, dtype=torch.double)
        out_m, state_m = model(x)

        # Manual loop through the cell with precomputed routing.
        xt = x.transpose(0, 1)
        mask_x, idx_x, mask_h, idx_h = route_sequence(cell, xt)
        x_active = torch.gather(xt, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)
        h, c = x.new_zeros(B, H), x.new_zeros(B, H)
        outputs = []
        for t in range(T):
            if model_cls is DSLSTM:
                h, (h, c) = cell.forward_active(x_active[t], idx_x[t], h, c, mask_h[t], idx_h[t])
            else:
                h, _ = cell.forward_active(x_active[t], idx_x[t], h, mask_h[t], idx_h[t])
            outputs.append(h)
        out_ref = torch.stack(outputs, dim=0).transpose(0, 1)
        assert torch.allclose(out_m, out_ref, atol=ATOL)


def main():
    test_dslstm_active_matches_generic()
    test_dsgru_active_matches_generic()
    test_dslstm_active_backward_matches_generic()
    test_sequence_models_match_cell_loop()
    print("All model equivalence tests passed.")


if __name__ == "__main__":
    main()
