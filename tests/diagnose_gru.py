"""Pinpoint where DSGRUCell.forward_active diverges from the generic forward.

Runs a single step through both paths and a dense manual reference computed
from the cell's own weights, then reports per-gate discrepancies.
"""

import torch
import torch.nn.functional as F

from dsrnn.nn.cells.dsgru import DSGRUCell
from dsrnn.nn.router import RouterConfig

B, T, D, H = 4, 7, 8, 16
SPARSITY = 0.9


def dense_gru_step(x, h, cell, gate_order):
    """Standard GRU step with full weights. gate_order: 'rzn' or 'zrn'."""
    gi = F.linear(x, cell.weight_ih, cell.bias_ih)
    gh = F.linear(h, cell.weight_hh, cell.bias_hh)
    i1, i2, i3 = gi.chunk(3, dim=-1)
    h1, h2, h3 = gh.chunk(3, dim=-1)
    if gate_order == "rzn":
        r = torch.sigmoid(i1 + h1)
        z = torch.sigmoid(i2 + h2)
        n = torch.tanh(i3 + r * h3)
    else:
        z = torch.sigmoid(i1 + h1)
        r = torch.sigmoid(i2 + h2)
        n = torch.tanh(i3 + r * h3)
    return (1 - z) * n + z * h


def main():
    torch.manual_seed(0)
    router_config = RouterConfig(num_channels=H, input_dim=D, context_dim=4,
                                 sparsity=SPARSITY, backward="selected_only")
    cell = DSGRUCell(input_size=D, hidden_size=H, router_config=router_config,
                     implementation="gathered").double()
    torch.manual_seed(1)
    x = torch.randn(B, D, dtype=torch.double)
    h0 = torch.zeros(B, H, dtype=torch.double)

    input_routing, hidden_routing = cell.compute_routing_batched(x)
    mask_x, idx_x = input_routing.mask, input_routing.idx
    mask_h, idx_h = hidden_routing.mask, hidden_routing.idx
    x_active = torch.gather(x, dim=-1, index=idx_x) * torch.gather(mask_x, dim=-1, index=idx_x)

    h_g, _ = cell(x, h0, routing=((mask_x, idx_x), (mask_h, idx_h)))
    h_a, _ = cell.forward_active(x_active, idx_x, h0, mask_h, idx_h)

    # Dense reference on masked inputs (unmasked entries contribute 0).
    x_masked = x * mask_x
    h_masked = h0 * mask_h  # h0 is zero here, but keep for generality
    for order in ("rzn", "zrn"):
        h_ref = dense_gru_step(x_masked, h_masked, cell, order)
        print(f"dense[{order}] vs generic: {(h_ref - h_g).abs().max().item():.3e}")
        print(f"dense[{order}] vs active:  {(h_ref - h_a).abs().max().item():.3e}")

    diff = (h_g - h_a).abs()
    print(f"\ngeneric vs active: max={diff.max().item():.3e}, "
          f"mean={diff.mean().item():.3e}, "
          f"num_mismatch={(diff > 1e-9).sum().item()}/{diff.numel()}")
    print("per-hidden-unit max diff:\n", diff.max(dim=0).values)

    # Decompose: does the discrepancy come from the candidate gate?
    # Recompute generic with h0 = random nonzero to expose reset-gate handling.
    torch.manual_seed(7)
    h0b = torch.randn(B, H, dtype=torch.double)
    h_g2, _ = cell(x, h0b, routing=((mask_x, idx_x), (mask_h, idx_h)))
    h_a2, _ = cell.forward_active(x_active, idx_x, h0b, mask_h, idx_h)
    print(f"\nnonzero h0: generic vs active max diff = "
          f"{(h_g2 - h_a2).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
