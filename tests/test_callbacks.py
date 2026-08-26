"""Tests for the analysis callbacks (WW + sparsity monitors).

Optional dependencies are skipped if absent (weightwatcher, networkx, imageio).
Runs on random tensors, CPU, no clutter.
"""

import torch

from dsrnn.nn.router import RouterConfig
from dsrnn.training.sequence import SequenceModule

B, T, D, H, C = 8, 10, 12, 16, 4


def _ds_module():
    return SequenceModule(
        model_type="dslstm", input_size=D, hidden_size=H, output_size=C,
        return_sequences=False, loss_type="ce",
        model_kwargs={"router_config": RouterConfig(
            num_channels=H, input_dim=D, context_dim=4, sparsity=0.9,
            backward="selected_only")},
        metric_params={"accuracy": {"task": "multiclass", "num_classes": C,
                                    "average": "macro"}},
        test_with_corruptions=False,
    )


def test_graph_metrics_smoke():
    import pytest
    pytest.importorskip("networkx")
    from dsrnn.utils.graphmetrics import compute_graph_metrics
    adj = torch.rand(2, 8, 8).clamp(min=0)
    out = compute_graph_metrics(adj, advanced=False)
    for key in ("density", "shannon_entropy_entries", "topological_entropy"):
        assert key in out
        assert out[key].shape == (2,)
        assert torch.isfinite(out[key]).all()


def test_plot_utils_smoke():
    from dsrnn.utils.visualization import plot_adjacency_heatmap, plot_channel_mask_heatmap
    import matplotlib
    matplotlib.use("Agg")
    fig1 = plot_adjacency_heatmap(torch.rand(8, 8))
    fig2 = plot_channel_mask_heatmap(torch.rand(T, H))
    assert fig1 is not None and fig2 is not None


def test_ww_extract_modules():
    import pytest
    pytest.importorskip("weightwatcher")
    from dsrnn.training.ww import analyze_model_ww, extract_linear_modules
    module = _ds_module()
    mods = extract_linear_modules(module.backbone)
    assert len(mods) > 0
    details, summary = analyze_model_ww(module.backbone)
    assert details is not None and isinstance(summary, dict)


def test_sparsity_mask_collection():
    from dsrnn.training.sparsity import _collect_hidden_mask_over_time
    module = _ds_module()
    x = torch.randn(1, T, D)
    mask = _collect_hidden_mask_over_time(module.backbone, x)
    assert mask.shape == (T, H)
    assert (mask >= 0).all() and (mask <= 1).all()


def main():
    test_graph_metrics_smoke()
    test_plot_utils_smoke()
    test_ww_extract_modules()
    test_sparsity_mask_collection()
    print("All callback tests passed.")


if __name__ == "__main__":
    main()
