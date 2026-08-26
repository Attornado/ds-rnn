"""Smoke tests for the S4-family LRA module (LRAModule / DSS4).

Runs on random tensors, CPU, tiny config, ``fast_dev_run``, ``logger=False``,
``enable_checkpointing=False`` and a temp root dir: no clutter left behind.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from dsrnn.nn.router import RouterConfig
from dsrnn.training.lra import LRAModule, SSM_PARAM_PATTERN
from dsrnn.training.optim import build_param_groups

D_MODEL, D_STATE, L, VOCAB, C = 16, 8, 32, 20, 3


def _router_cfg():
    # DSS4 routes from the (pooled) d_model-dimensional input and masks the
    # d_model hidden channels, so num_channels == input_dim == d_model.
    return RouterConfig(num_channels=D_MODEL, input_dim=D_MODEL, context_dim=4,
                        sparsity=0.9, backward="selected_only")


def _module(model_type, **overrides):
    kwargs = dict(model_type=model_type, vocab_size=VOCAB, d_model=D_MODEL,
                  d_state=D_STATE, output_size=C, n_layers=1, l_max=L,
                  router_config=_router_cfg() if "ds" in model_type else None)
    kwargs.update(overrides)
    return LRAModule(**kwargs)


def test_lra_forward_all_model_types():
    x = torch.randint(0, VOCAB, (2, L))
    for model_type in ("s4", "dss4", "liqs4", "dsliqs4"):
        m = _module(model_type)
        out = m(x)
        assert out.shape == (2, C), (model_type, out.shape)
        assert torch.isfinite(out).all(), model_type


def test_lra_continuous_input():
    # sCIFAR-style: continuous (B, L, 1) input, Linear in-projection.
    m = _module("s4", vocab_size=-1, input_size=1, output_size=10)
    out = m(torch.randn(2, L, 1))
    assert out.shape == (2, 10)


def test_lra_norm_types():
    x = torch.randint(0, VOCAB, (2, L))
    for norm_type in ("batch", "layer"):
        m = _module("s4", norm_type=norm_type)
        assert m(x).shape == (2, C)


def test_lra_ssm_param_groups():
    """Kernel params (log_dt/B/C/P) land in the low-LR, zero-wd group."""
    m = _module("dss4", n_layers=2)
    groups = build_param_groups(m.named_parameters(), {
        "ssm": {"pattern": SSM_PARAM_PATTERN, "lr": 1e-3, "weight_decay": 0.0},
        "all": {"lr": 4e-3, "weight_decay": 0.01},
    })
    ssm = next(g for g in groups if g["lr"] == 1e-3)
    rest = next(g for g in groups if g["lr"] == 4e-3)
    assert len(ssm["params"]) > 0, "no SSM kernel params matched the pattern"
    ssm_ids = {id(p) for p in ssm["params"]}
    named = dict(m.named_parameters())
    kernel_names = [n for n in named
                    if "kernel" in n and any(s in n for s in ("log_dt", "B", "C", "P"))]
    assert kernel_names, "expected some kernel params in the model"
    assert all(id(named[n]) in ssm_ids for n in kernel_names)
    # Non-kernel params (embedding, head, norms) stay in the base group.
    assert any("embed" in n or "head" in n for n, p in named.items()
               if id(p) in {id(q) for q in rest["params"]})


def test_lra_trainer_smoke():
    import tempfile
    from lightning import Trainer

    x = torch.randint(0, VOCAB, (16, L))
    y = torch.randint(0, C, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    m = _module("dss4")
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          default_root_dir=tmp, accelerator="cpu")
        trainer.fit(m, train_dataloaders=loader, val_dataloaders=loader)
        trainer.test(m, dataloaders=loader)


def main():
    test_lra_forward_all_model_types()
    test_lra_continuous_input()
    test_lra_norm_types()
    test_lra_ssm_param_groups()
    test_lra_trainer_smoke()
    print("All LRA/S4 tests passed.")


if __name__ == "__main__":
    main()
