"""Comprehensive smoke tests for the training stack.

Everything runs on random tensors, CPU, with ``fast_dev_run``,
``logger=False``, ``enable_checkpointing=False`` and a temp ``default_root_dir``,
so tests leave no checkpoints or logs in the working directory.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from dsrnn.nn.router import RouterConfig
from dsrnn.training.optim import build_optimizer, build_param_groups
from dsrnn.training.pruning import (
    MaskCallback,
    RigLCallback,
    apply_masks_and_handle_grow,
    apply_masks_to_model,
    count_nonzero_masks,
    make_initial_masks,
    rigl_step,
)
from dsrnn.training.scheduler import LRSchedulerConfig, build_lr_scheduler
from dsrnn.training.sequence import SequenceModule, build_backbone

B, T, D, H, C = 8, 10, 12, 32, 4


def _router_config(backward="selected_only"):
    return RouterConfig(num_channels=H, input_dim=D, context_dim=4,
                        sparsity=0.9, backward=backward)


def _make_module(model_type, scheduler=False, **overrides):
    kwargs = dict(
        model_type=model_type,
        input_size=D,
        hidden_size=H,
        output_size=C,
        return_sequences=False,
        loss_type="ce",
        metric_params={"accuracy": {"task": "multiclass", "num_classes": C,
                                    "average": "macro"}},
        optimizer_kwargs={"all": {"lr": 1e-3, "weight_decay": 0.0}},
        scheduler_config=None,
        test_with_corruptions=True,
    )
    if model_type in ("dslstm", "dsgru"):
        kwargs["model_kwargs"] = {"router_config": _router_config()}
    else:
        kwargs["model_kwargs"] = {"num_layers": 1, "dropout": 0.0}
    if scheduler:
        kwargs["scheduler_config"] = LRSchedulerConfig(
            scheduler_type="reduce_on_plateau",
            scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 2},
            monitor="val/loss")
    kwargs.update(overrides)
    return SequenceModule(**kwargs)


def _random_loader():
    x = torch.randn(4 * B, T, D)
    y = torch.randint(0, C, (4 * B,))
    return DataLoader(TensorDataset(x, y), batch_size=B)


def test_build_backbone_all_types():
    x = torch.randn(B, T, D)
    for model_type in ("lstm", "gru", "ltc"):
        backbone = build_backbone(model_type, D, H, model_kwargs={"num_layers": 1}
                                  if model_type != "ltc" else {})
        out, _ = backbone(x)
        assert out.shape == (B, T, H), (model_type, out.shape)
    for model_type in ("dslstm", "dsgru"):
        backbone = build_backbone(model_type, D, H,
                                  model_kwargs={"router_config": _router_config()})
        out, _ = backbone(x)
        assert out.shape == (B, T, H), (model_type, out.shape)


def test_param_groups_s4_pattern():
    """S4-style grouping: kernel/SSM params at lower LR, zero weight decay."""
    torch.manual_seed(0)
    model = torch.nn.ModuleDict({
        "kernel": torch.nn.Linear(4, 4),
        "head": torch.nn.Linear(4, 4),
    })
    groups = build_param_groups(
        model.named_parameters(),
        {"ssm": {"pattern": r"kernel", "lr": 1e-3, "weight_decay": 0.0},
         "all": {"lr": 4e-3, "weight_decay": 0.01}},
    )
    ssm = next(g for g in groups if g["lr"] == 1e-3)
    rest = next(g for g in groups if g["lr"] == 4e-3)
    ssm_names = {id(p) for g in groups if g["lr"] == 1e-3 for p in g["params"]}
    named = dict(model.named_parameters())
    assert id(named["kernel.weight"]) in ssm_names
    assert id(named["kernel.bias"]) in ssm_names
    assert id(named["head.weight"]) not in ssm_names
    assert ssm["weight_decay"] == 0.0 and rest["weight_decay"] == 0.01


def test_build_optimizer_flat_and_grouped():
    model = torch.nn.Linear(4, 4)
    opt = build_optimizer(model.parameters(), "adamw", {"lr": 1e-3})
    assert isinstance(opt, torch.optim.AdamW)
    opt2 = build_optimizer(model.named_parameters(), "adamw",
                           {"all": {"lr": 1e-3}})
    assert isinstance(opt2, torch.optim.AdamW)
    try:
        build_optimizer(model.parameters(), "not_an_optimizer", {})
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


def test_scheduler_builders():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rop = build_lr_scheduler(opt, LRSchedulerConfig(scheduler_type="reduce_on_plateau"))
    assert isinstance(rop, torch.optim.lr_scheduler.ReduceLROnPlateau)
    cos = build_lr_scheduler(opt, LRSchedulerConfig(
        scheduler_type="cosine", scheduler_kwargs={"T_max": 10}))
    assert isinstance(cos, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert build_lr_scheduler(opt, LRSchedulerConfig(scheduler_type="none")) is None
    oc = build_lr_scheduler(opt, LRSchedulerConfig(scheduler_type="one_cycle"),
                            epochs=5, steps_per_epoch=20)
    assert isinstance(oc, torch.optim.lr_scheduler.OneCycleLR)


def test_sequence_module_mask_enforcement():
    """Masks zero the covered params and the hooks keep them zero."""
    torch.manual_seed(0)
    module = _make_module("gru")
    masks = make_initial_masks(module, sparsity=0.5, deterministic=True)
    module.masks = masks
    apply_masks_to_model(module, masks)
    for name, p in module.named_parameters():
        if name in masks:
            assert torch.all(p.data[~masks[name]] == 0), name
    kept, total = count_nonzero_masks(masks)
    assert 0 < kept < total


def test_rigl_step_conserves_budget():
    """RigL rewiring keeps the number of active weights constant."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 4))
    masks = make_initial_masks(model, sparsity=0.9, deterministic=True)
    kept_before, _ = count_nonzero_masks(masks)
    loss = model(torch.randn(8, 16)).sum()
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if n in masks}
    new_masks = rigl_step(model, masks, grads, rewire_fraction=0.1)
    kept_after, _ = count_nonzero_masks(new_masks)
    assert kept_before == kept_after
    changed = any((masks[k] != new_masks[k]).any() for k in masks)
    assert changed


def test_rigl_apply_and_optimizer_clear():
    """apply_masks_and_handle_grow inits grown weights and clears optimizer state."""
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 16)).sum().backward()
    opt.step()

    masks = make_initial_masks(model, sparsity=0.5, deterministic=True)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if n in masks}
    new_masks = rigl_step(model, masks, grads, rewire_fraction=0.2)
    apply_masks_and_handle_grow(model, masks, new_masks, grads=grads,
                                optimizer_list=[opt])
    for name, p in model.named_parameters():
        if name in new_masks:
            assert torch.all(p.data[~new_masks[name]] == 0), name
            state = opt.state.get(p, {})
            for v in state.values():
                if isinstance(v, torch.Tensor) and v.shape == p.shape:
                    assert torch.all(v[~new_masks[name]] == 0), name


def test_sequence_module_trainer_smoke():
    """One fast dev run per backbone: train + val + test, no clutter."""
    import tempfile
    from lightning import Trainer

    loader = _random_loader()
    for model_type in ("lstm", "gru", "dslstm", "dsgru"):
        module = _make_module(model_type, scheduler=True)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                fast_dev_run=True,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                default_root_dir=tmp,
                accelerator="cpu",
            )
            trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
            trainer.test(module, dataloaders=loader)


def test_sequence_module_rigl_callback_smoke():
    """RigL callback runs inside a real Trainer and conserves the budget."""
    import tempfile
    from lightning import Trainer

    torch.manual_seed(0)
    module = _make_module("gru")
    masks = make_initial_masks(module, sparsity=0.9, deterministic=True)
    module.masks = masks
    apply_masks_to_model(module, masks)
    kept_before, _ = count_nonzero_masks(masks)

    loader = _random_loader()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp,
            accelerator="cpu",
            callbacks=[MaskCallback(masks),
                       RigLCallback(masks, update_frequency=1, rewire_fraction=0.1)],
        )
        # pass a val loader too, to avoid Lightning's "no val_dataloader" note
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    kept_after, _ = count_nonzero_masks(masks)
    assert kept_before == kept_after
    for name, p in module.named_parameters():
        if name in masks:
            assert torch.all(p.data[~masks[name]] == 0), name


def test_build_cl_model():
    """Continual-learning wrappers build and run (requires avalanche)."""
    import pytest
    pytest.importorskip("avalanche")
    from dsrnn.training.continual import build_cl_model

    model = build_cl_model("dslstm", input_size=D, hidden_size=H, output_size=C,
                           router_config=_router_config())
    out = model(torch.randn(B, T, D))
    assert out.shape == (B, C)


def main():
    test_build_backbone_all_types()
    test_param_groups_s4_pattern()
    test_build_optimizer_flat_and_grouped()
    test_scheduler_builders()
    test_sequence_module_mask_enforcement()
    test_rigl_step_conserves_budget()
    test_rigl_apply_and_optimizer_clear()
    test_sequence_module_trainer_smoke()
    test_sequence_module_rigl_callback_smoke()
    test_build_cl_model()
    print("All training-stack tests passed.")


if __name__ == "__main__":
    main()
