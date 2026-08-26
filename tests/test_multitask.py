"""Smoke tests for the multitask module and the mask-analysis callback.

Constructs the dict-based multitask batch directly (no dataset download).
Random tensors, CPU, fast_dev_run, no clutter.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsrnn.nn.router import RouterConfig
from dsrnn.training.multitask import (FixedTaskWeighting, MultitaskModule,
                                      UncertaintyTaskWeighting)

B, T_, D, H = 8, 10, 8, 16


def _batch():
    x = torch.randn(B, T_, D)
    y = torch.randn(B, T_, 4)
    x_lengths = torch.tensor([[T_, D]] * B)
    y_lengths = torch.tensor([[T_, 4]] * B)
    task_id = torch.tensor([0] * (B // 2) + [1] * (B // 2))
    task = np.array(["s1"] * (B // 2) + ["s2"] * (B // 2))
    return {"x": x, "y": y, "x_lengths": x_lengths, "y_lengths": y_lengths,
            "task_id": task_id, "task": task}


def _module(model_type="lstm", implementation="gathered"):
    task_configs = {
        "s1": dict(input_size=D, output_size=4, loss_type="mse", return_sequences=True),
        "s2": dict(input_size=D, output_size=4, loss_type="mse", return_sequences=True),
    }
    model_kwargs = {}
    if model_type in ("dslstm", "dsgru", "dsltc"):
        # the shared backbone's input is hidden_size (projected input + task emb)
        model_kwargs = {"router_config": RouterConfig(
            num_channels=H, input_dim=H, context_dim=4, sparsity=0.9,
            backward="selected_only")}
    return MultitaskModule(
        task_configs=task_configs, model_type=model_type, hidden_size=H,
        task_weighting=FixedTaskWeighting(["s1", "s2"], [1.0, 1.0]),
        task_embedding_dim=4, implementation=implementation, model_kwargs=model_kwargs)


def test_multitask_forward_and_step():
    module = _module("lstm")
    batch = _batch()
    out = module(x=batch["x"], task_ids=batch["task_id"], tasks=batch["task"],
                 x_lengths=batch["x_lengths"])
    assert set(out.keys()) == {"s1", "s2"}
    res = module.training_step(batch, 0)
    assert torch.isfinite(res["loss"])


def test_uncertainty_weighting():
    w = UncertaintyTaskWeighting(["s1", "s2"])
    total, weights = w({"s1": torch.tensor(1.0), "s2": torch.tensor(2.0)})
    assert torch.isfinite(total)
    assert set(weights) == {"s1", "s2"}
    total.backward()  # learnable log_vars get gradient


def test_multitask_trainer_smoke_ds():
    """fast_dev_run on a DS backbone (dslstm)."""
    import tempfile
    from lightning import Trainer

    module = _module("dslstm")
    loader = DataLoader([_batch()], batch_size=None, collate_fn=lambda b: b[0])
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          default_root_dir=tmp, accelerator="cpu")
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        trainer.test(module, dataloaders=loader)


def test_multitask_trainer_smoke_dsltc():
    """dsltc through build_backbone (selected_only uses the gathered path)."""
    import tempfile
    from lightning import Trainer

    module = _module("dsltc")  # gathered + selected_only is supported
    loader = DataLoader([_batch()], batch_size=None, collate_fn=lambda b: b[0])
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          default_root_dir=tmp, accelerator="cpu")
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)


def test_multitask_mask_analysis_callback():
    """The cross-task mask-analysis callback runs on a DS multitask model."""
    import tempfile
    from lightning import Trainer
    from dsrnn.training.sparsity import MultitaskMaskAnalysisCallback

    module = _module("dslstm")
    loader = DataLoader([_batch()], batch_size=None, collate_fn=lambda b: b[0])

    class _DM:
        def val_dataloader(self):
            return loader

    with tempfile.TemporaryDirectory() as tmp:
        cb = MultitaskMaskAnalysisCallback(datamodule=_DM(), every_n_epochs=1,
                                           max_batches_per_task=1, log_validation=True)
        trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          default_root_dir=tmp, accelerator="cpu",
                          callbacks=[cb], fast_dev_run=True)
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)


def main():
    test_multitask_forward_and_step()
    test_uncertainty_weighting()
    test_multitask_trainer_smoke_ds()
    test_multitask_trainer_smoke_dsltc()
    test_multitask_mask_analysis_callback()
    print("All multitask tests passed.")


if __name__ == "__main__":
    main()
