"""Symbolic oracle and end-to-end checks for retention experiments."""

import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from dsrnn.data.long_memory import LongMemoryDataset, LongMemoryDataModule
from dsrnn.training.long_memory import LongMemoryModule
from dsrnn.training.selective_memory import MODEL_VARIANTS


@pytest.mark.parametrize("delay", [0, 8, 32, 128, 256])
def test_symbolic_oracle_and_exact_delay(delay):
    ds = LongMemoryDataset(4, delays=[delay], horizon=delay, seed=19)
    for item in ds:
        values, latest = {}, {}
        for t in range(ds.sequence_length):
            r, w = int(item["read_register"][t]), int(item["write_register"][t])
            if r >= 0:
                assert r in values
                torch.testing.assert_close(item["target"][t], values[r])
                assert item["source_timestep"][t] == latest[r]
                assert item["retention_delay"][t] == t - latest[r] - 1
            if w >= 0:
                assert r != w
                values[w] = item["input"][t, -ds.value_size:]
                latest[w] = t
        mask = item["target_query"]
        assert mask.sum() == 1
        assert item["retention_delay"][mask].item() == delay
        assert (item["query_mask"] & ~mask).any()
        qt = int(mask.nonzero()[0])
        st = int(item["source_timestep"][qt])
        reg = item["read_register"][qt]
        assert not (item["write_register"][st + 1:qt + 1] == reg).any()


def test_seed_balance_and_split_isolation():
    a = LongMemoryDataset(12, delays=[0, 2, 8], horizon=8, seed=9)
    b = LongMemoryDataset(12, delays=[0, 2, 8], horizon=8, seed=9)
    assert torch.equal(a.inputs, b.inputs)
    assert a.retention_delay[a.target_query].bincount()[[0, 2, 8]].tolist() == [4, 4, 4]
    dm = LongMemoryDataModule(n_train=8, n_val=8, n_test=8, delays=[0, 8], horizon=8)
    dm.setup("fit")
    assert not hasattr(dm, "test_dataset")
    dm.setup("test")
    assert not torch.equal(dm.train_dataset.inputs, dm.val_dataset.inputs)
    assert not torch.equal(dm.val_dataset.inputs, dm.test_dataset.inputs)


@pytest.mark.parametrize("implementation", ["reference", "sparse"])
@pytest.mark.parametrize("variant", MODEL_VARIANTS)
def test_long_forward_backward_and_budgets(variant, implementation):
    ds = LongMemoryDataset(2, delays=[128], horizon=128)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    module = LongMemoryModule(ds.input_size, 16, ds.value_size, variant,
                             implementation=implementation)
    prediction, trace = module(batch["input"], return_routing=True)
    assert prediction.shape == batch["target"].shape
    prediction[batch["query_mask"]].square().mean().backward()
    for side in ("read", "write"):
        assert (trace[side + "_mask"].sum(-1) == getattr(module.model, side + "_budget")).all()
    for router in (module.model.read_router, module.model.write_router, module.model.shared_router):
        if router is not None:
            assert router.backbone.weight.grad is not None
            assert torch.isfinite(router.backbone.weight.grad).all()


def test_lightning_metrics_and_bounded_trace(tmp_path):
    ds = LongMemoryDataset(8, delays=[0, 8], horizon=8)
    loader = DataLoader(ds, batch_size=4)
    module = LongMemoryModule(ds.input_size, 8, ds.value_size, "dual",
                             analysis_path=str(tmp_path / "trace.pt"))
    trainer = Trainer(max_epochs=1, accelerator="cpu", logger=False,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    metrics = trainer.test(module, dataloaders=loader, verbose=False)[0]
    assert "test/delay_8/target_mse" in metrics
    assert metrics["test/balanced_mse"] == pytest.approx(
        .5 * (metrics["test/target_mse"] + metrics["test/distractor_mse"]), rel=1e-6)
    artifact = torch.load(tmp_path / "trace.pt", weights_only=False)
    assert artifact["input"].shape[0] == 4
    assert artifact["trace_scope"] == "first_test_batch"
    other = trainer.test(module, dataloaders=DataLoader(ds, batch_size=3), verbose=False)[0]
    for key in ("test/balanced_mse", "test/distractor_mse", "test/target_mse",
                "test/delay_8/target_mse", "test/target_source_preserved_read_fraction"):
        assert other[key] == pytest.approx(metrics[key], rel=1e-5, abs=1e-7)
