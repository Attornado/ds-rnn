"""CogScale target masking, reproducibility, and model integration."""
from copy import deepcopy
import numpy as np
import pytest
import torch

cog = pytest.importorskip("cogscale")
from experiments.train_cogscale import TASKS, build_data, target_loss, CogScaleModule


@pytest.mark.parametrize("task", TASKS)
def test_library_data_and_all_variants(task):
    original = deepcopy(cog.evals.cogscale_small[task]["params"])
    data, _ = build_data(task, "small", 2026, 4, 2, 2)
    again, _ = build_data(task, "small", 2026, 4, 2, 2)
    assert original == cog.evals.cogscale_small[task]["params"]
    assert np.array_equal(data["X_train"], again["X_train"])
    x, y = [torch.tensor(data[k], dtype=torch.float32) for k in ("X_train", "Y_train")]
    for variant in ("dense", "read_only", "write_only", "shared", "dual"):
        model = CogScaleModule(x.size(-1), 8, y.size(-1), variant, recurrent_cell="gru")
        logits = model(x)
        logits.retain_grad()
        loss, mask, labels = target_loss(logits, y)
        loss.backward()
        assert torch.count_nonzero(logits.grad[~mask]) == 0
        expected = (logits[mask].argmax(-1) != labels).float().mean().item()
        assert cog.compute_score(y.numpy(), logits.detach().numpy(), "classification") == pytest.approx(expected)
        for router in (model.model.read_router, model.model.write_router, model.model.shared_router):
            if router is not None:
                assert router.backbone.weight.grad is not None
                assert torch.isfinite(router.backbone.weight.grad).all()


def test_partial_minus100_targets_and_empty_mask():
    y = torch.tensor([[[-100., -100., -100.], [-100., 1., -100.]]])
    logits = torch.randn_like(y, requires_grad=True)
    loss, mask, labels = target_loss(logits, y)
    assert labels.tolist() == [1]
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    with pytest.raises(ValueError, match="no scored"):
        target_loss(logits, torch.full_like(y, -100))


@pytest.mark.parametrize("difficulty", ["small", "medium", "large"])
@pytest.mark.parametrize("task", TASKS)
def test_size_defaults_follow_library_presets(task, difficulty):
    from experiments.train_cogscale import build_parser, TASK_SIZE_OPTIONS
    args = build_parser().parse_args(["--difficulty", difficulty])
    overrides = {name: getattr(args, name) for name in TASK_SIZE_OPTIONS}
    assert all(value is None for value in overrides.values())
    preset = deepcopy(getattr(cog.evals, "cogscale_" + difficulty)[task]["params"])
    data, params = build_data(task, difficulty, 42, 2, 1, 1, overrides=overrides)
    for name in TASK_SIZE_OPTIONS:
        if name in preset:
            assert params[name] == preset[name]
    assert data["X_train"].shape[0] == 2


@pytest.mark.parametrize("task,length,features", [
    ("simple_copy", 19, 7), ("selective_copy", 13, 8),
    ("induction_heads", 8, 6), ("associative_recall", 8, 6),
])
def test_overrides_change_actual_shapes_without_mutating_presets(task, length, features):
    preset = deepcopy(cog.evals.cogscale_medium[task]["params"])
    data, params = build_data(task, "medium", 42, 2, 1, 1, overrides={
        "sequence_length": 8, "n_symbols": 6, "delay": 2,
        "n_markers": 2, "num_pairs": 3})
    assert data["X_train"].shape == (2, length, features)
    assert data["Y_train"].shape == (2, length, 6)
    assert params["sequence_length"] == 8
    assert params["n_symbols"] == 6
    assert cog.evals.cogscale_medium[task]["params"] == preset


@pytest.mark.parametrize("task,overrides", [
    ("simple_copy", {"sequence_length": 0}),
    ("simple_copy", {"delay": -1}),
    ("simple_copy", {"n_symbols": 0}),
    ("selective_copy", {"sequence_length": 3}),
    ("induction_heads", {"sequence_length": 7}),
    ("induction_heads", {"sequence_length": 2}),
    ("associative_recall", {"sequence_length": 8}),
    ("associative_recall", {"n_symbols": 4}),
])
def test_invalid_sizes_fail_without_mutating_presets(task, overrides):
    preset = deepcopy(cog.evals.cogscale_medium[task]["params"])
    with pytest.raises(ValueError, match=task):
        build_data(task, "medium", 42, 2, 1, 1, overrides=overrides)
    assert cog.evals.cogscale_medium[task]["params"] == preset


def test_yaml_sizes_and_cli_override(tmp_path, monkeypatch):
    from experiments.train_cogscale import build_parser, _load_config
    config = tmp_path / "sizes.yaml"
    config.write_text("difficulty: small\nsequence_length: 8\nn_symbols: 5\ndelay: null\n")
    monkeypatch.setattr("sys.argv", ["train_cogscale", "--config", str(config),
                                     "--n_symbols", "6"])
    args = _load_config(build_parser())
    assert (args.difficulty, args.sequence_length, args.n_symbols, args.delay) == (
        "small", 8, 6, None)
