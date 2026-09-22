"""CogScale target masking, reproducibility, and model integration."""
from copy import deepcopy
import numpy as np
import pytest
import torch

cog = pytest.importorskip("cogscale")
from dsrnn.data.cogscale import TASKS, build_data
from dsrnn.training.cogscale import target_loss, CogScaleModule


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("cell,implementation", [
    ("rnn", "reference"), ("rnn", "sparse"), ("gru", "reference"), ("gru", "sparse")])
@pytest.mark.parametrize("variant", ["dense", "read_only", "write_only", "shared", "dual"])
def test_routed_model_shapes_budgets_and_gradients(device, cell, implementation, variant):
    from dsrnn.training.cogscale import CogScaleRNN
    model = CogScaleRNN(7, 8, 3, variant, recurrent_cell=cell,
                       implementation=implementation).to(device)
    output, trace = model(torch.randn(3, 6, 7, device=device), return_routing=True)
    assert output.shape == (3, 6, 3)
    assert output.device.type == device
    for side in ("read", "write"):
        budget = getattr(model, side + "_budget")
        assert trace[side + "_mask"].shape == (3, 6, 8)
        assert torch.all(trace[side + "_mask"].sum(-1) == budget)
        assert trace[side + "_indices"].shape == (3, 6, budget)
    output.square().mean().backward()
    for router in (model.read_router, model.write_router, model.shared_router):
        if router is not None:
            assert router.backbone.weight.grad is not None
            assert torch.isfinite(router.backbone.weight.grad).all()


@pytest.mark.parametrize("variant", ["shared", "dual"])
@pytest.mark.parametrize("backward", ["ste", "selected_only"])
@pytest.mark.parametrize("cell", ["rnn", "gru"])
def test_sparse_model_matches_reference(variant, backward, cell):
    from dsrnn.training.cogscale import CogScaleRNN
    reference = CogScaleRNN(7, 8, 3, variant, router_backward=backward, recurrent_cell=cell).double()
    sparse = CogScaleRNN(7, 8, 3, variant, router_backward=backward,
                         implementation="sparse", recurrent_cell=cell).double()
    sparse.load_state_dict(reference.state_dict())
    x = torch.randn(3, 6, 7, dtype=torch.double)
    outputs = []
    inputs = []
    for model in (reference, sparse):
        inputs.append(x.clone().requires_grad_())
        outputs.append(model(inputs[-1]))
        outputs[-1].square().sum().backward()
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(inputs[0].grad, inputs[1].grad)
    for left, right in zip(reference.parameters(), sparse.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


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


def test_run_directories_are_unique_and_config_hashed(tmp_path, monkeypatch):
    from experiments.train_cogscale import build_parser, _RUN_ID_ENV, _prepare_run_directory
    parser = build_parser()
    args = parser.parse_args([])
    args.log_dir = str(tmp_path)
    args.run_id = "review_run"
    monkeypatch.delenv(_RUN_ID_ENV, raising=False)
    run_root, run_id, reference_hash, _ = _prepare_run_directory(args)
    assert run_id == "review_run"
    assert run_root.is_dir()

    monkeypatch.delenv(_RUN_ID_ENV, raising=False)
    with pytest.raises(FileExistsError):
        _prepare_run_directory(args)

    monkeypatch.delenv(_RUN_ID_ENV, raising=False)
    args.run_id = "sparse_run"
    args.implementation = "sparse"
    sparse_root, _, sparse_hash, _ = _prepare_run_directory(args)
    assert sparse_root.is_dir()
    assert sparse_hash != reference_hash


def test_overrides_must_apply_to_selected_tasks():
    from dsrnn.data.cogscale import resolve_task_overrides
    with pytest.raises(ValueError, match="does not apply"):
        resolve_task_overrides(["induction_heads"], "small", {"delay": 2})
    sizes = resolve_task_overrides(["simple_copy", "induction_heads"], "small", {"delay": 2})
    assert sizes["simple_copy"]["delay"] == 2
    assert "delay" not in sizes["induction_heads"]


def test_wandb_flags_do_not_change_scientific_hash(tmp_path, monkeypatch):
    from experiments.train_cogscale import build_parser, _scientific_config, _load_config
    parser = build_parser()
    assert _scientific_config(parser.parse_args([])) == _scientific_config(
        parser.parse_args(["--wandb", "--wandb_project", "example", "--wandb_mode", "offline"]))
    config = tmp_path / "logging.yaml"
    config.write_text("wandb: true\n")
    monkeypatch.setattr("sys.argv", ["train_cogscale", "--config", str(config), "--no-wandb"])
    assert _load_config(parser).wandb is False


def test_wandb_lifecycle_and_metadata(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments.train_cogscale import build_parser, _experiment_loggers
    args = build_parser().parse_args(["--wandb", "--wandb_mode", "offline"])
    runs = []
    def init(**kwargs):
        run = SimpleNamespace(config=kwargs["config"], summary={}, codes=[])
        run.finish = lambda exit_code: run.codes.append(exit_code)
        runs.append(run)
        return run
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, run=None))
    monkeypatch.setattr("lightning.pytorch.loggers.WandbLogger", lambda **kw: kw)
    manifest = {"config": {"seed": 42}, "run_id": "example", "config_hash": "hash",
                "runtime": {}, "protocol": {}, "tasks": {"simple_copy": {"parameters": {"delay": 3}}}}
    for variant in ("dense", "dual"):
        with _experiment_loggers(args, tmp_path, manifest, "simple_copy", variant) as (loggers, run):
            assert len(loggers) == 2
            assert run.config["task_metadata"]["parameters"]["delay"] == 3
            assert run.config["variant"] == variant
    assert len(runs) == 2 and all(run.codes == [0] for run in runs)
    with pytest.raises(RuntimeError, match="training failed"):
        with _experiment_loggers(args, tmp_path, manifest, "simple_copy", "dual"):
            raise RuntimeError("training failed")
    assert runs[-1].codes == [1]
