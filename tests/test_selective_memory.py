"""Tests for the synthetic selective-memory experiment."""

from copy import deepcopy

import pytest
import torch

from dsrnn.data.selective_memory import QUERY, UPDATE, SelectiveMemoryDataset
from dsrnn.nn.cells.dual_read_write_sparse import SparseDualReadWriteRNNCell
from dsrnn.nn.cells.dual_read_write_gru import DualReadWriteGRUCell
from dsrnn.training.selective_memory import (
    IMPLEMENTATIONS,
    MODEL_VARIANTS,
    SelectiveMemoryModule,
    SelectiveMemoryRNN,
    _merge_trace_shards,
    masked_query_mse,
    memory_alignment_statistics,
    query_loss_statistics,
    routing_statistics,
)
from experiments.train_selective_memory import (
    _RUN_ID_ENV,
    _prepare_run_directory,
    build_parser,
)


def test_program_queries_one_register_while_updating_another():
    dataset = SelectiveMemoryDataset(8, n_registers=4, value_size=3, seed=7)
    for sample in range(len(dataset)):
        item = dataset[sample]
        joint = item["operations"][:, QUERY] & item["operations"][:, UPDATE]
        assert joint.sum().item() == 2
        assert torch.all(item["read_register"][joint] != item["write_register"][joint])
        # The two final queries must observe the values introduced by updates.
        value_offset = 3 + 2 * dataset.n_registers
        assert torch.equal(item["target"][4], item["input"][3, value_offset:])
        assert torch.equal(item["target"][5], item["input"][2, value_offset:])


def test_all_baselines_expose_complete_routing_trace_and_backward():
    torch.manual_seed(8)
    dataset = SelectiveMemoryDataset(4, n_registers=3, value_size=2, seed=8)
    x = torch.stack([dataset[index]["input"] for index in range(4)])
    targets = torch.stack([dataset[index]["target"] for index in range(4)])
    query_mask = torch.stack([dataset[index]["query_mask"] for index in range(4)])
    operations = torch.stack([dataset[index]["operations"] for index in range(4)])

    for implementation in IMPLEMENTATIONS:
        for variant in MODEL_VARIANTS:
            model = SelectiveMemoryRNN(
                dataset.input_size, hidden_size=8, value_size=2, variant=variant,
                read_sparsity=0.5, write_sparsity=0.5, router_context_size=4,
                implementation=implementation)
            assert isinstance(model.cell, SparseDualReadWriteRNNCell) == (
                implementation == "sparse")
            predictions, trace = model(x, return_routing=True)
            assert predictions.shape == targets.shape
            for side in ("read", "write"):
                assert trace[f"{side}_mask"].shape == (4, 6, 8)
                assert trace[f"{side}_scores"].shape == (4, 6, 8)
                assert trace[f"{side}_logits"].shape == (4, 6, 8)
            expected_read = (4 if variant in ("read_only", "shared", "dual")
                             else 8)
            expected_write = (4 if variant in ("write_only", "shared", "dual")
                              else 8)
            assert torch.all(trace["read_mask"].sum(dim=-1) == expected_read)
            assert torch.all(trace["write_mask"].sum(dim=-1) == expected_write)
            assert trace["read_indices"].shape == (4, 6, expected_read)
            assert trace["write_indices"].shape == (4, 6, expected_write)
            stats = routing_statistics(trace, operations)
            states = ("00", "10", "01", "11")
            assert set(f"state_{state}" for state in states) <= stats.keys()
            assert torch.allclose(sum(stats[f"state_{state}"] for state in states),
                                  torch.tensor(1.0))
            loss = masked_query_mse(predictions, targets, query_mask)
            loss.backward()
            if model.read_router is not None:
                assert model.read_router.backbone.weight.grad is not None
            if model.write_router is not None:
                assert model.write_router.backbone.weight.grad is not None
            if model.shared_router is not None:
                assert torch.equal(trace["read_mask"], trace["write_mask"])
                assert torch.equal(trace["read_indices"], trace["write_indices"])
                assert model.shared_router.backbone.weight.grad is not None
                assert model.read_budget == model.write_budget == 4


@pytest.mark.parametrize("variant", ["shared", "dual"])
@pytest.mark.parametrize("backward", ["ste", "selected_only"])
def test_sparse_selective_memory_matches_reference(backward, variant):
    torch.manual_seed(81)
    dataset = SelectiveMemoryDataset(4, n_registers=3, value_size=2, seed=81)
    x = torch.stack([dataset[index]["input"] for index in range(4)]).double()
    reference = SelectiveMemoryRNN(
        dataset.input_size, 8, 2, variant, router_context_size=4,
        router_backward=backward, implementation="reference").double()
    sparse = SelectiveMemoryRNN(
        dataset.input_size, 8, 2, variant, router_context_size=4,
        router_backward=backward, implementation="sparse").double()
    sparse.load_state_dict(reference.state_dict())
    models = []
    for model in (reference, sparse):
        data = x.clone().requires_grad_()
        output = model(data)
        output.square().sum().backward()
        models.append((model, data, output))
    torch.testing.assert_close(models[1][2], models[0][2])
    torch.testing.assert_close(models[1][1].grad, models[0][1].grad)
    for left, right in zip(reference.parameters(), sparse.parameters()):
        torch.testing.assert_close(right.grad, left.grad)


@pytest.mark.parametrize("variant", MODEL_VARIANTS)
def test_gru_selective_memory_forward_backward_and_budgets(variant):
    torch.manual_seed(82)
    dataset = SelectiveMemoryDataset(3, n_registers=3, value_size=2, seed=82)
    x = torch.stack([dataset[index]["input"] for index in range(3)])
    model = SelectiveMemoryRNN(
        dataset.input_size, 8, 2, variant, router_context_size=4,
        recurrent_cell="gru")
    assert isinstance(model.cell, DualReadWriteGRUCell)
    predictions, trace = model(x, return_routing=True)
    assert predictions.shape == (3, 6, 2)
    assert torch.all(trace["read_mask"].sum(-1) == model.read_budget)
    assert torch.all(trace["write_mask"].sum(-1) == model.write_budget)
    predictions.square().mean().backward()
    for router in (model.read_router, model.write_router, model.shared_router):
        if router is not None:
            assert router.backbone.weight.grad is not None
            assert torch.isfinite(router.backbone.weight.grad).all()


def test_gru_rejects_sparse_implementation():
    with pytest.raises(ValueError, match="only a reference implementation"):
        SelectiveMemoryRNN(7, 8, 2, "dual", recurrent_cell="gru",
                           implementation="sparse")


def test_routing_statistics_align_joint_operations():
    read = torch.tensor([[[1., 0.], [0., 1.]]])
    write = torch.tensor([[[0., 1.], [0., 1.]]])
    operations = torch.zeros(1, 2, 3, dtype=torch.bool)
    operations[0, 0, QUERY] = True
    operations[0, 0, UPDATE] = True
    operations[0, 1, QUERY] = True
    stats = routing_statistics({"read_mask": read, "write_mask": write}, operations)
    assert stats["query/state_10"] == 0.25
    assert stats["query/state_01"] == 0.25
    assert stats["query/state_00"] == 0.25
    assert stats["query/state_11"] == 0.25
    assert stats["update/state_10"] == 0.5
    assert stats["update/state_01"] == 0.5
    assert stats["query_only/state_10"] == 0.0
    assert stats["query_only/state_01"] == 0.0
    assert stats["query_only/state_00"] == 0.5
    assert stats["query_only/state_11"] == 0.5
    assert stats["query_update/state_10"] == 0.5
    assert stats["query_update/state_01"] == 0.5


def test_query_loss_statistics_split_pure_and_joint_queries():
    predictions = torch.tensor([[[0.0], [2.0], [3.0]]])
    targets = torch.zeros_like(predictions)
    operations = torch.zeros(1, 3, 3, dtype=torch.bool)
    operations[0, 1, QUERY] = True
    operations[0, 1, UPDATE] = True
    operations[0, 2, QUERY] = True
    stats = query_loss_statistics(predictions, targets, operations)
    assert stats["query_update_mse"] == 4.0
    assert stats["query_only_mse"] == 9.0


def test_memory_alignment_uses_latest_write_to_queried_register():
    read = torch.tensor([[[1., 1., 0., 0.],
                          [0., 0., 1., 1.],
                          [0., 0., 1., 1.]]])
    write = torch.tensor([[[1., 1., 0., 0.],
                           [0., 0., 1., 1.],
                           [1., 1., 0., 0.]]])
    operations = torch.zeros(1, 3, 3, dtype=torch.bool)
    operations[0, 2, QUERY] = True
    read_register = torch.tensor([[-1, -1, 0]])
    write_register = torch.tensor([[0, 0, -1]])
    stats = memory_alignment_statistics(
        {"read_mask": read, "write_mask": write},
        read_register, write_register, operations)
    assert stats["memory/query/source_read_fraction"] == 1.0
    assert stats["memory/query/source_protected_read_fraction"] == 1.0
    assert stats["memory/query/source_preserved_fraction"] == 1.0
    assert stats["memory/query/source_preserved_read_fraction"] == 1.0
    assert stats["memory/query/protected_read_excess"] == 0.5
    assert stats["memory/query_only/source_read_fraction"] == 1.0
    assert stats["memory/query_update/source_read_fraction"] == 0.0


def test_shared_mask_requires_one_common_budget():
    with pytest.raises(ValueError, match="equal read/write sparsity"):
        SelectiveMemoryRNN(
            input_size=7, hidden_size=8, value_size=2, variant="shared",
            read_sparsity=0.25, write_sparsity=0.5)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_lightning_smoke_writes_per_timestep_trace(tmp_path, implementation):
    from lightning import Trainer
    from torch.utils.data import DataLoader

    dataset = SelectiveMemoryDataset(8, n_registers=3, value_size=2, seed=9)
    loader = DataLoader(dataset, batch_size=4)
    trace_path = tmp_path / "routing_trace.pt"
    module = SelectiveMemoryModule(
        input_size=dataset.input_size, hidden_size=8, value_size=2,
        variant="dual", read_sparsity=0.5, write_sparsity=0.5,
        router_context_size=4, optimizer_kwargs={"all": {"lr": 1e-3}},
        analysis_path=str(trace_path), implementation=implementation)
    trainer = Trainer(
        fast_dev_run=True, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=tmp_path, accelerator="cpu")
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(module, dataloaders=loader)
    artifact = torch.load(trace_path, weights_only=False)
    for key in ("read_mask", "write_mask", "read_indices", "write_indices",
                "read_scores", "write_scores", "task_loss", "operations",
                "sample_index"):
        assert key in artifact
    assert artifact["implementation"] == implementation
    assert artifact["read_mask"].shape[:2] == (4, 6)


def test_trace_shards_are_sorted_and_deduplicated():
    shards = [
        {"sample_index": torch.tensor([2, 0]), "value": torch.tensor([[20], [0]])},
        {"sample_index": torch.tensor([1, 2]), "value": torch.tensor([[10], [999]])},
    ]
    merged = _merge_trace_shards(deepcopy(shards))
    assert merged["sample_index"].tolist() == [0, 1, 2]
    assert merged["value"].squeeze(1).tolist() == [0, 10, 20]


def test_run_directories_are_unique_and_config_hashed(tmp_path, monkeypatch):
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
