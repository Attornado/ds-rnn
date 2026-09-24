"""Pixel task preprocessing, routing, and single-run training integration."""
import json

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset

from experiments import train_pixels as runner
from dsrnn.data.classification import SequentialMNISTDataModule
from dsrnn.data.lra import CIFAR10SeqDataModule


class FakeMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.transform = transform

    def __len__(self):
        return 20

    def __getitem__(self, index):
        pixels = torch.arange(784).remainder(256).byte().reshape(28, 28).numpy()
        image = Image.fromarray(pixels)
        return self.transform(image), index % 10


def test_permutation_and_split_survive_setup(monkeypatch):
    monkeypatch.setattr("dsrnn.data.classification.datasets.MNIST", FakeMNIST)
    scan = SequentialMNISTDataModule(subseq_len=1, split_seed=4)
    scan.setup()
    modules = [SequentialMNISTDataModule(subseq_len=1, sequential_order="random",
                                        split_seed=4, permutation_seed=7) for _ in range(2)]
    for data in modules:
        data.setup("fit")
        indices = data.train_dataset.indices.copy()
        data.setup("validate")
        data.setup("test")
        data.setup("fit")
        assert data.train_dataset.indices == indices == scan.train_dataset.indices
        expected = scan.test_dataset[0][0][data.permutation]
        assert expected.shape == (784, 1)
        assert not torch.equal(expected, scan.test_dataset[0][0])
        for dataset in (data.train_dataset, data.val_dataset, data.test_dataset):
            torch.testing.assert_close(dataset[0][0], expected)
    torch.testing.assert_close(modules[0].permutation, modules[1].permutation)


@pytest.mark.parametrize("cell", runner.RECURRENT_CELLS)
@pytest.mark.parametrize("variant", runner.MODEL_VARIANTS)
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_model_routing_and_equivalence(cell, variant, device):
    reference = runner.PixelRNN(1, 8, 10, variant, recurrent_cell=cell).to(device)
    sparse = runner.PixelRNN(1, 8, 10, variant, recurrent_cell=cell, implementation="sparse").to(device)
    sparse.load_state_dict(reference.state_dict())
    x = torch.randn(3, 6, 1, device=device)
    logits, trace = reference(x, return_routing=True)
    assert logits.shape == (3, 10)
    torch.testing.assert_close(logits, sparse(x), atol=1e-6, rtol=1e-5)
    for side in ("read", "write"):
        assert torch.all(trace[side + "_mask"].sum(-1) == getattr(reference, side + "_budget"))
    for model in (reference, sparse):
        torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 2, 4], device=device)).backward()
        for router in (model.read_router, model.write_router, model.shared_router):
            if router is not None:
                grad = router.backbone.weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for left, right in zip(reference.parameters(), sparse.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("task", ["sMNIST", "psMNIST", "sCIFAR"])
@pytest.mark.parametrize("skip_test", [False, True])
def test_runner_fit_and_best_checkpoint(tmp_path, monkeypatch, task, skip_test):
    monkeypatch.setattr("dsrnn.data.classification.datasets.MNIST", FakeMNIST)
    monkeypatch.setattr(SequentialMNISTDataModule, "prepare_data", lambda self: None)
    monkeypatch.setattr(CIFAR10SeqDataModule, "prepare_data", lambda self: None)
    # Shorten only this integration fixture; preprocessing is tested separately.
    monkeypatch.setattr("dsrnn.data.classification.PixelSequence.__call__",
                        lambda self, image: image.flatten()[:4].reshape(-1, 1))
    monkeypatch.setattr(CIFAR10SeqDataModule, "_load_split",
                        lambda self, split: TensorDataset(torch.rand(20, 4, 1), torch.arange(20) % 10))
    argv = ["train_pixels", "--task", task, "--hidden_size", "4", "--batch_size", "8",
            "--epochs", "1", "--accelerator", "cpu", "--log_dir", str(tmp_path), "--run_id", "smoke"]
    monkeypatch.setattr("sys.argv", argv + (["--skip_test"] if skip_test else []))
    runner.main()
    result = json.loads((tmp_path / "smoke/results.json").read_text())
    assert result["complete"]
    assert result["result"]["validation"]["val/loss"] == pytest.approx(result["result"]["best_validation_loss"])
    assert bool(result["result"]["evaluation"]) is not skip_test
    with pytest.raises(FileExistsError):
        runner.main()


def test_yaml_cli_precedence(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("task: psMNIST\nmodel: dense\nhidden_size: 16\nwandb: true\n")
    monkeypatch.setattr("sys.argv", ["pixels", "--config", str(config), "--hidden_size", "8", "--no-wandb"])
    args = runner._load_config(runner.build_parser())
    assert (args.task, args.model, args.hidden_size, args.wandb) == ("psMNIST", "dense", 8, False)


def test_cifar_preprocessing_and_seeded_split(monkeypatch):
    import numpy as np

    pixels = np.concatenate([np.full((4, 1024), channel, dtype=np.uint8)
                             for channel in (255, 128, 0)], axis=1)
    monkeypatch.setattr(CIFAR10SeqDataModule, "_unpickle",
                        staticmethod(lambda path: {b"data": pixels, b"labels": [0, 1, 2, 3]}))
    first = CIFAR10SeqDataModule(split_seed=42)
    second = CIFAR10SeqDataModule(split_seed=42)
    first.setup("validate")
    second.setup("fit")
    assert first.train_dataset.indices == second.train_dataset.indices
    assert set(first.train_dataset.indices).isdisjoint(first.val_dataset.indices)
    indices = first.train_dataset.indices.copy()
    first.setup("fit")
    first.setup("test")
    assert first.train_dataset.indices == indices
    image, label = first.test_dataset[0]
    assert image.shape == (1024, 1) and image.dtype == torch.float32
    torch.testing.assert_close(image, torch.full_like(image, (0.2989 * 255 + 0.5870 * 128) / 255))
    assert label.item() == 0


@pytest.mark.parametrize("length", [784, 1024])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_full_pixel_sequence_backward(length, batch_size):
    model = runner.PixelRNN(1, 4, 10, "dual", recurrent_cell="gru")
    output = model(torch.rand(batch_size, length, 1))
    assert output.shape == (batch_size, 10)
    output.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
