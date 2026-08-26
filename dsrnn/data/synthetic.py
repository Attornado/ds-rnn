import hashlib
import json
import os
import lightning.pytorch as pl
import torch
from torch.utils.data import TensorDataset, DataLoader


class TemporalSyntheticDataset(torch.utils.data.Dataset):
    """Synthetic temporal tasks from random dynamical-system teachers.

    Each task maps sequences x ~ N(0, I) to targets through an independent
    teacher system with parameters interpolated between a shared component and
    a task-specific one (shared_fraction controls task similarity).
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        input_dim: int,
        output_dim: int,
        temporal_type: str = "linear_dynamical",
        normalize_output: bool = True,
        hidden_dim: int = 64,
        spectral_radius: float = 0.9,
        noise_std: float = 0.0,
        task_id: int = 0,
        shared_fraction: float = 0.0,
        global_seed: int = 42,
    ):
        super().__init__()

        self.n_samples = n_samples
        self.seq_len = seq_len

        g_task = torch.Generator().manual_seed(task_id)
        g_shared = torch.Generator().manual_seed(global_seed)

        alpha = shared_fraction

        A_shared = torch.randn(hidden_dim, hidden_dim, generator=g_shared)
        B_shared = torch.randn(input_dim, hidden_dim, generator=g_shared)
        C_shared = torch.randn(hidden_dim, output_dim, generator=g_shared)

        A_task = torch.randn(hidden_dim, hidden_dim, generator=g_task)
        B_task = torch.randn(input_dim, hidden_dim, generator=g_task)
        C_task = torch.randn(hidden_dim, output_dim, generator=g_task)

        self.A = alpha * A_shared + (1 - alpha) * A_task
        self.B = alpha * B_shared + (1 - alpha) * B_task
        self.C = alpha * C_shared + (1 - alpha) * C_task

        eigvals = torch.linalg.eigvals(self.A).abs()
        max_eig = eigvals.max().real + 1e-6
        self.A = self.A * (spectral_radius / max_eig)

        self.temporal_type = temporal_type
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.normalize_output = normalize_output

        self.X = torch.randn(n_samples, seq_len, input_dim, generator=g_task)
        self.y = torch.zeros(n_samples, seq_len, output_dim)

        for i in range(n_samples):
            h = torch.zeros(hidden_dim)

            for t in range(seq_len):
                x_t = self.X[i, t]

                if temporal_type == "linear_dynamical":
                    h = self.A.t() @ h + self.B.t() @ x_t
                elif temporal_type == "rnn_teacher":
                    h = torch.tanh(self.A.t() @ h + self.B.t() @ x_t)
                else:
                    raise ValueError(f"Unknown temporal_type: {temporal_type}")

                y_t = self.C.t() @ h
                self.y[i, t] = y_t

        if self.noise_std > 0:
            self.y += self.noise_std * torch.randn_like(self.y)

        if self.normalize_output:
            mean = self.y.mean(dim=(0, 1), keepdim=True)
            std = self.y.std(dim=(0, 1), keepdim=True) + 1e-6
            self.y = (self.y - mean) / std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataset_id(hparams: dict) -> str:
    excluded = ["batch_size"]
    filtered = {k: v for k, v in hparams.items() if k not in excluded}

    s = json.dumps(filtered, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


class CachedTemporalTaskDataModule(pl.LightningDataModule):
    """Cached datamodule for synthetic temporal tasks."""

    def __init__(
        self,
        data_dir="./synthetic_cache",
        n_samples=10000,
        seq_len=50,
        input_dim=16,
        output_dim=4,
        batch_size=32,
        val_split=0.1,
        test_split=0.1,
        temporal_type="linear_dynamical",
        hidden_dim=64,
        spectral_radius=0.9,
        noise_std=0.0,
        task_id=0,
        shared_fraction=0.0,
        global_seed=42,
        split_seed=123,
    ):
        super().__init__()
        self.save_hyperparameters()
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

    def setup(self, stage=None):
        dataset_id = make_dataset_id(dict(self.hparams))
        path = os.path.join(self.data_dir, f"{dataset_id}.pt")

        if os.path.exists(path):
            print(f"Loading cached dataset: {path}")
            data = torch.load(path)
        else:
            print(f"Generating dataset: {path}")

            dataset = TemporalSyntheticDataset(
                n_samples=self.hparams.n_samples,
                seq_len=self.hparams.seq_len,
                input_dim=self.hparams.input_dim,
                output_dim=self.hparams.output_dim,
                temporal_type=self.hparams.temporal_type,
                hidden_dim=self.hparams.hidden_dim,
                spectral_radius=self.hparams.spectral_radius,
                noise_std=self.hparams.noise_std,
                task_id=self.hparams.task_id,
                shared_fraction=self.hparams.shared_fraction,
                global_seed=self.hparams.global_seed,
            )

            X, y = dataset.X, dataset.y

            g = torch.Generator().manual_seed(self.hparams.split_seed)
            perm = torch.randperm(len(X), generator=g)

            val_size = int(self.hparams.val_split * len(X))
            test_size = int(self.hparams.test_split * len(X))
            test_idx = perm[:test_size]
            val_idx = perm[test_size:test_size + val_size]
            train_idx = perm[test_size + val_size:]

            data = {
                "X_train": X[train_idx],
                "y_train": y[train_idx],
                "X_val": X[val_idx],
                "y_val": y[val_idx],
                "X_test": X[test_idx],
                "y_test": y[test_idx],
            }

            torch.save(data, path)

        self.train_dataset = TensorDataset(data["X_train"], data["y_train"])
        self.val_dataset = TensorDataset(data["X_val"], data["y_val"])
        self.test_dataset = TensorDataset(data["X_test"], data["y_test"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
