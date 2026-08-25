import os
from typing import Optional

import lightning as lit
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class CheetahDataModule(lit.LightningDataModule):
    """DataModule for the HalfCheetah kinematic modeling dataset.

    Loads poses from .npy files and creates sequences for next-step prediction
    (x: t..t+N, y: t+1..t+N+1).
    """

    def __init__(
        self,
        data_path: str = "./data/cheetah",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        sequence_length: int = 32,
        inc: int = 10,
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.seq_len = sequence_length
        self.inc = inc

        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    def prepare_data(self):
        """Verify the data is present. There is no automatic download."""
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(
                f"Data folder '{self.data_path}' was not found. "
                "Make sure the Cheetah .npy files are present."
            )

    def _cut_in_sequences(self, x: np.ndarray):
        sequences_x = []
        sequences_y = []

        for s in range(0, x.shape[0] - self.seq_len - 1, self.inc):
            start = s
            end = start + self.seq_len
            sequences_x.append(x[start:end])
            sequences_y.append(x[start + 1: end + 1])

        return sequences_x, sequences_y

    def _load_files(self, files: list):
        all_x, all_y = [], []
        for f_path in files:
            arr = np.load(f_path).astype(np.float32)
            x_list, y_list = self._cut_in_sequences(arr)
            all_x.extend(x_list)
            all_y.extend(y_list)

        return np.stack(all_x, axis=1), np.stack(all_y, axis=1)

    def setup(self, stage: Optional[str] = None):
        all_files = sorted(
            [
                os.path.join(self.data_path, d)
                for d in os.listdir(self.data_path)
                if d.endswith(".npy")
            ]
        )

        train_files = all_files[15:25]
        test_files = all_files[5:15]
        valid_files = all_files[:5]

        self.train_x, self.train_y = self._load_files(train_files)
        self.val_x, self.val_y = self._load_files(valid_files)
        self.test_x, self.test_y = self._load_files(test_files)

        def to_dataset(x_arr, y_arr):
            x_t = torch.tensor(x_arr.transpose(1, 0, 2), dtype=torch.float32)
            y_t = torch.tensor(y_arr.transpose(1, 0, 2), dtype=torch.float32)
            return TensorDataset(x_t, y_t)

        self.train_dataset = to_dataset(self.train_x, self.train_y)
        self.val_dataset = to_dataset(self.val_x, self.val_y)
        self.test_dataset = to_dataset(self.test_x, self.test_y)

        print(
            f"Datasets created: {len(self.train_dataset)} train, {len(self.val_dataset)} val, "
            f"{len(self.test_dataset)} test."
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
