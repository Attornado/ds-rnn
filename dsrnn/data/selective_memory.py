"""Synthetic register programs for the selective-memory experiment."""

from typing import Dict

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


STORE = 0
QUERY = 1
UPDATE = 2
OPERATION_NAMES = ("store", "query", "update")


class SelectiveMemoryDataset(Dataset):
    """Six-step programs that query one register while updating another.

    Every program uses two distinct registers ``A`` and ``B``:

    ``STORE A``, ``STORE B``, ``QUERY A + UPDATE B``,
    ``QUERY B + UPDATE A``, ``QUERY A``, ``QUERY B``.

    Query targets reflect all preceding stores and updates. Values are drawn
    uniformly from ``[-0.8, 0.8]`` so they lie in the useful range of the tanh
    reference recurrent cell.
    """

    sequence_length = 6

    def __init__(self, n_samples: int, n_registers: int = 4,
                 value_size: int = 2, seed: int = 42):
        super().__init__()
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if n_registers < 2:
            raise ValueError("n_registers must be at least 2.")
        if value_size < 1:
            raise ValueError("value_size must be positive.")
        self.n_samples = n_samples
        self.n_registers = n_registers
        self.value_size = value_size
        self.input_size = 3 + 2 * n_registers + value_size

        generator = torch.Generator().manual_seed(seed)
        first = torch.randint(n_registers, (n_samples,), generator=generator)
        second_offset = torch.randint(n_registers - 1, (n_samples,), generator=generator)
        second = second_offset + (second_offset >= first).long()
        values = 1.6 * torch.rand(n_samples, 4, value_size, generator=generator) - 0.8

        self.inputs = torch.zeros(n_samples, self.sequence_length, self.input_size)
        self.targets = torch.zeros(n_samples, self.sequence_length, value_size)
        self.query_mask = torch.zeros(n_samples, self.sequence_length, dtype=torch.bool)
        self.operations = torch.zeros(n_samples, self.sequence_length, 3, dtype=torch.bool)
        self.read_register = torch.full((n_samples, self.sequence_length), -1, dtype=torch.long)
        self.write_register = torch.full((n_samples, self.sequence_length), -1, dtype=torch.long)

        for sample in range(n_samples):
            reg_a, reg_b = int(first[sample]), int(second[sample])
            value_a, value_b, update_a, update_b = values[sample]
            self._write_step(sample, 0, STORE, reg_a, value_a)
            self._write_step(sample, 1, STORE, reg_b, value_b)
            self._joint_step(sample, 2, reg_a, value_a, reg_b, update_b)
            self._joint_step(sample, 3, reg_b, update_b, reg_a, update_a)
            self._query_step(sample, 4, reg_a, update_a)
            self._query_step(sample, 5, reg_b, update_b)

    @property
    def _write_address_offset(self) -> int:
        return 3

    @property
    def _read_address_offset(self) -> int:
        return 3 + self.n_registers

    @property
    def _value_offset(self) -> int:
        return 3 + 2 * self.n_registers

    def _write_step(self, sample: int, timestep: int, operation: int,
                    register: int, value: torch.Tensor) -> None:
        self.operations[sample, timestep, operation] = True
        self.inputs[sample, timestep, operation] = 1.0
        self.inputs[sample, timestep, self._write_address_offset + register] = 1.0
        self.inputs[sample, timestep, self._value_offset:] = value
        self.write_register[sample, timestep] = register

    def _query_step(self, sample: int, timestep: int, register: int,
                    target: torch.Tensor) -> None:
        self.operations[sample, timestep, QUERY] = True
        self.inputs[sample, timestep, QUERY] = 1.0
        self.inputs[sample, timestep, self._read_address_offset + register] = 1.0
        self.targets[sample, timestep] = target
        self.query_mask[sample, timestep] = True
        self.read_register[sample, timestep] = register

    def _joint_step(self, sample: int, timestep: int, query_register: int,
                    target: torch.Tensor, update_register: int,
                    update_value: torch.Tensor) -> None:
        self._query_step(sample, timestep, query_register, target)
        self._write_step(sample, timestep, UPDATE, update_register, update_value)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "sample_index": torch.tensor(index, dtype=torch.long),
            "input": self.inputs[index],
            "target": self.targets[index],
            "query_mask": self.query_mask[index],
            "operations": self.operations[index],
            "read_register": self.read_register[index],
            "write_register": self.write_register[index],
        }


class SelectiveMemoryDataModule(pl.LightningDataModule):
    """Deterministic train/validation/test splits for selective memory."""

    def __init__(self, n_train: int = 256, n_val: int = 64,
                 n_test: int = 64, n_registers: int = 4,
                 value_size: int = 2, batch_size: int = 32,
                 num_workers: int = 0, data_seed: int = 42,
                 split_seed: int = 123):
        super().__init__()
        self.save_hyperparameters()

    @property
    def input_size(self) -> int:
        return 3 + 2 * self.hparams.n_registers + self.hparams.value_size

    def setup(self, stage=None) -> None:
        total = self.hparams.n_train + self.hparams.n_val + self.hparams.n_test
        dataset = SelectiveMemoryDataset(
            total, n_registers=self.hparams.n_registers,
            value_size=self.hparams.value_size, seed=self.hparams.data_seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [self.hparams.n_train, self.hparams.n_val, self.hparams.n_test],
            generator=torch.Generator().manual_seed(self.hparams.split_seed),
        )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        generator = None
        if shuffle:
            # Keep sample order matched across variants even though their
            # different router counts consume different model-init RNG draws.
            generator = torch.Generator().manual_seed(self.hparams.split_seed + 1)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=self.hparams.num_workers,
                          pin_memory=torch.cuda.is_available(), generator=generator)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
