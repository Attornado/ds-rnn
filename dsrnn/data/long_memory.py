"""Random register programs with controlled target-retention delays."""

import torch

from dsrnn.data.selective_memory import (
    SelectiveMemoryDataset, SelectiveMemoryDataModule, STORE, UPDATE,
)


class LongMemoryDataset(SelectiveMemoryDataset):
    """Fixed-length, unpadded programs; target metadata is never an input.

    Initialize every register in random order. Insert a randomly timed target
    UPDATE and a later target QUERY separated by exactly ``delay`` operations.
    Other operations randomly query/update distractors, including standalone
    UPDATEs so the target write has no special input marker. Query values always
    refer to the pre-update state.
    """

    def __init__(self, n_samples, n_registers=8, value_size=4, seed=42,
                 delays=(0, 8, 32, 128), horizon=128):
        if n_samples < 1 or n_registers < 3 or value_size < 1:
            raise ValueError("Need positive samples/value size and at least 3 registers.")
        if not delays or len(set(delays)) != len(delays) or min(delays) < 0:
            raise ValueError("delays must be distinct nonnegative integers.")
        if max(delays) > horizon:
            raise ValueError("horizon must cover every requested delay.")
        self.n_samples, self.n_registers, self.value_size = n_samples, n_registers, value_size
        self.input_size = 3 + 2 * n_registers + value_size
        self.sequence_length = n_registers + horizon + 18
        shape = (n_samples, self.sequence_length)
        self.inputs = torch.zeros(*shape, self.input_size)
        self.targets = torch.zeros(*shape, value_size)
        self.query_mask = torch.zeros(shape, dtype=torch.bool)
        self.operations = torch.zeros(*shape, 3, dtype=torch.bool)
        self.read_register = torch.full(shape, -1, dtype=torch.long)
        self.write_register = torch.full(shape, -1, dtype=torch.long)
        self.target_query = torch.zeros(shape, dtype=torch.bool)
        self.retention_delay = torch.full(shape, -1, dtype=torch.long)
        self.source_timestep = torch.full(shape, -1, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        # Balanced delay counts, randomly ordered independently of program RNG.
        delay_order = torch.randperm(n_samples, generator=generator)
        for sample in range(n_samples):
            delay = delays[int(delay_order[sample]) % len(delays)]
            target_register = int(torch.randint(n_registers, (), generator=generator))
            others = [r for r in range(n_registers) if r != target_register]
            source_t = int(torch.randint(n_registers, self.sequence_length - delay - 1,
                                         (), generator=generator))
            query_t = source_t + delay + 1
            forced_query = next(k for k in range(n_registers, n_registers + 3)
                                if k not in (source_t, query_t))
            values = torch.zeros(n_registers, value_size)
            latest = torch.full((n_registers,), -1, dtype=torch.long)
            order = torch.randperm(n_registers, generator=generator).tolist()
            for t in range(self.sequence_length):
                write_reg = read_reg = None
                op = UPDATE
                if t < n_registers:
                    write_reg, op = order[t], STORE
                elif t == source_t:
                    write_reg = target_register
                else:
                    read_reg = (target_register if t == query_t else
                                others[int(torch.randint(len(others), (), generator=generator))])
                    # Both pure and joint queries; addresses are always distinct.
                    choice = float(torch.rand((), generator=generator))
                    if choice < 0.75:
                        candidates = [r for r in others if r != read_reg]
                        write_reg = candidates[int(torch.randint(len(candidates), (), generator=generator))]
                    # Guarantee one distractor query even in tiny test programs.
                    if choice < 0.25 and t not in (query_t, forced_query):
                        read_reg = None
                if read_reg is not None:
                    self._query_step(sample, t, read_reg, values[read_reg])
                    self.source_timestep[sample, t] = latest[read_reg]
                    self.retention_delay[sample, t] = t - latest[read_reg] - 1
                if write_reg is not None:
                    value = torch.rand(value_size, generator=generator) * 1.6 - 0.8
                    self._write_step(sample, t, op, write_reg, value)
                    values[write_reg] = value
                    latest[write_reg] = t
            self.target_query[sample, query_t] = True

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item.update(target_query=self.target_query[index],
                    retention_delay=self.retention_delay[index],
                    source_timestep=self.source_timestep[index])
        return item


class LongMemoryDataModule(SelectiveMemoryDataModule):
    def __init__(self, *, delays=(0, 8, 32, 128), horizon=128, **kwargs):
        super().__init__(**kwargs)
        self.delays, self.horizon = tuple(delays), horizon

    def setup(self, stage=None):
        # Separate deterministic RNG streams, invariant to other split sizes.
        for offset, split in enumerate(("train", "val", "test")):
            if stage == "test" and split != "test":
                continue
            if stage == "fit" and split == "test":
                continue
            if hasattr(self, split + "_dataset"):
                continue
            seed = self.hparams.data_seed + 1000003 * (self.hparams.split_seed + offset)
            setattr(self, split + "_dataset", LongMemoryDataset(
                self.hparams["n_" + split], self.hparams.n_registers,
                self.hparams.value_size, seed=seed, delays=self.delays,
                horizon=self.horizon))
