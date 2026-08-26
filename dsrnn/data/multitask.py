from typing import Dict, List, Any, Optional, Tuple
import lightning as lit
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from dsrnn.data import SequentialMNISTDataModule, HARDataModule, HandGestureDataModule, OccupancyDataModule, \
    OzoneDataModule, TrafficDataModule, PowerConsumptionDataModule, CheetahDataModule
from dsrnn.data.synthetic import CachedTemporalTaskDataModule, TemporalSyntheticDataset


class MultiTaskDataset(Dataset):
    """
    Dataset wrapper combining multiple task datasets while preserving task identity.

    Each sample is returned as a dict with keys 'x', 'y', 'task', 'task_id'.
    """

    def __init__(self, datasets: Dict[str, Dataset]):
        self.datasets = datasets
        self.mapping = []

        for task_id, (task, ds) in enumerate(datasets.items()):
            for i in range(len(ds)):
                self.mapping.append((task_id, task, i))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        task_id, task, sample_idx = self.mapping[idx]
        x, y = self.datasets[task][sample_idx]

        return dict(
            x=x,
            y=y,
            task=task,
            task_id=torch.tensor(task_id),
        )


class BalancedBatchSampler(Sampler):
    """
    Batch sampler that balances samples across multiple tasks.

    Each batch contains an equal number of samples from each task. If cycle is
    True, smaller datasets are cycled to match the largest one.
    """

    def __init__(self, dataset, batch_size, cycle: bool = False):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.cycle = cycle

        self.task_indices = {}
        for idx, (task_id, task, _) in enumerate(dataset.mapping):
            self.task_indices.setdefault(task_id, []).append(idx)

        self.tasks = list(self.task_indices.keys())
        self.samples_per_task = batch_size // len(self.tasks)

        sizes = [len(v) for v in self.task_indices.values()]
        driver = max(sizes) if cycle else min(sizes)
        self.num_batches = driver // self.samples_per_task

    def __iter__(self):
        if self.cycle:
            needed = self.num_batches * self.samples_per_task
            shuffled = {
                t: np.resize(np.random.permutation(v), needed)
                for t, v in self.task_indices.items()
            }
        else:
            shuffled = {
                t: np.random.permutation(v)
                for t, v in self.task_indices.items()
            }

        for i in range(self.num_batches):
            batch = []
            start = i * self.samples_per_task
            end = start + self.samples_per_task
            for t in self.tasks:
                batch.extend(shuffled[t][start:end])
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def to_2d(t: torch.Tensor) -> torch.Tensor:
    """Canonicalize a tensor to shape [T, D]."""
    if t.dim() == 0:
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 1:
        return t.unsqueeze(-1)
    return t


def pad_to(t: torch.Tensor, target_T: int, target_D: int) -> torch.Tensor:
    """Pad a 2D tensor to [target_T, target_D] with zeros."""
    T, D = t.shape
    return torch.nn.functional.pad(t, (0, target_D - D, 0, target_T - T))


def multitask_collate(batch: List[Dict[str, Any]],
                      max_t_x: Optional[int] = None,
                      max_d_x: Optional[int] = None,
                      max_t_y: Optional[int] = None,
                      max_d_y: Optional[int] = None) -> Dict[str, Any]:
    """Collate function for batching multitask samples with variable shapes."""
    max_t_x = max_t_x or -1
    max_d_x = max_d_x or -1
    max_t_y = max_t_y or -1
    max_d_y = max_d_y or -1

    xs = [to_2d(b["x"].float()) for b in batch]
    ys = [to_2d((b["y"] if isinstance(b["y"], torch.Tensor) else torch.tensor(b["y"])).float()) for b in batch]

    max_t_x_ = max(max(t.shape[0] for t in xs), max_t_x)
    max_d_x_ = max(max(t.shape[1] for t in xs), max_d_x)
    max_t_y_ = max(max(t.shape[0] for t in ys), max_t_y)
    max_d_y_ = max(max(t.shape[1] for t in ys), max_d_y)

    xs_padded = torch.stack([pad_to(x, max_t_x_, max_d_x_) for x in xs])
    ys_padded = torch.stack([pad_to(y, max_t_y_, max_d_y_) for y in ys])

    x_lengths = torch.tensor([list(x.shape) for x in xs])
    y_lengths = torch.tensor([list(y.shape) for y in ys])

    task_ids = torch.stack([b["task_id"] for b in batch])

    return dict(
        x=xs_padded,
        y=ys_padded,
        x_lengths=x_lengths,
        y_lengths=y_lengths,
        task_id=task_ids,
        task=np.array([b["task"] for b in batch]),
    )


class MultiTaskCollate:
    """Callable wrapper around multitask_collate with fixed max shapes."""

    def __init__(self,
                 max_t_x: Optional[int] = None,
                 max_d_x: Optional[int] = None,
                 max_t_y: Optional[int] = None,
                 max_d_y: Optional[int] = None):
        self._max_t_x = max_t_x
        self._max_d_x = max_d_x
        self._max_t_y = max_t_y
        self._max_d_y = max_d_y

    @property
    def max_t_x(self) -> Optional[int]:
        return self._max_t_x

    @property
    def max_d_x(self) -> Optional[int]:
        return self._max_d_x

    @property
    def max_t_y(self) -> Optional[int]:
        return self._max_t_y

    @property
    def max_d_y(self) -> Optional[int]:
        return self._max_d_y

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return multitask_collate(batch=batch,
                                 max_t_x=self.max_t_x,
                                 max_d_x=self.max_d_x,
                                 max_t_y=self.max_t_y,
                                 max_d_y=self.max_d_y)


class MultiTaskDataModule(lit.LightningDataModule):
    """DataModule combining multiple task datamodules for multitask training."""

    def __init__(self, tasks: List[str],
                 batch_size: int,
                 num_workers: int,
                 cycle_smaller: bool = True,
                 kwargs_per_dataset: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.save_hyperparameters()
        self.tasks = tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cycle_smaller = cycle_smaller
        self.kwargs_per_dataset = kwargs_per_dataset if kwargs_per_dataset is not None else {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.max_t_x = None
        self.max_d_x = None
        self.max_t_y = None
        self.max_d_y = None
        self.dms = {}

    def _infer_max_t_and_d(self,
                           stage: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        examples = None
        if stage == "fit":
            examples = [ds[0] for name, ds in self.train_dataset.datasets.items()]
        elif stage == "test":
            examples = [ds[0] for name, ds in self.test_dataset.datasets.items()]

        if examples is not None:
            xs = [to_2d(b[0].float()) for b in examples]
            ys = [to_2d((b[1] if isinstance(b[1], torch.Tensor) else torch.tensor(b[1])).float())
                  for b in examples]

            max_t_x_ = max(t.shape[0] for t in xs)
            max_d_x_ = max(t.shape[1] for t in xs)
            max_t_y_ = max(t.shape[0] for t in ys)
            max_d_y_ = max(t.shape[1] for t in ys)
            return max_t_x_, max_d_x_, max_t_y_, max_d_y_

        return None, None, None, None

    def setup(self, stage: Optional[str] = None):
        self.dms = {}
        for task in self.tasks:
            kwargs_task = self.kwargs_per_dataset.get(task, {})
            dm = create_datamodule(task, self.batch_size, self.num_workers, **kwargs_task)
            dm.setup(stage)
            self.dms[task] = dm

        train_sets = {t: dm.train_dataset for t, dm in self.dms.items()}
        val_sets = {t: dm.val_dataset for t, dm in self.dms.items()}
        test_sets = {t: dm.test_dataset for t, dm in self.dms.items()}

        if stage == "fit":
            self.train_dataset = MultiTaskDataset(train_sets)
            self.val_dataset = MultiTaskDataset(val_sets)

        if stage == "test":
            self.test_dataset = MultiTaskDataset(test_sets)

        self.max_t_x, self.max_d_x, self.max_t_y, self.max_d_y = self._infer_max_t_and_d(stage)

    def train_dataloader(self):
        sampler = BalancedBatchSampler(
            self.train_dataset,
            self.batch_size,
            cycle=self.cycle_smaller,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=multitask_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=MultiTaskCollate(max_t_x=self.max_t_x,
                                        max_d_x=self.max_d_x,
                                        max_t_y=self.max_t_y,
                                        max_d_y=self.max_d_y)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=MultiTaskCollate(max_t_x=self.max_t_x,
                                        max_d_x=self.max_d_x,
                                        max_t_y=self.max_t_y,
                                        max_d_y=self.max_d_y)
        )


def create_datamodule(task: str, batch_size: int, num_workers: int, **kwargs) -> lit.LightningDataModule:
    """Factory for per-task datamodules used by MultiTaskDataModule."""
    if task == "seqmnist":
        return SequentialMNISTDataModule(
            root="_data/mnist",
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=0.1,
            subseq_len=28,
        )
    elif task == "permseqmnist":
        return SequentialMNISTDataModule(
            root="_data/mnist",
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=0.1,
            sequential_order="random",
            subseq_len=28,
        )
    elif task == "har":
        return HARDataModule(
            data_path="_data/har",
            batch_size=batch_size,
            num_workers=num_workers,
            sequence_length=32,
        )
    elif task == "gesture":
        return HandGestureDataModule(
            data_path="_data/gesture",
            batch_size=batch_size,
            num_workers=num_workers,
            sequence_length=32,
        )
    elif task == "occupancy":
        return OccupancyDataModule(
            data_path="_data/occupancy",
            batch_size=batch_size,
            num_workers=num_workers,
            sequence_length=32,
        )
    elif task == "ozone":
        return OzoneDataModule(
            data_path="_data/ozone",
            batch_size=batch_size,
            num_workers=num_workers,
            sequence_length=32,
        )
    elif task == "traffic":
        return TrafficDataModule(
            data_path="_data/traffic",
            batch_size=batch_size,
            num_workers=num_workers,
            sequence_length=32,
        )
    elif task == "power":
        return PowerConsumptionDataModule(
            data_path="_data/power",
            batch_size=batch_size,
            num_workers=num_workers,
            sequence_length=32,
        )
    elif task == "cheetah":
        return CheetahDataModule(
            data_path="_data/cheetah",
            batch_size=batch_size,
            num_workers=num_workers,
            sequence_length=32,
        )
    elif "synth" in task:
        seed = 42
        if "synth-linear" in task:
            temporal_type = "linear_dynamical"
        else:
            temporal_type = "rnn_teacher"
        task_id = int(task.split("-")[-1])
        return CachedTemporalTaskDataModule(
            data_dir=f"_data/{task}",
            batch_size=batch_size,
            temporal_type=temporal_type,
            task_id=task_id,
            global_seed=seed,
            **kwargs,
        )
    else:
        raise ValueError(task)
