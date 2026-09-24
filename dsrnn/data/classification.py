import io
import os
import urllib.request
from ucimlrepo import fetch_ucirepo
import zipfile
from typing import Optional, Dict, Callable, Tuple, List
import lightning as lit
import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torchvision import datasets, transforms


class LabelRemappingDataset(Dataset):

    def __init__(self, dataset: Dataset, label_map: Dict[int, int]):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, self.label_map[label]


class PixelSequence:
    """Flatten an image tensor into pixel chunks with an optional fixed order."""

    def __init__(self, subseq_len: int, permutation: Optional[torch.Tensor] = None):
        self.subseq_len = subseq_len
        self.permutation = permutation

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        pixels = image.flatten()
        if self.permutation is not None:
            pixels = pixels[self.permutation]
        return pixels.reshape(-1, self.subseq_len)


class SequentialMNISTDataModule(lit.LightningDataModule):
    """LightningDataModule for the sequential MNIST benchmark.

    Each MNIST image (28x28) is flattened into a sequence presented in chunks
    of subseq_len pixels. The target is the digit class (0-9).
    """

    def __init__(
            self,
            root: str = "_data",
            batch_size: int = 64,
            num_workers: int = 4,
            pin_memory: bool = False,
            val_split: float = 0.1,
            sequential_order: str = "scan",
            subseq_len: int = 28,
            data_path: str = "",
            split_seed: Optional[int] = None,
            permutation_seed: Optional[int] = None,
    ):
        super().__init__()

        assert (28 * 28) % subseq_len == 0, (
            f"Invalid sub-sequence length: it must divide image size {28 * 28}."
        )
        self._root: str = root
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers
        self._pin_memory: bool = pin_memory
        self._val_split: float = val_split
        self._sequential_order: str = sequential_order
        self._subseq_len: int = subseq_len
        self.split_seed = split_seed
        if sequential_order not in ("scan", "random"):
            raise ValueError("sequential_order must be 'scan' or 'random'.")
        if not 0 < val_split < 1:
            raise ValueError("val_split must be between 0 and 1.")
        generator = (torch.Generator().manual_seed(permutation_seed)
                     if permutation_seed is not None else None)
        self.permutation = (torch.randperm(784, generator=generator)
                            if sequential_order == "random" else None)

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                PixelSequence(subseq_len, self.permutation),
            ]
        )

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    @property
    def root(self) -> str:
        return self._root

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def pin_memory(self) -> bool:
        return self._pin_memory

    @property
    def val_split(self) -> float:
        return self._val_split

    @property
    def sequential_order(self) -> str:
        return self._sequential_order

    @property
    def subseq_len(self) -> int:
        return self._subseq_len

    @property
    def transform(self) -> transforms.Compose:
        return self._transform

    def prepare_data(self):
        datasets.MNIST(self.root, train=True, download=True)
        datasets.MNIST(self.root, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit", "validate") and self.train_dataset is None:
            full_train = datasets.MNIST(
                self.root,
                train=True,
                transform=self.transform,
            )
            val_size = int(self.val_split * len(full_train))
            train_size = len(full_train) - val_size
            self.train_dataset, self.val_dataset = random_split(
                full_train, [train_size, val_size],
                generator=(torch.Generator().manual_seed(self.split_seed)
                           if self.split_seed is not None else None),
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(
                self.root,
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
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


class HandGestureDataModule(lit.LightningDataModule):
    _TRAINING_FILES = [
        "a3_va3.csv",
        "b1_va3.csv",
        "b3_va3.csv",
        "c1_va3.csv",
        "c3_va3.csv",
        "a2_va3.csv",
        "a1_va3.csv",
    ]
    _PHASE_MAP = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}

    def __init__(
        self,
        data_path: str = "_data/gesture",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        val_split: float = 0.1,
        test_split: float = 0.15,
        sequence_length: int = 32,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.val_split = val_split
        self.test_split = test_split
        self.seq_len = sequence_length
        self.interleaved_train = True

        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def setup(self, stage: Optional[str] = None):
        x_list, y_list = [], []

        for fname in self._TRAINING_FILES:
            df = pd.read_csv(os.path.join(self.data_path, fname), header=0)
            phases = df["Phase"].values
            y = np.array([self._PHASE_MAP[p] for p in phases], dtype=np.int64)
            x = df.values[:, :-1].astype(np.float32)

            num_seq = x.shape[0] // self.seq_len
            for s in range(num_seq):
                start = s * self.seq_len
                end = start + self.seq_len
                x_list.append(x[start:end])
                y_list.append(y[start:end])

                if self.interleaved_train and s < num_seq - 1:
                    offset = start + self.seq_len // 2
                    x_list.append(x[offset: offset + self.seq_len])
                    y_list.append(y[offset: offset + self.seq_len])

        x_arr = np.stack(x_list, axis=1)
        y_arr = np.stack(y_list, axis=1)

        flat = x_arr.reshape(-1, x_arr.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        x_arr = (x_arr - self.mean) / (self.std + 1e-8)

        X_np = x_arr.transpose(1, 0, 2)
        Y_np = y_arr.transpose(1, 0)

        X = torch.tensor(X_np, dtype=torch.float32)
        Y = torch.tensor(Y_np, dtype=torch.long)

        full_dataset = TensorDataset(X, Y)

        total = len(full_dataset)
        val_len = int(self.val_split * total)
        test_len = int(self.test_split * total)
        train_len = total - val_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
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

    def get_normalization_params(self):
        return self.mean, self.std


class OccupancyDataModule(lit.LightningDataModule):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"
    ZIP_NAME = "occupancy_data.zip"
    DIR_NAME = "occupancy_data"

    def __init__(
            self,
            data_path: str = "_data/occupancy",
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = False,
            val_split: float = 0.1,
            test_split: float = 0.15,
            sequence_length: int = 32,
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.val_split = val_split
        self.test_split = test_split
        self.seq_len = sequence_length

        self.inc_train = 1
        self.inc_test = 8

        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def _read_file(self, filename: str):
        df = pd.read_csv(filename)
        x = np.stack(
            [
                df["Temperature"].values,
                df["Humidity"].values,
                df["Light"].values,
                df["CO2"].values,
                df["HumidityRatio"].values,
            ],
            axis=-1,
        ).astype(np.float32)
        y = df["Occupancy"].values.astype(np.int64)
        return x, y

    def _cut_in_sequences(self, x: np.ndarray, y: np.ndarray, inc: int):
        seq_x, seq_y = [], []
        for s in range(0, x.shape[0] - self.seq_len, inc):
            seq_x.append(x[s: s + self.seq_len])
            seq_y.append(y[s: s + self.seq_len])
        return np.stack(seq_x, axis=1), np.stack(seq_y, axis=1)

    def setup(self, stage: Optional[str] = None):
        train_x, train_y = self._read_file(
            os.path.join(self.data_path, "datatraining.txt")
        )
        test0_x, test0_y = self._read_file(os.path.join(self.data_path, "datatest.txt"))
        test1_x, test1_y = self._read_file(
            os.path.join(self.data_path, "datatest2.txt")
        )

        flat = train_x
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        train_x = (train_x - self.mean) / self.std
        test0_x = (test0_x - self.mean) / self.std
        test1_x = (test1_x - self.mean) / self.std

        tx, ty = self._cut_in_sequences(train_x, train_y, inc=self.inc_train)
        t0x, t0y = self._cut_in_sequences(test0_x, test0_y, inc=self.inc_test)
        t1x, t1y = self._cut_in_sequences(test1_x, test1_y, inc=self.inc_test)

        test_x = np.concatenate([t0x, t1x], axis=1)
        test_y = np.concatenate([t0y, t1y], axis=1)

        def to_dataset(x_arr, y_arr):
            X = torch.tensor(x_arr.transpose(1, 0, 2), dtype=torch.float32)
            Y = torch.tensor(y_arr.transpose(1, 0), dtype=torch.long)
            return TensorDataset(X, Y)

        full_train_dataset = to_dataset(tx, ty)

        total = len(full_train_dataset)
        val_len = int(self.val_split * total)
        train_len = total - val_len

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, lengths=[train_len, val_len]
        )
        self.test_dataset = to_dataset(test_x, test_y)

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

    def get_normalization_params(self):
        return self.mean, self.std


class HARDataModule(lit.LightningDataModule):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    ZIP_NAME = "UCI_HAR_Dataset.zip"
    DIR_NAME = "UCI HAR Dataset"

    def __init__(
            self,
            data_path: str = "_data/har",
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = False,
            val_split: float = 0.1,
            test_split: float = 0.15,
            sequence_length: int = 16,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.val_split = val_split
        self.test_split = test_split
        self.seq_len = sequence_length
        self.inc_train = self.seq_len // 2
        self.inc_test = self.seq_len

        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    def prepare_data(self):
        zip_path = os.path.join(self.data_path, self.ZIP_NAME)
        extract_path = os.path.join(self.data_path, self.DIR_NAME)
        if os.path.isdir(extract_path):
            return
        os.makedirs(self.data_path, exist_ok=True)
        urllib.request.urlretrieve(self.URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.data_path)
        os.remove(zip_path)

    def _cut_in_sequences(self, x: np.ndarray, y: np.ndarray, inc: int):
        seq_x, seq_y = [], []
        for s in range(0, x.shape[0] - self.seq_len, inc):
            seq_x.append(x[s: s + self.seq_len])
            seq_y.append(y[s: s + self.seq_len])
        return np.stack(seq_x, axis=1), np.stack(seq_y, axis=1)

    def setup(self, stage: Optional[str] = None):
        base = os.path.join(self.data_path, self.DIR_NAME)

        X_train = np.loadtxt(os.path.join(base, "train/X_train.txt"))
        y_train = (np.loadtxt(os.path.join(base, "train/y_train.txt")) - 1).astype(np.int64)
        X_test = np.loadtxt(os.path.join(base, "test/X_test.txt"))
        y_test = (np.loadtxt(os.path.join(base, "test/y_test.txt")) - 1).astype(np.int64)

        train_x_seq, train_y_seq = self._cut_in_sequences(X_train, y_train, inc=self.inc_train)
        test_x_seq, test_y_seq = self._cut_in_sequences(X_test, y_test, inc=self.inc_test)

        def to_dataset(x_arr, y_arr):
            X = torch.tensor(x_arr.transpose(1, 0, 2), dtype=torch.float32)
            Y = torch.tensor(y_arr.transpose(1, 0), dtype=torch.long)
            return TensorDataset(X, Y)

        full_train_dataset = to_dataset(train_x_seq, train_y_seq)

        total = len(full_train_dataset)
        val_len = int(self.val_split * total)
        train_len = total - val_len

        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_len, val_len])
        self.test_dataset = to_dataset(test_x_seq, test_y_seq)

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


class OzoneDataModule(lit.LightningDataModule):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.data"
    FILE_NAME = "eighthr.data"

    def __init__(
            self,
            data_path: str = "_data/ozone",
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = False,
            val_split: float = 0.10,
            test_split: float = 0.15,
            sequence_length: int = 32,
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.val_split = val_split
        self.test_split = test_split
        self.seq_len = sequence_length
        self.inc = 4

        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def prepare_data(self):
        os.makedirs(self.data_path, exist_ok=True)
        file_path = os.path.join(self.data_path, self.FILE_NAME)
        if not os.path.isfile(file_path):
            print(f"Downloading {self.URL} to {file_path}...")
            urllib.request.urlretrieve(self.URL, file_path)
            print("Download complete.")

    def _to_float(self, v: str):
        return 0.0 if v == "?" else float(v)

    def _load_trace(self):
        all_x, all_y = [], []
        file_path = os.path.join(self.data_path, self.FILE_NAME)

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 74:
                    continue

                label = int(float(parts[-1]))
                feats = [self._to_float(p) for p in parts[1:-1]]

                all_x.append(np.array(feats, dtype=np.float32))
                all_y.append(label)

        all_x = np.stack(all_x, axis=0)
        all_y = np.array(all_y, dtype=np.int64)

        self.mean = np.mean(all_x, axis=0)
        self.std = np.std(all_x, axis=0)

        self.std[self.std == 0] = 1.0
        all_x = (all_x - self.mean) / self.std

        return all_x, all_y

    def _cut_in_sequences(self, x: np.ndarray, y: np.ndarray, inc: int):
        sequences_x, sequences_y = [], []
        for s in range(0, x.shape[0] - self.seq_len, inc):
            start = s
            end = start + self.seq_len
            sequences_x.append(x[start:end])
            sequences_y.append(y[start:end])

        return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)

    def setup(self, stage: Optional[str] = None):
        x, y = self._load_trace()
        x_seq, y_seq = self._cut_in_sequences(x, y, inc=self.inc)

        X_all = torch.tensor(x_seq.transpose(1, 0, 2), dtype=torch.float32)
        Y_all = torch.tensor(y_seq.transpose(1, 0), dtype=torch.long)
        full_ds = TensorDataset(X_all, Y_all)

        tot = len(full_ds)
        val_len = int(self.val_split * tot)
        test_len = int(self.test_split * tot)
        train_len = tot - val_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_ds, [train_len, val_len, test_len]
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

    def get_normalization_params(self):
        return self.mean, self.std


class UCIHAR(Dataset):
    def __init__(self,
                 root: str = "_data/UCI_HAR_Dataset",
                 split: str = "train",
                 url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"):
        self.root = root
        self.url = url
        self.split = split
        self.download()
        self.x, self.y = self._load_split()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def download(self):
        zip_path = f"{self.root}/UCIHAR.zip"
        os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(f"{self.root}/UCI HAR Dataset"):
            print("Downloading UCI HAR dataset...")
            urllib.request.urlretrieve(self.url, zip_path)

            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(f"{self.root}")
            os.remove(zip_path)

        print("Dataset ready.")

    def _load_split(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """split = 'train' or 'test'; returns X [N, 128, 9], y [N]."""
        split = self.split
        base = os.path.join(self.root, "UCI HAR Dataset", split, "Inertial Signals")

        signals = [
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
        ]

        x = []
        for s in signals:
            path = os.path.join(base, f"{s}_{split}.txt")
            data = np.loadtxt(path)
            x.append(data)

        x = np.stack(x, axis=-1).astype(np.float32)
        x = torch.tensor(x)

        y_path = os.path.join(self.root, "UCI HAR Dataset", split, f"y_{split}.txt")
        y = np.loadtxt(y_path).astype(int) - 1
        y = torch.tensor(y, dtype=torch.long)

        return x, y


def _flexible_loadtxt(path_or_bytes):
    """Try several delimiters/approaches to load a 2D numeric array where the
    first column is the label and the remaining columns are series values."""
    tried = []
    for delim in [',', '\t', None]:
        try:
            if isinstance(path_or_bytes, (bytes, bytearray)):
                s = io.BytesIO(path_or_bytes)
                arr = np.loadtxt(s, delimiter=delim)
            else:
                arr = np.loadtxt(path_or_bytes, delimiter=delim)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception as e:
            tried.append((delim, str(e)))
    try:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            s = io.BytesIO(path_or_bytes).read().decode('utf-8').strip().splitlines()
        else:
            with open(path_or_bytes, 'r', encoding='utf-8') as f:
                s = f.read().strip().splitlines()
        rows = []
        for line in s:
            if not line:
                continue
            parts = [p for p in line.replace(',', ' ').replace('\t', ' ').split(' ') if p != '']
            rows.append([float(p) for p in parts])
        arr = np.array(rows, dtype=float)
        return arr
    except Exception as e:
        raise RuntimeError(f"Failed to parse file. Tried delimiters {tried}. Last error: {e}")


class UCRDataset(Dataset):
    """Generic UCR-style dataset wrapper.

    Parameters
    ----------
    name : str
        Dataset name (used for filename/folder inference), e.g. 'FaceFour'.
    root : str
        Root directory to store/download datasets.
    train : bool
        If True return training split; else test split.
    download : bool
        If True try to download the dataset zip from a default UCR-like URL.
    transform : callable
        Optional transform applied to each sample (x, y).
    normalize : bool
        If True, z-score each time series instance individually.
    url : Optional[str]
        Custom URL to download a zip or the raw TRAIN/TEST files.
    """
    DEFAULT_BASE = "https://www.timeseriesclassification.com/aeon-toolkit"

    def __init__(self,
                 name: str,
                 root: str,
                 train: bool = True,
                 download: bool = True,
                 transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
                 normalize: bool = False,
                 url: Optional[str] = None):
        self.name = name
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.name)
        self.train_split = train
        self.transform = transform
        self.normalize = normalize
        self.url = url

        os.makedirs(self.dataset_dir, exist_ok=True)

        self.train_filename_candidates = [
            f"{self.name}_TRAIN.txt",
            f"{self.name}_TRAIN.tsv",
            f"{self.name}_TRAIN",
            f"{self.name}-TRAIN.txt",
            "TRAIN",
        ]
        self.test_filename_candidates = [
            f"{self.name}_TEST.txt",
            f"{self.name}_TEST.tsv",
            f"{self.name}_TEST",
            f"{self.name}-TEST.txt",
            "TEST",
        ]

        if download:
            self._maybe_download()

        self.train_path = self._find_file(self.train_filename_candidates)
        self.test_path = self._find_file(self.test_filename_candidates)

        if (self.train_path is None) or (self.test_path is None):
            raise FileNotFoundError(
                f"Could not find TRAIN/TEST files for dataset {self.name} under {self.dataset_dir}. "
                "Place them there or provide a working download URL.")

        train_arr = _flexible_loadtxt(self.train_path)
        test_arr = _flexible_loadtxt(self.test_path)

        self.x_train, self.y_train = self._split_xy(train_arr)
        self.x_test, self.y_test = self._split_xy(test_arr)

        unique_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
        unique_labels_sorted = np.sort(unique_labels)
        self.label_to_idx = {lab: i for i, lab in enumerate(unique_labels_sorted)}
        self.y_train = np.array([self.label_to_idx[l] for l in self.y_train], dtype=np.int64)
        self.y_test = np.array([self.label_to_idx[l] for l in self.y_test], dtype=np.int64)

        if self.train_split:
            self.x = torch.tensor(self.x_train).float()
            self.y = torch.tensor(self.y_train)
        else:
            self.x = torch.tensor(self.x_test).float()
            self.y = torch.tensor(self.y_test)

    def _maybe_download(self):
        tried_urls = []
        if self.url:
            tried_urls.append(self.url)
            try:
                self._download_and_extract(self.url)
                return
            except Exception as e:
                print(f"[UCRDataset] Warning: custom URL download failed: {e}")

        guesses = [
            f"{self.DEFAULT_BASE}/{self.name}.zip",
            f"{self.DEFAULT_BASE}/{self.name}.zip",
        ]
        for u in guesses:
            tried_urls.append(u)
            try:
                self._download_and_extract(u)
                return
            except Exception:
                pass
        print(f"[UCRDataset] Did not download files automatically. Tried URLs: {tried_urls}. "
              f"If you have TRAIN/TEST locally, place them in {self.dataset_dir}.")

    def _download_and_extract(self, url: str):
        print(f"[UCRDataset] Downloading {url} ...")
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Download failed with code {r.status_code}")
        content_type = r.headers.get('Content-Type', '')
        if 'zip' in content_type or url.lower().endswith('.zip'):
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.dataset_dir)
            print(f"[UCRDataset] Extracted zip to {self.dataset_dir}")
        else:
            fname = os.path.basename(url) or 'downloaded_file'
            with open(os.path.join(self.dataset_dir, fname), 'wb') as f:
                f.write(r.content)
            print(f"[UCRDataset] Saved {fname} into {self.dataset_dir}")

    def _find_file(self, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            path = os.path.join(self.dataset_dir, c)
            if os.path.exists(path):
                return path
        for fname in os.listdir(self.dataset_dir):
            if 'TRAIN' in fname.upper():
                return os.path.join(self.dataset_dir, fname)
        return None

    @staticmethod
    def _split_xy(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if arr.shape[1] < 2:
            raise ValueError("Expected at least 2 columns (label + values) in UCR file.")
        y = arr[:, 0]
        x = arr[:, 1:]
        if x.ndim == 2:
            x = x.astype(np.float32)
            x = x[:, np.newaxis, :]
        return x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        if self.normalize:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            std[std == 0] = 1.0
            x = (x - mean) / std
        x_t = x
        y_t = self.y[idx]
        if self.transform is not None:
            return self.transform(x_t, y_t)
        return x_t, y_t

    def num_classes(self) -> int:
        return len(self.label_to_idx)


class FaceFour(UCRDataset):
    def __init__(self, root="_data/FaceFour", train=True, download=True, **kwargs):
        super().__init__(name="FaceFour", root=root, train=train, download=download, **kwargs)


class SyntheticControl(UCRDataset):
    def __init__(self, root="_data/SyntheticControl", train=True, download=True, **kwargs):
        super().__init__(name="SyntheticControl", root=root, train=train, download=download, **kwargs)


class Beef(UCRDataset):
    def __init__(self, root="_data/Beef", train=True, download=True, **kwargs):
        super().__init__(name="Beef", root=root, train=train, download=download, **kwargs)


class _TransformSubset(torch.utils.data.Dataset):
    """Dataset wrapper applying a transform callable to (x, y)."""

    def __init__(self, subset_or_dataset,
                 transform: Optional[Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]]] = None):
        self.subset = subset_or_dataset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            return self.transform(x, y)
        return x, y


class UCRDataModule(lit.LightningDataModule):
    """LightningDataModule for UCR-style time series datasets."""

    def __init__(self,
                 dataset_name: str,
                 root: str,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 val_frac: float = 0.1,
                 seed: int = 42,
                 download: bool = True,
                 normalize: bool = False,
                 transform: Optional[Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]]] = None,
                 train_transform: Optional[Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]]] = None,
                 val_transform: Optional[Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]]] = None,
                 test_transform: Optional[Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]]] = None,
                 shuffle_train: bool = True,
                 drop_last: bool = False,
                 **dataset_kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.download = download
        self.normalize = normalize
        self.transform = transform
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes_ = None
        self.input_shape = None

    def prepare_data(self):
        _ = UCRDataset(name=self.dataset_name,
                       root=self.root,
                       train=True,
                       download=self.download,
                       normalize=self.normalize,
                       transform=None,
                       **self.dataset_kwargs)
        _ = UCRDataset(name=self.dataset_name,
                       root=self.root,
                       train=False,
                       download=False,
                       normalize=self.normalize,
                       transform=None,
                       **self.dataset_kwargs)

    def setup(self, stage: Optional[str] = None):
        full_train = UCRDataset(name=self.dataset_name,
                                root=self.root,
                                train=True,
                                download=False,
                                normalize=self.normalize,
                                transform=None,
                                **self.dataset_kwargs)
        test_ds = UCRDataset(name=self.dataset_name,
                             root=self.root,
                             train=False,
                             download=False,
                             normalize=self.normalize,
                             transform=None,
                             **self.dataset_kwargs)

        self.num_classes_ = full_train.num_classes()
        if len(full_train) > 0:
            x0, _ = full_train[0]
            self.input_shape = tuple(x0.shape)

        if (self.val_frac is not None) and (self.val_frac > 0.0):
            n_total = len(full_train)
            n_val = max(1, int(np.floor(self.val_frac * n_total)))
            n_train = n_total - n_val
            generator = torch.Generator().manual_seed(self.seed)
            train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=generator)
            train_ds = _TransformSubset(train_ds, transform=self.train_transform or self.transform)
            val_ds = _TransformSubset(val_ds, transform=self.val_transform or self.transform)
        else:
            train_ds = _TransformSubset(full_train, transform=self.train_transform or self.transform)
            val_ds = None

        test_ds = _TransformSubset(test_ds, transform=self.test_transform or self.transform)

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle_train,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def num_classes(self) -> int:
        return self.num_classes_

    def input_shape_(self):
        return self.input_shape


class PenDigits(torch.utils.data.Dataset):

    def __init__(self,
                 train: bool = True,
                 normalize: bool = True,
                 test_frac: float = 0.2,
                 transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None):
        self.normalize = normalize
        self.transform = transform

        pen_digits = fetch_ucirepo(id=81)

        x = pen_digits.data.features
        y = pen_digits.data.targets["Class"].to_numpy()

        cols = sorted(x.columns, key=lambda c: int(''.join(ch for ch in str(c) if ch.isdigit())))
        x = x[cols].to_numpy(dtype=np.float32)
        x = x.reshape(x.shape[0], 8, 2)

        train_max_idx = int(x.shape[0] * (1 - test_frac))
        x_train = x[:train_max_idx]
        y_train = y[:train_max_idx]
        x_test = x[train_max_idx:]
        y_test = y[train_max_idx:]

        if train:
            self.x = torch.tensor(x_train)
            self.y = torch.tensor(y_train)
        else:
            self.x = torch.tensor(x_test)
            self.y = torch.tensor(y_test)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        y = self.y[idx]

        if self.normalize:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            std[std == 0] = 1.0
            x = (x - mean) / std
        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y
