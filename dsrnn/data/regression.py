import datetime as dt
import os
import urllib.request
import zipfile
from typing import Optional
import lightning as lit
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class CheetahDataModule(lit.LightningDataModule):
    """DataModule for the HalfCheetah kinematic modeling dataset.

    Loads poses from .npy files and creates sequences for next-step prediction
    (x: t..t+N, y: t+1..t+N+1).
    """

    def __init__(
        self,
        data_path: str = "_data/cheetah",
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


class TrafficDataModule(lit.LightningDataModule):
    def __init__(
            self,
            data_path: str = "_data/traffic",
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

    def prepare_data(self):
        csv_path = os.path.join(self.data_path, "Metro_Interstate_Traffic_Volume.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Download the dataset manually."
            )

    def _load_trace(self):
        fp = os.path.join(self.data_path, "Metro_Interstate_Traffic_Volume.csv")
        df = pd.read_csv(fp)

        holiday = (df["holiday"].isna()).astype(np.float32)

        temp = df["temp"].values.astype(np.float32)
        temp -= temp.mean()

        rain = df["rain_1h"].values.astype(np.float32)
        snow = df["snow_1h"].values.astype(np.float32)
        clouds = df["clouds_all"].values.astype(np.float32)

        dt_list = [
            dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in df["date_time"]
        ]
        weekday = np.array([d.weekday() for d in dt_list], dtype=np.float32)
        hour = np.array([d.hour for d in dt_list], dtype=np.float32)
        noon = np.sin(hour * np.pi / 24).astype(np.float32)

        features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1)

        tv = df["traffic_volume"].values.astype(np.float32)
        tv -= tv.mean()
        tv /= tv.std()

        return features, tv

    def _cut_in_sequences(self, x: np.ndarray, y: np.ndarray, inc: int):
        xs, ys = [], []
        for s in range(0, x.shape[0] - self.seq_len, inc):
            xs.append(x[s: s + self.seq_len])
            ys.append(y[s: s + self.seq_len])
        return np.stack(xs, axis=1), np.stack(ys, axis=1)

    def setup(self, stage: Optional[str] = None):
        x, y = self._load_trace()

        x_seq, y_seq = self._cut_in_sequences(x, y, inc=self.inc)

        def to_ds(x_arr, y_arr):
            X = torch.tensor(x_arr.transpose(1, 0, 2), dtype=torch.float32)
            Y = torch.tensor(y_arr.transpose(1, 0), dtype=torch.float32)
            if Y.ndim == 2:
                Y = Y.unsqueeze(-1)
            return TensorDataset(X, Y)

        full_dataset = to_ds(x_seq, y_seq)

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


class PowerConsumptionDataModule(lit.LightningDataModule):
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    ZIP_NAME = "household_power_consumption.zip"
    TXT_NAME = "household_power_consumption.txt"

    def __init__(self,
                 data_path: str = "_data/power",
                 batch_size: int = 64,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 val_split: float = 0.10,
                 test_split: float = 0.15,
                 sequence_length: int = 32, ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.val_split = val_split
        self.test_split = test_split
        self.seq_len = sequence_length

        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    def prepare_data(self):
        os.makedirs(self.data_path, exist_ok=True)
        zip_path = os.path.join(self.data_path, self.ZIP_NAME)
        txt_path = os.path.join(self.data_path, self.TXT_NAME)

        if not os.path.isfile(txt_path):
            urllib.request.urlretrieve(self.URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_path)
            os.remove(zip_path)

    def _convert_to_floats(self, feature_col, memory):
        for i in range(len(feature_col)):
            if feature_col[i] == "?" or feature_col[i] == "\n":
                feature_col[i] = memory[i]
            else:
                feature_col[i] = float(feature_col[i])
                memory[i] = feature_col[i]
        return feature_col, memory

    def _load_crappy_csv(self):
        all_x = []
        path = os.path.join(self.data_path, self.TXT_NAME)
        with open(path, "r") as f:
            lineno = -1
            memory = [i for i in range(7)]
            for line in f:
                lineno += 1
                if lineno == 0:
                    continue
                arr = line.split(";")
                if len(arr) < 8:
                    continue
                feature_col = arr[2:]
                feature_col, memory = self._convert_to_floats(feature_col, memory)
                all_x.append(np.array(feature_col, dtype=np.float32))

        all_x = np.stack(all_x, axis=0)
        all_x -= np.mean(all_x, axis=0)
        all_x /= np.std(all_x, axis=0)

        all_y = all_x[:, 0].reshape([-1, 1])
        all_x = all_x[:, 1:]

        return all_x, all_y

    def _cut_in_sequences(self, x: np.ndarray, y: np.ndarray, seq_len: int, inc: int = 1):
        sequences_x = []
        sequences_y = []

        for s in range(0, x.shape[0] - seq_len, inc):
            start = s
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_y.append(y[start:end])

        return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)

    def setup(self, stage: Optional[str] = None):
        x, y = self._load_crappy_csv()

        tx, ty = self._cut_in_sequences(x, y, seq_len=self.seq_len, inc=self.seq_len)

        total_seqs = tx.shape[1]
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        valid_x = torch.tensor(tx[:, permutation[:valid_size]].transpose(1, 0, 2))
        valid_y = torch.tensor(ty[:, permutation[:valid_size]].transpose(1, 0, 2))
        test_x = torch.tensor(tx[:, permutation[valid_size:valid_size + test_size]].transpose(1, 0, 2))
        test_y = torch.tensor(ty[:, permutation[valid_size:valid_size + test_size]].transpose(1, 0, 2))
        train_x = torch.tensor(tx[:, permutation[valid_size + test_size:]].transpose(1, 0, 2))
        train_y = torch.tensor(ty[:, permutation[valid_size + test_size:]].transpose(1, 0, 2))

        self.train_dataset = TensorDataset(train_x, train_y)
        self.val_dataset = TensorDataset(valid_x, valid_y)
        self.test_dataset = TensorDataset(test_x, test_y)

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
