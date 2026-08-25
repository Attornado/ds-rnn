from pathlib import Path
from typing import Tuple, List, Union, Dict, Any
import urllib.request

import lightning as lit
import numpy as np
import pandas as pd
import pytorch_forecasting as pf
from torch.utils.data import Sampler


def generate_var(max_t: int = 10_000,
                 dim: int = 5,
                 p: int = 1,
                 noise_std: float = 0.1,
                 seed: int = 42) -> Tuple[pd.DataFrame, List[str], List[str]]:
    rng = np.random.default_rng(seed)
    a = rng.normal(scale=0.3, size=(p, dim, dim))
    x = np.zeros((max_t, dim))

    for t in range(p, max_t):
        for i in range(p):
            x[t] += a[i] @ x[t - i - 1]
        x[t] += rng.normal(scale=noise_std, size=dim)

    target_cols = [f"v{i}" for i in range(dim)]
    df = pd.DataFrame(x, columns=target_cols)
    df['time_idx'] = np.arange(len(df))
    df['group'] = 0

    return df, target_cols, []


def generate_lorenz(max_t: int = 10_000,
                    dt: float = 0.01,
                    sigma: float = 10.0,
                    rho: float = 28.0,
                    beta: float = 8.0 / 3.0) -> Tuple[pd.DataFrame, List[str], List[str]]:
    x = np.zeros((max_t, 3))
    x[0] = [1.0, 1.0, 1.0]

    for t in range(max_t - 1):
        dx = sigma * (x[t, 1] - x[t, 0])
        dy = x[t, 0] * (rho - x[t, 2]) - x[t, 1]
        dz = x[t, 0] * x[t, 1] - beta * x[t, 2]
        x[t + 1] = x[t] + dt * np.array([dx, dy, dz])

    target_cols = ['x', 'y', 'z']
    df = pd.DataFrame(x, columns=target_cols)
    df['time_idx'] = np.arange(len(df))
    df['group'] = 0

    return df, target_cols, []


def generate_mackey_glass(max_t: int = 10_000,
                          dim: int = 5,
                          tau: int = 17,
                          beta: float = 0.2,
                          gamma: float = 0.1,
                          n: int = 10,
                          seed: int = 42) -> Tuple[pd.DataFrame, List[str], List[str]]:
    rng = np.random.default_rng(seed)
    x = np.ones((max_t + tau + 1, dim)) * 1.2

    for t in range(tau, max_t + tau):
        x[t + 1] = x[t] + beta * x[t - tau] / (1 + x[t - tau] ** n) - gamma * x[t]

    data = x[tau + 1:]

    target_cols = [f"mg_{i}" for i in range(dim)]
    df = pd.DataFrame(data, columns=target_cols)
    df['time_idx'] = np.arange(len(df))
    df['group'] = 0

    return df, target_cols, []


def load_ett(root: str, name: str = "ETTh1") -> Tuple[pd.DataFrame, List[str], List[str]]:
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.csv"

    if not path.exists():
        urllib.request.urlretrieve(url, path)

    df = pd.read_csv(path)
    df = df.sort_values("date").reset_index(drop=True)
    df['time_idx'] = np.arange(len(df))
    df['group'] = 0

    target_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

    return df, target_columns, []


def load_air_quality(root: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/"
        "AirQualityUCI.zip"
    )
    root = Path(root)
    zip_path = root / "air_quality.zip"
    data_path = root / "AirQualityUCI.csv"

    root.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        urllib.request.urlretrieve(url, zip_path)
        import zipfile

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)

    df = pd.read_csv(data_path, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna()

    target_cols = [
        'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)'
    ]

    feature_cols = [
        'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
        'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]

    df['time_idx'] = np.arange(len(df))
    df['group'] = 0

    return df[target_cols + feature_cols], target_cols, feature_cols


class ForecastingDataModule(lit.LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 root: str,
                 min_encoder_length: int = 168,
                 max_encoder_length: int = 168,
                 min_prediction_length: int = 24,
                 max_prediction_length: int = 24,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 batch_sampler: Union[Sampler, str, None] = None,
                 datagen_kwargs: Dict[str, Any] = None,
                 **kwargs,):
        super().__init__()
        self.dataset = dataset_name.lower()
        self.root = root
        self.min_encoder_length = min_encoder_length
        self.max_encoder_length = max_encoder_length
        self.min_prediction_length = min_prediction_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.datagen_kwargs = datagen_kwargs
        self.kwargs = kwargs

        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare_data(self):
        if self.dataset.startswith("ett"):
            load_ett(self.root, f"ETT{self.dataset[-2:]}")
        elif self.dataset == "air_quality":
            load_air_quality(self.root)

    def setup(self, stage: str = None):
        if self.dataset.startswith("ett"):
            data, target_cols, feature_cols = load_ett(self.root, f"ETT{self.dataset[-2:]}")
        elif self.dataset == "air_quality":
            data, target_cols, feature_cols = load_air_quality(self.root)
        elif self.dataset == "var":
            data, target_cols, feature_cols = generate_var(**self.datagen_kwargs)
        elif self.dataset == "lorenz":
            data, target_cols, feature_cols = generate_lorenz(**self.datagen_kwargs)
        elif self.dataset == "mackey_glass":
            data, target_cols, feature_cols = generate_mackey_glass(**self.datagen_kwargs)
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        training_cutoff = int(len(data) * 0.7)
        validation_cutoff = int(len(data) * 0.85)

        training = pf.TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx='time_idx',
            target=target_cols,
            group_ids=['group'],
            min_encoder_length=self.min_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.min_prediction_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_unknown_reals=target_cols,
            time_varying_known_reals=feature_cols,
            **self.kwargs
        )

        validation = pf.TimeSeriesDataSet.from_dataset(
            training,
            data[lambda x: (x.time_idx > training_cutoff) & (x.time_idx <= validation_cutoff)],
            predict=True,
            stop_randomization=True
        )

        test = pf.TimeSeriesDataSet.from_dataset(
            training,
            data[lambda x: x.time_idx > validation_cutoff],
            predict=True,
            stop_randomization=True
        )

        self.train_ds, self.val_ds, self.test_ds = training, validation, test

    def train_dataloader(self):
        return self.train_ds.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            batch_sampler=self.batch_sampler
        )

    def val_dataloader(self):
        return self.val_ds.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            batch_sampler=self.batch_sampler
        )

    def test_dataloader(self):
        return self.test_ds.to_dataloader(
            train=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            batch_sampler=self.batch_sampler
        )
