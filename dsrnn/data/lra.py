import os
import pickle
import tarfile
import urllib.request
from functools import reduce
from glob import glob
from typing import Callable, List, Optional, Tuple
import lightning as lit
import numpy as np
import pandas as pd
import torch
import torchvision
from einops.layers.torch import Reduce, Rearrange
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


def _download(url: str, dest_path: str):
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_path):
        print(f"[skip] File already exists: {dest_path}")
        return
    print(f"[download] {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[ok] Downloaded: {dest_path}")


def _is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    try:
        return os.path.commonpath([abs_directory, abs_target]) == abs_directory
    except ValueError:
        return False


def _safe_extract(tar: tarfile.TarFile, path: str = ".", members=None):
    for member in members or tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            raise Exception("Path traversal revealed in tar archive!")
    tar.extractall(path=path, members=members)


def _extract_gz(gz_path: str, extract_to: str, members: Optional[List[str]] = None):
    """Extract a .tar.gz safely."""
    print(f"[extract] {gz_path} -> {extract_to}")
    with tarfile.open(gz_path, "r:gz") as tar:
        if members:
            wanted = [
                m
                for m in tar.getmembers()
                if any(m.name.startswith(x) for x in members)
            ]
            _safe_extract(tar, path=extract_to, members=wanted)
        else:
            _safe_extract(tar, path=extract_to)
    print(f"[ok] Extracted to: {extract_to}")


def _pad_trunc(ids: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def _to_tensor_dataset_xy_seq1(x_list: List[np.ndarray],
                               y_list: List[int],
                               normalize_255: bool = True,
                               is_integer_x: bool = False, ) -> TensorDataset:
    """Create a TensorDataset with X [N, L, 1] float32 and y [N] int64."""
    x = torch.as_tensor(np.stack(x_list, axis=0),
                        dtype=torch.float32 if not is_integer_x else torch.int64)
    if normalize_255:
        x = x / 255.0
    x = x.unsqueeze(-1) if not is_integer_x else x
    y = torch.as_tensor(np.array(y_list, dtype=np.int64))
    return TensorDataset(x, y)


def prepare_datasets(base_dir: str = "datasets"):
    os.makedirs(base_dir, exist_ok=True)

    cifar_dir = os.path.join(base_dir, "cifar-10-batches-py")
    if not os.path.exists(cifar_dir):
        tgz_path = os.path.join(base_dir, "cifar-10-python.tar.gz")
        _download("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tgz_path)
        _extract_gz(tgz_path, base_dir)
    else:
        print("[skip] CIFAR-10 already exists.")

    listops_dir = os.path.join(base_dir, "lra_release", "listops-1000")
    if not os.path.exists(listops_dir):
        gz_path = os.path.join(base_dir, "lra_release.gz")
        _download(
            "https://storage.googleapis.com/long-range-arena/lra_release.gz", gz_path
        )
        _extract_gz(
            gz_path,
            base_dir,
            members=["lra_release/listops-1000", "lra_release/tsv_data"],
        )
    else:
        print("[skip] ListOps (LRA) already exists.")

    imdb_dir = os.path.join(base_dir, "aclImdb")
    if not os.path.exists(imdb_dir):
        tgz_path = os.path.join(base_dir, "aclImdb_v1.tar.gz")
        _download(
            "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", tgz_path
        )
        _extract_gz(tgz_path, base_dir)
    else:
        print("[skip] IMDB already exists.")

    print("[done] Dataset preparation completed.")


def int_space_tokenizer(text: str, max_len: int):
    parts = text.strip().split()
    ids = [int(p) for p in parts[:max_len]]
    return ids


class IMDBDataModule(lit.LightningDataModule):
    """DataModule for character-level IMDb (binary classification).

    Output batch: X float32 [B, L, 1], y int64 [B].
    """

    def __init__(self,
                 data_path: str = "./datasets/aclImdb",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 val_split: float = 0.1,
                 max_length: int = 4096,
                 pad_id: int = 0,
                 encoding: str = "utf-8", ):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters({"task": "imdb"})
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.val_split = val_split
        self.max_length = max_length
        self.pad_id = pad_id
        self.encoding = encoding

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    @property
    def vocab_size(self):
        return 257

    def prepare_data(self):
        pass

    def _load_split(self, split: str) -> TensorDataset:
        split_dir = os.path.join(self.data_path, "train" if split == "train" else "test")
        neg_dir = os.path.join(split_dir, "neg")
        pos_dir = os.path.join(split_dir, "pos")

        paths_labels = [(p, 0) for p in glob(os.path.join(neg_dir, "*.txt"))]
        paths_labels += [(p, 1) for p in glob(os.path.join(pos_dir, "*.txt"))]

        rng = np.random.default_rng(42)
        rng.shuffle(paths_labels)

        x, y = [], []
        for path, label in paths_labels:
            with open(path, "rb") as f:
                b = f.read()

            ids = list(b[: self.max_length])
            ids = _pad_trunc(ids, self.max_length, self.pad_id)

            x.append(np.asarray(ids, dtype=np.int64))
            y.append(int(label))

        return _to_tensor_dataset_xy_seq1(x, y, is_integer_x=True, normalize_255=False)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            full_train = self._load_split("train")
            total = len(full_train)
            val_len = int(self.val_split * total)
            train_len = total - val_len
            self.train_dataset, self.val_dataset = random_split(
                full_train,
                [train_len, val_len]
            )
        if stage in (None, "test"):
            self.test_dataset = self._load_split("test")

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None: did you call setup('test')?")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class ListOpsDataModule(lit.LightningDataModule):
    """DataModule for ListOps (LRA).

    Output batch: X float32 [B, L, 1], y int64 [B].
    """

    def __init__(self,
                 data_path: str = "./datasets/lra_release/listops-1000",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 tokenizer: Optional[Callable[[str, int], List[int]]] = int_space_tokenizer,
                 max_length: int = 2048,
                 pad_id: int = 0, ):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters({"task": "listops"})
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = pad_id
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[TensorDataset] = None
        self.vocab_size = None

    def prepare_data(self):
        pass

    def _load_tsv(self, path: str) -> Tuple[List[np.ndarray], List[int]]:
        assert self.tokenizer is not None, "A callable tokenizer is required for ListOps."
        df = pd.read_csv(path, delimiter="\t")
        x, y = [], []
        max_id = 0
        for _, row in df.iterrows():
            src = str(row["Source"])
            tgt = int(row["Target"])
            ids = self.tokenizer(src, self.max_length)
            ids = _pad_trunc(list(ids), self.max_length, self.pad_id)
            max_id = max(max_id, max(ids))
            x.append(np.asarray(ids, dtype=np.int64))
            y.append(tgt)
        self.vocab_size = max_id + 1
        return x, y

    def setup(self, stage: Optional[str] = None):
        train_path = os.path.join(self.data_path, "basic_train.tsv")
        val_path = os.path.join(self.data_path, "basic_val.tsv")
        test_path = os.path.join(self.data_path, "basic_test.tsv")

        if stage in (None, "fit"):
            xtr, ytr = self._load_tsv(train_path)
            xva, yva = self._load_tsv(val_path)

            self.train_dataset = _to_tensor_dataset_xy_seq1(
                xtr, ytr, is_integer_x=True, normalize_255=False
            )
            self.val_dataset = _to_tensor_dataset_xy_seq1(
                xva, yva, is_integer_x=True, normalize_255=False
            )

        if stage in (None, "test"):
            xt, yt = self._load_tsv(test_path)
            self.test_dataset = _to_tensor_dataset_xy_seq1(
                xt, yt, is_integer_x=True, normalize_255=False
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None: did you call setup('test')?")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class CIFAR10SeqDataModule(lit.LightningDataModule):
    """DataModule for sequential grayscale CIFAR-10.

    Output batch: X float32 [B, 1024, 1] (values in [0, 1] if as_float), y int64 [B].
    """

    def __init__(self,
                 data_path: str = "./datasets/cifar-10-batches-py",
                 batch_size: int = 1024,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 as_float: bool = True,
                 split_seed: Optional[int] = None,
                 val_split: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters({"task": "scifar10"})
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.as_float = as_float
        self.split_seed = split_seed
        if not 0 < val_split < 1:
            raise ValueError("val_split must be between 0 and 1.")
        self.val_split = val_split

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    @staticmethod
    def _unpickle(file: str):
        with open(file, "rb") as fo:
            d = pickle.load(fo, encoding="bytes")
        return d

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.data_path, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_path, train=False, download=True)

    def _load_split(self, split: str) -> TensorDataset:
        if split == "train":
            batch_files = [
                os.path.join(self.data_path, "cifar-10-batches-py", f"data_batch_{i}") for i in range(1, 6)
            ]
        else:
            batch_files = [os.path.join(self.data_path, "cifar-10-batches-py", "test_batch")]

        dicts = [self._unpickle(p) for p in batch_files]
        merged = reduce(
            lambda x_, y_: {
                b"data": np.concatenate(
                    [np.asarray(x_[b"data"]), np.asarray(y_[b"data"])], axis=0
                ),
                b"labels": np.concatenate(
                    [np.asarray(x_[b"labels"]), np.asarray(y_[b"labels"])], axis=0
                ),
            },
            dicts,
        )

        data = merged[b"data"].reshape((-1, 3, 1024))
        labels = np.asarray(merged[b"labels"], dtype=np.int64)

        r, g, b = data[:, 0], data[:, 1], data[:, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        if self.as_float:
            x = torch.as_tensor(gray / 255.0, dtype=torch.float32)
        else:
            x = torch.as_tensor(gray.astype(np.int64), dtype=torch.long)

        x = x.unsqueeze(-1)
        y = torch.as_tensor(labels, dtype=torch.long)
        return TensorDataset(x, y)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit", "validate") and self.train_dataset is None:
            full_train = self._load_split("train")
            total = len(full_train)
            val_len = int(self.val_split * total)
            train_len = total - val_len
            self.train_dataset, self.val_dataset = random_split(
                full_train,
                [train_len, val_len],
                generator=(torch.Generator().manual_seed(self.split_seed)
                           if self.split_seed is not None else None),
            )
        if stage in (None, "test"):
            self.test_dataset = self._load_split("test")

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None: did you call setup('test')?")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class NpyGrayscaleDataset(Dataset):
    def __init__(self,
                 x_path: str,
                 y_path: str,
                 as_image: bool = False,
                 size: int = 32,
                 transform: Optional[Callable] = None):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.as_image = as_image
        self.size = size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.copy(self.x[idx]))
        if self.as_image:
            x = x.view(1, self.size, self.size)
        x = x if self.as_image else x.unsqueeze(-1)
        x = self.transform(x)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class PathFinder(lit.LightningDataModule):

    def __init__(self,
                 data_path: str = "./datasets/pathfinder",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 tokenize: bool = False,
                 pool: int = 1,
                 as_image: bool = False,
                 img_size: int = 32):
        super().__init__()
        self.save_hyperparameters({"task": "pathfinder"})
        self.save_hyperparameters()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tokenize = tokenize
        self.pool = pool
        self.as_image = as_image
        self.img_size = img_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def n_tokens(self):
        if self.tokenize:
            return 256
        return None

    def default_transforms(self):
        transform_list: list = [torchvision.transforms.Lambda(lambda x: x / 255)]
        if self.pool > 1:
            transform_list.append(
                Reduce(
                    "1 (h h2) (w w2) -> 1 h w",
                    "mean",
                    h2=self.pool,
                    w2=self.pool,
                )
            )
        if self.tokenize:
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: (x * 255).long().squeeze(-1))
            )
        if not self.as_image and self.pool > 1:
            transform_list.append(
                Rearrange("1 h w -> (h w)")
                if self.tokenize
                else Rearrange("1 h w -> (h w) 1")
            )
        return None if len(transform_list) < 1 else torchvision.transforms.Compose(transform_list)

    def prepare_data(self):
        if not os.path.exists(self.data_path) and os.path.isdir(self.data_path):
            print(f"Warning: Pathfinder{self.img_size} data not found at {self.data_path}!")

    def setup(self, stage=None):
        transform = self.default_transforms()
        if stage == "fit":
            self.train_dataset = NpyGrayscaleDataset(
                x_path=os.path.join(self.data_path, "x_train.npy"),
                y_path=os.path.join(self.data_path, "y_train.npy"),
                as_image=self.as_image,
                size=self.img_size,
                transform=transform,
            )
            self.val_dataset = NpyGrayscaleDataset(
                x_path=os.path.join(self.data_path, "x_val.npy"),
                y_path=os.path.join(self.data_path, "y_val.npy"),
                as_image=self.as_image,
                size=self.img_size,
                transform=transform,
            )
        elif stage == "test":
            self.test_dataset = NpyGrayscaleDataset(
                x_path=os.path.join(self.data_path, "x_test.npy"),
                y_path=os.path.join(self.data_path, "y_test.npy"),
                as_image=self.as_image,
                size=self.img_size,
                transform=transform,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None: did you call setup('test')?")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class NpyAANDataset(Dataset):
    def __init__(self,
                 x_path: str,
                 y_path: str,
                 l_max: int = 4000,
                 transform: Optional[Callable] = None):
        """Tokenized AAN dataset stored in npy files."""
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.l_max = l_max
        self.transform = transform if transform is not None else lambda x: x

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.copy(self.x[idx])).long()  # (2, L)
        x0, x1 = x[0], x[1]
        x = torch.cat([x0[:self.l_max], x1[:self.l_max]], dim=0)
        x = self.transform(x)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class AAN(lit.LightningDataModule):

    def __init__(self,
                 data_path: str = "./datasets/pathfinder",
                 l_max: int = 4000,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 transform: Optional[Callable] = None, ):
        super().__init__()
        self.save_hyperparameters({"task": "aan"})
        self.save_hyperparameters()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform
        self.l_max = l_max

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def vocab_size(self):
        return 128

    def prepare_data(self):
        if not os.path.exists(self.data_path) and os.path.isdir(self.data_path):
            print(f"Warning: AAN data not found at {self.data_path}!")

    def setup(self, stage=None):
        transform = self.transform
        if stage == "fit":
            self.train_dataset = NpyAANDataset(
                x_path=os.path.join(self.data_path, "x_train.npy"),
                y_path=os.path.join(self.data_path, "y_train.npy"),
                l_max=self.l_max,
                transform=transform,
            )
            self.val_dataset = NpyAANDataset(
                x_path=os.path.join(self.data_path, "x_val.npy"),
                y_path=os.path.join(self.data_path, "y_val.npy"),
                l_max=self.l_max,
                transform=transform,
            )
        elif stage == "test":
            self.test_dataset = NpyAANDataset(
                x_path=os.path.join(self.data_path, "x_test.npy"),
                y_path=os.path.join(self.data_path, "y_test.npy"),
                l_max=self.l_max,
                transform=transform,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None: did you call setup('fit')?")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None: did you call setup('test')?")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
