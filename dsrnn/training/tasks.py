"""Shared benchmark task registry for the classification/regression suites."""
import importlib
from typing import Any, Dict

# task name -> datamodule location, constructor kwargs, and model/loss/metric
# settings (matching the paper benchmarks).
TASKS: Dict[str, Dict[str, Any]] = {
    "seqmnist": {
        "data": ("dsrnn.data.classification", "SequentialMNISTDataModule"),
        "data_kwargs": {"root": "_data/mnist", "val_split": 0.1,
                        "sequential_order": "scan", "subseq_len": 28},
        "input_size": 28, "output_size": 10, "return_sequences": False,
        "loss_type": "ce",
        "metric_params": {
            "accuracy": {"average": "macro", "task": "multiclass", "num_classes": 10},
            "f1": {"average": "macro", "task": "multiclass", "num_classes": 10},
            "ece": {"task": "multiclass", "num_classes": 10, "norm": "l1"},
            "mce": {"task": "multiclass", "num_classes": 10},
        },
    },
    "har": {
        "data": ("dsrnn.data.classification", "HARDataModule"),
        "data_kwargs": {"data_path": "_data/har", "val_split": 0.1,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 561, "output_size": 6, "return_sequences": True,
        "loss_type": "ce",
        "metric_params": {
            "accuracy": {"average": "macro", "task": "multiclass", "num_classes": 6},
            "f1": {"average": "macro", "task": "multiclass", "num_classes": 6},
        },
    },
    "gesture": {
        "data": ("dsrnn.data.classification", "HandGestureDataModule"),
        "data_kwargs": {"data_path": "_data/gesture", "val_split": 0.15,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 32, "output_size": 5, "return_sequences": True,
        "loss_type": "ce",
        "metric_params": {
            "accuracy": {"average": "macro", "task": "multiclass", "num_classes": 5},
            "f1": {"average": "macro", "task": "multiclass", "num_classes": 5},
        },
    },
    "occupancy": {
        "data": ("dsrnn.data.classification", "OccupancyDataModule"),
        "data_kwargs": {"data_path": "_data/occupancy", "val_split": 0.1,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 5, "output_size": 1, "return_sequences": True,
        "loss_type": "bce",
        "metric_params": {"accuracy": {"task": "binary"}, "f1": {"task": "binary"}},
    },
    "ozone": {
        "data": ("dsrnn.data.classification", "OzoneDataModule"),
        "data_kwargs": {"data_path": "_data/ozone", "val_split": 0.1,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 72, "output_size": 1, "return_sequences": True,
        "loss_type": "bce",
        "metric_params": {"accuracy": {"task": "binary"}, "f1": {"task": "binary"}},
    },
    "traffic": {
        "data": ("dsrnn.data.regression", "TrafficDataModule"),
        "data_kwargs": {"data_path": "_data/traffic", "val_split": 0.1,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 7, "output_size": 1, "return_sequences": True,
        "loss_type": "mse",
        "metric_params": {"mse": {}, "mae": {}},
    },
    "power": {
        "data": ("dsrnn.data.regression", "PowerConsumptionDataModule"),
        "data_kwargs": {"data_path": "_data/power", "val_split": 0.1,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 6, "output_size": 1, "return_sequences": True,
        "loss_type": "mse",
        "metric_params": {"mse": {}, "mae": {}},
    },
    "cheetah": {
        "data": ("dsrnn.data.regression", "CheetahDataModule"),
        "data_kwargs": {"data_path": "_data/cheetah", "sequence_length": 32},
        "input_size": 17, "output_size": 17, "return_sequences": True,
        "loss_type": "mse",
        "metric_params": {"mse": {}, "mae": {}},
    },
}


def load_datamodule(task: str, batch_size: int, num_workers: int):
    """Instantiate the datamodule for a task from the registry."""
    cfg = TASKS[task]
    module_path, class_name = cfg["data"]
    cls = getattr(importlib.import_module(module_path), class_name)
    kwargs = dict(cfg["data_kwargs"])
    kwargs.update({"batch_size": batch_size, "num_workers": num_workers,
                   "pin_memory": False})
    return cls(**kwargs)
