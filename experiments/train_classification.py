"""Train a sequence model on the classification/regression benchmarks.

Uniform argparse entry point; an optional YAML config pre-loads defaults and
any flag overrides it.

Example (SeqMNIST, DS-LSTM, paper hyperparameters):

    python experiments/train_classification.py \\
        --task seqmnist --model_type dslstm --hidden_size 128 \\
        --sparsity 0.9 --lr 1e-2 --epochs 500
"""

import argparse
from typing import Any, Dict

import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from dsrnn.training.sequence import SequenceModule
from dsrnn.training.scheduler import LRSchedulerConfig

# Task registry: maps a task name to its datamodule path, constructor kwargs,
# and the model/loss/metric settings. Datamodule class names match the old
# connectomencps.data layout; verify against dsrnn.data if renamed.
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
        "metric_params": {
            "accuracy": {"task": "binary"},
            "f1": {"task": "binary"},
        },
    },
    "ozone": {
        "data": ("dsrnn.data.classification", "OzoneDataModule"),
        "data_kwargs": {"data_path": "_data/ozone", "val_split": 0.1,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 72, "output_size": 1, "return_sequences": True,
        "loss_type": "bce",
        "metric_params": {
            "accuracy": {"task": "binary"},
            "f1": {"task": "binary"},
        },
    },
    "traffic": {
        "data": ("dsrnn.data.classification", "TrafficDataModule"),
        "data_kwargs": {"data_path": "_data/traffic", "val_split": 0.1,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 7, "output_size": 1, "return_sequences": True,
        "loss_type": "mse",
        "metric_params": {"mse": {}, "mae": {}},
    },
    "power": {
        "data": ("dsrnn.data.classification", "PowerConsumptionDataModule"),
        "data_kwargs": {"data_path": "_data/power", "val_split": 0.1,
                        "test_split": 0.15, "sequence_length": 32},
        "input_size": 6, "output_size": 1, "return_sequences": True,
        "loss_type": "mse",
        "metric_params": {"mse": {}, "mae": {}},
    },
    "cheetah": {
        "data": ("dsrnn.data.cheetah", "CheetahDataModule"),
        "data_kwargs": {"data_path": "_data/cheetah", "sequence_length": 32},
        "input_size": 17, "output_size": 17, "return_sequences": True,
        "loss_type": "mse",
        "metric_params": {"mse": {}, "mae": {}},
    },
}


def _load_datamodule(task: str, batch_size: int, num_workers: int):
    import importlib
    cfg = TASKS[task]
    module_path, class_name = cfg["data"]
    cls = getattr(importlib.import_module(module_path), class_name)
    kwargs = dict(cfg["data_kwargs"])
    kwargs.update({"batch_size": batch_size, "num_workers": num_workers,
                   "pin_memory": False})
    return cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    parser.add_argument("--task", type=str, default="seqmnist", choices=list(TASKS))
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "gru", "dslstm", "dsgru"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--implementation", type=str, default="gathered",
                        choices=["gathered", "dense"])

    # DS router (only used for ds* model types)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--context_dim", type=int, default=4)
    parser.add_argument("--router_backward", type=str, default="selected_only",
                        choices=["selected_only", "ste"])
    parser.add_argument("--regularizer_ds", type=str, default=None, choices=[None, "entropy"])

    # Optimization
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--early_stopping_patience", type=int, default=25)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ds-rnn")
    parser.add_argument("--log_dir", type=str, default="_logs")

    args = parser.parse_args()

    # Optional YAML defaults, overridden by any explicitly-set CLI flag.
    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        # Only apply YAML values for args the user did not explicitly pass.
        import sys
        explicitly_set = set()
        for i, token in enumerate(sys.argv[1:], start=1):
            if token.startswith("--"):
                explicitly_set.add(token[2:])
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    task_cfg = TASKS[args.task]

    model_kwargs: Dict[str, Any] = {}
    if args.model_type in ("dslstm", "dsgru"):
        model_kwargs["router_config"] = {
            "sparsity": args.sparsity,
            "context_dim": args.context_dim,
            "backward": args.router_backward,
        }
    else:
        model_kwargs.update({"num_layers": 1, "dropout": 0.0})

    model = SequenceModule(
        model_type=args.model_type,
        input_size=task_cfg["input_size"],
        hidden_size=args.hidden_size,
        output_size=task_cfg["output_size"],
        use_final_proj=True,
        return_sequences=task_cfg["return_sequences"],
        batch_first=True,
        implementation=args.implementation,
        model_kwargs=model_kwargs,
        regularizer_ds=args.regularizer_ds,
        loss_type=task_cfg["loss_type"],
        optimizer_type=args.optimizer,
        optimizer_kwargs={"all": {"lr": args.lr, "weight_decay": args.weight_decay}},
        scheduler_config=LRSchedulerConfig(
            scheduler_type="reduce_on_plateau",
            scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 10,
                              "threshold": 1e-4, "threshold_mode": "rel",
                              "cooldown": 0, "min_lr": 0, "eps": 1e-8},
            interval="epoch", frequency=1, monitor="val/loss",
        ),
        metric_params=task_cfg["metric_params"],
    )

    dm = _load_datamodule(args.task, batch_size=args.batch_size,
                          num_workers=args.num_workers)

    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        logger = WandbLogger(project=args.wandb_project,
                             name=f"{args.model_type}({args.hidden_size})-{args.task}",
                             config=vars(args))
    else:
        logger = CSVLogger(args.log_dir, name=args.task)

    callbacks = [EarlyStopping(monitor="val/loss", min_delta=0.0,
                               patience=args.early_stopping_patience,
                               verbose=True, mode="min")]

    devices = [int(d) for d in args.devices.split(",")]
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=devices,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
