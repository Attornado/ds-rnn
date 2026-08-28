"""
Train a sequence model on the classification/regression benchmarks.

Example (sequential MNIST, DS-LSTM):

    python experiments/train_classification.py --task seqmnist --model_type dslstm --hidden_size 128 \
        --sparsity 0.9 --lr 1e-2 --epochs 500
"""
import argparse
import os
import re
import sys
from typing import Any, Dict
import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from dsrnn.training.scheduler import LRSchedulerConfig
from dsrnn.training.sequence import SequenceModule
from dsrnn.training.tasks import TASKS, load_datamodule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    parser.add_argument("--task", type=str, default="seqmnist", choices=list(TASKS))
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "gru", "ltc", "dslstm", "dsgru", "dsltc"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--implementation", type=str, default="gathered",
                        choices=["gathered", "dense"])

    # DS router (only used for ds* model types)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--context_dim", type=int, default=4)
    parser.add_argument("--router_backward", type=str, default="selected_only",
                        choices=["selected_only", "ste"])

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
        explicitly_set = {token[2:].split("=")[0] for token in sys.argv[1:]
                          if token.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    task_cfg = TASKS[args.task]

    model_kwargs: Dict[str, Any] = {}
    if args.model_type in ("dslstm", "dsgru", "dsltc"):
        model_kwargs["router_config"] = {
            "sparsity": args.sparsity,
            "context_dim": args.context_dim,
            "backward": args.router_backward,
        }

    # DSLTCCell only forbids gathered+STE; dense is required only with ste.
    implementation = args.implementation
    if args.model_type == "dsltc" and args.router_backward == "ste":
        implementation = "dense"

    model = SequenceModule(
        model_type=args.model_type,
        input_size=task_cfg["input_size"],
        hidden_size=args.hidden_size,
        output_size=task_cfg["output_size"],
        use_final_proj=True,
        return_sequences=task_cfg["return_sequences"],
        batch_first=True,
        implementation=implementation,
        model_kwargs=model_kwargs,
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

    dm = load_datamodule(args.task, batch_size=args.batch_size,
                         num_workers=args.num_workers)

    run_name = f"{args.model_type}({args.hidden_size})-{args.task}"
    run_tag = re.sub(r"[^A-Za-z0-9]+", "_", run_name).strip("_")

    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        logger = WandbLogger(project=args.wandb_project,
                             name=run_name,
                             config=vars(args))
    else:
        logger = CSVLogger(args.log_dir, name=run_tag)

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, run_tag),
        filename=f"{run_tag}_best",
        monitor="val/loss", mode="min", save_top_k=1)
    callbacks = [EarlyStopping(monitor="val/loss", min_delta=0.0,
                               patience=args.early_stopping_patience,
                               verbose=True, mode="min"),
                 checkpoint_cb]

    devices = [int(d) for d in args.devices.split(",")]
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=devices,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
