"""RigL dynamic-sparse-training baseline.

Argparse entry point; an optional YAML config pre-loads defaults and any
explicitly-passed flag overrides it.

Example (sequential MNIST, GRU, RigL at 90% sparsity):

    python experiments/train_pruning.py --task seqmnist --model_type gru \\
        --sparsity 0.9
"""

import argparse
import sys
from typing import Dict, Tuple

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from dsrnn.training.pruning import (
    MaskCallback,
    RigLCallback,
    apply_masks_to_model,
    count_nonzero_masks,
    make_initial_masks,
)
from dsrnn.training.scheduler import LRSchedulerConfig
from dsrnn.training.sequence import SequenceModule
from dsrnn.training.tasks import TASKS, load_datamodule


def _build_module(task: str, model_type: str, hidden_size: int) -> SequenceModule:
    cfg = TASKS[task]
    return SequenceModule(
        model_type=model_type,
        input_size=cfg["input_size"],
        hidden_size=hidden_size,
        output_size=cfg["output_size"],
        use_final_proj=True,
        return_sequences=cfg["return_sequences"],
        batch_first=True,
        model_kwargs={"num_layers": 1, "dropout": 0.0},
        loss_type=cfg["loss_type"],
        optimizer_type="adamw",
        optimizer_kwargs={"all": {"lr": 5e-3, "weight_decay": 0.0}},
        scheduler_config=LRSchedulerConfig(
            scheduler_type="reduce_on_plateau",
            scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 10,
                              "threshold": 1e-4, "threshold_mode": "rel",
                              "cooldown": 0, "min_lr": 0, "eps": 1e-8},
            interval="epoch", frequency=1, monitor="val/loss"),
        metric_params=cfg["metric_params"],
    )


def run_rigl(args) -> Tuple[SequenceModule, Dict[str, torch.Tensor]]:
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    devices = [int(d) for d in args.devices.split(",")]

    module = _build_module(args.task, args.model_type, args.hidden_size)
    dm = load_datamodule(args.task, batch_size=args.batch_size,
                         num_workers=args.num_workers)
    dm.setup("fit")

    # RigL maintains a fixed sparsity; initialize masks at the target level.
    masks = make_initial_masks(module, sparsity=args.sparsity, deterministic=True)
    module.masks = masks
    apply_masks_to_model(module, masks)
    kept, total = count_nonzero_masks(masks)
    print(f"Initial masks: kept {kept}/{total} ({100. * kept / total:.2f}%)")

    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        logger = WandbLogger(
            project=args.wandb_project,
            name=f"RigL-{args.model_type}({args.hidden_size})-s={args.sparsity},"
                 f"p={args.rewire_fraction},f={args.update_frequency}"
                 f"-seed={args.seed}-task={args.task}",
            config=vars(args))
    else:
        logger = CSVLogger(args.log_dir, name=f"rigl_{args.task}")

    rigl_cfg = {
        "rewire_fraction": args.rewire_fraction,
        "update_frequency": args.update_frequency,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "grow_init": args.grow_init,
        "scale_init": args.scale_init,
    }
    callbacks = [MaskCallback(masks), RigLCallback(masks, **rigl_cfg),
                 EarlyStopping(monitor="val/loss", min_delta=0.0,
                               patience=args.early_stopping_patience,
                               verbose=True, mode="min")]
    trainer = Trainer(max_epochs=args.epochs, callbacks=callbacks, logger=logger,
                      enable_checkpointing=False, accelerator="auto", devices=devices,
                      accumulate_grad_batches=args.accumulate_grad_batches)
    trainer.fit(module, datamodule=dm)

    kept, total = count_nonzero_masks(masks)
    print(f"After RigL training: kept {kept}/{total} ({100. * kept / total:.2f}%)")

    dm.setup("test")
    trainer.test(module, datamodule=dm)
    return module, masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    parser.add_argument("--task", type=str, default="seqmnist", choices=list(TASKS))
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.9,
                        help="Target sparsity maintained by RigL.")

    # RigL
    parser.add_argument("--rewire_fraction", type=float, default=0.05)
    parser.add_argument("--update_frequency", type=int, default=100)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int, default=None)
    parser.add_argument("--grow_init", type=str, default="random")
    parser.add_argument("--scale_init", type=float, default=0.01)

    # Training
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ds-rnn-pruning")
    parser.add_argument("--log_dir", type=str, default="_logs")

    args = parser.parse_args()

    if args.config is not None:
        import yaml
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        explicitly_set = {t[2:].split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    _, masks = run_rigl(args)
    kept, total = count_nonzero_masks(masks)
    print(f"Final kept {kept}/{total} ({100. * kept / total:.2f}%)")


if __name__ == "__main__":
    main()
