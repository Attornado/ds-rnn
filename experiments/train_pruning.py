"""RigL / iterative-magnitude-pruning (IMP) baseline training.

Ported from the old ``train_pruning.py`` to the new stack: the LightningModule
is ``SequenceModule`` (which carries the mask hooks), and the RigL/IMP
machinery lives in ``dsrnn.training.pruning``. Argparse-first; optional
``--config`` YAML (flat key: value) overridden by any flag.

Example (SeqMNIST, LSTM, RigL at 90% sparsity):

    python experiments/train_pruning.py --task seqmnist --model_type lstm \\
        --rigl --init_sparsity 0.9

Example (IMP lottery, 5 rounds):

    python experiments/train_pruning.py --task seqmnist --model_type lstm \\
        --rounds 5 --epochs_per_round 3 --prune_amount_per_round 0.2
"""

import argparse
import sys
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import lightning as lit
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from dsrnn.training.pruning import (
    MaskCallback,
    RigLCallback,
    apply_masks_to_model,
    count_nonzero_masks,
    global_magnitude_prune,
    make_initial_masks,
)
from dsrnn.training.scheduler import LRSchedulerConfig
from dsrnn.training.sequence import SequenceModule
from dsrnn.training.tasks import TASKS, load_datamodule


def _build_module(task: str, model_type: str, hidden_size: int, seed: int) -> SequenceModule:
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


def _train_one_round(module, datamodule, masks, epochs, devices, logger,
                     accumulate_grad_batches=1):
    cbs = [MaskCallback(masks),
           EarlyStopping(monitor="val/loss", min_delta=0.0, patience=25,
                         verbose=True, mode="min")]
    trainer = Trainer(max_epochs=epochs, callbacks=cbs, logger=logger,
                      enable_checkpointing=False, accelerator="auto",
                      devices=devices, accumulate_grad_batches=accumulate_grad_batches)
    trainer.fit(module, datamodule=datamodule)
    return module, trainer


def _train_with_rigl(module, datamodule, masks, rigl_cfg, epochs, devices, logger,
                     accumulate_grad_batches=1):
    cbs = [MaskCallback(masks), RigLCallback(masks, **rigl_cfg),
           EarlyStopping(monitor="val/loss", min_delta=0.0, patience=25,
                         verbose=True, mode="min")]
    trainer = Trainer(max_epochs=epochs, callbacks=cbs, logger=logger,
                      enable_checkpointing=False, accelerator="auto",
                      devices=devices, accumulate_grad_batches=accumulate_grad_batches)
    trainer.fit(module, datamodule=datamodule)
    return module, trainer


def run_pruning(args) -> Tuple[SequenceModule, Dict[str, torch.Tensor]]:
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    devices = [int(d) for d in args.devices.split(",")]

    module = _build_module(args.task, args.model_type, args.hidden_size, args.seed)
    dm = load_datamodule(args.task, batch_size=args.batch_size,
                         num_workers=args.num_workers)
    dm.setup("fit")

    # Save initial state for IMP rewinding
    initial_state = deepcopy(module.state_dict())

    # Initial masks (all-True unless --init_sparsity > 0)
    init_sparsity = args.init_sparsity if args.init_sparsity and args.init_sparsity > 0 else None
    masks = make_initial_masks(module, sparsity=init_sparsity, deterministic=True)
    module.masks = masks
    apply_masks_to_model(module, masks)

    logger = None
    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        rigl_str = (f"-RigL(s={args.init_sparsity},p={args.rigl_rewire_fraction},"
                    f"f={args.rigl_update_frequency})" if args.rigl else "")
        imp_str = (f"-IMP(r={args.rounds},e={args.epochs_per_round},"
                   f"p={args.prune_amount_per_round})" if args.rounds > 0 else "")
        logger = WandbLogger(
            project=args.wandb_project,
            name=f"Pruning-{args.model_type}({args.hidden_size}){rigl_str}{imp_str}"
                 f"-seed={args.seed}-task={args.task}",
            config=vars(args))
    else:
        logger = CSVLogger(args.log_dir, name=f"pruning_{args.task}")

    trainer = None
    # Optional RigL phase
    if args.rigl:
        rigl_cfg = {
            "rewire_fraction": args.rigl_rewire_fraction,
            "update_frequency": args.rigl_update_frequency,
            "start_step": args.rigl_start_step,
            "end_step": args.rigl_end_step,
            "grow_init": args.rigl_grow_init,
            "scale_init": args.rigl_scale_init,
        }
        print("=== Starting RigL phase ===")
        module, trainer = _train_with_rigl(
            module, dm, masks, rigl_cfg, epochs=args.rigl_epochs, devices=devices,
            logger=logger, accumulate_grad_batches=args.accumulate_grad_batches)
        kept, total = count_nonzero_masks(masks)
        print(f"After RigL: kept {kept}/{total} ({100. * kept / total:.2f}%)")

    # IMP rounds
    print("=== Starting IMP rounds ===")
    for r in range(args.rounds):
        print(f"--- IMP round {r + 1}/{args.rounds} ---")
        apply_masks_to_model(module, masks)
        module, trainer = _train_one_round(
            module, dm, masks, epochs=args.epochs_per_round, devices=devices,
            logger=logger, accumulate_grad_batches=args.accumulate_grad_batches)
        masks = global_magnitude_prune(module, masks, amount=args.prune_amount_per_round)
        module.masks = masks
        if args.rewind_to_initial:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name in masks:
                        p.data.copy_(initial_state[name].to(p.device) * masks[name])
        kept, total = count_nonzero_masks(masks)
        print(f"After IMP round {r + 1}: kept {kept}/{total} ({100. * kept / total:.2f}%)")

    if args.rounds > 0:
        module, trainer = _train_one_round(
            module, dm, masks, epochs=args.epochs_per_round, devices=devices,
            logger=logger, accumulate_grad_batches=args.accumulate_grad_batches)

    # Test
    dm.setup("test")
    if trainer is None:
        trainer = Trainer(logger=logger, enable_checkpointing=False,
                          accelerator="auto", devices=devices)
    trainer.test(module, datamodule=dm)

    return module, masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    parser.add_argument("--task", type=str, default="seqmnist", choices=list(TASKS))
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--hidden_size", type=int, default=128)

    # IMP
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs_per_round", type=int, default=3)
    parser.add_argument("--prune_amount_per_round", type=float, default=0.2)
    parser.add_argument("--rewind_to_initial", action="store_true", default=True)
    parser.add_argument("--init_sparsity", type=float, default=0.0)

    # RigL
    parser.add_argument("--rigl", action="store_true", default=False)
    parser.add_argument("--rigl_rewire_fraction", type=float, default=0.05)
    parser.add_argument("--rigl_epochs", type=int, default=10)
    parser.add_argument("--rigl_update_frequency", type=int, default=100)
    parser.add_argument("--rigl_start_step", type=int, default=0)
    parser.add_argument("--rigl_end_step", type=int, default=None)
    parser.add_argument("--rigl_grow_init", type=str, default="random")
    parser.add_argument("--rigl_scale_init", type=float, default=0.01)

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ds-rnn-pruning")
    parser.add_argument("--log_dir", type=str, default="_logs")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            import yaml
            file_cfg = yaml.safe_load(f) or {}
        explicitly_set = {t[2:].split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    _, masks = run_pruning(args)
    kept, total = count_nonzero_masks(masks)
    print(f"Final kept {kept}/{total} ({100. * kept / total:.2f}%)")


if __name__ == "__main__":
    main()
