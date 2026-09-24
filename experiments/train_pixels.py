"""Single-model pixel classification on sMNIST, psMNIST, or grayscale sCIFAR.

Run from the repository root: python -m experiments.train_pixels --task psMNIST --model dual
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid
from typing import Optional, Tuple, Dict, Union

import torch
import lightning as lit
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from dsrnn.data.classification import SequentialMNISTDataModule
from dsrnn.data.lra import CIFAR10SeqDataModule
from dsrnn.utils.experiment import (
    _load_config, _devices, _scientific_config, _config_hash,
    _runtime_metadata, _experiment_loggers,
)
from dsrnn.nn.cells.dual_read_write import DualReadWriteRNNCell, SparseDualReadWriteRNNCell
from dsrnn.nn.cells.dual_read_write_gru import DualReadWriteGRUCell, SparseDualReadWriteGRUCell
from dsrnn.nn.router import Router, RouterConfig, RouterOutput
from dsrnn.training.optim import build_optimizer
from dsrnn.training.scheduler import LRSchedulerConfig, build_lr_scheduler


MODEL_VARIANTS = ("dense", "read_only", "write_only", "shared", "dual")
IMPLEMENTATIONS = ("reference", "sparse")
RECURRENT_CELLS = ("rnn", "gru")


class PixelRNN(torch.nn.Module):
    """Classify an image from the full hidden state after its last pixel."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 value_size: int,
                 variant: str,
                 read_sparsity: float = 0.5,
                 write_sparsity: float = 0.5,
                 router_context_size: int = 8,
                 router_backward: str = "ste",
                 implementation: str = "reference",
                 recurrent_cell: str = "rnn"):
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {MODEL_VARIANTS}.")
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"Unknown implementation {implementation!r}; "
                             f"expected one of {IMPLEMENTATIONS}.")
        if recurrent_cell not in RECURRENT_CELLS:
            raise ValueError(f"Unknown recurrent cell {recurrent_cell!r}; "
                             f"expected one of {RECURRENT_CELLS}.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.value_size = value_size
        self.variant = variant
        self.router_backward = router_backward
        self.implementation = implementation
        self.recurrent_cell = recurrent_cell

        # Construct shared cell/output before variant-specific routers
        if recurrent_cell == "gru":
            cell_type = (DualReadWriteGRUCell if implementation == "reference"
                         else SparseDualReadWriteGRUCell)
        else:
            cell_type = DualReadWriteRNNCell if implementation == "reference" else SparseDualReadWriteRNNCell
        self.cell = cell_type(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, value_size)

        # Router
        base = dict(num_channels=hidden_size, input_dim=input_size,
                    context_dim=router_context_size, backward=router_backward,
                    binary=True, budget_rounding="ceil")
        if variant == "shared" and read_sparsity != write_sparsity:
            raise ValueError("The shared-mask variant requires equal read/write sparsity.")
        self.read_router = (Router(RouterConfig(sparsity=read_sparsity, **base))
                            if variant in ("read_only", "dual") else None)
        self.write_router = (Router(RouterConfig(sparsity=write_sparsity, **base))
                             if variant in ("write_only", "dual") else None)
        self.shared_router = (Router(RouterConfig(sparsity=read_sparsity, **base))
                              if variant == "shared" else None)

    @property
    def read_budget(self) -> int:
        """Return the exact number of readable hidden coordinates per sample."""
        router = (self.read_router if self.read_router is not None
                  else self.shared_router)
        return self.hidden_size if router is None else router.k

    @property
    def write_budget(self) -> int:
        """Return the exact number of writable hidden coordinates per sample."""
        router = (self.write_router if self.write_router is not None
                  else self.shared_router)
        return self.hidden_size if router is None else router.k

    @staticmethod
    def _fixed_routing(x: torch.Tensor, hidden_size: int) -> RouterOutput:
        mask = x.new_ones(x.size(0), hidden_size)
        idx = torch.arange(hidden_size, device=x.device).expand(x.size(0), -1)
        return RouterOutput(mask=mask, idx=idx, scores=mask, logits=mask)

    def forward(self,
                x: torch.Tensor,
                h0: Optional[torch.Tensor] = None,
                return_routing: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Return (B, classes) logits and optionally per-pixel routing traces."""
        if x.ndim != 3 or x.size(-1) != self.input_size or x.size(1) == 0:
            raise ValueError(f"x must have shape (B, T, {self.input_size}).")
        h = x.new_zeros(x.size(0), self.hidden_size) if h0 is None else h0
        traces = {name: [] for name in ("read_mask", "write_mask", "read_indices", "write_indices",
                                        "read_scores", "write_scores", "read_logits", "write_logits")}
        for timestep in range(x.size(1)):
            xt = x[:, timestep]
            shared = self.shared_router(xt) if self.shared_router is not None else None
            read = (shared if shared is not None else
                    self.read_router(xt) if self.read_router is not None else
                    self._fixed_routing(xt, self.hidden_size))
            write = (shared if shared is not None else
                     self.write_router(xt) if self.write_router is not None else
                     self._fixed_routing(xt, self.hidden_size))

            cell_kwargs: dict = {"read_mask": read.mask, "write_mask": write.mask}
            if self.implementation == "sparse":
                cell_kwargs.update(
                    read_backward=(self.router_backward
                                   if self.read_router is not None or shared is not None
                                   else None),
                    write_backward=(self.router_backward
                                    if self.write_router is not None or shared is not None
                                    else None),
                )
            h, _ = self.cell(xt, h, **cell_kwargs)
            if return_routing:
                for prefix, routing in (("read", read), ("write", write)):
                    traces[prefix + "_mask"].append(routing.mask)
                    traces[prefix + "_indices"].append(routing.idx)
                    traces[prefix + "_scores"].append(routing.scores)
                    traces[prefix + "_logits"].append(routing.logits)
        # Read out the full post-write state, including coordinates preserved by WRITE=0.
        predictions = self.output(h)
        if not return_routing:
            return predictions
        return predictions, {key: torch.stack(value, dim=1) for key, value in traces.items()}


class PixelModule(lit.LightningModule):
    """Final-state cross-entropy classification with epoch-level metrics."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 value_size: int,
                 variant: str,
                 read_sparsity: float = 0.5,
                 write_sparsity: float = 0.5,
                 router_context_size: int = 8,
                 router_backward: str = "ste",
                 optimizer_type: str = "adamw",
                 optimizer_kwargs: Optional[dict] = None,
                 scheduler_config: Optional[LRSchedulerConfig] = None,
                 implementation: str = "reference",
                 recurrent_cell: str = "rnn"):
        super().__init__()
        self.save_hyperparameters()
        self.model = PixelRNN(
            input_size=input_size, hidden_size=hidden_size, value_size=value_size,
            variant=variant, read_sparsity=read_sparsity,
            write_sparsity=write_sparsity, router_context_size=router_context_size,
            router_backward=router_backward, implementation=implementation,
            recurrent_cell=recurrent_cell)

    def forward(self,
                x: torch.Tensor,
                return_routing: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Evaluate the routed recurrent model.

        Args:
            x: Input tensor with shape ``(batch, timesteps, input_size)``.
            return_routing: Whether to include timestep-level routing traces.

        Returns:
            Model logits, or ``(logits, trace)`` when ``return_routing`` is true.
        """
        return self.model(x, return_routing=return_routing)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """
        Build the configured optimizer and optional learning-rate scheduler.

        Returns:
            A Lightning optimizer configuration. When a scheduler is enabled, the result also contains its interval,
            frequency, and monitored validation metric.
        """
        optimizer = build_optimizer(
            self.named_parameters(),
            optimizer_type=self.hparams.optimizer_type,
            optimizer_kwargs=self.hparams.optimizer_kwargs
        )

        if self.hparams.scheduler_config:
            scheduler = build_lr_scheduler(
                optimizer=optimizer,
                config=self.hparams.scheduler_config,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=int(self.trainer.estimated_stepping_batches)
            )
            if scheduler is not None:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": self.hparams.scheduler_config.interval,
                        "frequency": self.hparams.scheduler_config.frequency,
                        "monitor": self.hparams.scheduler_config.monitor,
                    },
                }
        return {"optimizer": optimizer}

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        x, targets = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(-1) == targets).float().mean()
        for name, value in (("loss", loss), ("accuracy", accuracy), ("error_rate", 1 - accuracy)):
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True,
                     batch_size=x.size(0), prog_bar=stage != "train")
        return {"loss": loss}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="Optional YAML config; explicit CLI flags override it.")
    parser.add_argument("--task", choices=("sMNIST", "psMNIST", "sCIFAR"), default="sMNIST")
    parser.add_argument("--model", choices=MODEL_VARIANTS, default="dual")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--read_sparsity", type=float, default=0.5)
    parser.add_argument("--write_sparsity", type=float, default=0.5)
    parser.add_argument("--router_context_size", type=int, default=8)
    parser.add_argument("--router_backward", choices=("ste", "selected_only"), default="ste")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default="reference")
    parser.add_argument("--recurrent_cell", choices=RECURRENT_CELLS, default="gru")
    parser.add_argument("--data_dir", default="_data")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--permutation_seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--log_dir", default="_results/pixels")
    parser.add_argument("--run_id", default=None, help="Unique run ID; existing runs are never overwritten.")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", default="dsrnn-pixels")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline"), default="online")
    return parser


def build_datamodule(args):
    """Keep existing image preprocessing, with explicit split/permutation seeds."""
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  val_split=args.val_split, split_seed=args.split_seed)
    if args.task == "sCIFAR":
        return CIFAR10SeqDataModule(data_path=args.data_dir, as_float=True, **common)
    return SequentialMNISTDataModule(
        root=args.data_dir, subseq_len=1,
        sequential_order="random" if args.task == "psMNIST" else "scan",
        permutation_seed=args.permutation_seed, **common)


def _save_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    torch.set_float32_matmul_precision("medium")
    parser = build_parser()
    args = _load_config(parser)
    # YAML values go through the same choice checks as CLI values.
    for action in parser._actions:
        if action.choices and getattr(args, action.dest) not in action.choices:
            parser.error(f"Invalid {action.dest}: {getattr(args, action.dest)!r}.")
    if _devices(str(args.devices)) != 1:
        parser.error("Pixel runner currently requires --devices 1.")
    if min(args.epochs, args.batch_size, args.hidden_size, args.router_context_size) < 1:
        parser.error("Epochs, batch size, hidden size, and router context size must be positive.")
    if not 0 < args.val_split < 1 or args.num_workers < 0:
        parser.error("val_split must be in (0, 1) and num_workers must be nonnegative.")
    if args.lr <= 0 or args.weight_decay < 0 or args.lr_scheduler_patience < 0:
        parser.error("Invalid optimizer or scheduler settings.")
    if not all(0 <= value < 1 for value in (args.read_sparsity, args.write_sparsity)):
        parser.error("Read and write sparsity must be in [0, 1).")
    if args.model == "shared" and args.read_sparsity != args.write_sparsity:
        parser.error("The shared variant requires equal read/write sparsity.")
    if args.wandb:
        try:
            import wandb
        except ImportError:
            parser.error("W&B logging requires the optional dependency: pip install '.[wandb]'.")

    seed_everything(args.seed, workers=True)
    config = _scientific_config(args)
    fingerprint = _config_hash(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = args.run_id or f"{timestamp}_{fingerprint}_{uuid.uuid4().hex[:8]}"
    if run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        parser.error("run_id may contain only letters, numbers, '.', '_' and '-'.")
    root = Path(args.log_dir) / run_id
    root.mkdir(parents=True, exist_ok=False)
    directory = root / args.task / args.model
    manifest = {
        "run_id": run_id, "config_hash": fingerprint, "config": config,
        "runtime": _runtime_metadata(),
        "protocol": {"monitor": "val/loss", "score": "accuracy",
                     "loss": "final_state_cross_entropy", "early_stopping": False,
                     "readout": "full_post_write_hidden_state",
                     "data": "validation held out from official training set; official test set",
                     "preprocessing": "grayscale pixels in [0, 1]; no augmentation"},
        "tasks": {args.task: {"input_shape": [784 if args.task != "sCIFAR" else 1024, 1],
                              "num_classes": 10}},
    }
    _save_json(root / "manifest.json", manifest)
    _save_json(root / "results.json", {**manifest, "complete": False})

    # Create data and one model.
    data = build_datamodule(args)
    module = PixelModule(
        input_size=1, hidden_size=args.hidden_size, value_size=10, variant=args.model,
        recurrent_cell=args.recurrent_cell, implementation=args.implementation,
        read_sparsity=args.read_sparsity, write_sparsity=args.write_sparsity,
        router_context_size=args.router_context_size, router_backward=args.router_backward,
        optimizer_kwargs={"all": {"lr": args.lr, "weight_decay": args.weight_decay}},
        scheduler_config=LRSchedulerConfig(
            scheduler_type="reduce_on_plateau",
            scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": args.lr_scheduler_patience},
            interval="epoch", frequency=1, monitor="val/loss"))
    checkpoint = ModelCheckpoint(dirpath=directory / "checkpoints", filename="best",
                                 monitor="val/loss", mode="min", save_top_k=1)
    with _experiment_loggers(args, directory, manifest, args.task, args.model) as (loggers, wandb_run):
        trainer = Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
                          logger=loggers,
                          callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")])
        trainer.fit(module, datamodule=data)
        validation = trainer.validate(module, datamodule=data, ckpt_path="best", verbose=False)[0]
        evaluation = ({} if args.skip_test else
                      trainer.test(module, datamodule=data, ckpt_path="best", verbose=False)[0])
        result = {"task": args.task, "variant": args.model,
                  "best_validation_loss": float(checkpoint.best_model_score),
                  "validation": validation, "evaluation": evaluation,
                  "best_checkpoint": checkpoint.best_model_path,
                  "total_parameters": sum(p.numel() for p in module.parameters()),
                  "read_budget": module.model.read_budget, "write_budget": module.model.write_budget,
                  "optimizer_steps": trainer.global_step}
        _save_json(root / "results.json", {**manifest, "complete": True, "result": result})
        if wandb_run is not None:
            wandb_run.summary.update({"result": result, "complete": True})


if __name__ == "__main__":
    main()
