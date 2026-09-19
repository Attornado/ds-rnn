"""Train the four selective-memory baselines with matched methodology.

The defaults and ``experiments/configs/selective_memory_smoke.yaml`` are a
small CPU-friendly smoke experiment. No large sweep is defined here.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from dsrnn.data.selective_memory import SelectiveMemoryDataModule
from dsrnn.training.scheduler import LRSchedulerConfig
from dsrnn.training.selective_memory import (
    IMPLEMENTATIONS,
    MODEL_VARIANTS,
    SelectiveMemoryModule,
)


_RUN_ID_ENV = "DSRNN_SELECTIVE_MEMORY_RUN_ID"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; explicit CLI flags override it.")
    parser.add_argument("--models", nargs="+", choices=MODEL_VARIANTS,
                        default=list(MODEL_VARIANTS))
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_registers", type=int, default=4)
    parser.add_argument("--value_size", type=int, default=2)
    parser.add_argument("--read_sparsity", type=float, default=0.5)
    parser.add_argument("--write_sparsity", type=float, default=0.5)
    parser.add_argument("--router_context_size", type=int, default=8)
    parser.add_argument("--router_backward", choices=["ste", "selected_only"], default="ste")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default="reference")
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--n_val", type=int, default=64)
    parser.add_argument("--n_test", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--log_dir", type=str, default="_results/selective_memory_logs")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Optional unique run identifier; must not already exist.")
    return parser


def _load_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if args.config is None:
        return args
    with open(args.config, encoding="utf-8") as stream:
        file_config = yaml.safe_load(stream) or {}
    explicitly_set = {token[2:].split("=")[0] for token in sys.argv[1:]
                      if token.startswith("--")}
    for key, value in file_config.items():
        if hasattr(args, key) and key not in explicitly_set:
            setattr(args, key, value)
    return args


def _devices(value: str):
    if value == "auto":
        return "auto"
    if "," in value:
        return [int(item) for item in value.split(",")]
    return int(value)


def _scientific_config(args: argparse.Namespace) -> dict:
    """Return the effective settings that can affect experiment results."""
    return {key: value for key, value in vars(args).items()
            if key not in {"config", "log_dir", "run_id"}}


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _git_state() -> dict:
    """Capture source revision metadata without making Git a runtime requirement."""
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository, check=True,
            capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": dirty}


def _prepare_run_directory(args: argparse.Namespace):
    """Allocate one collision-free directory shared by spawned DDP workers."""
    config = _scientific_config(args)
    fingerprint = _config_hash(config)
    inherited = os.environ.get(_RUN_ID_ENV)
    owns_directory = inherited is None
    if inherited is not None:
        run_id = inherited
    elif args.run_id is not None:
        run_id = args.run_id
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{timestamp}_{fingerprint}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, numbers, '.', '_' and '-'.")
    os.environ[_RUN_ID_ENV] = run_id
    run_root = Path(args.log_dir) / run_id
    if owns_directory:
        run_root.mkdir(parents=True, exist_ok=False)
    else:
        run_root.mkdir(parents=True, exist_ok=True)
    return run_root, run_id, fingerprint, config


def _runtime_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "git": _git_state(),
    }


def main() -> None:
    args = _load_config(build_parser())
    torch.set_float32_matmul_precision("medium")
    run_root, run_id, fingerprint, scientific_config = _prepare_run_directory(args)
    global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if global_rank == 0:
        with (run_root / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump({
                "run_id": run_id,
                "config_hash": fingerprint,
                "config": scientific_config,
                "runtime": _runtime_metadata(),
            }, stream, indent=2)
    data = SelectiveMemoryDataModule(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n_registers=args.n_registers, value_size=args.value_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_seed=args.seed, split_seed=args.split_seed)
    comparison = []

    for variant in args.models:
        # The shared recurrent and output parameters receive the same initial
        # values in every baseline. Data splits and loader shuffling also reset.
        seed_everything(args.seed, workers=True)
        run_name = (f"selective-memory-{variant}-{args.implementation}"
                    f"-h{args.hidden_size}-seed{args.seed}-{fingerprint}")
        run_tag = re.sub(r"[^A-Za-z0-9]+", "_", run_name).strip("_")
        run_dir = run_root / run_tag
        analysis_path = run_dir / "artifacts" / "routing_trace.pt"
        module = SelectiveMemoryModule(
            input_size=data.input_size, hidden_size=args.hidden_size,
            value_size=args.value_size, variant=variant,
            read_sparsity=args.read_sparsity, write_sparsity=args.write_sparsity,
            router_context_size=args.router_context_size,
            router_backward=args.router_backward,
            optimizer_kwargs={"all": {"lr": args.lr,
                                       "weight_decay": args.weight_decay}},
            scheduler_config=LRSchedulerConfig(
                scheduler_type="reduce_on_plateau",
                scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 1},
                interval="epoch", frequency=1, monitor="val/query_mse"),
            analysis_path=str(analysis_path), implementation=args.implementation)
        checkpoint = ModelCheckpoint(
            dirpath=run_dir / "checkpoints", filename=f"{run_tag}_best",
            monitor="val/query_mse", mode="min", save_top_k=1)
        trainer = Trainer(
            max_epochs=args.epochs, accelerator=args.accelerator,
            devices=_devices(args.devices),
            logger=CSVLogger(run_dir, name="metrics"),
            callbacks=[
                EarlyStopping(monitor="val/query_mse", mode="min",
                              patience=args.early_stopping_patience),
                checkpoint,
            ])
        trainer.fit(module, datamodule=data)
        results = trainer.test(module, datamodule=data, ckpt_path="best")[0]
        router_parameters = sum(
            parameter.numel() for name, parameter in module.named_parameters()
            if "router" in name)
        comparison.append({
            "variant": variant,
            "implementation": args.implementation,
            "total_parameters": sum(parameter.numel() for parameter in module.parameters()),
            "router_parameters": router_parameters,
            "read_budget": (args.hidden_size if module.model.read_router is None
                            else module.model.read_router.k),
            "write_budget": (args.hidden_size if module.model.write_router is None
                             else module.model.write_router.k),
            "analysis_path": str(analysis_path),
            "best_checkpoint": checkpoint.best_model_path,
            **{key: float(value) for key, value in results.items()},
        })

    if trainer.is_global_zero:
        with (run_root / "comparison.json").open("w", encoding="utf-8") as stream:
            json.dump({
                "run_id": run_id,
                "config_hash": fingerprint,
                "config": scientific_config,
                "results": comparison,
            }, stream, indent=2)


if __name__ == "__main__":
    main()
