"""CogScale classification benchmarks with target-only supervision."""
import argparse
from contextlib import contextmanager
from typing import Final
import hashlib
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
import json
from time import perf_counter
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from dsrnn.data.cogscale import TASKS, TASK_SIZE_OPTIONS, build_data, resolve_task_overrides
from dsrnn.training.cogscale import IMPLEMENTATIONS, MODEL_VARIANTS, RECURRENT_CELLS, CogScaleModule
from dsrnn.training.scheduler import LRSchedulerConfig


# Constants
_RUN_ID_ENV: Final[str] = "DSRNN_COGSCALE_RUN_ID"


# Helpers
def _load_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if args.config is None:
        return args
    with open(args.config, encoding="utf-8") as stream:
        file_config = yaml.safe_load(stream) or {}
    explicitly_set = {token[2:].split("=")[0] for token in sys.argv[1:]
                      if token.startswith("--")}
    if "no-wandb" in explicitly_set:
        explicitly_set.add("wandb")
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
            if key not in {"config", "log_dir", "run_id"}
            and not key.startswith("wandb")}


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


@contextmanager
def _experiment_loggers(args, directory, manifest, task, variant):
    """Provide CSV and optional W&B loggers for one task/model pair.

    Args:
        args: Parsed CLI configuration, including optional W&B settings.
        directory: Local output directory for this task/model pair.
        manifest: Experiment configuration, runtime, protocol, and task metadata.
        task: Current CogScale task name.
        variant: Current routing baseline name.

    Yields:
        Pair of the Trainer logger list and the optional W&B run. Each W&B
        run is finished on success or failure before the next model starts.
    """
    loggers = [CSVLogger(directory, name="metrics")]
    run = None
    exit_code = 1
    try:
        if args.wandb:
            import wandb
            from lightning.pytorch.loggers import WandbLogger

            if wandb.run is not None:
                raise RuntimeError("An existing W&B run is active; finish it before this sweep.")
            directory.mkdir(parents=True, exist_ok=True)
            run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                group=args.wandb_group or manifest["run_id"],
                name=f"{manifest['run_id']}/{task}/{variant}",
                dir=str(directory), mode=args.wandb_mode,
                config={**manifest["config"], "run_id": manifest["run_id"],
                        "config_hash": manifest["config_hash"],
                        "runtime": manifest["runtime"], "protocol": manifest["protocol"],
                        "task": task, "variant": variant,
                        "task_metadata": manifest["tasks"][task],
                        "logging": {key: value for key, value in vars(args).items()
                                    if key.startswith("wandb")}},
            )
            loggers.append(WandbLogger(experiment=run, log_model=False))
        yield loggers, run
        exit_code = 0
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; explicit CLI flags override it.")
    parser.add_argument("--models", nargs="+", choices=MODEL_VARIANTS,
                        default=list(MODEL_VARIANTS))
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--read_sparsity", type=float, default=0.5)
    parser.add_argument("--write_sparsity", type=float, default=0.5)
    parser.add_argument("--router_context_size", type=int, default=8)
    parser.add_argument("--router_backward", choices=["ste", "selected_only"], default="ste")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default="reference")
    parser.add_argument("--recurrent_cell", choices=RECURRENT_CELLS, default="gru")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--log_dir", type=str, default="_results/cogscale")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Optional unique run identifier; must not already exist.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--difficulty", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--sequence_length", type=int, default=None,
                        help="Override each task's preset sequence_length; for copy tasks "
                             "this is input content length, not total unrolled length.")
    parser.add_argument("--n_symbols", type=int, default=None,
                        help="Override vocabulary size for all selected tasks; "
                             "associative recall includes two reserved symbols.")
    parser.add_argument("--delay", type=int, default=None,
                        help="Override delay for simple_copy and selective_copy only.")
    parser.add_argument("--n_markers", type=int, default=None,
                        help="Override marked token count for selective_copy only.")
    parser.add_argument("--num_pairs", type=int, default=None,
                        help="Override key-value pair count for associative_recall only.")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False,
                        help="Add W&B logging; CSV and comparison.json remain enabled.")
    parser.add_argument("--wandb_project", default="dsrnn-cogscale")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None,
                        help="W&B group; defaults to the local run ID.")
    parser.add_argument("--wandb_mode", choices=("online", "offline"), default="online")
    return parser


def main():
    torch.set_float32_matmul_precision('medium')
    parser = build_parser()
    args = _load_config(parser)
    if _devices(args.devices) != 1:
        parser.error("CogScale runner currently requires --devices 1.")
    if min(args.n_train, args.n_val, args.n_test, args.epochs, args.batch_size) < 1:
        parser.error("Split sizes, epochs, and batch size must be positive.")
    if args.wandb:
        try:
            import wandb
        except ImportError:
            parser.error("W&B logging requires the optional dependency: pip install '.[wandb]'.")
    import cogscale as cog
    overrides = {name: getattr(args, name) for name in TASK_SIZE_OPTIONS}
    try:
        resolved_sizes = resolve_task_overrides(args.tasks, args.difficulty, overrides)
    except ValueError as error:
        parser.error(str(error))
    args.cogscale_version = version("cogscale")
    args.loss = "target_cross_entropy"
    root, run_id, fingerprint, config = _prepare_run_directory(args)
    manifest = {"run_id": run_id, "config_hash": fingerprint, "config": config,
                "runtime": _runtime_metadata(), "tasks": {},
                "protocol": {"monitor": "val/target_ce", "score": "cogscale_error_rate",
                             "mask": "any target channel != -100",
                             "data": "fixed CogScale splits shared across variants",
                             "early_stopping": False, "readout": "full_post_write_hidden_state"}}
    rows = []
    for task in args.tasks:
        print(f"Generating CogScale {task} ({args.difficulty}), "
              f"sizes={resolved_sizes[task]}...", flush=True)
        start = perf_counter()
        data, params = build_data(task, args.difficulty, args.split_seed,
                                   args.n_train, args.n_val, args.n_test, overrides=overrides)
        datasets = {split: TensorDataset(torch.tensor(data["X_" + split], dtype=torch.float32),
                                         torch.tensor(data["Y_" + split], dtype=torch.float32))
                    for split in ("train", "valid", "test") if split != "test" or not args.skip_test}
        manifest["tasks"][task] = {"parameters": params, "category": data["category"],
                                   "input_shape": list(data["X_train"].shape),
                                   "target_shape": list(data["Y_train"].shape)}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Generated in {perf_counter() - start:.2f}s; input shape {data['X_train'].shape}", flush=True)
        for variant in args.models:
            seed_everything(args.seed, workers=True)
            directory = root / task / variant
            module = CogScaleModule(
                input_size=data["X_train"].shape[-1], hidden_size=args.hidden_size,
                value_size=data["Y_train"].shape[-1], variant=variant,
                recurrent_cell=args.recurrent_cell, implementation=args.implementation,
                read_sparsity=args.read_sparsity, write_sparsity=args.write_sparsity,
                router_context_size=args.router_context_size, router_backward=args.router_backward,
                optimizer_kwargs={"all": {"lr": args.lr, "weight_decay": args.weight_decay}},
                scheduler_config=LRSchedulerConfig(
                    scheduler_type="reduce_on_plateau", scheduler_kwargs={
                        "mode": "min", "factor": 0.5, "patience": args.lr_scheduler_patience},
                    interval="epoch", frequency=1, monitor="val/target_ce"))
            def loader(split):
                return DataLoader(datasets[split], batch_size=args.batch_size,
                                  shuffle=split == "train", num_workers=args.num_workers,
                                  generator=torch.Generator().manual_seed(args.split_seed + 1))
            checkpoint = ModelCheckpoint(dirpath=directory / "checkpoints", filename="best",
                                         monitor="val/target_ce", mode="min", save_top_k=1)
            with _experiment_loggers(args, directory, manifest, task, variant) as (loggers, wandb_run):
                trainer = Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
                                  logger=loggers,
                                  callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")])
                trainer.fit(module, train_dataloaders=loader("train"), val_dataloaders=loader("valid"))
                # Final-epoch callback metrics need not describe the selected model.
                # Re-evaluate the best checkpoint on validation, including skip_test runs.
                validation = trainer.validate(module, dataloaders=loader("valid"),
                                              ckpt_path="best", verbose=False)[0]
                validation["val/accuracy"] = 1.0 - validation["val/error_rate"]
                row = {"task": task, "variant": variant,
                       "best_validation_target_ce": float(checkpoint.best_model_score),
                       "validation": validation,
                       "best_checkpoint": checkpoint.best_model_path,
                       "total_parameters": sum(p.numel() for p in module.parameters()),
                       "read_budget": module.model.read_budget, "write_budget": module.model.write_budget,
                       "optimizer_steps": trainer.global_step, "evaluation": {}}
                if not args.skip_test:
                    row["evaluation"] = trainer.test(module, dataloaders=loader("test"),
                                                      ckpt_path="best", verbose=False)[0]
                    # Use the library's public scorer on the same restored checkpoint.
                    module.eval()
                    scores, counts = [], []
                    with torch.no_grad():
                        for x, y in loader("test"):
                            prediction = module(x.to(module.device)).cpu().numpy()
                            scores.append(float(cog.compute_score(y.numpy(), prediction, data["category"])))
                            counts.append(int((y != -100).any(-1).sum()))
                    row["evaluation"]["cogscale_score"] = sum(s*c for s,c in zip(scores, counts)) / sum(counts)
                rows.append(row)
                (root / "comparison.json").write_text(json.dumps({
                    **manifest, "complete": len(rows) == len(args.tasks) * len(args.models),
                    "results": rows}, indent=2))
                if wandb_run is not None:
                    wandb_run.summary.update({"result": row, "complete": True})
if __name__ == "__main__":
    main()
