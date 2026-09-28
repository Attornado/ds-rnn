"""Configuration, provenance, and logging shared by experiment runners."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch
import yaml
from lightning.pytorch.loggers import CSVLogger


def load_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
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


def devices(value: str):
    if value == "auto":
        return "auto"
    if "," in value:
        return [int(item) for item in value.split(",")]
    return int(value)


def scientific_config(args: argparse.Namespace) -> dict:
    """Return the effective settings that can affect experiment results."""
    return {key: value for key, value in vars(args).items()
            if key not in {"config", "log_dir", "run_id"}
            and not key.startswith("wandb")}


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def git_state() -> dict:
    """Capture source revision metadata without making Git a runtime requirement."""
    repository = Path(__file__).resolve().parents[2]
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


def runtime_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "git": git_state(),
    }


@contextmanager
def experiment_loggers(args, directory, manifest, task, variant):
    """Provide CSV and optional W&B loggers for one task/model pair.

    Args:
        args: Parsed CLI configuration, including optional W&B settings.
        directory: Local output directory for this task/model pair.
        manifest: Experiment configuration, runtime, protocol, and task metadata.
        task: Current task name.
        variant: Current routing baseline name.

    Yields:
        Pair of the Trainer logger list and the optional W&B run. Each W&B
        run is finished on success or failure before the next model starts.
    """
    loggers: list = [CSVLogger(directory, name="metrics")]
    run = None
    exit_code = 1
    try:
        if args.wandb:
            import wandb
            from lightning.pytorch.loggers import WandbLogger

            if wandb.run is not None:
                raise RuntimeError("An existing W&B run is active; finish it before starting an experiment.")
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
