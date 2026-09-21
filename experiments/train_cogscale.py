"""CogScale classification benchmarks with target-only supervision."""

from copy import deepcopy
from importlib.metadata import version
import json
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from dsrnn.training.selective_memory import SelectiveMemoryModule
from dsrnn.training.scheduler import LRSchedulerConfig
from experiments.train_selective_memory import (
    build_parser as build_base_parser, _load_config, _devices, _prepare_run_directory,
    _runtime_metadata,
)

TASKS = ("simple_copy", "selective_copy", "induction_heads", "associative_recall")
TASK_SIZE_OPTIONS = ("sequence_length", "n_symbols", "delay", "n_markers", "num_pairs")


def resolve_task_sizes(task, difficulty, overrides=None):
    """Overlay applicable non-None options on the library's difficulty preset."""
    import cogscale as cog
    params = deepcopy(getattr(cog.evals, "cogscale_" + difficulty)[task]["params"])
    for name, value in (overrides or {}).items():
        if name not in TASK_SIZE_OPTIONS:
            raise ValueError(f"Unknown task size option: {name}")
        if value is not None and name in params:
            params[name] = value
    sizes = {name: params[name] for name in TASK_SIZE_OPTIONS if name in params}
    for name, value in sizes.items():
        minimum = 0 if name == "delay" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{task}: {name} must be an integer >= {minimum}.")
    if task == "selective_copy" and sizes["n_markers"] > sizes["sequence_length"]:
        raise ValueError("selective_copy: n_markers must not exceed sequence_length.")
    if task == "induction_heads" and (sizes["sequence_length"] < 4
                                      or sizes["sequence_length"] % 2):
        raise ValueError("induction_heads: sequence_length must be even and >= 4.")
    if task == "associative_recall":
        if sizes["sequence_length"] < 2 * sizes["num_pairs"] + 2:
            raise ValueError("associative_recall: sequence_length must be >= 2*num_pairs + 2.")
        if sizes["n_symbols"] < sizes["num_pairs"] + 2:
            raise ValueError("associative_recall: n_symbols must be >= num_pairs + 2 "
                             "(two reserved symbols).")
    return sizes


def build_data(task, difficulty, seed, n_train, n_val, n_test, *, overrides=None):
    import cogscale as cog
    # CogScale 0.2.0 mutates its preset dictionary in build_task. Restore it so
    # repeated calls in the same process cannot contaminate subsequent tasks.
    registry = getattr(cog.evals, "cogscale_" + difficulty)
    original = deepcopy(registry[task]["params"])
    sizes = resolve_task_sizes(task, difficulty, overrides)
    try:
        data = cog.build_task(task, difficulty=difficulty, seed=seed,
                              n_train=n_train, n_valid=n_val, n_test=n_test, **sizes)
        params = deepcopy(registry[task]["params"])
    finally:
        registry[task]["params"] = original
    if data["category"] != "classification":
        raise ValueError("This runner supports CogScale classification tasks only.")
    return data, params


def target_loss(logits, targets):
    # Some tasks leave -100 in non-target classes even at scored timesteps.
    # Match CogScale's timestep mask and argmax, rather than using soft labels.
    mask = (targets != -100).any(-1)
    if not mask.any():
        raise ValueError("Batch contains no scored CogScale targets.")
    labels = targets[mask].argmax(-1)
    return F.cross_entropy(logits[mask], labels), mask, labels


class CogScaleModule(SelectiveMemoryModule):
    def _shared_step(self, batch, stage):
        x, targets = batch
        logits = self(x)
        loss, mask, labels = target_loss(logits, targets)
        error = (logits[mask].argmax(-1) != labels).float().mean()
        count = int(mask.sum())
        self.log(f"{stage}/target_ce", loss, on_step=False, on_epoch=True,
                 batch_size=count, prog_bar=stage != "train")
        self.log(f"{stage}/error_rate", error, on_step=False, on_epoch=True,
                 batch_size=count)
        return {"loss": loss}


def build_parser():
    parser = build_base_parser()
    parser.set_defaults(hidden_size=128, recurrent_cell="gru", epochs=20,
                        lr=1e-3, lr_scheduler_patience=5, seed=100,
                        log_dir="_results/cogscale", n_train=1000, n_val=200, n_test=1000)
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
    return parser


def main():
    parser = build_parser()
    args = _load_config(parser)
    if _devices(args.devices) != 1:
        parser.error("CogScale runner currently requires --devices 1.")
    if args.recurrent_cell == "gru" and args.implementation != "reference":
        parser.error("GRU requires --implementation reference.")
    if min(args.n_train, args.n_val, args.n_test, args.epochs, args.batch_size) < 1:
        parser.error("Split sizes, epochs, and batch size must be positive.")
    import cogscale as cog
    overrides = {name: getattr(args, name) for name in TASK_SIZE_OPTIONS}
    try:
        resolved_sizes = {task: resolve_task_sizes(task, args.difficulty, overrides)
                          for task in args.tasks}
        for name, value in overrides.items():
            if value is not None and not any(name in sizes for sizes in resolved_sizes.values()):
                raise ValueError(f"--{name} does not apply to any selected task.")
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
            trainer = Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
                              logger=CSVLogger(directory, name="metrics"),
                              callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")])
            trainer.fit(module, train_dataloaders=loader("train"), val_dataloaders=loader("valid"))
            row = {"task": task, "variant": variant,
                   "best_validation_target_ce": float(checkpoint.best_model_score),
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


if __name__ == "__main__":
    main()
