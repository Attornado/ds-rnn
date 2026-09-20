"""Long-delay register benchmark; preserves the six-step entry point."""

import json
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from experiments.train_selective_memory import (
    build_parser, _load_config, _devices, _prepare_run_directory, _runtime_metadata,
)
from dsrnn.data.long_memory import LongMemoryDataModule
from dsrnn.training.long_memory import LongMemoryModule
from dsrnn.training.scheduler import LRSchedulerConfig


def main():
    parser = build_parser()
    parser.add_argument("--train_delays", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--eval_delays", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--skip_test", action="store_true",
                        help="Development run: train/validate without inspecting test data.")
    args = _load_config(parser)
    if _devices(args.devices) != 1:
        parser.error("Long-memory metrics currently require a single device (--devices 1).")
    if not args.train_delays or min(args.train_delays) < 0 or max(args.train_delays) > args.horizon:
        parser.error("Training delays must be nonnegative and fit the horizon.")
    if not args.eval_delays or min(args.eval_delays) < 0:
        parser.error("Evaluation delays must be nonnegative.")
    if args.n_train % args.batch_size:
        parser.error("n_train must be divisible by batch_size for equal token budgets.")
    torch.set_float32_matmul_precision("medium")
    root, run_id, fingerprint, config = _prepare_run_directory(args)
    length = args.n_registers + args.horizon + 18
    protocol = {
        "task": "random_register_retention_v1", "loss": "0.5*target_mse+0.5*distractor_mse",
        "early_stopping": False, "validation_every_tokens": args.n_train * length,
        "training_tokens_per_model": args.epochs * args.n_train * length,
        "sequence_length": length, "readout": "full_post_write_hidden_state",
        "test_trace_scope": "first_batch_per_condition",
    }
    (root / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "config_hash": fingerprint, "config": config,
        "protocol": protocol, "runtime": _runtime_metadata()}, indent=2))

    def data(delays, horizon):
        return LongMemoryDataModule(
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            n_registers=args.n_registers, value_size=args.value_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
            data_seed=args.seed, split_seed=args.split_seed,
            delays=delays, horizon=horizon)

    comparison = []
    for variant in args.models:
        seed_everything(args.seed, workers=True)
        run_dir = root / variant
        module = LongMemoryModule(
            input_size=3 + 2 * args.n_registers + args.value_size,
            hidden_size=args.hidden_size, value_size=args.value_size, variant=variant,
            read_sparsity=args.read_sparsity, write_sparsity=args.write_sparsity,
            router_context_size=args.router_context_size, router_backward=args.router_backward,
            implementation=args.implementation,
            optimizer_kwargs={"all": {"lr": args.lr, "weight_decay": args.weight_decay}},
            scheduler_config=LRSchedulerConfig(
                scheduler_type="reduce_on_plateau", scheduler_kwargs={
                    "mode": "min", "factor": 0.5, "patience": args.lr_scheduler_patience},
                interval="epoch", frequency=1, monitor="val/balanced_mse"))
        checkpoint = ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="best",
                                     monitor="val/balanced_mse", mode="min", save_top_k=1)
        trainer = Trainer(max_epochs=args.epochs, accelerator=args.accelerator,
                          devices=1, logger=CSVLogger(run_dir, name="metrics"),
                          callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")])
        trainer.fit(module, datamodule=data(args.train_delays, args.horizon))
        row = {"variant": variant, "best_checkpoint": checkpoint.best_model_path,
               "best_validation_balanced_mse": float(checkpoint.best_model_score),
               "total_parameters": sum(p.numel() for p in module.parameters()),
               "router_parameters": sum(p.numel() for n, p in module.named_parameters() if "router" in n),
               "read_budget": module.model.read_budget, "write_budget": module.model.write_budget,
               "optimizer_steps": trainer.global_step, "evaluation": {}}
        if not args.skip_test:
            conditions = [(d, max(args.horizon, d)) for d in args.eval_delays]
            if max(args.eval_delays) > args.horizon:
                conditions.append((min(args.train_delays), max(args.eval_delays)))
            for delay, horizon in dict.fromkeys(conditions):
                label = f"delay_{delay}_horizon_{horizon}"
                trace_path = run_dir / "artifacts" / (label + ".pt")
                module.hparams.analysis_path = str(trace_path)
                # setup('test') generates only the held-out stream.
                metrics = trainer.test(module, datamodule=data([delay], horizon),
                                       ckpt_path="best", verbose=False)[0]
                row["evaluation"][label] = {
                    "delay": delay, "sequence_length": args.n_registers + horizon + 18,
                    "length_generalization": horizon > args.horizon,
                    "analysis_path": str(trace_path), **metrics}
        comparison.append(row)
        # Preserve completed variants if a later run fails.
        (root / "comparison.json").write_text(json.dumps({
            "run_id": run_id, "config": config, "protocol": protocol,
            "complete": len(comparison) == len(args.models), "results": comparison}, indent=2))


if __name__ == "__main__":
    main()
