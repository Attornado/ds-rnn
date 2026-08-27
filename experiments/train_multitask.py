"""
Train a shared (DS-)RNN jointly on multiple tasks.

Example (natural multi-task set, DS-LSTM, uncertainty weighting):

    python experiments/train_multitask.py \\
        --tasks seqmnist har gesture ozone occupancy --model_type dslstm \\
        --weighting uncertainty

Example (synthetic teacher-student tasks):

    python experiments/train_multitask.py --tasks synth-rnn --n_tasks_synth 5 \\
        --model_type dslstm
"""
import argparse
import sys
from typing import Any, Dict, List, Optional
import torch
import yaml
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from dsrnn.data.multitask import MultiTaskDataModule
from dsrnn.nn.router import RouterConfig
from dsrnn.training.multitask import FixedTaskWeighting, MultitaskModule, UncertaintyTaskWeighting
from dsrnn.training.scheduler import LRSchedulerConfig


def get_task_config(task: str, **kwargs) -> Dict[str, Any]:
    """Per-task config; synthetic tasks take their dims from kwargs."""
    base = {
        "seqmnist": dict(input_size=28, output_size=10, loss_type="ce", return_sequences=False),
        "permseqmnist": dict(input_size=28, output_size=10, loss_type="ce", return_sequences=False),
        "har": dict(input_size=561, output_size=6, loss_type="ce", return_sequences=True),
        "gesture": dict(input_size=32, output_size=5, loss_type="ce", return_sequences=True),
        "occupancy": dict(input_size=5, output_size=1, loss_type="bce", return_sequences=True),
        "ozone": dict(input_size=72, output_size=1, loss_type="bce", return_sequences=True),
        "traffic": dict(input_size=7, output_size=1, loss_type="mse", return_sequences=True),
        "power": dict(input_size=6, output_size=1, loss_type="mse", return_sequences=True),
        "cheetah": dict(input_size=17, output_size=17, loss_type="mse", return_sequences=True),
    }
    if task in base:
        return base[task]
    if "synth" in task:
        return dict(input_size=kwargs.get("input_dim", 16),
                    output_size=kwargs.get("output_dim", 4),
                    loss_type="mse", return_sequences=True)
    raise ValueError(task)


def build_model(tasks: List[str], model_type: str, hidden_size: int,
                task_emb_dim: int, task_weighting, seed: int = 42,
                dynamic_sparsity: float = 0.9, router_dim: int = 4,
                n_router_layers: int = 1, router_backward: str = "selected_only",
                shared_in_head: bool = False, shared_out_head: bool = False,
                kwargs_per_dataset: Optional[Dict[str, Any]] = None) -> MultitaskModule:
    kwargs_per_dataset = kwargs_per_dataset or {}
    task_configs = {t: get_task_config(t, **kwargs_per_dataset.get(t, {})) for t in tasks}

    model_kwargs: Dict[str, Any] = {}
    implementation = "gathered"
    if model_type in ("lstm", "gru"):
        model_kwargs = {"dropout": 0.0, "num_layers": 1}
    elif model_type == "ltc":
        model_kwargs = {"input_mapping": "affine", "output_mapping": "affine",
                        "ode_unfolds": 6, "epsilon": 1e-8,
                        "implicit_param_constraints": True}
    elif model_type in ("dslstm", "dsgru", "dsltc"):
        model_kwargs = {"router_config": RouterConfig(
            num_channels=hidden_size, input_dim=hidden_size,
            context_dim=router_dim, n_layers=n_router_layers,
            sparsity=dynamic_sparsity, backward=router_backward)}
        # DSLTCCell only forbids gathered+STE; dense is required only with ste.
        if model_type == "dsltc" and router_backward == "ste":
            implementation = "dense"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return MultitaskModule(
        task_configs=task_configs,
        model_type=model_type,
        hidden_size=hidden_size,
        task_weighting=task_weighting,
        batch_first=True,
        implementation=implementation,
        model_kwargs=model_kwargs,
        optimizer_type="adamw",
        optimizer_kwargs={"all": {"lr": 5e-3, "weight_decay": 0.0}},
        scheduler_config=LRSchedulerConfig(
            scheduler_type="reduce_on_plateau",
            scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 10,
                              "threshold": 1e-4, "threshold_mode": "rel",
                              "cooldown": 0, "min_lr": 0, "eps": 1e-8},
            interval="epoch", frequency=1, monitor="val/loss"),
        shared_in_head=shared_in_head,
        shared_out_head=shared_out_head,
        task_embedding_dim=task_emb_dim,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    # Multitask
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--weighting", choices=["uncertainty", "fixed", "equal"],
                        default="uncertainty")
    parser.add_argument("--task_weights", nargs="*", type=float, default=None,
                        help="One float per task, required with --weighting fixed.")
    parser.add_argument("--cycle_smaller_ds", action="store_true", default=False)
    parser.add_argument("--n_tasks_synth", type=int, default=1)
    parser.add_argument("--synth_n_samples", type=int, default=1000)
    parser.add_argument("--synth_input_dim", type=int, default=16)
    parser.add_argument("--synth_hidden_dim", type=int, default=64)
    parser.add_argument("--synth_output_dim", type=int, default=4)
    parser.add_argument("--synth_seq_len", type=int, default=50)
    parser.add_argument("--synth_noise_std", type=float, default=0.0)
    parser.add_argument("--synth_shared_fraction", type=float, default=0.0)

    # Model
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--task_emb_dim", type=int, default=16)
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "gru", "ltc", "dsltc", "dslstm", "dsgru"])
    parser.add_argument("--dynamic_sparsity", type=float, default=0.9)
    parser.add_argument("--router_dim", type=int, default=4)
    parser.add_argument("--n_router_layers", type=int, default=1)
    parser.add_argument("--router_backward", type=str, default="selected_only",
                        choices=["selected_only", "ste"])
    parser.add_argument("--shared_in_head", action="store_true", default=False)
    parser.add_argument("--shared_out_head", action="store_true", default=False)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    # Analysis
    parser.add_argument("--analyze_masks", action="store_true", default=False,
                        help="Cross-task routing-overlap analysis (DS models).")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DS-RNN-Multitask")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="_multitasklogs")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        explicitly_set = {t[2:].split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Expand synthetic tasks
    tasks: List[str] = []
    for task in args.tasks:
        if "synth" in task:
            tasks.extend(f"{task}-{i}" for i in range(args.n_tasks_synth))
        else:
            tasks.append(task)

    kwargs_per_dataset = {
        t: {"n_samples": args.synth_n_samples, "seq_len": args.synth_seq_len,
            "input_dim": args.synth_input_dim, "output_dim": args.synth_output_dim,
            "hidden_dim": args.synth_hidden_dim, "noise_std": args.synth_noise_std,
            "shared_fraction": args.synth_shared_fraction}
        for t in tasks if "synth" in t
    }
    dm = MultiTaskDataModule(tasks=tasks, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             cycle_smaller=args.cycle_smaller_ds,
                             kwargs_per_dataset=kwargs_per_dataset)

    if args.weighting == "uncertainty":
        task_weighting = UncertaintyTaskWeighting(tasks)
    elif args.weighting == "fixed":
        assert args.task_weights is not None and len(args.task_weights) == len(tasks), \
            "--task_weights must have one value per task with --weighting fixed"
        task_weighting = FixedTaskWeighting(tasks, args.task_weights)
    else:
        task_weighting = FixedTaskWeighting(tasks, [1.0] * len(tasks))

    model = build_model(
        tasks=tasks, model_type=args.model_type, hidden_size=args.hidden_size,
        task_emb_dim=args.task_emb_dim, task_weighting=task_weighting, seed=args.seed,
        dynamic_sparsity=args.dynamic_sparsity, router_dim=args.router_dim,
        n_router_layers=args.n_router_layers, router_backward=args.router_backward,
        shared_in_head=args.shared_in_head, shared_out_head=args.shared_out_head,
        kwargs_per_dataset=kwargs_per_dataset,
    )

    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        task_str = ",".join(tasks)
        weights_str = args.weighting if not args.task_weights else ",".join(args.task_weights)
        run_name = args.wandb_run_name or (
            f"Multitask-{args.model_type}({args.hidden_size})-seed={args.seed}"
            f"-tasks=({task_str})-weights=({weights_str})")
        logger = WandbLogger(project=args.wandb_project, name=run_name,
                             entity=args.wandb_entity, config=vars(args))
    else:
        logger = CSVLogger(args.log_dir, name="rnn_multitask")

    callbacks: list[Callback] = [EarlyStopping(monitor="val/loss", patience=25, mode="min", verbose=True)]
    if args.analyze_masks:
        from dsrnn.training.sparsity import MultitaskMaskAnalysisCallback
        callbacks.append(MultitaskMaskAnalysisCallback(
            datamodule=dm, every_n_epochs=1,
            max_batches_per_task=2, log_validation=True
        ))

    devices = [int(d) for d in args.devices.split(",")]
    trainer = Trainer(max_epochs=args.epochs, logger=logger, callbacks=callbacks,
                      accelerator="auto", devices=devices)

    dm.setup("fit")
    trainer.fit(model, dm)
    dm.setup("test")
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
