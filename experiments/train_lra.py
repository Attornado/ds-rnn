"""
Train an S4-family model on a Long Range Arena task.

Example (sCIFAR-10, DS-S4):

    python experiments/train_lra.py --task cifar --model_type dss4 \\
        --sparsity 0.9 --router_dim 16
"""
import argparse
import importlib
import sys
from typing import Any, Dict
import lightning as lit
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from dsrnn.nn.router import RouterConfig
from dsrnn.training.lra import LRAModule


# Per-task configuration. All use BatchNorm1d by default (norm_type="batch").
TASKS: Dict[str, Dict[str, Any]] = {
    "aan": {
        "data": ("dsrnn.data.lra", "AAN"),
        "data_kwargs": {"data_path": "_data/aan", "l_max": 4000},
        "vocab_size": 128, "input_size": -1, "output_size": 2,
        "l_max": 8000, "epochs": 30, "max_steps": 20000, "batch_size": 64,
        "lr": 0.01, "ssm_lr": 0.001, "weight_decay": 0.05,
        "gradient_clip": 0.0, "norm_type": "batch",
    },
    "imdb": {
        "data": ("dsrnn.data.lra", "IMDBDataModule"),
        "data_kwargs": {"data_path": "_data/aclImdb", "max_length": 4096},
        "vocab_size": 257, "input_size": -1, "output_size": 2,
        "l_max": 4096, "epochs": 40, "max_steps": 20000, "batch_size": 64,
        "lr": 0.01, "ssm_lr": 0.001, "weight_decay": 0.05,
        "gradient_clip": 0.0, "norm_type": "batch",
    },
    "cifar": {
        "data": ("dsrnn.data.lra", "CIFAR10SeqDataModule"),
        "data_kwargs": {"data_path": "_data/cifar10", "as_float": True},
        "vocab_size": -1, "input_size": 1, "output_size": 10,
        "l_max": 1024, "epochs": 100, "max_steps": None, "batch_size": 50,
        "lr": 0.004, "ssm_lr": 0.001, "weight_decay": 0.01,
        "gradient_clip": 1.0, "norm_type": "batch",
    },
    "listops": {
        "data": ("dsrnn.data.lra", "ListOpsDataModule"),
        "data_kwargs": {"data_path": "./datasets/lra_release/listops-1000",
                        "max_length": 2048, "pad_id": 0},
        "vocab_size": 16, "input_size": -1, "output_size": 10,
        "l_max": 2048, "epochs": 50, "max_steps": None, "batch_size": 50,
        "lr": 0.004, "ssm_lr": 0.001, "weight_decay": 0.01,
        "gradient_clip": 1.0, "norm_type": "batch",
    },
}


def _load_datamodule(task: str, batch_size: int, num_workers: int):
    cfg = TASKS[task]
    module_path, class_name = cfg["data"]
    cls = getattr(importlib.import_module(module_path), class_name)
    kwargs = dict(cfg["data_kwargs"])
    kwargs.update({"batch_size": batch_size, "num_workers": num_workers,
                   "pin_memory": False})
    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    parser.add_argument("--task", type=str, required=True, choices=list(TASKS))
    parser.add_argument("--model_type", type=str, default="s4",
                        choices=["s4", "dss4", "liqs4", "dsliqs4"])
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm_type", type=str, default=None, choices=["batch", "layer"],
                        help="Defaults to the per-task setting.")
    parser.add_argument("--liquid_degree", type=int, default=2)

    # DS-S4 router
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--router_dim", type=int, default=16)
    parser.add_argument("--n_router_layers", type=int, default=1)
    parser.add_argument("--router_backward", type=str, default="ste",
                        choices=["ste", "selected_only"])
    parser.add_argument("--sparsity_target", type=str, default="k_f", choices=["k", "k_f"])
    parser.add_argument("--sparsity_dim", type=str, default="H", choices=["H", "C"])
    parser.add_argument("--use_timewise_mask", action="store_true", default=False)
    parser.add_argument("--use_sparse_implementation", action="store_true", default=False)

    # Optimization (None -> use the per-task default)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ssm_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_clip", type=float, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Directory for the best-validation checkpoint (Lightning default if unset).")
    parser.add_argument("--test_with_corruptions", action="store_true", default=False)

    # Logging (default: no logger, results go to the CSV file)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="DS-RNN-LRA")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        explicitly_set = {t[2:].split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    task_cfg = TASKS[args.task]

    # Resolve per-task defaults for anything not overridden.
    def resolve(flag_value, key):
        return task_cfg[key] if flag_value is None else flag_value

    lr = resolve(args.lr, "lr")
    ssm_lr = resolve(args.ssm_lr, "ssm_lr")
    weight_decay = resolve(args.weight_decay, "weight_decay")
    epochs = resolve(args.epochs, "epochs")
    max_steps = resolve(args.max_steps, "max_steps")
    batch_size = resolve(args.batch_size, "batch_size")
    gradient_clip = resolve(args.gradient_clip, "gradient_clip")
    norm_type = resolve(args.norm_type, "norm_type")

    lit.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    dm = _load_datamodule(args.task, batch_size=batch_size, num_workers=args.num_workers)

    # Resolve vocab size (ListOps computes it from the data during setup).
    vocab_size = task_cfg["vocab_size"]
    if vocab_size is not None and vocab_size > 0:
        dm.prepare_data()
        dm.setup("fit")
        vocab_size = getattr(dm, "vocab_size", None) or vocab_size

    router_config = None
    if args.model_type in ("dss4", "dsliqs4"):
        router_config = RouterConfig(
            num_channels=args.d_model,   # masks the hidden (d_model) dimension
            input_dim=args.d_model,      # routes from the (pooled) S4 input
            context_dim=args.router_dim,
            n_layers=args.n_router_layers,
            sparsity=args.sparsity,
            backward=args.router_backward,
        )

    model = LRAModule(
        model_type=args.model_type,
        input_size=task_cfg["input_size"],
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_state=args.d_state,
        output_size=task_cfg["output_size"],
        n_layers=args.n_layers,
        l_max=task_cfg["l_max"],
        dropout=args.dropout,
        norm_type=norm_type,
        liquid_degree=args.liquid_degree,
        router_config=router_config,
        sparsity_target=args.sparsity_target,
        sparsity_dim=args.sparsity_dim,
        use_timewise_mask=args.use_timewise_mask,
        use_sparse_implementation=args.use_sparse_implementation,
        lr=lr,
        ssm_lr=ssm_lr,
        weight_decay=weight_decay,
        test_with_corruptions=args.test_with_corruptions,
        results_file=args.results_file,
    )

    # Always keep the best-validation checkpoint so the test step evaluates it.
    checkpoint_cb = ModelCheckpoint(dirpath=args.ckpt_dir, monitor="val/acc",
                                    mode="max", save_top_k=1)
    callbacks = [checkpoint_cb]

    # Logger: none by default (results go to the CSV file); wandb opt-in.
    logger = None
    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        sparsity_str = f"-sp{args.sparsity}" if args.model_type in ("dss4", "dsliqs4") else ""
        run_name = (f"{args.model_type}(d={args.d_model},h={args.d_state})-"
                    f"{args.task}{sparsity_str}-seed={args.seed}")
        logger = WandbLogger(project=args.wandb_project, name=run_name,
                             entity=args.wandb_entity, config=vars(args))

    devices = [int(d) for d in args.devices.split(",")]
    trainer_kwargs: Dict[str, Any] = dict(
        accelerator="auto", devices=devices, logger=logger,
        callbacks=callbacks, max_epochs=epochs,
        enable_checkpointing=True,
    )
    if max_steps is not None:
        trainer_kwargs["max_steps"] = max_steps
    if gradient_clip and gradient_clip > 0:
        trainer_kwargs["gradient_clip_val"] = gradient_clip
    trainer = lit.Trainer(**trainer_kwargs)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
