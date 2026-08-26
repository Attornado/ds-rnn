"""
Structured-sparsity ablation baselines (SNIP / learned-global / random).

Example (sequential MNIST, LSTM, random dynamic routing at 90% sparsity):

    python experiments/train_ablation_masking.py --task seqmnist \\
        --model_type lstm --masker random --sparsity 0.9
"""
import argparse
import csv
import os
import sys
import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from dsrnn.training.ablation import AblationLitModule, StructuredSNIPMasker
from dsrnn.training.tasks import TASKS, load_datamodule


class RunSummaryCSV(Callback):
    """Write the final test metrics to a one-row CSV."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def on_test_end(self, trainer, pl_module) -> None:
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(metrics.keys()))
            writer.writerow(list(metrics.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    parser.add_argument("--task", type=str, default="seqmnist", choices=list(TASKS))
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--masker", type=str, required=True,
                        choices=["static", "learned", "random"])
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--hidden_size", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ds-rnn-ablations")
    parser.add_argument("--log_root", type=str, default="_ablation_logs")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        explicitly_set = {t[2:].split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    torch.set_float32_matmul_precision("medium")
    seed_everything(args.seed)

    task_cfg = TASKS[args.task]
    dm = load_datamodule(args.task, batch_size=args.batch_size,
                         num_workers=args.num_workers)
    dm.setup("fit")

    module = AblationLitModule(
        model_type=args.model_type, masker_type=args.masker,
        input_size=task_cfg["input_size"], hidden_size=args.hidden_size,
        output_size=task_cfg["output_size"], sparsity=args.sparsity,
        return_sequences=task_cfg["return_sequences"],
        loss_type=task_cfg["loss_type"], metric_params=task_cfg["metric_params"],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    if isinstance(module.rnn.masker, StructuredSNIPMasker):
        module.maybe_init_static_mask(dm.train_dataloader())

    run_name = (f"Ablation-{args.masker}-{args.model_type}({args.hidden_size})"
                f"-s{args.sparsity}-{args.task}-seed{args.seed}")
    log_root = os.path.join(args.log_root, run_name)
    os.makedirs(log_root, exist_ok=True)

    loggers = [CSVLogger(save_dir=log_root, name="csv")]
    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        loggers.append(WandbLogger(project=args.wandb_project, name=run_name,
                                   config=vars(args)))

    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(log_root, "checkpoints"),
                              monitor="val/loss", mode="min", save_top_k=1,
                              filename="best")
    early_stop_cb = EarlyStopping(monitor="val/loss", mode="min",
                                  patience=args.patience, verbose=True)
    summary_cb = RunSummaryCSV(csv_path=os.path.join(log_root, "test_summary.csv"))

    devices = [int(d) for d in args.devices.split(",")]
    trainer = Trainer(max_epochs=args.epochs, logger=loggers,
                      callbacks=[ckpt_cb, early_stop_cb, summary_cb],
                      accelerator="auto", devices=devices)

    trainer.fit(module, datamodule=dm)
    dm.setup("test")
    trainer.test(module, datamodule=dm, ckpt_path=ckpt_cb.best_model_path or "best")


if __name__ == "__main__":
    main()
