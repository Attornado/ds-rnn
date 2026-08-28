"""
Train a DeltaRNN (DeltaGRU / DeltaLSTM) baseline.

Example (SeqMNIST, DeltaGRU):

    python experiments/train_deltarnn.py --task seqmnist --model_type gru \\
        --hidden_size 128 --thx 0.1 --thh 0.1
"""
import argparse
import sys
import lightning as lit
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dsrnn.training.deltarnn import DeltaRNNClassifier
from dsrnn.training.tasks import TASKS, load_datamodule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    parser.add_argument("--task", type=str, default="seqmnist", choices=list(TASKS))
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--hidden_size", type=int, default=128)

    # DeltaRNN thresholds / regularization
    parser.add_argument("--thx", type=float, default=0.0)
    parser.add_argument("--thh", type=float, default=0.0)
    parser.add_argument("--qa", type=int, default=0)
    parser.add_argument("--lambda_reg", type=float, default=1e-5)

    # Optimization
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early_stopping_patience", type=int, default=25)

    # Misc
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--csv_output", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ds-rnn-deltarnn")
    parser.add_argument("--log_dir", type=str, default="_logs")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        explicitly_set = {t[2:].split("=")[0] for t in sys.argv[1:] if t.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    lit.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    task_cfg = TASKS[args.task]
    dm = load_datamodule(args.task, batch_size=args.batch_size,
                         num_workers=args.num_workers)
    dm.prepare_data()
    dm.setup()

    # Map the task to the DeltaRNN task_type (classification / sequence / regression)
    if task_cfg["loss_type"] == "mse":
        task_type = "regression"
    elif args.task == "seqmnist":
        task_type = "classification"
    else:
        task_type = "sequence"

    batch = next(iter(dm.train_dataloader()))
    input_size = batch[0].shape[-1]
    binary = task_cfg["output_size"] == 1
    num_classes = 2 if binary else task_cfg["output_size"]

    log_hparams = {"task": args.task, "seed": args.seed}
    model = DeltaRNNClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_classes=num_classes,
        task_type=task_type,
        num_layers=1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        thx=args.thx,
        thh=args.thh,
        qa=args.qa,
        lambda_reg=args.lambda_reg,
        log_hparams=log_hparams,
        csv_output=args.csv_output,
        model_type=args.model_type,
    )

    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        logger = WandbLogger(
            project=args.wandb_project,
            name=f"delta{args.model_type}({args.hidden_size})-{args.task}-seed={args.seed}",
            config=vars(args))
        logger_kw = {"logger": logger}
    else:
        logger_kw = {"logger": False}

    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    devices = [int(d) for d in args.devices.split(",")]
    trainer = lit.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=devices,
        log_every_n_steps=10,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience,
                                 verbose=True),
                   checkpoint_cb],
        enable_checkpointing=True,
        **logger_kw,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
