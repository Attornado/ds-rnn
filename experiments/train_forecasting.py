"""
Train a (DS-)RNN on a forecasting benchmark (pytorch_forecasting).

Example (ETTh1, DS-LSTM, encoder 168 / prediction 24):

    python experiments/train_forecasting.py --dataset etth1 \\
        --model_type dslstm --hidden_size 64 --sparsity 0.9
"""
import argparse
import os
import re
import sys
from typing import Any, Dict
import lightning as lit
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from dsrnn.data.forecasting import ForecastingDataModule
from dsrnn.nn.router import RouterConfig
from dsrnn.training.forecasting import ForecastModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    # Data
    parser.add_argument("--dataset", type=str, default="etth1")
    parser.add_argument("--root", type=str, default="_data/etth1")
    parser.add_argument("--input_size", type=int, default=7,
                        help="Encoder feature count (7 for ETT).")
    parser.add_argument("--output_size", type=int, default=7,
                        help="Target count (7 for ETT).")
    parser.add_argument("--max_encoder_length", type=int, default=168)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)

    # Model
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "gru", "dslstm", "dsgru"])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--implementation", type=str, default="gathered",
                        choices=["gathered", "dense"])

    # DS router
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--context_dim", type=int, default=4)
    parser.add_argument("--router_backward", type=str, default="selected_only",
                        choices=["selected_only", "ste"])

    # Optimization (handled by the pytorch_forecasting BaseModel)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--reduce_on_plateau_patience", type=int, default=10)
    parser.add_argument("--reduce_on_plateau_reduction", type=float, default=2.0)
    parser.add_argument("--reduce_on_plateau_min_lr", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=25)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ds-rnn-forecasting")
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

    dm = ForecastingDataModule(
        dataset_name=args.dataset,
        root=args.root,
        min_encoder_length=args.max_encoder_length,
        max_encoder_length=args.max_encoder_length,
        min_prediction_length=args.prediction_length,
        max_prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup("fit")

    model_kwargs: Dict[str, Any] = {}
    if args.model_type in ("dslstm", "dsgru"):
        model_kwargs["router_config"] = RouterConfig(
            num_channels=args.hidden_size,
            input_dim=args.input_size,
            context_dim=args.context_dim,
            sparsity=args.sparsity,
            backward=args.router_backward,
        )
    else:
        model_kwargs.update({"num_layers": 1, "dropout": 0.0})

    import pytorch_forecasting as pf
    model = ForecastModule.from_dataset(
        dataset=dm.train_ds,
        model_type=args.model_type,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        prediction_length=args.prediction_length,
        batch_first=True,
        implementation=args.implementation,
        model_kwargs=model_kwargs,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        reduce_on_plateau_patience=args.reduce_on_plateau_patience,
        reduce_on_plateau_reduction=args.reduce_on_plateau_reduction,
        reduce_on_plateau_min_lr=args.reduce_on_plateau_min_lr,
        loss=pf.metrics.SMAPE(),
        logging_metrics=torch.nn.ModuleList([
            pf.metrics.SMAPE(), pf.metrics.MAE(), pf.metrics.RMSE()]),
    )

    sp = f"-sp{args.sparsity}" if args.model_type in ("dslstm", "dsgru") else ""
    run_name = f"{args.model_type}({args.hidden_size}){sp}-{args.dataset}-seed={args.seed}"
    run_tag = re.sub(r"[^A-Za-z0-9]+", "_", run_name).strip("_")

    if args.use_wandb:
        from lightning.pytorch.loggers.wandb import WandbLogger
        logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            config=vars(args))
    else:
        logger = CSVLogger(args.log_dir, name=run_tag)

    # pytorch_forecasting logs the validation loss as "val_loss" (underscore).
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, run_tag),
        filename=f"{run_tag}_best",
        monitor="val_loss", mode="min", save_top_k=1)
    callbacks = [EarlyStopping(monitor="val_loss", min_delta=0.0,
                               patience=args.early_stopping_patience,
                               verbose=True, mode="min"),
                 checkpoint_cb]
    devices = [int(d) for d in args.devices.split(",")]
    trainer = lit.Trainer(max_epochs=args.epochs, accelerator="auto",
                          devices=devices, logger=logger, callbacks=callbacks)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
