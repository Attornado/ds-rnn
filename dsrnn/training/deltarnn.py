"""DeltaRNN classifier LightningModule for the sparse-baseline experiments.

Separate from ``dsrnn.training.sequence.SequenceModule`` because DeltaRNN has
a different cell interface (an extra regularization term and temporal-sparsity
statistics).
"""

import csv
import os

import lightning as lit
import torch
import torchmetrics
from torch import nn

from dsrnn.nn.deltarnn import DeltaGRU, DeltaLSTM


class DeltaRNNClassifier(lit.LightningModule):
    def __init__(self,
                 input_size=28,
                 hidden_size=128,
                 num_layers=1,
                 num_classes=10,
                 lr=1e-3,
                 weight_decay=0.0,
                 thx=0.0,
                 thh=0.0,
                 qa=0,
                 lambda_reg=1e-5,
                 debug=True,
                 model_type: str = "gru",
                 task_type="classification",
                 csv_output=None,
                 log_hparams=None):
        super().__init__()
        self.save_hyperparameters()

        assert task_type in ["classification", "sequence", "regression"]
        self.log_hparams = log_hparams if log_hparams is not None else {}

        rnn_cls = {"gru": DeltaGRU, "lstm": DeltaLSTM}.get(model_type)
        if rnn_cls is None:
            raise ValueError(f"model_type {model_type} not supported")
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            thx=thx,
            thh=thh,
            qa=qa,  # keep quantization off for training
            debug=int(debug),
        )

        self.fc = nn.Linear(hidden_size, num_classes)

        task = "binary" if num_classes == 2 else "multiclass"
        if task_type != "regression":
            self.train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, average="macro")
            self.val_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, average="macro")
            self.test_acc_metric = torchmetrics.Accuracy(task=task, num_classes=num_classes, average="macro")
            self.train_f1 = torchmetrics.F1Score(task=task, num_classes=num_classes, average="macro")
            self.val_f1 = torchmetrics.F1Score(task=task, num_classes=num_classes, average="macro")
            self.test_f1 = torchmetrics.F1Score(task=task, num_classes=num_classes, average="macro")
        else:
            self.train_mae = lambda preds, y: torch.mean(torch.abs(preds - y))
            self.val_mae = lambda preds, y: torch.mean(torch.abs(preds - y))
            self.test_mae = lambda preds, y: torch.mean(torch.abs(preds - y))

    def forward(self, x):
        x = x.transpose(0, 1)
        out, _, reg = self.rnn(x)  # [T, B, H]

        if self.hparams.task_type == "classification":
            last = out[-1]            # [B, H]
            logits = self.fc(last)    # [B, C]
        else:  # sequence labeling
            logits = self.fc(out)     # [T, B, C]
        return logits, reg

    def _compute_loss(self, logits, y, reg):
        if self.hparams.task_type == "classification":
            loss = nn.functional.cross_entropy(logits, y)
        else:
            logits = logits.transpose(0, 1)  # [B, T, C]
            B, T, C = logits.shape
            if self.hparams.task_type == "regression":
                loss = nn.functional.mse_loss(logits, y)
            else:
                loss = nn.functional.cross_entropy(logits.reshape(B * T, C),
                                                   y.reshape(B * T))
        return loss + self.hparams.lambda_reg * reg

    def _preds(self, logits):
        if self.hparams.task_type == "classification":
            return torch.argmax(logits, dim=1)
        if self.hparams.task_type == "regression":
            return logits.transpose(0, 1)
        return torch.argmax(logits, dim=-1).transpose(0, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, reg = self(x)
        loss = self._compute_loss(logits, y, reg)
        preds = self._preds(logits)
        if self.hparams.task_type == "regression":
            self.log("train_mae", self.train_mae(preds, y).item(), prog_bar=True)
        else:
            self.train_acc(preds, y)
            self.train_f1(preds, y)
            self.log("train_acc", self.train_acc, prog_bar=True)
            self.log("train_f1", self.train_f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, reg = self(x)
        loss = self._compute_loss(logits, y, reg)
        preds = self._preds(logits)
        self.log("val_loss", loss, prog_bar=True)
        if self.hparams.task_type == "regression":
            self.log("val_mae", self.val_mae(preds, y).item(), prog_bar=True)
        else:
            self.val_acc(preds, y)
            self.val_f1(preds, y)
            self.log("val_acc", self.val_acc, prog_bar=True)
            self.log("val_f1", self.val_f1, prog_bar=True)
        self._log_sparsity("val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, reg = self(x)
        loss = self._compute_loss(logits, y, reg)
        preds = self._preds(logits)
        self._log_sparsity("test")
        if self.hparams.task_type == "regression":
            self.log("test_mae", self.test_mae(preds, y).item())
        else:
            self.test_acc_metric(preds, y)
            self.test_f1(preds, y)
            self.log("test_acc", self.test_acc_metric)
            self.log("test_f1", self.test_f1)
        self.log("test_loss", loss)

    def _log_sparsity(self, stage):
        if self.rnn.debug:
            sp = self.rnn.get_temporal_sparsity()
            if "SP_T_DX" in sp:
                self.log(f"{stage}_sp_dx", sp["SP_T_DX"], prog_bar=True)
                self.log(f"{stage}_sp_dh", sp["SP_T_DH"], prog_bar=True)
                self.log(f"{stage}_sp_total", sp["SP_T_DV"], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, eps=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch",
                             "frequency": 1, "monitor": "val_loss"},
        }

    def _write_test_results_to_csv(self, metrics: dict, filename="test_results.csv"):
        file_exists = os.path.isfile(filename)
        with open(filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def on_test_epoch_end(self):
        acc = f1 = mae = None
        if self.hparams.task_type == "regression":
            mae = self.trainer.callback_metrics["test_mae"].item()
        else:
            acc = self.trainer.callback_metrics["test_acc"].item()
            f1 = self.trainer.callback_metrics["test_f1"].item()
        loss = self.trainer.callback_metrics["test_loss"].item()

        sp_dx = sp_dh = sp_total = None
        if self.rnn.debug:
            sp = self.rnn.get_temporal_sparsity()
            sp_dx = sp.get("SP_T_DX", None)
            sp_dh = sp.get("SP_T_DH", None)
            sp_total = sp.get("SP_T_DV", None)

        metrics = {
            "model_type": self.hparams.model_type,
            "hidden_size": self.hparams.hidden_size,
            "num_layers": self.hparams.num_layers,
            "thx": self.hparams.thx,
            "thh": self.hparams.thh,
            "qa": self.hparams.qa,
            "lambda_reg": self.hparams.lambda_reg,
            **self.log_hparams,
            "sp_dx": sp_dx,
            "sp_dh": sp_dh,
            "sp_total": sp_total,
            "test_acc": acc,
            "test_f1": f1,
            "test_loss": loss,
            "test_mae": mae,
        }

        if self.hparams.csv_output is not None:
            self._write_test_results_to_csv(metrics, filename=self.hparams.csv_output)
            print(f"\n\nTest results saved to CSV in path {self.hparams.csv_output}:", metrics, "\n\n")
