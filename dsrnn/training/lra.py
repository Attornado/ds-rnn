"""LRA LightningModule for the S4 model family (S4 / DS-S4 / Liquid variants).

This module is intentionally separate from the RNN/DSRNN stack: it is built
only from ``dsrnn.nn.dss4.DSS4`` blocks and shares nothing with the recurrent
cells or the DSRNN wrapper.

A single ``LRAModule`` unifies the three wrapper classes used by the original
per-dataset scripts (``RNNClassifier``, ``S4LRAClassifier``, ``LRAModule``),
which all share the same block structure (norm -> S4 -> dropout -> GELU ->
linear -> residual) and all used BatchNorm1d. ``norm_type`` selects
batch/layer norm; the default is batch, matching the original scripts.

Dense S4 is ``DSS4(router_config=None)``: with no router the layer applies no
mask, so the same class covers "s4" (dense) and "dss4" (dynamic-sparse), with
"liqs4"/"dsliqs4" adding the liquid kernel.
"""

from typing import Any, Callable, Dict, Optional

import lightning as lit
import torch
from torch import nn

from dsrnn.nn.dss4 import DSS4
from dsrnn.nn.router import RouterConfig
from dsrnn.training.optim import build_optimizer
from dsrnn.training.scheduler import LRSchedulerConfig, build_lr_scheduler

# SSM kernel parameters get their own group: lower LR, zero weight decay.
# Mirrors the original rule: name contains "kernel" and one of log_dt/B/C/P.
SSM_PARAM_PATTERN = r"kernel.*(?:log_dt|[BCP])"


class S4Block(nn.Module):
    """One S4 residual block: norm -> S4 -> dropout -> GELU -> linear -> +residual.

    Operates on (B, L, H); the S4 layer is transposed (B, H, L) internally.
    """

    def __init__(self, d_model: int, d_state: int, l_max: Optional[int],
                 dropout: float, norm_type: str = "batch",
                 s4_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(d_model)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type '{norm_type}'.")

        self.s4 = DSS4(d_model=d_model, d_state=d_state, l_max=l_max,
                       bidirectional=False, transposed=True, **(s4_kwargs or {}))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        residual = x
        if self.norm_type == "batch":
            x = self.norm(x.transpose(-2, -1)).transpose(-2, -1)  # BatchNorm over H
        else:
            x = self.norm(x)
        x = x.transpose(-2, -1)   # (B, H, L) for the transposed S4
        x, _ = self.s4(x)
        x = x.transpose(-2, -1)   # back to (B, L, H)
        x = self.dropout(self.activation(x))
        x = self.linear(x)
        return x + residual


class LRAModule(lit.LightningModule):
    """S4-family LightningModule for the Long Range Arena benchmarks.

    Supports "s4" (dense), "dss4" (dynamic-sparse), "liqs4" and "dsliqs4"
    (liquid-kernel variants). Token inputs (vocab_size > 0) use an Embedding;
    continuous inputs (e.g. sequential CIFAR) use a Linear in-projection.
    """

    def __init__(self,
                 model_type: str = "s4",
                 input_size: int = 1,
                 vocab_size: int = -1,
                 d_model: int = 128,
                 d_state: int = 64,
                 output_size: int = 10,
                 n_layers: int = 4,
                 l_max: Optional[int] = 1024,
                 dropout: float = 0.1,
                 pooling: str = "mean",
                 norm_type: str = "batch",
                 liquid_kernel: Optional[str] = None,
                 liquid_degree: int = 2,
                 router_config: Optional[RouterConfig] = None,
                 sparsity_target: str = "k_f",
                 sparsity_dim: str = "H",
                 use_timewise_mask: bool = False,
                 use_sparse_implementation: bool = False,
                 lr: float = 0.004,
                 ssm_lr: float = 0.001,
                 weight_decay: float = 0.01,
                 test_with_corruptions: bool = False,
                 results_file: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["router_config"])

        model_type = model_type.lower()
        if model_type not in ("s4", "dss4", "liqs4", "dsliqs4"):
            raise ValueError(f"Unsupported model_type '{model_type}'.")
        is_ds = model_type in ("dss4", "dsliqs4")
        if "liq" in model_type and liquid_kernel is None:
            liquid_kernel = "polyb"

        # Input projection
        if vocab_size is not None and vocab_size > 0:
            self.embed: nn.Module = nn.Embedding(vocab_size, d_model)
        else:
            self.embed = nn.Linear(input_size, d_model)

        # S4 blocks; DS variants attach a router to each layer.
        s4_kwargs: Dict[str, Any] = dict(
            liquid_kernel=liquid_kernel, liquid_degree=liquid_degree,
            sparsity_target=sparsity_target, sparsity_dim=sparsity_dim,
            use_timewise_mask=use_timewise_mask,
            use_sparse_implementation=use_sparse_implementation,
        )
        if is_ds:
            if router_config is None:
                raise ValueError(f"router_config is required for model_type '{model_type}'.")
            s4_kwargs["router_config"] = router_config

        self.layers = nn.ModuleList([
            S4Block(d_model, d_state, l_max, dropout, norm_type, s4_kwargs)
            for _ in range(n_layers)
        ])

        if pooling in (None, "none"):
            self.pool = lambda x: x
        elif pooling == "mean":
            self.pool = lambda x: x.mean(dim=1)
        elif pooling == "last":
            self.pool = lambda x: x[:, -1]
        else:
            raise ValueError(f"Unknown pooling '{pooling}'.")
        self.head = nn.Linear(d_model, output_size)

        self.criterion = nn.BCEWithLogitsLoss() if output_size == 1 else nn.CrossEntropyLoss()

        # Test-time corruptions (robustness evaluation)
        self.test_corruptions: Dict[str, Callable] = {}
        if test_with_corruptions:
            for nl in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                self.test_corruptions[f"gauss_noise{nl}"] = \
                    lambda x, nl=nl: x + nl * torch.randn_like(x)
            for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
                self.test_corruptions[f"channel_corrupt{p}"] = (
                    lambda x, p=p: torch.where(
                        (torch.rand(x.shape[-1], device=x.device) < p)
                        .view(*([1] * (x.dim() - 1)), x.shape[-1]),
                        torch.randn_like(x), x)
                )

        # RigL masks (attached externally; no-op when empty)
        self.masks: Dict[str, Optional[torch.Tensor]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.embed, nn.Embedding):
            if torch.is_floating_point(x):
                x = x.argmax(dim=-1).long()  # one-hot/float -> token indices
            x = x.long()
        else:
            x = x.float()
        x = self.embed(x)                     # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.pool(x))

    def _accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            preds = (torch.sigmoid(logits) >= 0.5).float()
            return (preds == y).float().mean()
        return (logits.argmax(dim=-1) == y).float().mean()

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            loss = self.criterion(logits.squeeze(-1), y.to(logits.dtype))
        else:
            loss = self.criterion(logits, y)
        acc = self._accuracy(logits, y)
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/acc", acc, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, sync_dist=True)

        if stage == "test" and self.hparams.test_with_corruptions:
            for name, fn in self.test_corruptions.items():
                logits_c = self(fn(x))
                self.log(f"test/{name}_acc", self._accuracy(logits_c, y),
                         on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.named_parameters(),
            optimizer_type="adamw",
            optimizer_kwargs={
                "ssm": {"pattern": SSM_PARAM_PATTERN,
                        "lr": self.hparams.ssm_lr, "weight_decay": 0.0},
                "all": {"lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay},
            },
        )
        scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(scheduler_type="cosine",
                              scheduler_kwargs={"T_max": max(1, self.trainer.max_epochs)}),
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_test_end(self) -> None:
        """Append a config + test-metrics row to the results CSV."""
        if not self.hparams.results_file:
            return
        import csv
        import os

        def _scalar(v):
            return float(v.cpu().detach().item()) if hasattr(v, "cpu") else float(v)

        metrics = dict(self.trainer.callback_metrics)
        row = {
            "model_type": self.hparams.model_type,
            "liquid_kernel": self.hparams.liquid_kernel,
            "d_model": self.hparams.d_model,
            "d_state": self.hparams.d_state,
            "n_layers": self.hparams.n_layers,
            "lr": self.hparams.lr,
            "ssm_lr": self.hparams.ssm_lr,
            "weight_decay": self.hparams.weight_decay,
            "test_loss": _scalar(metrics.get("test/loss")),
            "test_acc": _scalar(metrics.get("test/acc")),
        }
        if self.hparams.test_with_corruptions:
            for name in self.test_corruptions:
                key = f"test/{name}_acc"
                if key in metrics:
                    row[f"test_{name}_acc"] = _scalar(metrics[key])

        path = self.hparams.results_file
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = not os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Saved test results to: {path}")

    # -- RigL hooks (no-op until masks are attached) --

    def apply_masks(self) -> None:
        if not self.masks:
            return
        with torch.no_grad():
            for name, param in self.named_parameters():
                m = self.masks.get(name, None)
                if m is not None:
                    param.data.mul_(m.to(param.device))

    def on_before_zero_grad(self, optimizer) -> None:
        self.apply_masks()

    def on_after_backward(self) -> None:
        self.apply_masks()
