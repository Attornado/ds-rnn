"""Sequence-level LightningModule for the classification/regression benchmarks."""
from typing import Any, Callable, Dict, Optional, Tuple, Union
import einops
import lightning as lit
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
from dsrnn.models.dslstm_model import DSLSTM
from dsrnn.models.dsgru_model import DSGRU
from dsrnn.models.dsltc_model import DSLTC
from dsrnn.models.ltc_model import LTC
from dsrnn.nn.router import RouterConfig
from dsrnn.training.optim import build_optimizer
from dsrnn.training.scheduler import LRSchedulerConfig, build_lr_scheduler
from dsrnn.training.metrics import resolve_metric


class _TorchRNNAdapter(nn.Module):
    """Thin adapter giving torch.nn.LSTM/GRU the ``(out, state)`` interface."""

    def __init__(self, kind: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.0,
                 batch_first: bool = True):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cls(input_size=input_size, hidden_size=hidden_size,
                       num_layers=num_layers, dropout=dropout,
                       batch_first=batch_first)
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, x: torch.Tensor, hx=None):
        return self.rnn(x, hx)


def _build_router_config(router: Union[RouterConfig, Dict[str, Any]],
                         input_size: int, hidden_size: int) -> RouterConfig:
    """Build a RouterConfig, filling in the routed dimensions if omitted."""
    if isinstance(router, RouterConfig):
        return router
    router = dict(router)
    router.setdefault("num_channels", hidden_size)
    router.setdefault("input_dim", input_size)
    return RouterConfig(**router)


def build_backbone(model_type: str,
                   input_size: int,
                   hidden_size: int,
                   batch_first: bool = True,
                   implementation: str = "gathered",
                   model_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Construct a headless backbone returning ``(outputs, state)``.

    For the DS models, ``model_kwargs`` may contain a ``router_config`` entry
    (a RouterConfig or a dict of its kwargs; ``num_channels``/``input_dim``
    default to the hidden and input sizes). For ``ltc``, extra ``model_kwargs``
    go to ncps' LTC. ``dsltc`` requires ``implementation="dense"`` when the
    router backward rule is "ste" (the gathered DSLTC path only supports
    "selected_only").
    """
    model_type = model_type.lower()
    model_kwargs = dict(model_kwargs or {})

    if model_type in ("lstm", "gru"):
        return _TorchRNNAdapter(model_type, input_size=input_size,
                                hidden_size=hidden_size, batch_first=batch_first,
                                **model_kwargs)

    if model_type == "ltc":
        return LTC(input_size=input_size, hidden_size=hidden_size,
                   batch_first=batch_first, output_size=None, **model_kwargs)

    if model_type in ("dslstm", "dsgru", "dsltc"):
        router = model_kwargs.pop("router_config", None)
        if router is None:
            raise ValueError(f"{model_type} requires a 'router_config' in model_kwargs.")
        router = _build_router_config(router, input_size, hidden_size)
        cls = {"dslstm": DSLSTM, "dsgru": DSGRU, "dsltc": DSLTC}[model_type]
        return cls(input_size=input_size, hidden_size=hidden_size,
                   router_config=router, implementation=implementation,
                   batch_first=batch_first, output_size=None, **model_kwargs)

    raise ValueError(f"Unknown model type: {model_type}")


class SequenceModule(lit.LightningModule):
    """LightningModule for the sequence benchmarks (classification + regression)."""

    def __init__(
        self,
        model_type: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        use_final_proj: bool = True,
        return_sequences: bool = True,
        batch_first: bool = True,
        implementation: str = "gathered",
        model_kwargs: Optional[Dict[str, Any]] = None,
        test_transform: Optional[Callable] = None,
        loss_type: str = "ce",
        loss_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_type: str = "adamw",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[LRSchedulerConfig] = None,
        metric_params: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        test_with_corruptions: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.test_transform = test_transform
        metric_params = metric_params if metric_params is not None else {"mse": {}}

        self.backbone = build_backbone(
            model_type, input_size=input_size, hidden_size=hidden_size,
            batch_first=batch_first, implementation=implementation,
            model_kwargs=model_kwargs,
        )
        h_dim = hidden_size

        self.projection: nn.Module = nn.Identity()
        if use_final_proj:
            self.projection = nn.Linear(h_dim, output_size)

        loss_kwargs = loss_kwargs or {}
        loss_type = loss_type.lower()
        if loss_type == "mse":
            self.loss_fn: nn.Module = nn.MSELoss(**loss_kwargs)
        elif loss_type in ("mae", "l1"):
            self.loss_fn = nn.L1Loss(**loss_kwargs)
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss(**loss_kwargs)
        elif loss_type in ("ce", "crossentropy", "cross_entropy",
                           "bce", "binary_crossentropy", "binarycrossentropy"):
            if output_size == 1:
                self.loss_fn = nn.BCEWithLogitsLoss(**loss_kwargs)
            else:
                self.loss_fn = nn.CrossEntropyLoss(**loss_kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

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
            for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
                self.test_corruptions[f"channel_mask{p}"] = (
                    lambda x, p=p: torch.where(
                        (torch.rand(x.shape[-1], device=x.device) < p)
                        .view(*([1] * (x.dim() - 1)), x.shape[-1]),
                        torch.zeros_like(x), x)
                )

        self.metrics = {
            m: resolve_metric(m, **({} if metric_params[m] is None else metric_params[m]))
            for m in metric_params
        }

        self.masks: Dict[str, Optional[torch.Tensor]] = {}

    @property
    def sparsity(self) -> float:
        if not self.masks:
            return 0.0
        kept = sum(m.sum().item() for m in self.masks.values())
        total = sum(m.numel() for m in self.masks.values())
        return 1.0 - kept / total

    def forward(self, x: torch.Tensor, h0=None, **kwargs) -> Tuple[torch.Tensor, Any]:
        outputs, state = self.backbone(x, h0, **kwargs)

        if self.hparams.return_sequences:
            b, t, h = outputs.shape
            out = self.projection(outputs.reshape(-1, h)).view(b, t, -1)
        else:
            last = outputs[:, -1] if self.hparams.batch_first else outputs[-1]
            out = self.projection(last)
        return out, state

    def _shared_step(self, batch: Any, stage: str) -> STEP_OUTPUT:
        x, y = batch
        y_hat, _ = self(x)

        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss) and y.dim() < y_hat.dim() \
                and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
            y = y.to(y_hat)
        if isinstance(self.loss_fn, nn.CrossEntropyLoss) and self.hparams.return_sequences:
            y_hat = einops.rearrange(y_hat, "... t c -> ... c t")

        loss = self.loss_fn(y_hat, y)

        log_dict = {f"{stage}/loss": loss.item()}
        for name, metric in self.metrics.items():
            log_dict[f"{stage}/{name}"] = metric(y_hat, y).item()
        on_step = stage == "train"
        self.log_dict(dictionary=log_dict, on_step=on_step, sync_dist=True, prog_bar=True)

        if stage == "test" and self.hparams.test_with_corruptions:
            for corruption_name, corruption_fn in self.test_corruptions.items():
                x_c = corruption_fn(x)
                y_hat_c, _ = self(x_c)
                if isinstance(self.loss_fn, nn.BCEWithLogitsLoss) and y.dim() < y_hat_c.dim() \
                        and y_hat_c.shape[-1] == 1:
                    y_hat_c = y_hat_c.squeeze(-1)
                if isinstance(self.loss_fn, nn.CrossEntropyLoss) and self.hparams.return_sequences:
                    y_hat_c = einops.rearrange(y_hat_c, "... t c -> ... c t")
                clog = {f"{stage}/{corruption_name}_{name}": metric(y_hat_c, y).item()
                        for name, metric in self.metrics.items()}
                self.log_dict(dictionary=clog, on_step=on_step, sync_dist=True, prog_bar=True)

        return {"loss": loss}

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        tag = ""
        if self.test_transform is not None:
            tag = repr(self.test_transform)
            batch = self.test_transform(batch)
        return self._shared_step(batch, f"{tag}test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(
            self.named_parameters(),
            optimizer_type=self.hparams.optimizer_type,
            optimizer_kwargs=self.hparams.optimizer_kwargs,
        )
        if self.hparams.scheduler_config:
            scheduler = build_lr_scheduler(
                optimizer,
                self.hparams.scheduler_config,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.trainer.estimated_stepping_batches,
            )
            if scheduler is not None:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": self.hparams.scheduler_config.interval,
                        "frequency": self.hparams.scheduler_config.frequency,
                        "monitor": self.hparams.scheduler_config.monitor,
                    },
                }
        return {"optimizer": optimizer}

    def apply_masks(self) -> None:
        if not self.masks:
            return
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.masks:
                    param.data.mul_(self.masks[name].to(param.device))

    def on_before_zero_grad(self, optimizer) -> None:
        self.apply_masks()

    def on_after_backward(self) -> None:
        self.apply_masks()
