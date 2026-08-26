"""Multi-task LightningModule: one shared backbone with per-task heads.

Ported from ``connectomencps.nn.sequence.multitask``. A single recurrent
backbone (dense or dynamic-sparse, via ``build_backbone``) is shared across
tasks; each task gets its own input projection and output head, and a task
embedding conditions the shared backbone. Per-task losses are combined through
a task-weighting module (uncertainty / fixed / equal).
"""

from typing import Any, Dict, List, Optional, Tuple

import einops
import lightning as lit
import numpy as np
import torch
import torchmetrics
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn

from dsrnn.training.optim import build_optimizer
from dsrnn.training.scheduler import LRSchedulerConfig, build_lr_scheduler
from dsrnn.training.sequence import build_backbone


class UncertaintyTaskWeighting(nn.Module):
    """Learnable homoscedastic uncertainty weighting (Kendall et al., 2018).

    Loss per task: (1 / 2σ_k²) * L_k + log σ_k, parametrized as
    log_var_k = log(σ_k²) for numerical stability.
    """

    def __init__(self, task_names: List[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.zeros(1, device=next(iter(losses.values())).device)
        weights = {}
        for task, loss in losses.items():
            log_var = self.log_vars[task]
            precision = torch.exp(-log_var)
            total = total + 0.5 * precision * loss + 0.5 * log_var
            weights[task] = precision.item()
        return total.squeeze(), weights


class FixedTaskWeighting(nn.Module):
    """Normalized fixed per-task weights (baseline / ablation)."""

    def __init__(self, task_names: List[str], weights: List[float]):
        super().__init__()
        assert len(task_names) == len(weights)
        total = sum(weights)
        self.weights = {t: w / total for t, w in zip(task_names, weights)}

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = sum(self.weights[t] * losses[t] for t in losses)
        return total, dict(self.weights)


class MultitaskModule(lit.LightningModule):
    """One shared (DS-)RNN across tasks, with per-task heads and task weighting."""

    def __init__(self,
                 task_configs: Dict[str, Dict[str, Any]],
                 model_type: str,
                 hidden_size: int,
                 task_weighting: nn.Module,
                 batch_first: bool = True,
                 implementation: str = "gathered",
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 optimizer_type: str = "adamw",
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 scheduler_config: Optional[LRSchedulerConfig] = None,
                 shared_out_head: bool = False,
                 shared_in_head: bool = False,
                 task_embedding_dim: int = 16):
        super().__init__()
        self.save_hyperparameters(ignore=["task_weighting"])

        self.task_configs = task_configs
        self.tasks = list(task_configs.keys())
        self.task_to_id = {t: i for i, t in enumerate(self.tasks)}
        self.task_weighting = task_weighting

        model_kwargs = model_kwargs or {}
        h_dim = hidden_size

        # Task embedding
        task_embedding_dim = max(task_embedding_dim, 0)
        self.task_embedding = nn.Embedding(
            len(self.tasks), task_embedding_dim if task_embedding_dim > 0 else h_dim)

        # Per-task input projections (map to h_dim - task_embedding_dim; the
        # task embedding is concatenated to reach h_dim, the backbone's input)
        self.input_heads = nn.ModuleDict()
        if shared_in_head:
            max_inp = max(cfg["input_size"] for cfg in task_configs.values())
            head = nn.Linear(max_inp, h_dim - task_embedding_dim)
            for task in task_configs:
                self.input_heads[task] = head
        else:
            for task, cfg in task_configs.items():
                self.input_heads[task] = nn.Linear(cfg["input_size"],
                                                   h_dim - task_embedding_dim)

        # Shared backbone (its input is h_dim: projected input + task embedding)
        self.model = build_backbone(model_type, input_size=hidden_size,
                                    hidden_size=hidden_size, batch_first=batch_first,
                                    implementation=implementation,
                                    model_kwargs=model_kwargs)

        # Per-task output heads
        self.output_heads = nn.ModuleDict()
        if shared_out_head:
            max_out = max(cfg["output_size"] for cfg in task_configs.values())
            head = nn.Linear(h_dim, max_out)
            for task in task_configs:
                self.output_heads[task] = head
        else:
            for task, cfg in task_configs.items():
                self.output_heads[task] = nn.Linear(h_dim, cfg["output_size"])

        # Per-task losses
        self.losses = nn.ModuleDict()
        for task, cfg in task_configs.items():
            lt = cfg["loss_type"]
            if lt == "ce":
                self.losses[task] = nn.CrossEntropyLoss()
            elif lt == "bce":
                self.losses[task] = nn.BCEWithLogitsLoss()
            elif lt == "mse":
                self.losses[task] = nn.MSELoss()
            elif lt == "huber":
                self.losses[task] = nn.HuberLoss()
            elif lt in ("l1", "mae"):
                self.losses[task] = nn.L1Loss()
            else:
                raise ValueError(f"Unknown loss type: {lt}")

        # Per-task metrics
        self.metrics = nn.ModuleDict()
        for task, cfg in task_configs.items():
            m = nn.ModuleDict()
            if cfg["loss_type"] == "ce":
                m["acc"] = torchmetrics.classification.MulticlassAccuracy(
                    num_classes=cfg["output_size"], average="macro")
                m["f1"] = torchmetrics.classification.MulticlassF1Score(
                    num_classes=cfg["output_size"], average="macro")
            elif cfg["loss_type"] == "bce":
                m["acc"] = torchmetrics.classification.BinaryAccuracy()
                m["f1"] = torchmetrics.classification.BinaryF1Score()
            elif cfg["loss_type"] == "mse":
                m["mse"] = torchmetrics.MeanSquaredError()
                m["mae"] = torchmetrics.MeanAbsoluteError()
            self.metrics[task] = m

        # Pruning masks (attached externally for RigL; no-op when empty)
        self.masks: Dict[str, Optional[torch.Tensor]] = {}

    @property
    def sparsity(self) -> float:
        if not self.masks:
            return 0.0
        kept = sum(m.sum().item() for m in self.masks.values())
        total = sum(m.numel() for m in self.masks.values())
        return 1.0 - kept / total

    def get_rnn_input(self, x_task: torch.Tensor, task_id: torch.Tensor, task: str) -> torch.Tensor:
        proj = self.input_heads[task](x_task)
        emb = self.task_embedding(task_id)
        emb = emb.unsqueeze(0).unsqueeze(0).expand(x_task.size(0), x_task.size(1), -1)
        if self.hparams.task_embedding_dim > 0:
            return torch.cat([proj, emb], dim=-1)
        return proj + emb

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor, tasks: np.ndarray,
                x_lengths: torch.Tensor, h0: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = {}
        for task_id in torch.unique(task_ids):
            mask = task_ids == task_id
            x_length_t = x_lengths[mask][0][0]
            x_length_d = x_lengths[mask][0][1]
            task = tasks[mask.cpu()][0]
            x_task = x[mask]
            if not self.hparams.shared_in_head:
                x_task = x_task[:, :x_length_t, :x_length_d]

            rnn_input = self.get_rnn_input(x_task=x_task, task=task, task_id=task_id)
            out, _ = self.model(rnn_input, h0, **kwargs)

            if self.task_configs[task]["return_sequences"]:
                b, t, h = out.shape
                out = self.output_heads[task](out.reshape(-1, h)).view(b, t, -1)
            else:
                last = out[:, -1] if self.hparams.batch_first else out[-1]
                out = self.output_heads[task](last)
            outputs[task] = (mask, out)
        return outputs

    def _shared_step(self, batch: Any, stage: str) -> STEP_OUTPUT:
        x = batch["x"]
        y = batch["y"]
        x_lengths = batch["x_lengths"]
        y_lengths = batch["y_lengths"]
        task_ids = batch["task_id"]
        tasks = batch["task"]

        outputs = self(x=x, task_ids=task_ids, tasks=tasks, x_lengths=x_lengths)

        per_task_losses = {}
        for task, (mask, y_hat_task) in outputs.items():
            y_task = y[mask]
            y_lengths_t = y_lengths[mask][0][0]
            y_lengths_d = y_lengths[mask][0][1]
            y_task = y_task[:, :y_lengths_t, :y_lengths_d]

            if self.hparams.shared_out_head:
                if not self.task_configs[task]["return_sequences"]:
                    y_hat_task = y_hat_task.unsqueeze(-2)
                y_hat_task = y_hat_task[:, :y_lengths_t, :]
                if isinstance(self.losses[task], (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss)):
                    y_hat_task = y_hat_task[:, :y_lengths_t, :self.task_configs[task]["output_size"]]
                else:
                    y_hat_task = y_hat_task[:, :, :y_lengths_d]
                if not self.task_configs[task]["return_sequences"]:
                    y_hat_task = y_hat_task.squeeze(-2)

            if not self.task_configs[task]["return_sequences"]:
                y_task = y_task.squeeze(-2)
            if isinstance(self.losses[task], (nn.BCEWithLogitsLoss, nn.CrossEntropyLoss)):
                y_task = y_task.squeeze(-1)
            if isinstance(self.losses[task], nn.CrossEntropyLoss):
                y_task = y_task.long()

            if isinstance(self.losses[task], nn.BCEWithLogitsLoss) and y_task.dim() < y_hat_task.dim() \
                    and y_hat_task.shape[-1] == 1:
                y_hat_task = y_hat_task.squeeze(-1)
                y_task = y_task.to(y_hat_task)
            if isinstance(self.losses[task], nn.CrossEntropyLoss) and self.task_configs[task]["return_sequences"]:
                y_hat_task = einops.rearrange(y_hat_task, "... t c -> ... c t")

            loss = self.losses[task](y_hat_task, y_task)
            per_task_losses[task] = loss

            self.log(f"{stage}/{task}_loss", loss.item(), on_step=stage == "train",
                     on_epoch=stage != "train", prog_bar=True, sync_dist=True)
            for name, metric in self.metrics[task].items():
                val = metric(y_hat_task, y_task)
                val = val if not isinstance(val, torch.Tensor) else val.item()
                self.log(f"{stage}/{task}/{name}", val, on_step=stage == "train",
                         on_epoch=stage != "train", prog_bar=True, sync_dist=True)

        total_loss, weights = self.task_weighting(per_task_losses)
        for task in per_task_losses:
            self.log(f"{stage}/{task}_weight", weights[task], on_epoch=True, prog_bar=False)
        self.log(f"{stage}/loss", total_loss, on_step=stage == "train",
                 on_epoch=stage != "train", prog_bar=True, sync_dist=True)

        return {"loss": total_loss}

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(
            self.named_parameters(),
            optimizer_type=self.hparams.optimizer_type,
            optimizer_kwargs=self.hparams.optimizer_kwargs,
        )
        if self.hparams.scheduler_config:
            scheduler = build_lr_scheduler(
                optimizer, self.hparams.scheduler_config,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.trainer.estimated_stepping_batches)
            if scheduler is not None:
                return {"optimizer": optimizer,
                        "lr_scheduler": {"scheduler": scheduler,
                                         "interval": self.hparams.scheduler_config.interval,
                                         "frequency": self.hparams.scheduler_config.frequency,
                                         "monitor": self.hparams.scheduler_config.monitor}}
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
