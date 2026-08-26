"""Sparsity-mask monitoring callbacks, adapted to the new channel router.

Ported from ``connectomencps.callbacks.sparsity``. The old GNN router produced
edge-level (N, N) adjacency masks; the new router produces channel masks
(B, H), so the monitors visualize the channel mask over time (a channels x
time heatmap), its evolution, and per-class averages. Graph metrics are
computed on the mask-gated recurrent subnetwork (the cell's recurrent weight
restricted to active channels) when ``with_graph_metrics`` is set.

Also includes the multitask variant (``MultitaskMaskAnalysisCallback``) for
cross-task routing-overlap analysis.
"""

import os
from typing import Dict, List, Optional

import lightning as lit
import numpy as np
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt

from dsrnn.utils.sparsitymasks import (compute_inter_class_fuzzy_iou_similarity,
                                     compute_inter_class_jsd,
                                     compute_inter_class_similarity)
from dsrnn.utils.visualization import plot_channel_mask_heatmap


def _collect_hidden_mask_over_time(backbone, x: torch.Tensor) -> torch.Tensor:
    """Return the hidden-channel mask over time for one input, shape (T, H).

    Uses the cell's public batched routing (input-dependent, matching how the
    sequence models route). ``x`` is (1, T, D) batch-first.
    """
    cell = backbone.cell
    t = x.shape[1]
    flat = x.reshape(t, -1)  # (T, D)
    _, hidden_routing = cell.compute_routing_batched(flat)
    return hidden_routing.mask  # (T, H)


def _active_recurrent_adjacency(backbone, mask_t: torch.Tensor) -> torch.Tensor:
    """Recurrent weight restricted to active channels (for graph metrics).

    Uses the hidden-to-hidden weight's first gate block; mask_t is (H,).
    """
    cell = backbone.cell
    H = cell.hidden_size
    w_hh = cell.weight_hh[:H, :]  # first gate block (H, H)
    active = (mask_t > 0).to(w_hh)
    adj = w_hh * active.unsqueeze(0)  # gate the source channels
    idx = torch.nonzero(mask_t > 0, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return torch.zeros(1, 1)
    return adj[idx][:, idx].abs()


def _log_dir(trainer) -> str:
    log_dir = getattr(trainer, "log_dir", None)
    return str(log_dir) if log_dir else str(getattr(trainer, "default_root_dir", os.getcwd()))


def _log_fig(trainer, fig, key):
    if isinstance(trainer.logger, WandbLogger):
        import wandb
        trainer.logger.experiment.log({key: wandb.Image(fig)})
    else:
        path = os.path.join(_log_dir(trainer), key.replace("/", "_"))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path + ".png", dpi=150, bbox_inches="tight")


class SparsityMaskMonitor(lit.Callback):
    """Log channel-mask heatmaps (+ optional graph metrics on the subnetwork)."""

    def __init__(self, every_n_epochs: int = 1, log_evolution: bool = True,
                 log_validation: bool = False, with_graph_metrics: bool = False):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.log_evolution = log_evolution
        self.log_validation = log_validation
        self.with_graph_metrics = with_graph_metrics

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.log_validation:
            self._shared(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._shared(trainer, pl_module, "test")

    def _shared(self, trainer, pl_module, stage):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        backbone = getattr(pl_module, "backbone", None)
        if backbone is None or not hasattr(backbone, "cell"):
            return

        batch = next(iter(trainer.val_dataloaders if stage == "val"
                          else trainer.test_dataloaders))
        x, _ = batch
        x = x[0].unsqueeze(0).to(pl_module.device)  # (1, T, D)

        with torch.no_grad():
            mask = _collect_hidden_mask_over_time(backbone, x)  # (T, H)

        fig = plot_channel_mask_heatmap(mask, title=f"Channel mask over time ({stage})")
        _log_fig(trainer, fig, f"sparsity_mask_{stage}/heatmap")
        plt.close(fig)

        if self.with_graph_metrics:
            from dsrnn.utils.graphmetrics import compute_graph_metrics
            metrics_over_t = []
            for t in range(mask.shape[0]):
                adj = _active_recurrent_adjacency(backbone, mask[t])
                metrics_over_t.append(compute_graph_metrics(adj, advanced=False))
            avg = {}
            for key in metrics_over_t[0]:
                series = torch.stack([m[key].reshape(-1)[0].float() for m in metrics_over_t])
                avg[f"sparsity_mask_metrics_{stage}/{key}_mean"] = float(series.mean())
            trainer.logger.log_metrics(avg, step=trainer.global_step)


class SparsityMaskPerClassMonitor(lit.Callback):
    """Log the average channel mask per ground-truth class."""

    def __init__(self, n_cols: int = 5, every_n_epochs: int = 1,
                 log_validation: bool = False):
        super().__init__()
        self.n_cols = n_cols
        self.every_n_epochs = every_n_epochs
        self.log_validation = log_validation

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.log_validation:
            self._shared(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._shared(trainer, pl_module, "test")

    def _shared(self, trainer, pl_module, stage):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        backbone = getattr(pl_module, "backbone", None)
        if backbone is None or not hasattr(backbone, "cell"):
            return

        batch = next(iter(trainer.val_dataloaders if stage == "val"
                          else trainer.test_dataloaders))
        x, y = batch
        x = x.to(pl_module.device)

        with torch.no_grad():
            t, b = x.shape[1], x.shape[0]
            flat = x.reshape(t * b, -1)
            _, hidden_routing = backbone.cell.compute_routing_batched(flat)
            masks = hidden_routing.mask.view(t, b, -1).permute(1, 0, 2)  # (B, T, H)

        labels = y.view(-1).cpu()
        unique = torch.unique(labels).tolist()
        n = len(unique)
        n_rows = int(np.ceil(n / self.n_cols))
        fig, axs = plt.subplots(n_rows, self.n_cols, figsize=(3 * self.n_cols, 3 * n_rows))
        axs = np.atleast_1d(axs).flat
        for ax, label in zip(axs, unique):
            subset = masks[(labels == label).to(masks.device)]
            avg = subset.mean(dim=(0, 1)).cpu().numpy()  # (H,)
            ax.imshow(avg[:, None], aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"Class={label}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for j in range(n, n_rows * self.n_cols):
            axs[j].axis("off")
        fig.suptitle(f"Per-class average channel mask ({stage})")
        fig.tight_layout()
        _log_fig(trainer, fig, f"sparsity_mask_{stage}/per_class")
        plt.close(fig)


class MultitaskMaskAnalysisCallback(lit.Callback):
    """Cross-task routing-overlap analysis for the multitask DS models.

    Per task, collects the channel-routing masks on a few validation batches
    and logs a per-task channel x time heatmap plus the cross-task similarity
    matrices (cosine / fuzzy-IoU / JSD) and their mean off-diagonal scalars.
    This is the paper's input-dependent-subnetwork evidence (Sec. 4.3 / 5.4).

    Adapted to the new channel router: masks are collected by routing each
    task's conditioned input through the backbone cell's ``compute_routing_batched``.
    """

    def __init__(self,
                 datamodule: lit.LightningDataModule,
                 every_n_epochs: int = 5,
                 binarize: bool = False,
                 similarity_mode: str = "minmax",
                 max_batches_per_task: int = 2,
                 log_validation: bool = True):
        super().__init__()
        self.datamodule = datamodule
        self.every_n_epochs = every_n_epochs
        self.binarize = binarize
        self.similarity_mode = similarity_mode
        self.max_batches_per_task = max_batches_per_task
        self.log_validation = log_validation

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.log_validation:
            self._shared(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._shared(trainer, pl_module, "test")

    # -- mask collection --

    def _task_masks(self, pl_module) -> Dict[str, torch.Tensor]:
        """Collect a per-task channel-mask centroid, shape (T, H) per task."""
        device = pl_module.device
        cell = pl_module.model.cell
        task_masks: Dict[str, torch.Tensor] = {}

        task_xs: Dict[str, List[torch.Tensor]] = {t: [] for t in pl_module.tasks}
        counts = {t: 0 for t in pl_module.tasks}
        with torch.no_grad():
            for batch in self.datamodule.val_dataloader():
                x = batch["x"].to(device)
                task_ids = batch["task_id"].to(device)
                x_lengths = batch["x_lengths"].to(device)
                tasks = batch["task"]
                for task in pl_module.tasks:
                    if counts[task] >= self.max_batches_per_task:
                        continue
                    bmask = (task_ids == pl_module.task_to_id[task]).cpu()
                    if bmask.sum() == 0:
                        continue
                    x_task = x[bmask.cpu()]
                    if not pl_module.hparams.shared_in_head:
                        Tx, Dx = x_lengths[bmask][0].tolist()
                        x_task = x_task[:, :Tx, :Dx]
                    # condition on the task (project + task embedding)
                    rnn_input = pl_module.get_rnn_input(
                        x_task, pl_module.task_to_id[task] if not torch.is_tensor(pl_module.task_to_id[task]) else pl_module.task_to_id[task],
                        task)
                    task_id_tensor = torch.tensor(pl_module.task_to_id[task], device=device)
                    rnn_input = pl_module.get_rnn_input(x_task, task_id_tensor, task)
                    b, t, h = rnn_input.shape
                    flat = rnn_input.reshape(t * b, h)
                    _, hidden_routing = cell.compute_routing_batched(flat)
                    masks = hidden_routing.mask.view(t, b, h).permute(1, 0, 2)  # (B,T,H)
                    task_xs[task].append(masks)
                    counts[task] += 1
                if all(c >= self.max_batches_per_task for c in counts.values()):
                    break
        for task in pl_module.tasks:
            if task_xs[task]:
                m = torch.cat(task_xs[task], dim=0)  # (N, T, H)
                task_masks[task] = m
        return task_masks

    def _shared(self, trainer, pl_module, stage):
        epoch = trainer.current_epoch
        if (stage != "test" and epoch == 0) or epoch % self.every_n_epochs != 0:
            return
        backbone = getattr(pl_module, "model", None)
        if backbone is None or not hasattr(backbone, "cell"):
            return  # only DS models have a routing cell

        pl_module.eval()
        task_masks = self._task_masks(pl_module)
        if len(task_masks) < 1:
            return

        # Per-task heatmaps (channel x time)
        for task, m in task_masks.items():
            centroid = m.mean(dim=0)  # (T, H)
            fig = plot_channel_mask_heatmap(centroid,
                                            title=f"Task={task} | channel mask ({stage})")
            _log_fig(trainer, fig, f"masks_{stage}/{task}")
            plt.close(fig)

        # Cross-task similarity: reshape centroids to (1, 1, n_tasks, T, H)
        task_names = list(task_masks.keys())
        n_tasks = len(task_names)
        if n_tasks >= 2:
            max_t = max(task_masks[t].shape[1] for t in task_names)
            padded = []
            for t in task_names:
                c = task_masks[t].mean(dim=0)  # (T, H)
                pad_t = max_t - c.shape[0]
                c = torch.nn.functional.pad(c, (0, 0, 0, pad_t))
                padded.append(c)
            # (n_tasks, T, H) -> (1, 1, n_tasks, T, H)
            stacked = torch.stack(padded, dim=0).unsqueeze(0).unsqueeze(0)
            pseudo_labels = torch.arange(n_tasks, device=pl_module.device)
            off_diag = ~np.eye(n_tasks, dtype=bool)

            sims = {
                "cosine": compute_inter_class_similarity(stacked, pseudo_labels,
                                                         binarize=self.binarize).cpu().numpy(),
                "fuzzy_iou": compute_inter_class_fuzzy_iou_similarity(
                    stacked, pseudo_labels, binarize=self.binarize,
                    mode=self.similarity_mode).cpu().numpy(),
                "jsd": compute_inter_class_jsd(stacked, pseudo_labels,
                                               binarize=self.binarize,
                                               return_similarity=True).cpu().numpy(),
            }
            for name, mat in sims.items():
                fig, ax = plt.subplots(figsize=(max(4, n_tasks), max(4, n_tasks)))
                im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
                fig.colorbar(im, ax=ax)
                ax.set_xticks(range(n_tasks))
                ax.set_xticklabels(task_names, rotation=45, ha="right")
                ax.set_yticks(range(n_tasks))
                ax.set_yticklabels(task_names)
                ax.set_title(f"Cross-task {name} ({stage})")
                fig.tight_layout()
                _log_fig(trainer, fig, f"masks_{stage}/similarity/{name}")
                plt.close(fig)

                trainer.logger.log_metrics(
                    {f"masks_{stage}/similarity/{name}_mean_offdiag": float(mat[off_diag].mean())},
                    step=trainer.global_step)
