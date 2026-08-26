"""Sparsity-mask monitoring callbacks, adapted to the new channel router.

Ported from ``connectomencps.callbacks.sparsity``. The old GNN router produced
edge-level (N, N) adjacency masks; the new router produces channel masks
(B, H), so the monitors visualize the channel mask over time (a channels x
time heatmap), its evolution, and per-class averages. Graph metrics are
computed on the mask-gated recurrent subnetwork (the cell's recurrent weight
restricted to active channels) when ``with_graph_metrics`` is set.
"""

import os
from io import BytesIO
from typing import Optional

import imageio
import lightning as lit
import numpy as np
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt

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


class SparsityMaskMonitor(lit.Callback):
    """Log channel-mask heatmaps (+ evolution GIF, + optional graph metrics)."""

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

        if stage == "val":
            batch = next(iter(trainer.val_dataloaders))
        else:
            batch = next(iter(trainer.test_dataloaders))
        x, _ = batch
        x = x[0].unsqueeze(0).to(pl_module.device)  # (1, T, D)

        with torch.no_grad():
            mask = _collect_hidden_mask_over_time(backbone, x)  # (T, H)

        fig = plot_channel_mask_heatmap(mask, title=f"Channel mask over time ({stage})")
        self._log_fig(trainer, fig, f"sparsity_mask_{stage}/heatmap")
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

    def _log_fig(self, trainer, fig, key):
        if isinstance(trainer.logger, WandbLogger):
            import wandb
            trainer.logger.experiment.log({key: wandb.Image(fig)})
        else:
            path = os.path.join(_log_dir(trainer), key.replace("/", "_"))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fig.savefig(path + ".png", dpi=150, bbox_inches="tight")


def _log_dir(trainer) -> str:
    log_dir = getattr(trainer, "log_dir", None)
    return str(log_dir) if log_dir else str(getattr(trainer, "default_root_dir", os.getcwd()))


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

        if stage == "val":
            batch = next(iter(trainer.val_dataloaders))
        else:
            batch = next(iter(trainer.test_dataloaders))
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
        fig, axs = plt.subplots(n_rows, self.n_cols,
                                figsize=(3 * self.n_cols, 3 * n_rows))
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

        if isinstance(trainer.logger, WandbLogger):
            import wandb
            trainer.logger.experiment.log({f"sparsity_mask_{stage}/per_class": wandb.Image(fig)})
        else:
            path = os.path.join(_log_dir(trainer), f"sparsity_mask_classes_{stage}")
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(path, f"epoch_{trainer.current_epoch}.png"),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)
