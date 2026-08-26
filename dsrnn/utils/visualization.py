"""Plotting utilities for sparsity-mask analysis."""

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_adjacency_heatmap(adj: torch.Tensor,
                           cmap: str = "viridis",
                           figsize: tuple = (6, 6),
                           max_ticks: int = 10,
                           label_rotation: float = 90,
row_width: float = 8,
                           title: str = "Adjacency Matrix Heatmap") -> plt.Figure:
    """Heatmap of a weighted (N, N) adjacency matrix. Pure matplotlib."""
    mat = adj.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, origin="upper")

    n = mat.shape[0]
    ticks = np.arange(n) if n <= max_ticks else np.linspace(0, n - 1, max_ticks, dtype=int)
    ax.set_xlabel("To node")
    ax.set_ylabel("From node")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, rotation=label_rotation, fontsize=row_width)
    ax.set_yticklabels(ticks, rotation=0, fontsize=row_width)
    ax.set_aspect("equal")
    ax.set_xticks([i - 0.5 for i in range(1, mat.shape[1])], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, mat.shape[0])], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weight")
    fig.tight_layout()
    return fig


def plot_channel_mask_heatmap(mask: torch.Tensor,
                              title: str = "Channel mask over time",
                              cmap: str = "viridis") -> plt.Figure:
    """Heatmap of a (T, H) channel mask (channels on y, time on x).

    This is the channel-level analogue of the adjacency heatmap for the new
    router, which produces per-channel (not per-edge) masks.
    """
    mat = mask.detach().cpu().numpy().T  # (H, T)
    fig, ax = plt.subplots(figsize=(max(6, mat.shape[1] // 4), max(3, mat.shape[0] // 16)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, origin="lower")
    fig.colorbar(im, ax=ax, label="Routing score")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Hidden channel")
    ax.set_title(title)
    fig.tight_layout()
    return fig
