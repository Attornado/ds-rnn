"""Cross-class / cross-task routing-mask similarity utilities.

Ported from ``connectomencps.utils.visualization.sparsitymasks``: cosine
similarity, fuzzy IoU, and Jensen-Shannon divergence over per-class mask
centroids. These operate on mask tensors shaped (L, G, C, T, H) (layers x
gates x classes x time x channels); the new channel router yields one mask, so
L=G=1. Pure torch/numpy.
"""

from typing import Optional

import numpy as np
import torch


def compute_inter_class_similarity(res: torch.Tensor, labels: torch.Tensor,
                                   binarize: bool = False) -> torch.Tensor:
    """Mean pairwise cosine similarity matrix over per-class mask centroids."""
    timesteps = res.shape[-2]
    n_neurons = res.shape[-1]
    n_classes = int(labels.max().item() + 1)
    res_grouped = torch.zeros(res.shape[0], res.shape[1], n_classes, timesteps, n_neurons)
    for c in range(n_classes):
        idx = (labels.squeeze() == c)
        if idx.sum() == 0:
            continue
        hh_c = res[:, :, idx, :, :]  # (L, G, Bc, T, H)
        if binarize:
            hh_c = (hh_c > 0).float()
        res_grouped[:, :, c, :, :] = hh_c.mean(dim=2)

    res = torch.nn.functional.normalize(res_grouped, dim=-1)
    res_perm = res.permute(0, 1, 3, 2, 4)              # (L,G,T,C,H)
    sim = torch.matmul(res_perm, res_perm.transpose(-1, -2))  # (L,G,T,C,C)
    return sim.mean(dim=(0, 1, 2))                      # (C, C)


def compute_class_heatmaps(res: torch.Tensor, layer: int, labels: torch.Tensor,
                           n_gates: int = 4, n_neurons: int = 128,
                           timesteps: int = 28) -> torch.Tensor:
    """Per-class mask centroid heatmaps, layout (C, G, H, T)."""
    n_classes = int(labels.max().item() + 1)
    class_heats = []
    for c in range(n_classes):
        idx = (labels == c)
        if idx.sum() == 0:
            class_heats.append(torch.zeros(n_gates, n_neurons, timesteps))
            continue
        hh_c = res[layer, :, idx, :, :]   # (G, Bc, T, H)
        hh_c = hh_c.mean(dim=1)           # (G, T, H)
        class_heats.append(hh_c.permute(0, 2, 1).contiguous())  # (G, H, T)
    return torch.stack(class_heats, dim=0)[:, 0:n_gates]


def fuzzy_intersection(class_heats: torch.Tensor, mode: str = "minmax") -> torch.Tensor:
    if mode == "minmax":
        return class_heats.min(dim=0).values
    if mode == "multsum":
        return class_heats.log().sum(dim=0).exp()
    raise ValueError(f"Mode {mode} is not supported")


def fuzzy_union(class_heats: torch.Tensor, mode: str = "minmax") -> torch.Tensor:
    if mode == "minmax":
        return class_heats.max(dim=0).values
    if mode == "multsum":
        mult = class_heats.log().sum(dim=0).exp()
        return class_heats.sum(dim=0) - mult
    raise ValueError(f"Mode {mode} is not supported")


def fuzzy_xor(class_heats: torch.Tensor, mode: str = "minmax") -> torch.Tensor:
    """Symmetric difference: union - intersection."""
    return fuzzy_union(class_heats, mode=mode) - fuzzy_intersection(class_heats, mode=mode)


def compute_inter_class_fuzzy_iou_similarity(res: torch.Tensor, labels: torch.Tensor,
                                             binarize: bool = False, mode: str = "minmax",
                                             reduction: str = "mean") -> torch.Tensor:
    """Pairwise fuzzy-IoU similarity matrix over per-class mask centroids."""
    timesteps = res.shape[-2]
    n_neurons = res.shape[-1]
    n_classes = int(labels.max().item() + 1)

    res_grouped = torch.zeros(res.shape[0], res.shape[1], n_classes, timesteps, n_neurons,
                              device=res.device, dtype=res.dtype)
    for c in range(n_classes):
        idx = (labels.squeeze() == c)
        if idx.sum() == 0:
            continue
        hh_c = res[:, :, idx, :, :]
        if binarize:
            hh_c = (hh_c > 0).float()
        res_grouped[:, :, c, :, :] = hh_c.mean(dim=2)
    res = res_grouped

    sim = torch.zeros(n_classes, n_classes, device=res.device)
    for i in range(n_classes):
        for j in range(i, n_classes):
            pair = torch.stack([res[:, :, i, :, :], res[:, :, j, :, :]], dim=0)
            inter = (fuzzy_intersection(pair, mode=mode).sum(dim=-1)
                     / (fuzzy_union(pair, mode=mode) + 1e-12).sum(dim=-1))
            if reduction == "mean":
                score = inter.mean()
            elif reduction == "sum":
                score = inter.sum()
            else:
                raise ValueError(f"Unknown reduction {reduction}")
            sim[i, j] = score
            sim[j, i] = score
    return sim


def compute_inter_class_jsd(res: torch.Tensor, labels: torch.Tensor,
                            binarize: bool = False, eps: float = 1e-8,
                            return_similarity: bool = True) -> torch.Tensor:
    """Pairwise Jensen-Shannon divergence (as similarity if ``return_similarity``)."""
    timesteps = res.shape[-2]
    n_neurons = res.shape[-1]
    n_classes = int(labels.max().item() + 1)

    res_grouped = torch.zeros(res.shape[0], res.shape[1], n_classes, timesteps, n_neurons,
                              device=res.device, dtype=res.dtype)
    for c in range(n_classes):
        idx = (labels.squeeze() == c)
        if idx.sum() == 0:
            continue
        hh_c = res[:, :, idx, :, :]
        if binarize:
            hh_c = (hh_c > 0).float()
        res_grouped[:, :, c, :, :] = hh_c.mean(dim=2)

    res = res_grouped.clamp(min=0)
    res = res / (res.sum(dim=-1, keepdim=True) + eps)

    P = res.unsqueeze(2)  # (L,G,C,1,T,H)
    Q = res.unsqueeze(3)  # (L,G,1,C,T,H)
    M = 0.5 * (P + Q)

    def kl_div(A, B):
        return (A * (torch.log(A + eps) - torch.log(B + eps))).sum(dim=-1)

    jsd = 0.5 * (kl_div(P, M) + kl_div(Q, M))  # (L,G,C,C,T)
    jsd_mean = jsd.mean(dim=(0, 1, 4))          # (C,C)
    if return_similarity:
        jsd_mean = 1.0 - jsd_mean / torch.log(torch.tensor(2.0, device=res.device))
    return jsd_mean
