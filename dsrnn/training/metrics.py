"""Metric registry.

Torchmetrics functionals plus custom dice / Brier / adaptive-ECE metrics,
resolved by name.
"""

import functools
from typing import Any, Callable, Dict, Optional

import torch
import torchmetrics


def _binary_dice_score(preds: torch.Tensor,
                       targets: torch.Tensor,
                       smooth: float = 1.0,
                       p: float = 1.0,
                       from_logits: bool = True,
                       reduction: str = "mean") -> torch.Tensor:
    assert preds.shape[0] == targets.shape[0], "predict & target batch size don't match"

    if from_logits:
        preds = preds.sigmoid()

    preds = preds.contiguous().view(preds.shape[0], -1)
    target = targets.contiguous().view(targets.shape[0], -1)

    num = torch.sum(torch.mul(preds, target), dim=1) + smooth
    den = torch.sum(preds.pow(p) + target.pow(p), dim=1) + smooth

    score = (2 * num) / den

    if reduction == "mean":
        return score.mean()
    elif reduction == "sum":
        return score.sum()
    elif reduction == "none":
        return score
    else:
        raise Exception(f"Unexpected reduction {reduction}")


def dice_score(preds: torch.Tensor,
               targets: torch.Tensor,
               n_classes: int = 2,
               smooth: float = 1.0,
               p: float = 1.0,
               reduction: str = "mean",
               ignore_index: int = -1,
               weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if n_classes > 2:
        assert preds.shape == targets.shape, "predict & target shape do not match"
        dice = functools.partial(_binary_dice_score, smooth=smooth, p=p,
                                 reduction=reduction, from_logits=False)
        total_score = 0

        class_dim = 1 if len(preds.shape) < 4 else -3
        predict = preds.softmax(dim=-class_dim)

        for i in range(targets.shape[class_dim]):
            if i != ignore_index:
                dice_loss = dice(predict[:, i], targets[:, i])
                if weight is not None:
                    assert weight.shape[0] == targets.shape[class_dim], \
                        f"Expect weight shape [{targets.shape[class_dim]}], got [{weight.shape[0]}]"
                    dice_loss *= weight[i]
                total_score += dice_loss

        return total_score / targets.shape[class_dim]
    return _binary_dice_score(preds=preds, targets=targets, smooth=smooth, p=p,
                              from_logits=True, reduction=reduction)


def _is_multiclass_logits(preds: torch.Tensor) -> bool:
    return preds.dim() == 2 and preds.size(1) > 1


def _to_probabilities(preds: torch.Tensor, from_logits: bool) -> torch.Tensor:
    """Convert preds to probabilities (multiclass (N, C) or binary (N,))."""
    if from_logits:
        if _is_multiclass_logits(preds):
            return torch.nn.functional.softmax(preds, dim=1)
        return torch.sigmoid(preds).reshape(-1)
    if _is_multiclass_logits(preds):
        row_sum = preds.sum(dim=1, keepdim=True)
        close_to_one = torch.isclose(row_sum, torch.ones_like(row_sum), atol=1e-6)
        if close_to_one.all():
            return preds
        return preds / (row_sum + 1e-12)
    return preds.view(-1)


def brier_score(preds: torch.Tensor, target: torch.Tensor,
                from_logits: bool = True) -> torch.Tensor:
    """Brier score (multi-class or binary), torchmetrics-functional style."""
    probs = _to_probabilities(preds, from_logits)

    if _is_multiclass_logits(preds) or probs.dim() == 2:
        if target.dim() == 1:
            target_onehot = torch.nn.functional.one_hot(
                target.long(), num_classes=probs.size(1)).float()
        else:
            raise ValueError("For multiclass, target should be shape (N,) of class indices.")
        per_sample = torch.sum((probs - target_onehot) ** 2, dim=1)
    else:
        p = probs.view(-1)
        t = target.view(-1).float()
        per_sample = (p - t) ** 2

    return per_sample.mean()


def adaptive_expected_calibration_error(preds: torch.Tensor,
                                        target: torch.Tensor,
                                        n_bins: int = 15,
                                        from_logits: bool = True) -> torch.Tensor:
    """Adaptive (equal-mass) top-label expected calibration error."""
    probs = _to_probabilities(preds, from_logits)

    if probs.dim() == 2:
        confs, preds_top = torch.max(probs, dim=1)
    else:
        confs = probs.view(-1)
        preds_top = (confs >= 0.5).long()

    target = target.view(-1).long()
    if confs.numel() != target.numel():
        raise ValueError("preds and target must have same number of samples")

    correct = (preds_top == target).to(dtype=confs.dtype)

    n = confs.numel()
    if n == 0:
        raise ValueError("No samples provided (N=0)")

    n_bins = max(1, min(int(n_bins), n))

    sorted_conf, sorted_idx = torch.sort(confs)
    sorted_correct = correct[sorted_idx]

    boundaries = [int(round(i * n / n_bins)) for i in range(n_bins + 1)]

    device = confs.device
    dtype = confs.dtype
    ece_accum = torch.tensor(0.0, device=device, dtype=dtype)
    for i in range(n_bins):
        start, end = boundaries[i], boundaries[i + 1]
        if start >= end:
            continue
        bin_conf_vals = sorted_conf[start:end]
        bin_correct_vals = sorted_correct[start:end]
        bin_count = bin_conf_vals.numel()
        bin_gap = torch.abs(bin_correct_vals.mean() - bin_conf_vals.mean())
        ece_accum += (bin_count / float(n)) * bin_gap

    return ece_accum


_METRICS: Dict[str, Callable[..., torch.Tensor]] = {
    "accuracy": torchmetrics.functional.accuracy,
    "top_k_accuracy": torchmetrics.functional.accuracy,
    "auroc": torchmetrics.functional.auroc,
    "precision": torchmetrics.functional.precision,
    "recall": torchmetrics.functional.recall,
    "f1": torchmetrics.functional.f1_score,
    "fbeta": torchmetrics.functional.fbeta_score,
    "dice": dice_score,
    "jaccard": torchmetrics.functional.jaccard_index,
    "iou": torchmetrics.functional.jaccard_index,
    "mcc": torchmetrics.functional.matthews_corrcoef,
    "specificity": torchmetrics.functional.specificity,
    "ssim": torchmetrics.functional.structural_similarity_index_measure,
    "msssim": torchmetrics.functional.multiscale_structural_similarity_index_measure,
    "mse": torchmetrics.functional.mean_squared_error,
    "mae": torchmetrics.functional.mean_absolute_error,
    "ece": torchmetrics.functional.calibration_error,
    "mce": functools.partial(torchmetrics.functional.calibration_error, norm="max"),
    "ace": adaptive_expected_calibration_error,
    "brier": brier_score,
}


def resolve_metric(metric: str, **kwargs: Any) -> Callable[..., torch.Tensor]:
    """Resolve a metric name to a callable, binding ``kwargs``."""
    if metric not in _METRICS:
        raise NotImplementedError(
            f"Unknown metric '{metric}'. Available: {sorted(_METRICS)}"
        )
    return functools.partial(_METRICS[metric], **kwargs)
