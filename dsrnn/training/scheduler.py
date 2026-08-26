"""Learning-rate scheduler construction."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LinearLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

_SCHEDULER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "none": {},
    "cosine": {"T_max": 100, "eta_min": 0.0},
    "cosine_restarts": {"T_0": 50, "T_mult": 1, "eta_min": 0.0},
    "one_cycle": {
        "max_lr": None,
        "total_steps": None,
        "pct_start": 0.3,
        "anneal_strategy": "cos",
        "div_factor": 25.0,
        "final_div_factor": 1e4,
    },
    "multistep": {"milestones": [30, 80], "gamma": 0.1},
    "reduce_on_plateau": {
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
        "threshold": 1e-4,
        "cooldown": 0,
        "min_lr": 0.0,
        "eps": 1e-8,
    },
    "exponential": {"gamma": 0.95},
    "linear": {"start_factor": 1e-3, "end_factor": 1.0, "total_iters": 10},
    "cyclic": {
        "base_lr": None,
        "max_lr": None,
        "step_size_up": 2000,
        "mode": "triangular2",
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
    },
}


@dataclass
class LRSchedulerConfig:
    """Configuration for a learning-rate scheduler.

    Attributes:
        scheduler_type: one of the keys in ``_SCHEDULER_DEFAULTS``.
        scheduler_kwargs: constructor kwargs; missing entries fall back to the
            defaults table for the chosen type.
        interval: ``"step"`` or ``"epoch"`` (Lightning).
        frequency: steps/epochs between updates (Lightning).
        monitor: metric for ``reduce_on_plateau``; forced to ``None`` for the
            other schedulers.
    """

    scheduler_type: str = "none"
    scheduler_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    interval: str = "epoch"
    frequency: int = 1
    monitor: Optional[str] = "val/loss"

    def __post_init__(self) -> None:
        self.scheduler_type = self.scheduler_type.lower()
        if self.scheduler_type not in _SCHEDULER_DEFAULTS:
            raise NotImplementedError(
                f"Unsupported scheduler '{self.scheduler_type}'. "
                f"Available: {sorted(_SCHEDULER_DEFAULTS)}"
            )
        if self.scheduler_kwargs is None:
            self.scheduler_kwargs = dict(_SCHEDULER_DEFAULTS[self.scheduler_type])
        if self.scheduler_type != "reduce_on_plateau":
            self.monitor = None


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: LRSchedulerConfig,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Instantiate the LR scheduler described by ``config``.

    ``epochs`` and ``steps_per_epoch`` are required only for ``one_cycle``.
    Returns ``None`` for ``scheduler_type="none"``.
    """
    scheduler_type = config.scheduler_type
    params = dict(config.scheduler_kwargs or {})

    if scheduler_type == "none":
        return None

    if scheduler_type == "one_cycle":
        if params.get("max_lr") is None:
            params["max_lr"] = optimizer.defaults.get("lr", 0.1)
        if params.get("total_steps") is None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError(
                    "epochs and steps_per_epoch must be provided for one_cycle"
                )
            params["total_steps"] = epochs * steps_per_epoch
        return OneCycleLR(optimizer, **params)

    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, **params)
    if scheduler_type == "cosine_restarts":
        return CosineAnnealingWarmRestarts(optimizer, **params)
    if scheduler_type == "multistep":
        return MultiStepLR(optimizer, **params)
    if scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, **params)
    if scheduler_type == "exponential":
        return ExponentialLR(optimizer, **params)
    if scheduler_type == "linear":
        return LinearLR(optimizer, **params)
    if scheduler_type == "cyclic":
        max_lr = params.pop("max_lr", None)
        base_lr = params.pop("base_lr", None)
        if max_lr is not None and base_lr is None:
            base_lr = max_lr / 10
        if base_lr is not None and max_lr is None:
            max_lr = base_lr * 10
        if base_lr is None and max_lr is None:
            max_lr = optimizer.defaults.get("lr", 0.01)
            base_lr = max_lr / 10
        return CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, **params)

    raise NotImplementedError(f"Unsupported scheduler: {scheduler_type}")
