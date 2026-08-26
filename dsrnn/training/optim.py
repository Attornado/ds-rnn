"""Optimizers with parameter-group support.

Lowercase-string optimizer dispatch plus a helper to split model parameters
into groups by name pattern (e.g. to give SSM kernel parameters a lower
learning rate and zero weight decay).
"""

from typing import Any, Dict, Iterable, List, Tuple

import re
import torch

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "adagrad": torch.optim.Adagrad,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "asgd": torch.optim.ASGD,
    "adadelta": torch.optim.Adadelta,
    "adamax": torch.optim.Adamax,
}


def build_param_groups(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    optimizer_kwargs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Split parameters into optimizer groups keyed by name pattern.

    ``optimizer_kwargs`` maps a group name to that group's optimizer kwargs
    (e.g. ``{"all": {"lr": 5e-3}}``). Matching rule, first match wins:

    - the group named ``"all"`` is the catch-all fallback;
    - otherwise the group name is used as a regex searched against the
      parameter name;
    - a group dict may instead provide an explicit ``"pattern"`` key, which is
      then used as the regex and popped before the kwargs reach the optimizer.

    Example (SSM kernel params at lower LR and no weight decay)::

        optimizer_kwargs = {
            "ssm": {"pattern": r"kernel|log_dt|\\bB\\b|\\bC\\b|\\bP\\b",
                    "lr": 1e-3, "weight_decay": 0.0},
            "all": {"lr": 4e-3, "weight_decay": 0.01},
        }
    """
    named_parameters = list(named_parameters)
    groups: Dict[str, Dict[str, Any]] = {}
    patterns: Dict[str, str] = {}
    for name, kwargs in optimizer_kwargs.items():
        kwargs = dict(kwargs)
        patterns[name] = kwargs.pop("pattern", name)
        groups[name] = {"params": [], **kwargs}

    for param_name, param in named_parameters:
        for group_name, group in groups.items():
            if group_name == "all":
                continue
            if re.search(patterns[group_name], param_name):
                group["params"].append(param)
                break
        else:
            if "all" not in groups:
                raise KeyError(
                    f"Parameter '{param_name}' matched no group and no 'all' "
                    "fallback group was provided."
                )
            groups["all"]["params"].append(param)

    return [g for name, g in groups.items() if g["params"] or name == "all"]


def build_optimizer(
    named_parameters_or_params,
    optimizer_type: str = "adamw",
    optimizer_kwargs: Dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    """Create an optimizer.

    Args:
        named_parameters_or_params: an iterable of ``(name, param)`` tuples
            (used with grouped ``optimizer_kwargs``) or a plain iterable of
            parameters (used with flat ``optimizer_kwargs``). The flat path
            also tolerates ``(name, param)`` tuples (they are unwrapped).
        optimizer_type: one of the keys in ``_OPTIMIZERS`` (case-insensitive).
        optimizer_kwargs: either flat kwargs (``{"lr": ...}``) applied to all
            parameters, or grouped kwargs (``{"all": {...}, "ssm": {...}}``)
            in which case ``named_parameters_or_params`` must be named.
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type not in _OPTIMIZERS:
        raise NotImplementedError(
            f"Unsupported optimizer type '{optimizer_type}'. "
            f"Available: {sorted(_OPTIMIZERS)}"
        )
    cls = _OPTIMIZERS[optimizer_type]
    optimizer_kwargs = dict(optimizer_kwargs or {})

    params = list(named_parameters_or_params)
    named = bool(params) and all(
        isinstance(p, tuple) and len(p) == 2 and isinstance(p[1], torch.Tensor)
        for p in params
    )

    is_grouped = bool(optimizer_kwargs) and all(
        isinstance(v, dict) for v in optimizer_kwargs.values()
    )
    if is_grouped:
        groups = build_param_groups(params, optimizer_kwargs)
        return cls(params=groups)

    if named:
        params = [p for _, p in params]
    return cls(params=params, **optimizer_kwargs)
