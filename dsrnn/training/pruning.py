"""Pruning / dynamic sparse training primitives and Lightning callbacks.

Faithful port of ``connectomencps.nn.pruning`` and
``connectomencps.callbacks.pruning`` (IMP magnitude pruning + RigL rewiring),
used by the sparse-baseline experiments.
"""

from typing import Dict, List, Optional, Tuple

import logging

import lightning as lit
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _params_to_prune(model: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """Collect prunable parameters (excludes scalars and biases)."""
    to_prune = []
    for name, p in model.named_parameters():
        if p.dim() == 0:
            continue
        if "bias" in name.lower():
            continue
        to_prune.append((name, p))
    return to_prune


def make_initial_masks(model: torch.nn.Module,
                       sparsity: Optional[float] = None,
                       deterministic: bool = False) -> Dict[str, torch.Tensor]:
    """Create an initial boolean mask per prunable parameter.

    With ``sparsity=None`` every mask is all-True. Otherwise a fraction
    ``sparsity`` of entries is pruned (Bernoulli sampling, or top-k of random
    scores when ``deterministic=True``).
    """
    masks = {}
    for name, p in _params_to_prune(model):
        if sparsity is not None:
            if deterministic:
                k = int((1 - sparsity) * p.numel())
                scores = torch.rand(p.numel(), device=p.device)
                _, idx = torch.topk(scores, k=k)
                mask = torch.zeros(p.numel(), device=p.device, dtype=torch.bool)
                mask[idx] = True
                masks[name] = mask.view_as(p)
            else:
                masks[name] = torch.rand_like(p.data, device=p.device) > sparsity
        else:
            masks[name] = torch.ones_like(p.data, dtype=torch.bool, device=p.device)
    return masks


def apply_masks_to_model(model: torch.nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Multiply parameter data by masks in place (pruned weights stay zero)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.data.mul_(masks[name].to(p.device))


def count_nonzero_masks(masks: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Return (kept, total) element counts across all masks."""
    total = kept = 0
    for m in masks.values():
        total += m.numel()
        kept += int(m.sum().item())
    return kept, total


def global_magnitude_prune(model: torch.nn.Module,
                           current_masks: Dict[str, torch.Tensor],
                           amount: float) -> Dict[str, torch.Tensor]:
    """One IMP global-magnitude pruning step.

    Prunes a fraction ``amount`` of the currently active weights (smallest
    magnitude first, global threshold) and returns freshly-allocated masks.
    """
    vals = []
    names_and_tensors = _params_to_prune(model)
    for name, p in names_and_tensors:
        mask = current_masks.get(name, torch.ones_like(p.data, dtype=torch.bool, device=p.device))
        if mask.sum().item() == 0:
            continue
        vals.append(torch.abs(p.data[mask]).view(-1))
    if len(vals) == 0:
        return current_masks
    allvals = torch.cat(vals)
    k = int(np.floor(allvals.numel() * amount))
    if k == 0:
        return current_masks
    thr, _ = torch.kthvalue(allvals, k)
    new_masks = {}
    for name, p in names_and_tensors:
        cur_mask = current_masks.get(name, torch.ones_like(p.data, dtype=torch.bool, device=p.device))
        keep_mask = torch.abs(p.data) > thr
        new_masks[name] = cur_mask & keep_mask
    return new_masks


def rigl_step(model: torch.nn.Module,
              current_masks: Dict[str, torch.Tensor],
              grads: Dict[str, Optional[torch.Tensor]],
              rewire_fraction: float = 0.2,
              ignore_unselected_params: bool = False) -> Dict[str, torch.Tensor]:
    """One RigL rewiring step.

    Per parameter: prune the ``rewire_fraction`` smallest-magnitude active
    weights, then grow the same number among zero positions with the largest
    |grad| (uniform random fallback when gradients are unavailable).
    """
    new_masks = {}
    for name, p in _params_to_prune(model):
        if ignore_unselected_params and name not in current_masks:
            continue
        device = p.device
        mask = current_masks.get(name, torch.ones_like(p.data, dtype=torch.bool, device=device)).to(p.device)
        flat_mask = mask.view(-1)
        flat_w = p.data.view(-1)

        n_active = int(flat_mask.sum().item())
        n_rewire = int(np.floor(max(1.0, n_active * rewire_fraction))) if n_active > 0 else 0
        if n_rewire <= 0:
            new_masks[name] = mask.clone()
            continue

        active_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze()
        if active_idx.numel() == 0:
            new_masks[name] = mask.clone()
            continue

        if active_idx.numel() <= n_rewire:
            pruned_mask_flat = flat_mask.clone()
            pruned_mask_flat[active_idx] = False
        else:
            active_vals = torch.abs(flat_w[active_idx])
            sorted_idx = torch.argsort(active_vals)[:n_rewire]
            prune_flat_idx = active_idx[sorted_idx]
            pruned_mask_flat = flat_mask.clone()
            pruned_mask_flat[prune_flat_idx] = False

        zero_idx = torch.nonzero(~flat_mask, as_tuple=False).squeeze()
        grow_flat_idx = torch.tensor([], dtype=torch.long, device=device)
        if zero_idx.numel() > 0:
            g = grads.get(name, None)
            k_grow = min(n_rewire, int(zero_idx.numel()))
            if (g is None) or (g.numel() == 0):
                if k_grow > 0:
                    perm = torch.randperm(zero_idx.numel(), device=device)[:k_grow]
                    grow_flat_idx = zero_idx[perm]
            else:
                flat_g = g.view(-1).abs()
                zero_grads = flat_g[zero_idx]
                if k_grow > 0:
                    topk = torch.topk(zero_grads, k_grow)
                    grow_flat_idx = zero_idx[topk.indices]

        new_mask_flat = pruned_mask_flat.clone()
        if grow_flat_idx.numel() > 0:
            new_mask_flat[grow_flat_idx] = True
        new_masks[name] = new_mask_flat.view_as(mask)
    return new_masks


def _clear_optimizer_state_for_masks(optimizer: torch.optim.Optimizer,
                                     param_to_mask: Dict) -> None:
    """Zero optimizer state entries (momentum, exp_avg, ...) at pruned positions."""
    for p, state in optimizer.state.items():
        if p not in param_to_mask:
            continue
        mask = param_to_mask[p]
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.shape == p.shape:
                v.mul_(mask)


def apply_masks_and_handle_grow(model: torch.nn.Module,
                                old_masks: Dict[str, torch.Tensor],
                                new_masks: Dict[str, torch.Tensor],
                                grads: Optional[Dict[str, Optional[torch.Tensor]]] = None,
                                grow_init: str = "random",
                                scale_init: float = 0.01,
                                optimizer_list: Optional[List[torch.optim.Optimizer]] = None) -> None:
    """Apply new masks, initialize grown weights, zero pruned weights/grads.

    Grown positions are initialized from Normal(0, std * scale_init) when
    ``grow_init="random"`` (std of the current tensor, fallback 1e-3), else
    zero. Optimizer state at pruned positions is cleared if ``optimizer_list``
    is given.
    """
    param_to_oldmask = {}
    param_to_newmask = {}
    for name, p in model.named_parameters():
        if name in new_masks:
            param_to_oldmask[p] = old_masks.get(
                name, torch.ones_like(p.data, dtype=torch.bool, device=p.device)).to(p.device)
            param_to_newmask[p] = new_masks[name].to(p.device)

    with torch.no_grad():
        for p, old_m in param_to_oldmask.items():
            new_m = param_to_newmask[p]
            grown = (~old_m) & new_m
            pruned = old_m & (~new_m)

            if grown.any():
                if grow_init == "random":
                    std = float(p.data.std().cpu().item())
                    if std == 0.0:
                        std = 1e-3
                    rnd = torch.randn_like(p.data) * (std * scale_init)
                    p.data[grown] = rnd[grown]
                else:
                    p.data[grown] = 0.0

            if pruned.any():
                p.data[pruned] = 0.0

            p.data.mul_(new_m)

            if p.grad is not None:
                p.grad.data.mul_(new_m)

    if optimizer_list is not None:
        for opt in optimizer_list:
            _clear_optimizer_state_for_masks(opt, param_to_newmask)


def sync_masks_across_ranks(masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Broadcast masks from rank 0 to all ranks (no-op outside DDP)."""
    if not dist.is_available() or not dist.is_initialized():
        return masks
    rank = dist.get_rank()
    synced = {}
    for name, m in masks.items():
        t = m.to("cpu").to(torch.uint8)
        if rank == 0:
            dist.broadcast(t, src=0)
            synced[name] = t.to(torch.bool)
        else:
            tmp = torch.empty_like(t)
            dist.broadcast(tmp, src=0)
            synced[name] = tmp.to(torch.bool)
    return {name: synced[name].to(m.device) for name, m in masks.items()}


# ---------------------------------------------------------------------------
# Lightning callbacks
# ---------------------------------------------------------------------------

class MaskCallback(Callback):
    """Re-apply masks after each training batch (safety net for pruned weights)."""

    def __init__(self, masks: Dict[str, Optional[torch.Tensor]]):
        self.masks = masks

    def on_train_batch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule,
                           outputs, batch, batch_idx) -> None:
        if hasattr(pl_module, "masks"):
            pl_module.masks = self.masks
            pl_module.apply_masks()


class RigLCallback(Callback):
    """RigL dynamic sparse training callback.

    Runs a rewire step every ``update_frequency`` global steps
    (``on_after_backward``), then applies new masks, initializes grown
    weights, zeros pruned gradients and clears optimizer buffers.
    """

    def __init__(self,
                 masks: Dict[str, Optional[torch.Tensor]],
                 rewire_fraction: float = 0.2,
                 update_frequency: int = 100,
                 start_step: int = 0,
                 end_step: Optional[int] = None,
                 grow_init: str = "random",
                 scale_init: float = 0.01,
                 ignore_unselected_params: bool = False,
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        self.masks = masks  # shared mutable dict[name -> bool tensor]
        self.rewire_fraction = rewire_fraction
        self.update_frequency = update_frequency
        self.start_step = start_step
        self.end_step = end_step
        self.grow_init = grow_init
        self.scale_init = scale_init
        self.ignore_unselected_params = ignore_unselected_params
        self.logger = logger or _logger

    def on_after_backward(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        step = int(trainer.global_step)
        if step < self.start_step:
            return
        if (self.end_step is not None) and (step > self.end_step):
            return
        if (step - self.start_step) % self.update_frequency != 0:
            return

        grads = {}
        for name, p in pl_module.named_parameters():
            if name in self.masks:
                grads[name] = p.grad.detach().clone().to(p.device) if (p.grad is not None) else None

        new_masks = rigl_step(pl_module, self.masks, grads,
                              rewire_fraction=self.rewire_fraction,
                              ignore_unselected_params=self.ignore_unselected_params)

        try:
            new_masks = sync_masks_across_ranks(new_masks)
        except Exception as e:  # non-fatal
            self.logger.warning(f"mask sync failed: {e}")

        optimizers = trainer.optimizers if hasattr(trainer, "optimizers") else None
        optimizer_list = (optimizers if isinstance(optimizers, list) and len(optimizers) > 0
                          else ([optimizers] if optimizers is not None else None))
        apply_masks_and_handle_grow(pl_module, self.masks, new_masks, grads=grads,
                                    grow_init=self.grow_init, scale_init=self.scale_init,
                                    optimizer_list=optimizer_list)

        # Update the shared mask dict in place (keep the same object reference)
        self.masks.clear()
        for k, v in new_masks.items():
            self.masks[k] = v

        if hasattr(pl_module, "masks"):
            pl_module.masks = self.masks
            pl_module.apply_masks()

        kept, total = (sum(m.sum().item() for m in self.masks.values()),
                       sum(m.numel() for m in self.masks.values()))
        self.logger.info(f"RigL rewire at step {step}: kept {kept}/{total} "
                         f"({100. * kept / total:.2f}%)")
