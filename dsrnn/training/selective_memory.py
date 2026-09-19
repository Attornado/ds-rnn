"""Models, metrics, and training module for selective memory."""

from pathlib import Path
from typing import Dict, Optional

import lightning as lit
import torch
import torch.distributed as dist
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn

from dsrnn.data.selective_memory import OPERATION_NAMES, QUERY, UPDATE
from dsrnn.nn.cells.dual_read_write import DualReadWriteRNNCell
from dsrnn.nn.cells.dual_read_write_sparse import SparseDualReadWriteRNNCell
from dsrnn.nn.router import Router, RouterConfig, RouterOutput
from dsrnn.training.optim import build_optimizer
from dsrnn.training.scheduler import LRSchedulerConfig, build_lr_scheduler


MODEL_VARIANTS = ("dense", "read_only", "write_only", "shared", "dual")
IMPLEMENTATIONS = ("reference", "sparse")


class SelectiveMemoryRNN(nn.Module):
    """Tanh recurrence configured by routing variant and compute implementation."""

    def __init__(self, input_size: int, hidden_size: int, value_size: int,
                 variant: str, read_sparsity: float = 0.5,
                 write_sparsity: float = 0.5, router_context_size: int = 8,
                 router_backward: str = "ste", implementation: str = "reference"):
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {MODEL_VARIANTS}.")
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"Unknown implementation {implementation!r}; "
                             f"expected one of {IMPLEMENTATIONS}.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.value_size = value_size
        self.variant = variant
        self.router_backward = router_backward
        self.implementation = implementation

        # Construct shared components before variant-specific routers so that
        # reseeding produces identical recurrent/output initialization.
        cell_type = (DualReadWriteRNNCell if implementation == "reference"
                     else SparseDualReadWriteRNNCell)
        self.cell = cell_type(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, value_size)
        base = dict(num_channels=hidden_size, input_dim=input_size,
                    context_dim=router_context_size, backward=router_backward,
                    binary=True, budget_rounding="ceil")
        if variant == "shared" and read_sparsity != write_sparsity:
            raise ValueError("The shared-mask variant requires equal read/write sparsity.")
        self.read_router = (Router(RouterConfig(sparsity=read_sparsity, **base))
                            if variant in ("read_only", "dual") else None)
        self.write_router = (Router(RouterConfig(sparsity=write_sparsity, **base))
                             if variant in ("write_only", "dual") else None)
        self.shared_router = (Router(RouterConfig(sparsity=read_sparsity, **base))
                              if variant == "shared" else None)

    @property
    def read_budget(self) -> int:
        router = (self.read_router if self.read_router is not None
                  else self.shared_router)
        return self.hidden_size if router is None else router.k

    @property
    def write_budget(self) -> int:
        router = (self.write_router if self.write_router is not None
                  else self.shared_router)
        return self.hidden_size if router is None else router.k

    @staticmethod
    def _fixed_routing(x: torch.Tensor, hidden_size: int) -> RouterOutput:
        mask = x.new_ones(x.size(0), hidden_size)
        idx = torch.arange(hidden_size, device=x.device).expand(x.size(0), -1)
        return RouterOutput(mask=mask, idx=idx, scores=mask, logits=mask)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None,
                return_routing: bool = False):
        if x.ndim != 3 or x.size(-1) != self.input_size:
            raise ValueError(f"x must have shape (B, T, {self.input_size}).")
        h = x.new_zeros(x.size(0), self.hidden_size) if h0 is None else h0
        outputs = []
        traces = {name: [] for name in (
            "read_mask", "write_mask", "read_indices", "write_indices",
            "read_scores", "write_scores", "read_logits", "write_logits")}
        for timestep in range(x.size(1)):
            xt = x[:, timestep]
            shared = self.shared_router(xt) if self.shared_router is not None else None
            read = (shared if shared is not None else
                    self.read_router(xt) if self.read_router is not None else
                    self._fixed_routing(xt, self.hidden_size))
            write = (shared if shared is not None else
                     self.write_router(xt) if self.write_router is not None else
                     self._fixed_routing(xt, self.hidden_size))
            cell_kwargs = {"read_mask": read.mask, "write_mask": write.mask}
            if self.implementation == "sparse":
                cell_kwargs.update(
                    read_backward=(self.router_backward
                                   if self.read_router is not None or shared is not None
                                   else None),
                    write_backward=(self.router_backward
                                    if self.write_router is not None or shared is not None
                                    else None),
                )
            h, _ = self.cell(xt, h, **cell_kwargs)
            # Keep the established baseline readout explicit: it sees the full
            # post-write state, including coordinates preserved by WRITE=0.
            outputs.append(self.output(h))
            for prefix, routing in (("read", read), ("write", write)):
                traces[prefix + "_mask"].append(routing.mask)
                traces[prefix + "_indices"].append(routing.idx)
                traces[prefix + "_scores"].append(routing.scores)
                traces[prefix + "_logits"].append(routing.logits)
        predictions = torch.stack(outputs, dim=1)
        if not return_routing:
            return predictions
        return predictions, {key: torch.stack(value, dim=1)
                             for key, value in traces.items()}


def masked_query_mse(predictions: torch.Tensor, targets: torch.Tensor,
                     query_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over QUERY timesteps only."""
    per_timestep = (predictions - targets).square().mean(dim=-1)
    weights = query_mask.to(per_timestep.dtype)
    return (per_timestep * weights).sum() / weights.sum().clamp_min(1.0)


def routing_statistics(trace: Dict[str, torch.Tensor],
                       operations: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Measure activity and the four binary read/write states by operation."""
    read = trace["read_mask"] > 0.5
    write = trace["write_mask"] > 0.5
    stats = {
        "read_activity": read.float().mean(),
        "write_activity": write.float().mean(),
    }
    states = {
        "00": ~read & ~write,
        "10": read & ~write,
        "01": ~read & write,
        "11": read & write,
    }
    for state, active in states.items():
        stats[f"state_{state}"] = active.float().mean()
    timestep_groups = {name: operations[..., index]
                       for index, name in enumerate(OPERATION_NAMES)}
    timestep_groups["query_only"] = (operations[..., QUERY]
                                     & ~operations[..., UPDATE])
    timestep_groups["query_update"] = (operations[..., QUERY]
                                       & operations[..., UPDATE])
    for op_name, timesteps in timestep_groups.items():
        selected = timesteps.unsqueeze(-1).expand_as(read)
        denominator = selected.sum().clamp_min(1)
        stats[f"{op_name}/read_activity"] = (read & selected).sum() / denominator
        stats[f"{op_name}/write_activity"] = (write & selected).sum() / denominator
        for state, active in states.items():
            stats[f"{op_name}/state_{state}"] = (active & selected).sum() / denominator
    return stats


def query_loss_statistics(predictions: torch.Tensor, targets: torch.Tensor,
                          operations: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split query error between pure queries and joint query/update steps."""
    query = operations[..., QUERY]
    update = operations[..., UPDATE]
    return {
        "query_only_mse": masked_query_mse(predictions, targets, query & ~update),
        "query_update_mse": masked_query_mse(predictions, targets, query & update),
    }


def memory_alignment_statistics(trace: Dict[str, torch.Tensor],
                                read_register: torch.Tensor,
                                write_register: torch.Tensor,
                                operations: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Measure whether the latest write to a register is later read and protected.

    Each query is paired with the most recent earlier STORE/UPDATE of the same
    symbolic register. Fractions use the source write-mask size as denominator.
    The protected-read excess subtracts the overlap expected if source-written
    coordinates were unrelated to the current ``(R=1, W=0)`` coordinates.
    """
    read = trace["read_mask"] > 0.5
    write = trace["write_mask"] > 0.5
    query = operations[..., QUERY]
    update = operations[..., UPDATE]
    groups = {
        "query": query,
        "query_only": query & ~update,
        "query_update": query & update,
    }
    stats = {}
    _, sequence_length, hidden_size = read.shape
    for group_name, selected_timesteps in groups.items():
        denominator = read.new_zeros((), dtype=torch.float32)
        source_read = denominator.clone()
        source_protected = denominator.clone()
        source_preserved = denominator.clone()
        source_preserved_read = denominator.clone()
        chance_protected = denominator.clone()
        for query_t in range(1, sequence_length):
            query_register = read_register[:, query_t]
            eligible = selected_timesteps[:, query_t] & (query_register >= 0)
            for source_t in range(query_t):
                same_register = write_register[:, source_t] == query_register
                if source_t + 1 < query_t:
                    later_same_register = (
                        write_register[:, source_t + 1:query_t]
                        == query_register.unsqueeze(1)).any(dim=1)
                else:
                    later_same_register = torch.zeros_like(eligible)
                use_pair = eligible & same_register & ~later_same_register
                source = write[use_pair, source_t]
                current_read = read[use_pair, query_t]
                current_write = write[use_pair, query_t]
                protected = current_read & ~current_write
                preserved = source.clone()
                for intervening_t in range(source_t + 1, query_t + 1):
                    preserved &= ~write[use_pair, intervening_t]
                source_size = source.sum(dtype=torch.float32)
                denominator += source_size
                source_read += (source & current_read).sum(dtype=torch.float32)
                source_protected += (source & protected).sum(dtype=torch.float32)
                source_preserved += preserved.sum(dtype=torch.float32)
                source_preserved_read += (preserved & current_read).sum(
                    dtype=torch.float32)
                chance_protected += (source.sum(dim=1, dtype=torch.float32)
                                     * protected.sum(dim=1, dtype=torch.float32)
                                     / hidden_size).sum()
        normalizer = denominator.clamp_min(1.0)
        protected_fraction = source_protected / normalizer
        stats[f"memory/{group_name}/source_read_fraction"] = source_read / normalizer
        stats[f"memory/{group_name}/source_protected_read_fraction"] = protected_fraction
        stats[f"memory/{group_name}/source_preserved_fraction"] = (
            source_preserved / normalizer)
        stats[f"memory/{group_name}/source_preserved_read_fraction"] = (
            source_preserved_read / normalizer)
        stats[f"memory/{group_name}/protected_read_excess"] = (
            protected_fraction - chance_protected / normalizer)
    return stats


def _merge_trace_shards(shards):
    """Merge distributed test shards and remove sampler padding duplicates."""
    tensors = {key: torch.cat([shard[key] for shard in shards], dim=0)
               for key in shards[0]}
    if "sample_index" not in tensors:
        return tensors
    order = tensors["sample_index"].argsort()
    tensors = {key: value[order] for key, value in tensors.items()}
    sample_index = tensors["sample_index"]
    keep = torch.ones_like(sample_index, dtype=torch.bool)
    keep[1:] = sample_index[1:] != sample_index[:-1]
    return {key: value[keep] for key, value in tensors.items()}


class SelectiveMemoryModule(lit.LightningModule):
    """Lightning wrapper with query loss and routing-analysis artifacts."""

    def __init__(self, input_size: int, hidden_size: int, value_size: int,
                 variant: str, read_sparsity: float = 0.5,
                 write_sparsity: float = 0.5, router_context_size: int = 8,
                 router_backward: str = "ste", optimizer_type: str = "adamw",
                 optimizer_kwargs: Optional[dict] = None,
                 scheduler_config: Optional[LRSchedulerConfig] = None,
                 analysis_path: Optional[str] = None,
                 implementation: str = "reference"):
        super().__init__()
        self.save_hyperparameters()
        self.model = SelectiveMemoryRNN(
            input_size=input_size, hidden_size=hidden_size, value_size=value_size,
            variant=variant, read_sparsity=read_sparsity,
            write_sparsity=write_sparsity, router_context_size=router_context_size,
            router_backward=router_backward, implementation=implementation)
        self._test_traces = []

    def forward(self, x: torch.Tensor, return_routing: bool = False):
        return self.model(x, return_routing=return_routing)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> STEP_OUTPUT:
        predictions, trace = self(batch["input"], return_routing=True)
        loss = masked_query_mse(predictions, batch["target"], batch["query_mask"])
        metrics = {f"{stage}/query_mse": loss}
        diagnostics = query_loss_statistics(
            predictions, batch["target"], batch["operations"])
        if stage != "train":
            diagnostics.update(memory_alignment_statistics(
                trace, batch["read_register"], batch["write_register"],
                batch["operations"]))
        metrics.update({f"{stage}/{key}": value
                        for key, value in routing_statistics(trace, batch["operations"]).items()})
        metrics.update({f"{stage}/{key}": value
                        for key, value in diagnostics.items()})
        self.log_dict(metrics, on_step=stage == "train", on_epoch=True,
                      prog_bar=stage != "train", sync_dist=True,
                      batch_size=batch["input"].size(0))

        if stage == "test" and self.hparams.analysis_path:
            per_timestep = ((predictions - batch["target"]).square().mean(dim=-1)
                            * batch["query_mask"])
            record = {key: value.detach().cpu() for key, value in batch.items()}
            record.update({key: value.detach().cpu() for key, value in trace.items()})
            record["predictions"] = predictions.detach().cpu()
            record["task_loss"] = per_timestep.detach().cpu()
            self._test_traces.append(record)
        return {"loss": loss}

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self._shared_step(batch, "test")

    def on_test_epoch_start(self) -> None:
        self._test_traces.clear()

    def on_test_epoch_end(self) -> None:
        if not self.hparams.analysis_path or not self._test_traces:
            return
        local = {key: torch.cat([batch[key] for batch in self._test_traces], dim=0)
                 for key in self._test_traces[0]}
        shards = [local]
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            gathered = [None] * dist.get_world_size() if rank == 0 else None
            dist.gather_object(local, gathered, dst=0)
            if rank != 0:
                self._test_traces.clear()
                return
            shards = gathered
        tensors = _merge_trace_shards(shards)
        summary_tensors = routing_statistics(tensors, tensors["operations"])
        summary_tensors.update(query_loss_statistics(
            tensors["predictions"], tensors["target"], tensors["operations"]))
        summary_tensors.update(memory_alignment_statistics(
            tensors, tensors["read_register"], tensors["write_register"],
            tensors["operations"]))
        summary = {key: float(value) for key, value in summary_tensors.items()}
        artifact = {
            "variant": self.hparams.variant,
            "implementation": self.hparams.implementation,
            "readout_source": "full_post_write_hidden_state",
            "input_layout": {
                "operation_flags": [0, 3],
                "write_address": [3, 3 + (self.hparams.input_size - 3
                                           - self.hparams.value_size) // 2],
                "read_address": [3 + (self.hparams.input_size - 3
                                       - self.hparams.value_size) // 2,
                                 self.hparams.input_size - self.hparams.value_size],
                "value": [self.hparams.input_size - self.hparams.value_size,
                          self.hparams.input_size],
            },
            "state_labels": {
                "00": "dormant/preserved", "10": "read-only/protected",
                "01": "write-only/previous value ignored", "11": "read and write",
            },
            "operation_labels": OPERATION_NAMES,
            "summary": summary,
            **tensors,
        }
        path = Path(self.hparams.analysis_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, path)
        self._test_traces.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = build_optimizer(
            self.named_parameters(), optimizer_type=self.hparams.optimizer_type,
            optimizer_kwargs=self.hparams.optimizer_kwargs)
        if self.hparams.scheduler_config:
            scheduler = build_lr_scheduler(
                optimizer, self.hparams.scheduler_config,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.trainer.estimated_stepping_batches)
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
