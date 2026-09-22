"""Routed recurrent models and target-supervised CogScale training."""
from typing import Optional, Tuple, Dict, Union
import torch
import lightning as lit
from dsrnn.nn.cells.dual_read_write import DualReadWriteRNNCell
from dsrnn.nn.cells.dual_read_write_gru import DualReadWriteGRUCell
from dsrnn.nn.cells.dual_read_write_sparse import SparseDualReadWriteRNNCell
from dsrnn.nn.router import Router, RouterConfig, RouterOutput
from dsrnn.training.optim import build_optimizer
from dsrnn.training.scheduler import LRSchedulerConfig, build_lr_scheduler


MODEL_VARIANTS = ("dense", "read_only", "write_only", "shared", "dual")
IMPLEMENTATIONS = ("reference", "sparse")
RECURRENT_CELLS = ("rnn", "gru")


class CogScaleRNN(torch.nn.Module):
    """Recurrent model configured by cell, routing variant, and implementation."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 value_size: int,
                 variant: str,
                 read_sparsity: float = 0.5,
                 write_sparsity: float = 0.5,
                 router_context_size: int = 8,
                 router_backward: str = "ste",
                 implementation: str = "reference",
                 recurrent_cell: str = "rnn"):
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {MODEL_VARIANTS}.")
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"Unknown implementation {implementation!r}; "
                             f"expected one of {IMPLEMENTATIONS}.")
        if recurrent_cell not in RECURRENT_CELLS:
            raise ValueError(f"Unknown recurrent cell {recurrent_cell!r}; "
                             f"expected one of {RECURRENT_CELLS}.")
        if recurrent_cell == "gru" and implementation != "reference":
            raise ValueError("The GRU currently has only a reference implementation.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.value_size = value_size
        self.variant = variant
        self.router_backward = router_backward
        self.implementation = implementation
        self.recurrent_cell = recurrent_cell

        # Construct shared cell/output before variant-specific routers
        if recurrent_cell == "gru":
            cell_type = DualReadWriteGRUCell
        else:
            cell_type = DualReadWriteRNNCell if implementation == "reference" else SparseDualReadWriteRNNCell
        self.cell = cell_type(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, value_size)

        # Router
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

    def forward(self,
                x: torch.Tensor,
                h0: Optional[torch.Tensor] = None,
                return_routing: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if x.ndim != 3 or x.size(-1) != self.input_size:
            raise ValueError(f"x must have shape (B, T, {self.input_size}).")
        h = x.new_zeros(x.size(0), self.hidden_size) if h0 is None else h0
        outputs = []
        traces = {name: [] for name in ("read_mask", "write_mask", "read_indices", "write_indices",
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

            cell_kwargs: dict = {"read_mask": read.mask, "write_mask": write.mask}
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
            # Keep the established baseline readout explicit: it sees the full post-write state, including coordinates
            # preserved by WRITE=0.
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


def target_loss(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Some tasks leave -100 in non-target classes even at scored timesteps.
    # Match CogScale's timestep mask and argmax, rather than using soft labels.
    mask = (targets != -100).any(-1)
    if not mask.any():
        raise ValueError("Batch contains no scored CogScale targets.")
    labels = targets[mask].argmax(-1)
    return torch.nn.functional.cross_entropy(logits[mask], labels), mask, labels


class CogScaleModule(lit.LightningModule):
    """Target-supervised classification with routed recurrent baselines."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 value_size: int,
                 variant: str,
                 read_sparsity: float = 0.5,
                 write_sparsity: float = 0.5,
                 router_context_size: int = 8,
                 router_backward: str = "ste",
                 optimizer_type: str = "adamw",
                 optimizer_kwargs: Optional[dict] = None,
                 scheduler_config: Optional[LRSchedulerConfig] = None,
                 analysis_path: Optional[str] = None,
                 implementation: str = "reference",
                 recurrent_cell: str = "rnn"):
        super().__init__()
        # Kept in the signature to load existing CogScale checkpoints.
        self.save_hyperparameters(ignore=["analysis_path"])
        self.model = CogScaleRNN(
            input_size=input_size, hidden_size=hidden_size, value_size=value_size,
            variant=variant, read_sparsity=read_sparsity,
            write_sparsity=write_sparsity, router_context_size=router_context_size,
            router_backward=router_backward, implementation=implementation,
            recurrent_cell=recurrent_cell)

    def forward(self,
                x: torch.Tensor,
                return_routing: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        return self.model(x, return_routing=return_routing)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.named_parameters(),
            optimizer_type=self.hparams.optimizer_type,
            optimizer_kwargs=self.hparams.optimizer_kwargs
        )

        if self.hparams.scheduler_config:
            scheduler = build_lr_scheduler(
                optimizer=optimizer,
                config=self.hparams.scheduler_config,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=int(self.trainer.estimated_stepping_batches)
            )
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

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        x, targets = batch
        logits = self(x)
        loss, mask, labels = target_loss(logits, targets)
        error = (logits[mask].argmax(-1) != labels).float().mean()
        count = int(mask.sum())
        self.log(f"{stage}/target_ce", loss, on_step=False, on_epoch=True,
                 batch_size=count, prog_bar=stage != "train")
        self.log(f"{stage}/error_rate", error, on_step=False, on_epoch=True,
                 batch_size=count)
        return {"loss": loss}
