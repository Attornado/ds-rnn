"""
Structured-sparsity ablation baselines (paper Sec. 4.2.3).

Three masking strategies applied on top of a dense recurrent cell, used to isolate the contribution of (i) structured
sparsity, (ii) input-dependent masking, and (iii) the learned routing mechanism:

- ``StructuredSNIPMasker``: channel-level SNIP (van Amersfoort et al., 2020), a single gradient step on a real batch at
    init, mask frozen for the run.
- ``LearnedGlobalMasker``: one input-independent structured mask learned jointly with the weights (STE top-k).
- ``RandomDynamicMasker``: random per-example top-k selection (learned router replaced by chance).
"""
from typing import Any, Dict, List, Optional, Tuple
import lightning as lit
import torch
from torch import nn
from dsrnn.nn.cells.gru import GRUCell
from dsrnn.nn.cells.lstm import LSTMCell
from dsrnn.training.metrics import resolve_metric


def _topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, min(k, logits.shape[-1]))
    idx = torch.topk(logits, k=k, dim=-1).indices
    mask = torch.zeros_like(logits)
    mask.scatter_(-1, idx, 1.0)
    return mask


class BaseMasker(nn.Module):
    """Base class: produce (input_mask, hidden_mask) for one step."""

    def __init__(self, input_size: int, hidden_size: int, sparsity: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k_x = max(1, round((1.0 - sparsity) * input_size))
        self.k_h = max(1, round((1.0 - sparsity) * hidden_size))

    def get_masks(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class StructuredSNIPMasker(BaseMasker):
    """Structured SNIP: sensitivity-based static channel mask, frozen at init."""

    def __init__(self, input_size: int, hidden_size: int, sparsity: float):
        super().__init__(input_size, hidden_size, sparsity)
        self.register_buffer("mask_x", torch.ones(input_size))
        self.register_buffer("mask_h", torch.ones(hidden_size))
        self._initialized = False

    @torch.enable_grad()
    def compute_mask(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     loss_fn: nn.Module, return_sequences: bool, is_ce: bool) -> None:
        model.zero_grad(set_to_none=True)
        c_x = torch.ones(self.input_size, device=x.device, requires_grad=True)
        c_h = torch.ones(self.hidden_size, device=x.device, requires_grad=True)

        self._override_cx, self._override_ch = c_x, c_h
        y_hat = model(x)
        self._override_cx, self._override_ch = None, None

        if isinstance(loss_fn, nn.BCEWithLogitsLoss) and y.dim() < y_hat.dim() and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
            y = y.to(y_hat)
        if is_ce and return_sequences:
            y_hat = y_hat.transpose(-1, -2)

        loss = loss_fn(y_hat, y)
        loss.backward()

        sens_x = (c_x.grad * c_x).abs().detach()
        sens_h = (c_h.grad * c_h).abs().detach()
        model.zero_grad(set_to_none=True)

        mask_x = torch.zeros(self.input_size, device=x.device)
        mask_x[torch.topk(sens_x, self.k_x).indices] = 1.0
        mask_h = torch.zeros(self.hidden_size, device=x.device)
        mask_h[torch.topk(sens_h, self.k_h).indices] = 1.0

        self.mask_x.copy_(mask_x)
        self.mask_h.copy_(mask_h)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def get_masks(self, x_t, h_prev):
        b = x_t.shape[0]
        if getattr(self, "_override_cx", None) is not None:
            return self._override_cx.expand(b, -1), self._override_ch.expand(b, -1)
        return self.mask_x.expand(b, -1), self.mask_h.expand(b, -1)


class LearnedGlobalMasker(BaseMasker):
    """One structured mask learned jointly with the weights (shared over inputs)."""

    def __init__(self, input_size: int, hidden_size: int, sparsity: float):
        super().__init__(input_size, hidden_size, sparsity)
        self.logits_x = nn.Parameter(torch.zeros(input_size))
        self.logits_h = nn.Parameter(torch.zeros(hidden_size))

    def get_masks(self, x_t, h_prev):
        b = x_t.shape[0]
        soft_x = torch.sigmoid(self.logits_x)
        soft_h = torch.sigmoid(self.logits_h)
        hard_x = _topk_mask(soft_x.unsqueeze(0), self.k_x).squeeze(0)
        hard_h = _topk_mask(soft_h.unsqueeze(0), self.k_h).squeeze(0)
        m_x = hard_x + soft_x - soft_x.detach()
        m_h = hard_h + soft_h - soft_h.detach()
        return m_x.expand(b, -1), m_h.expand(b, -1)


class RandomDynamicMasker(BaseMasker):
    """Random per-example top-k selection (learned router replaced by chance)."""

    def get_masks(self, x_t, h_prev):
        b = x_t.shape[0]
        rand_x = torch.rand(b, self.input_size, device=x_t.device)
        rand_h = torch.rand(b, self.hidden_size, device=x_t.device)
        return _topk_mask(rand_x, self.k_x), _topk_mask(rand_h, self.k_h)


def build_masker(masker_type: str, input_size: int, hidden_size: int,
                 sparsity: float) -> BaseMasker:
    if masker_type == "static":
        return StructuredSNIPMasker(input_size, hidden_size, sparsity)
    if masker_type == "learned":
        return LearnedGlobalMasker(input_size, hidden_size, sparsity)
    if masker_type == "random":
        return RandomDynamicMasker(input_size, hidden_size, sparsity)
    raise ValueError(f"Unknown masker_type: {masker_type}")


class MaskedRNN(nn.Module):
    """Dense RNN whose input/hidden state are gated by a masking strategy."""

    def __init__(self, model_type: str, input_size: int, hidden_size: int,
                 sparsity: float, masker_type: str, batch_first: bool = True,
                 model_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        model_kwargs = model_kwargs or {}

        if model_type == "lstm":
            self.cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)
        elif model_type == "gru":
            self.cell = GRUCell(input_size=input_size, hidden_size=hidden_size)
        elif model_type == "ltc":
            from ncps.torch import LTCCell
            from ncps.wirings import FullyConnected
            wiring = FullyConnected(units=hidden_size, output_dim=hidden_size,
                                    self_connections=True)
            wiring.build(input_size)
            self.cell = LTCCell(wiring=wiring, in_features=input_size, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.masker = build_masker(masker_type, input_size, hidden_size, sparsity)

    def forward(self, x: torch.Tensor, h0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if self.model_type == "lstm":
            h = x.new_zeros(batch_size, self.hidden_size) if h0 is None else h0[0]
            c = x.new_zeros(batch_size, self.hidden_size) if h0 is None else h0[1]
        else:
            h = x.new_zeros(batch_size, self.hidden_size) if h0 is None else h0

        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            x_t = x[t]
            m_x, m_h = self.masker.get_masks(x_t, h)
            x_tilde = x_t * m_x
            h_tilde = h * m_h
            if self.model_type == "lstm":
                out, (h, c) = self.cell(x_tilde, (h_tilde, c))
            elif self.model_type == "gru":
                out, h = self.cell(x_tilde, h_tilde)
            else:  # ltc
                out, h = self.cell.forward(x_tilde, h_tilde)
            outputs.append(out)

        out_seq = torch.stack(outputs, dim=0)
        if self.batch_first:
            out_seq = out_seq.transpose(0, 1)
        return out_seq, h


class AblationLitModule(lit.LightningModule):
    """LightningModule wrapping a MaskedRNN + linear head for the ablations."""

    def __init__(self,
                 model_type: str,
                 masker_type: str,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 sparsity: float,
                 return_sequences: bool,
                 loss_type: str,
                 metric_params: Dict[str, Optional[Dict[str, Any]]],
                 batch_first: bool = True,
                 lr: float = 5e-3,
                 weight_decay: float = 0.0,
                 model_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.save_hyperparameters()

        self.rnn = MaskedRNN(model_type=model_type, input_size=input_size,
                             hidden_size=hidden_size, sparsity=sparsity,
                             masker_type=masker_type, batch_first=batch_first,
                             model_kwargs=model_kwargs)
        self.projection = nn.Linear(hidden_size, output_size)

        loss_type = loss_type.lower()
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type in ("ce", "crossentropy", "cross_entropy"):
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.metrics = {m: resolve_metric(m, **(metric_params[m] or {}))
                        for m in metric_params}

    def maybe_init_static_mask(self, dataloader) -> None:
        masker = self.rnn.masker
        if isinstance(masker, StructuredSNIPMasker) and not masker.initialized:
            x, y = next(iter(dataloader))
            x, y = x.to(self.device), y.to(self.device)
            masker.compute_mask(self, x, y, self.loss_fn,
                                self.hparams.return_sequences,
                                isinstance(self.loss_fn, nn.CrossEntropyLoss))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        if self.hparams.return_sequences:
            b, t, h = outputs.shape
            return self.projection(outputs.reshape(-1, h)).view(b, t, -1)
        last = outputs[:, -1] if self.hparams.batch_first else outputs[-1]
        return self.projection(last)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)

        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss) and y.dim() < y_hat.dim() and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
            y = y.to(y_hat)
        if isinstance(self.loss_fn, nn.CrossEntropyLoss) and self.hparams.return_sequences:
            y_hat = y_hat.transpose(-1, -2)

        loss = self.loss_fn(y_hat, y)
        log_dict = {f"{stage}/loss": loss.item()}
        for name, metric in self.metrics.items():
            log_dict[f"{stage}/{name}"] = metric(y_hat, y).item()
        self.log_dict(log_dict, on_step=(stage == "train"), on_epoch=True,
                      sync_dist=True, prog_bar=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, threshold=1e-4,
            min_lr=0, eps=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch",
                             "frequency": 1, "monitor": "val/loss"},
        }
