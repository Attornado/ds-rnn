"""Forecasting LightningModule built on pytorch_forecasting.

Kept out of ``dsrnn.training.__init__`` so ``pytorch_forecasting`` stays an
optional dependency (install the ``data`` extra).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from lightning import LightningModule
from dsrnn.training.sequence import build_backbone

try:
    import pytorch_forecasting as pf
    from pytorch_forecasting import TimeSeriesDataSet
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "dsrnn.training.forecasting requires pytorch_forecasting; "
        "install it with `pip install ds-rnn[data]`."
    ) from e


class ForecastModule(pf.BaseModelWithCovariates):
    """pytorch_forecasting BaseModel wrapping a (DS-)RNN encoder.

    The encoder is a headless backbone from ``build_backbone``; a direct
    multi-horizon projection maps the final hidden state to
    ``prediction_length * output_size`` outputs.
    """

    def __init__(self,
                 model_type: str,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 prediction_length: int = 24,
                 batch_first: bool = True,
                 implementation: str = "gathered",
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 test_transform: Optional[Callable] = None,
                 dataset_parameters: Optional[Dict[str, Any]] = None,
                 log_interval: Union[int, float] = 1,
                 log_val_interval: Union[int, float] = 1,
                 learning_rate: Union[float, List[float]] = 1e-3,
                 optimizer: str = "adam",
                 optimizer_params: Optional[Dict[str, Any]] = None,
                 reduce_on_plateau_patience: int = 1000,
                 reduce_on_plateau_reduction: float = 2.0,
                 reduce_on_plateau_min_lr: float = 1e-5,
                 weight_decay: float = 0.0,
                 log_gradient_flow: bool = False,
                 loss: Optional[pf.metrics.Metric] = None,
                 logging_metrics: Optional[torch.nn.ModuleList] = None,
                 monotone_constraints: Optional[Dict[str, int]] = None,
                 output_transformer: Optional[Callable] = None,
                 **kwargs):
        super().__init__(
            dataset_parameters=dataset_parameters,
            log_interval=log_interval,
            log_val_interval=log_val_interval,
            learning_rate=learning_rate,
            log_gradient_flow=log_gradient_flow,
            loss=loss if loss is not None else pf.metrics.SMAPE(),
            logging_metrics=logging_metrics if logging_metrics is not None else torch.nn.ModuleList([]),
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            reduce_on_plateau_reduction=reduce_on_plateau_reduction,
            reduce_on_plateau_min_lr=reduce_on_plateau_min_lr,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            weight_decay=weight_decay,
            monotone_constraints=monotone_constraints if monotone_constraints is not None else {},
            output_transformer=output_transformer,
        )
        self.save_hyperparameters()

        self.test_transform = test_transform
        self.backbone = build_backbone(
            model_type, input_size=input_size, hidden_size=hidden_size,
            batch_first=batch_first, implementation=implementation,
            model_kwargs=model_kwargs,
        )
        h_dim = hidden_size
        self.prediction_length = prediction_length
        self._direct_projection = torch.nn.Linear(h_dim, prediction_length * output_size)

    @property
    def direct_projection(self) -> torch.nn.Module:
        return self._direct_projection

    @property
    def rnn(self) -> torch.nn.Module:
        return self.backbone

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs) -> LightningModule:
        new_kwargs = cls.deduce_default_output_parameters(
            dataset=dataset, kwargs=kwargs, default_loss=None)
        kwargs.update(new_kwargs)
        return super().from_dataset(dataset=dataset, **kwargs)

    def _extract_final_hidden(self,
                              outputs: torch.Tensor,
                              state) -> torch.Tensor:
        """Return the final hidden state of shape (B, h_dim)."""
        if isinstance(state, tuple):          # LSTM: (h_n, c_n)
            h_n = state[0]
        elif isinstance(state, torch.Tensor):  # GRU / single-state
            h_n = state
        else:
            h_n = None

        if h_n is None:                        # fallback: last output step
            return outputs[:, -1, :] if self.hparams.batch_first else outputs[-1]
        if h_n.ndim == 3:                      # (num_layers, B, H)
            return h_n[-1]
        return h_n                             # (B, H)

    def forward(self, x: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                **kwargs) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        # Continuous variables for the encoder: (B, encoder_length, n_features)
        x_inp = x["encoder_cont"]

        outputs, state = self.backbone(x_inp)
        h_final = self._extract_final_hidden(outputs, state)  # (B, h_dim)

        out = self._direct_projection(h_final)  # (B, pred * output)
        batch_size = out.shape[0]
        output_size = out.shape[-1] // self.prediction_length
        y_hat = out.view(batch_size, self.prediction_length, output_size)
        y_hat = y_hat.unbind(dim=-1)  # list of (B, pred)

        y_hat = self.transform_output(y_hat, target_scale=x["target_scale"])
        return self.to_network_output(prediction=y_hat)


class ForecastPruningModule(ForecastModule):
    """ForecastModule with pruning-mask enforcement hooks (for RigL)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks: Dict[str, Optional[torch.Tensor]] = {}

    def apply_masks(self) -> None:
        if not self.masks:
            return
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.masks:
                    param.data.mul_(self.masks[name].to(param.device))

    def on_before_zero_grad(self, optimizer) -> None:
        self.apply_masks()

    def on_after_backward(self) -> None:
        super().on_after_backward()
        self.apply_masks()
