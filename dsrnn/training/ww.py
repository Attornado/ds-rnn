"""WeightWatcher analysis callback."""
import os
import lightning as lit
import torch
from matplotlib import pyplot as plt
from lightning.pytorch.loggers.wandb import WandbLogger
from dsrnn.models.dslstm_model import DSLSTM
from dsrnn.models.dsgru_model import DSGRU
from dsrnn.models.gru_model import GRU
from dsrnn.models.lstm_model import LSTM
from dsrnn.models.ltc_model import LTC


def _router_linears(router) -> torch.nn.ModuleList:
    """Collect the torch.nn.Linear layers inside a Router."""
    mods = torch.nn.ModuleList()
    for _, m in router.named_modules():
        if isinstance(m, torch.nn.Linear):
            mods.append(m)
    return mods


def _cell_weight_linears(cell, n_gates: int, with_router: bool = True) -> torch.nn.ModuleDict:
    """Split a cell's fused weight_ih/weight_hh into per-gate Linear modules."""
    modules = torch.nn.ModuleDict()
    H = cell.hidden_size
    wih_chunks = cell.weight_ih.split(H, dim=0)
    whh_chunks = cell.weight_hh.split(H, dim=0)
    bih_chunks = cell.bias_ih.split(H, dim=0) if cell.bias_ih is not None else [None] * n_gates
    bhh_chunks = cell.bias_hh.split(H, dim=0) if cell.bias_hh is not None else [None] * n_gates

    lih, lhh = torch.nn.ModuleList(), torch.nn.ModuleList()
    for wih, whh, bih, bhh in zip(wih_chunks, whh_chunks, bih_chunks, bhh_chunks):
        li = torch.nn.Linear(cell.input_size, H, bias=bih is not None)
        lh = torch.nn.Linear(H, H, bias=bhh is not None)
        li.weight = torch.nn.Parameter(wih.clone())
        lh.weight = torch.nn.Parameter(whh.clone())
        if bih is not None:
            li.bias = torch.nn.Parameter(bih.clone())
        if bhh is not None:
            lh.bias = torch.nn.Parameter(bhh.clone())
        dev = cell.weight_ih.device
        li.to(dev)
        lh.to(dev)
        lih.append(li)
        lhh.append(lh)
    modules["ih"] = lih
    modules["hh"] = lhh
    if with_router:
        modules["router"] = _router_linears(cell.hidden_router)
        modules["input_router"] = _router_linears(cell.input_router)
    return modules


def extract_linear_modules(model: torch.nn.Module) -> torch.nn.ModuleDict:
    """Extract linear-equivalent modules from a backbone for WW analysis.

    Accepts a SequenceModule backbone: DSLSTM/DSGRU (single ``.cell`` with a
    router), dense LSTM/GRU (single ``.cell``, no router), or dense LTC.
    """
    modules = torch.nn.ModuleDict()

    if isinstance(model, (DSLSTM, DSGRU)):
        n_gates = 4 if isinstance(model, DSLSTM) else 3
        modules["cell"] = _cell_weight_linears(model.cell, n_gates, with_router=True)
        if isinstance(model.proj, torch.nn.Linear):
            modules["proj"] = model.proj
    elif isinstance(model, (LSTM, GRU)):
        n_gates = 4 if isinstance(model, LSTM) else 3
        modules["cell"] = _cell_weight_linears(model.cell, n_gates, with_router=False)
        if isinstance(model.proj, torch.nn.Linear):
            modules["proj"] = model.proj
    elif isinstance(model, LTC):
        cell = model.rnn  # ncps LTC
        if isinstance(model.proj, torch.nn.Linear):
            modules["proj"] = model.proj
        params = getattr(cell, "_params", {})
        if "w" in params:
            lh = torch.nn.Linear(cell.state_size, cell.state_size, bias=False)
            lh.weight = torch.nn.Parameter(params["w"].clone())
            modules["hh"] = lh
        if "sensory_w" in params:
            li = torch.nn.Linear(model.input_size, cell.sensory_size, bias=False)
            li.weight = torch.nn.Parameter(params["sensory_w"].clone())
            modules["ih"] = li
    else:
        raise ValueError(f"Unsupported model class {type(model)} for WW analysis.")

    return modules


def analyze_model_ww(model: torch.nn.Module):
    """Run WeightWatcher on a backbone's linear modules. Returns (details, summary)."""
    import weightwatcher as ww

    modules = extract_linear_modules(model)
    watcher = ww.WeightWatcher(modules)
    details = watcher.analyze(modules, detX=True, plot=False)
    summary = watcher.get_summary(details)
    return details, summary


def _get_logger_dir(trainer: lit.Trainer) -> str:
    log_dir = getattr(trainer, "log_dir", None)
    if log_dir:
        return str(log_dir)
    return str(getattr(trainer, "default_root_dir", os.getcwd()))


class WeightWatcherCallback(lit.Callback):
    """Run WeightWatcher analysis every N epochs and log summary + details."""

    def __init__(self, every_n_epochs: int = 1, log_details: bool = True,
                 log_validation: bool = False):
        super().__init__()
        self._every_n_epochs = every_n_epochs
        self._log_details = log_details
        self._log_validation = log_validation

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._log_validation:
            self._shared(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._shared(trainer, pl_module, "test")

    def _shared(self, trainer, pl_module, stage):
        if trainer.current_epoch % self._every_n_epochs != 0:
            return
        backbone = getattr(pl_module, "backbone", None) or getattr(pl_module, "model", None)
        if backbone is None:
            return
        details, summary = analyze_model_ww(backbone)

        summary = {f"ww_metrics_{stage}/{m}": summary[m] for m in summary}
        trainer.logger.log_metrics(summary, step=trainer.current_epoch)

        cols = [c for c in details.columns
                if ("alpha" in c) or ("detX" in c)
                or c in ["layer_id", "longname", "name", "log_alpha_norm",
                         "log_spectral_norm", "stable_rank", "xmin"]]
        details = details[cols]

        if isinstance(trainer.logger, WandbLogger):
            import wandb
            trainer.logger.log_table(key=f"ww_metrics_{stage}/table", dataframe=details)
            if "alpha" in details.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(details["alpha"].dropna().values, bins=30)
                ax.set_title(f"Layer alpha distribution ({stage})")
                ax.set_xlabel("alpha")
                trainer.logger.experiment.log({f"ww_plot_{stage}/alpha_hist": wandb.Image(fig)})
                plt.close(fig)
        else:
            log_dir = _get_logger_dir(trainer)
            os.makedirs(log_dir, exist_ok=True)
            details.to_csv(os.path.join(log_dir, f"ww_details_{stage}_epoch{trainer.current_epoch}.csv"),
                           index=False)
