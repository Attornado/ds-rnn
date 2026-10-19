# DS-RNN: Dynamic Sparse Recurrent Neural Networks

Official implementation of *"DS-RNNs: Conditional Computation in Recurrent
Models via Input-Dependent Sparse Gating"* (published in TMLR: https://openreview.net/forum?id=8LkOGzFcVL&).

DS-RNNs equip recurrent architectures (LSTM, GRU, LTC) and state-space models (S4, Liquid-S4) with a lightweight **router** that selects, for each input, a small subnetwork of input and hidden channels that participates in the recurrent update. Because the masks are structured, the sparse recurrent matmuls become dense operations on active submatrices, which map onto standard batched GEMM kernels and yield real FLOPs and wall-clock savings on commodity hardware, with no specialized sparse primitives.

At 90% sparsity, DS-RNNs largely match (and sometimes exceed) dense performance across LSTM, GRU, LTC, and S4 backbones, and match or outperform RigL and
DeltaRNN baselines. Because different inputs activate largely disjoint subnetworks, gradient interference is reduced — which improves multi-task and continual learning.

## Highlights

- **Architecture-agnostic**: one routing mechanism applies to DS-LSTM, DS-GRU,
  DS-LTC, DS-S4, and DS-Liquid-S4.
- **Real efficiency**: structured channel masking reduces recurrent matmuls to
  dense operations on active submatrices (see `experiments/benchmark.py`).
- **Two backward rules** for the hard top-k selection: a straight-through
  estimator (`"ste"`) and selected-only differentiation (`"selected_only"`).
- **Full experiment suite**: every benchmark, baseline, and analysis from the
  paper has a script in `experiments/`.

## Installation

```bash
pip install -e .                        # core library
pip install -e ".[experiments,data]"    # benchmark training stack + datasets
pip install -e ".[cl]"                  # continual learning (Avalanche)
pip install -e ".[analysis,ww]"         # mask-analysis + WeightWatcher callbacks
pip install -e ".[wandb]"               # optional Weights & Biases logging
```

## Quickstart

```python
import torch
from dsrnn.models.dslstm_model import DSLSTM
from dsrnn.nn.router import RouterConfig

model = DSLSTM(
    input_size=28,
    hidden_size=128,
    router_config=RouterConfig(
        num_channels=128,   # hidden channels being routed
        input_dim=28,       # per-timestep input dimension
        context_dim=4,      # router context width (R in the paper)
        sparsity=0.9,       # fraction of channels deactivated
        backward="selected_only",  # or "ste"
    ),
    implementation="gathered",      # restrict matmuls to the active channels
    output_size=10,
)

x = torch.randn(64, 28, 28)         # (batch, time, features)
out, (h, c) = model(x)              # out: (batch, time, 10)
```

The cell-level API (`dsrnn.nn.cells`) exposes the same routing through a
standard `forward(x, state)` interface, so DS-RNN cells drop into existing
recurrent loops. The sequence models (`dsrnn.models`) additionally precompute
routing over the full sequence and gather the active input channels once,
amortizing the router overhead. `implementation="dense"` applies the same masks
without gathering (useful as a reference and required for the STE backward
rule in DS-LTC).

## Repository layout

```
dsrnn/
  nn/           cells, router, gathered/dense masked matmuls, SSM kernels
  models/       DSLSTM, DSGRU, DSLTC, dense LTC, generic multi-layer DSRNN
  data/         benchmark datamodules (classification, forecasting, LRA,
                multitask, synthetic, HalfCheetah)
  training/     LightningModules, pruning (RigL), ablations, optimizers,
                analysis callbacks
  utils/        logging, graph metrics, mask-similarity, visualization
experiments/    one entry point per benchmark (see below)
tests/          equivalence, gradient, and smoke tests
```

## Reproducing the paper

Every entry point is argparse-based; an optional `--config` YAML pre-loads
defaults and any explicitly-passed flag overrides it. All support
`--use_wandb` for logging (CSV/none by default) and `--seed` for
reproducibility.

| Experiment | Script | Example |
|---|---|---|
| DS vs. dense RNNs | `experiments/train_classification.py` | `--task seqmnist --model_type dslstm --sparsity 0.9` |
| Forecasting (ETT) | `experiments/train_forecasting.py` | `--dataset etth1 --model_type dslstm` |
| DS-S4 on LRA | `experiments/train_lra.py` | `--task cifar --model_type dss4 --sparsity 0.9` |
| DS vs. RigL | `experiments/train_pruning.py` | `--task seqmnist --model_type gru --sparsity 0.9` |
| DS vs. DeltaRNN | `experiments/train_deltarnn.py` | `--task seqmnist --model_type gru --thx 0.1 --thh 0.1` |
| Ablations (SNIP / learned / random) | `experiments/train_ablation_masking.py` | `--task seqmnist --masker random` |
| Multi-task | `experiments/train_multitask.py` | `--tasks seqmnist har gesture ozone occupancy` |
| Continual learning | `experiments/continual_learning.py` | `--benchmark splitmnist --strategy gem` |
| Efficiency (FLOPs / runtime) | `experiments/benchmark.py` | `--sweep both --plot fig.png` |

## Running the tests

```bash
pip install -e ".[experiments,data,analysis,ww]"
pytest tests/ -q
```

The suite covers cell/model equivalence (generic vs. gathered), gradient
correctness, the training stack, LRA, the baselines, the analysis callbacks,
and multitask. Tests that need an optional dependency skip cleanly if it is
absent.

## Citation

```bibtex
@article{dsrnn,
  title   = {DS-RNNs: Conditional Computation in Recurrent Models via
             Input-Dependent Sparse Gating},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  note    = {Under review},
}
```

## License

MIT
