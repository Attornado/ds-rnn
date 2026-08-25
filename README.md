# DS-RNN: Dynamic Sparse Recurrent Neural Networks

Official implementation of the paper
*"Dynamic Sparse Recurrent Neural Networks"* (TMLR).

DS-RNNs equip recurrent architectures (LSTM, GRU, LTC) and state-space models
(S4, Liquid-S4) with a lightweight **router** that selects, at every timestep
and for every input, a small subnetwork of hidden units that participates in
the recurrent update. Computation is restricted to the selected units, yielding
input-conditioned sparsity of around 90% with dense-level performance, and —
because different inputs activate largely disjoint parameter subspaces —
reduced gradient interference in multi-task and continual learning settings.

## Highlights

- **Architecture-agnostic routing**: the same mechanism applies to DS-LSTM,
  DS-GRU, DS-LTC, DS-S4 and DS-Liquid-S4.
- **Real FLOPs savings**: a gathered implementation restricts the recurrent
  matmuls to active channels rather than multiplying dense masks.
- **Interpretable routing**: the router gradient admits a marginal-utility
  interpretation — training aligns subnetwork selection with the causal
  influence of each unit on the loss (see the paper, Sec. 3.2).
- **Downstream benefits**: input-dependent subnetworks mitigate catastrophic
  forgetting in continual learning and interference in multi-task learning.

## Installation

```bash
pip install -e .                # core library (torch + numpy)
pip install -e ".[experiments,data]"   # benchmark training stack
pip install -e ".[cl]"                 # continual learning experiments
```

## Quickstart

```python
import torch
from dsrnn.models.dslstm_model import DSLSTM
from dsrnn.nn.router import RouterConfig

router_config = RouterConfig(
    num_channels=128,   # hidden size being routed
    input_dim=28,       # per-timestep input dimension
    context_dim=4,      # router context width (R in the paper)
    sparsity=0.9,       # fraction of hidden units deactivated
    backward="ste",     # or "selected_only"
)

model = DSLSTM(
    input_size=28,
    hidden_size=128,
    router_config=router_config,
    implementation="gathered",  # restrict matmuls to active channels
    output_size=10,
)

x = torch.randn(64, 784 // 28, 28)  # (batch, time, features)
out, h = model(x)                   # out: (batch, time, 10)
```

The generic cell API (`dsrnn.nn.cells`) exposes the same routing through the
standard `forward(x, state)` interface, so DS-RNN cells can be dropped into
existing recurrent loops; the specialized sequence models (`dsrnn.models`)
additionally pre-compute routing over the full sequence and gather the active
input channels once, amortizing the router overhead.

## Repository layout

```
dsrnn/
  nn/           cells, router, gathered sparse matmuls
  models/       DSLSTM, DSGRU, DSLTC, generic DSRNN wrapper
  data/         benchmark datamodules (classification, forecasting, LRA,
                multi-task, synthetic, HalfCheetah)
  training/     LightningModules, callbacks, continual-learning wrappers
experiments/    experiment entry points
configs/        exact YAML configs reproducing the paper's tables
tests/          equivalence and gradient tests
```

## Reproducing the paper

Per-benchmark instructions and configs live in `experiments/` and `configs/`:

- **DS-RNN vs. dense RNNs** — SeqMNIST, UCI HAR, Ozone, Power, Traffic,
  Occupancy, HalfCheetah kinematics.
- **Dynamic-sparse SSMs** — LRA (Retrieval, Image, Text, ListOps) with S4 and
  Liquid-S4 backbones.
- **Sparse baselines and ablations** — RigL, DeltaRNN, static SNIP, learned
  global mask, random routing.
- **Multi-task and continual learning** — joint training on heterogeneous
  tasks, synthetic teacher-student tasks, and Avalanche-based CL benchmarks
  (SplitMNIST, PermutedMNIST, RotatedMNIST, SplitCIFAR10, SplitFMNIST).

## Citation

```bibtex
@article{dsrnn,
  title  = {Dynamic Sparse Recurrent Neural Networks},
  author = {Terlizzi, Andrea and ...},
  journal= {Transactions on Machine Learning Research},
  year   = {2026},
}
```

## License

MIT
