"""Benchmark: FLOPs and wall-clock runtime of DS-LSTM vs dense LSTM.

Reproduces the data behind the paper's efficiency figure (Fig. 3):
  (a) FLOPs and runtime as a function of hidden size (fixed sequence length)
  (b) FLOPs and runtime as a function of sequence length (fixed hidden size)

All measurements use batch size 64, input size 32, sparsity 0.9, router width
R=4, and time the full forward pass (router + top-k + gather). The dense
baseline is an eager LSTM loop (``NaiveLSTM``), which is the apples-to-apples
reference: both it and ``DSLSTM(implementation="gathered")`` are per-timestep
loops, so the difference is exactly the routing + gathering overhead the
figure reports.

The measurement harness (profiler FLOPs + steady-state runtime) is adapted
from compare_implementations.py; the old DSLSTMGathered reference is dropped.

Example (both sweeps, write CSV + figure):

    python experiments/benchmark.py --sweep both --out _benchmarks/dslstm.csv \\
        --plot _benchmarks/dslstm.png
"""

import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsrnn.models.dslstm_model import DSLSTM
from dsrnn.nn.router import RouterConfig


# ---------------------------------------------------------------------------
# Dense eager LSTM reference (standard LSTM equations, per-timestep loop).
# ---------------------------------------------------------------------------

class NaiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        if hx is None:
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
        else:
            h, c = hx
        w_ih = self.weight_ih.view(4, self.hidden_size, self.input_size)
        w_hh = self.weight_hh.view(4, self.hidden_size, self.hidden_size)
        gates = []
        for i in range(4):
            i2h = F.linear(x, w_ih[i], self.bias_ih[i * self.hidden_size:(i + 1) * self.hidden_size])
            h2h = F.linear(h, w_hh[i], self.bias_hh[i * self.hidden_size:(i + 1) * self.hidden_size])
            gates.append(i2h + h2h)
        i, f, g, o = torch.chunk(torch.stack(gates, dim=-1), 4, dim=-1)
        i, f, g, o = i.squeeze(-1), f.squeeze(-1), g.squeeze(-1), o.squeeze(-1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)


class NaiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.cell = NaiveLSTMCell(input_size, hidden_size)
        self.batch_first = batch_first
        self.hidden_size = hidden_size

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch, _ = x.shape
        if hx is None:
            h = x.new_zeros(batch, self.hidden_size)
            c = x.new_zeros(batch, self.hidden_size)
        else:
            h, c = hx
        outputs = []
        for t in range(seq_len):
            h, (h, c) = self.cell(x[t], (h, c))
            outputs.append(h)
        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, (h, c)


# ---------------------------------------------------------------------------
# Measurement harness (from compare_implementations.py).
# ---------------------------------------------------------------------------

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def runtime(model, x, n_warmup, n_iter, device):
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        synchronize(device)
        return (time.perf_counter() - start) / n_iter


def flops(model, x):
    with torch.no_grad():
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                with_flops=True) as prof:
            model(x)
        return int(sum(event.flops for event in prof.key_averages()))


# ---------------------------------------------------------------------------
# Model construction + sweep.
# ---------------------------------------------------------------------------

def build_models(D: int, H: int, sparsity: float, router_dim: int,
                 device: torch.device) -> Dict[str, nn.Module]:
    return {
        "dense_lstm": NaiveLSTM(D, H).to(device).eval(),
        "ds_lstm": DSLSTM(
            input_size=D, hidden_size=H,
            router_config=RouterConfig(num_channels=H, input_dim=D,
                                       context_dim=router_dim, sparsity=sparsity,
                                       backward="selected_only"),
            implementation="gathered",
        ).to(device).eval(),
    }


def measure(D: int, H: int, T: int, B: int, sparsity: float, router_dim: int,
            device: torch.device, n_warmup: int, n_iter: int,
            compile_model: bool) -> List[Dict]:
    x = torch.randn(B, T, D, device=device)
    rows = []
    for name, model in build_models(D, H, sparsity, router_dim, device).items():
        m = torch.compile(model) if compile_model else model
        f = flops(m, x)
        t = runtime(m, x, n_warmup, n_iter, device)
        rows.append({"model": name, "flops": f, "runtime_ms": t * 1e3})
    return rows


def _plot(results: List[Dict], sweep: str, out_path: str) -> None:
    """Dual-axis figure: FLOPs (bars) + runtime (lines) vs the sweep parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    params = sorted({r["param"] for r in results})
    models = ["dense_lstm", "ds_lstm"]
    x = np.arange(len(params))
    width = 0.35

    fig, ax_flops = plt.subplots(figsize=(8, 5))
    ax_rt = ax_flops.twinx()

    for i, model in enumerate(models):
        fl = [next(r["flops"] for r in results if r["param"] == p and r["model"] == model)
              for p in params]
        rt = [next(r["runtime_ms"] for r in results if r["param"] == p and r["model"] == model)
              for p in params]
        ax_flops.bar(x + (i - 0.5) * width, fl, width, label=f"{model} FLOPs", alpha=0.6)
        ax_rt.plot(x, rt, marker="o", label=f"{model} runtime")

    ax_flops.set_xticks(x)
    ax_flops.set_xticklabels([str(p) for p in params])
    ax_flops.set_xlabel("Hidden size" if sweep == "hidden" else "Sequence length")
    ax_flops.set_ylabel("FLOPs")
    ax_rt.set_ylabel("Runtime (ms)")
    ax_flops.set_yscale("log")
    ax_flops.set_title("DS-LSTM vs dense LSTM: FLOPs (bars) and runtime (lines)")
    lines1, labels1 = ax_flops.get_legend_handles_labels()
    lines2, labels2 = ax_rt.get_legend_handles_labels()
    ax_flops.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="both",
                        choices=["hidden", "seq", "both"],
                        help="hidden: vary hidden size (fixed T=500); "
                             "seq: vary sequence length (fixed H=128); both.")
    parser.add_argument("--hidden_sizes", type=int, nargs="+",
                        default=[64, 128, 256, 512])
    parser.add_argument("--seq_lens", type=int, nargs="+",
                        default=[128, 256, 500, 1000])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--router_dim", type=int, default=4)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Also wrap models in torch.compile.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None, help="CSV output path.")
    parser.add_argument("--plot", type=str, default=None,
                        help="Optional figure output path (requires matplotlib).")
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="Single tiny config to validate the path quickly.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    device = torch.device(args.device)
    print(f"device={device} B={args.batch_size} D={args.input_size} "
          f"sparsity={args.sparsity} R={args.router_dim}")

    if args.smoke:
        rows = measure(D=8, H=16, T=32, B=4, sparsity=args.sparsity,
                       router_dim=args.router_dim, device=device,
                       n_warmup=1, n_iter=2, compile_model=False)
        for r in rows:
            print(f"[smoke] {r['model']}: flops={r['flops']:.3e} "
                  f"runtime={r['runtime_ms']:.3f}ms")
        print("smoke ok")
        return

    sweeps = []
    if args.sweep in ("hidden", "both"):
        sweeps.append(("hidden", args.hidden_sizes))  # vary H at fixed T=500
    if args.sweep in ("seq", "both"):
        sweeps.append(("seq", args.seq_lens))          # vary T at fixed H=128

    all_rows: List[Dict] = []
    for sweep, values in sweeps:
        for v in values:
            H = v if sweep == "hidden" else 128
            T = 500 if sweep == "hidden" else v
            rows = measure(D=args.input_size, H=H, T=T, B=args.batch_size,
                           sparsity=args.sparsity, router_dim=args.router_dim,
                           device=device, n_warmup=args.n_warmup,
                           n_iter=args.n_iter, compile_model=args.compile)
            dense = next(r for r in rows if r["model"] == "dense_lstm")
            ds = next(r for r in rows if r["model"] == "ds_lstm")
            flops_red = 100.0 * (1 - ds["flops"] / dense["flops"])
            rt_red = 100.0 * (1 - ds["runtime_ms"] / dense["runtime_ms"])
            print(f"[{sweep}={v}] dense: {dense['flops']:.3e} FLOPs, "
                  f"{dense['runtime_ms']:.3f}ms | ds: {ds['flops']:.3e} FLOPs, "
                  f"{ds['runtime_ms']:.3f}ms | FLOPs -{flops_red:.1f}%  "
                  f"runtime -{rt_red:.1f}% ({dense['runtime_ms']/ds['runtime_ms']:.2f}x)")
            for r in rows:
                r.update({"sweep": sweep, "param": v, "batch_size": args.batch_size,
                          "input_size": args.input_size, "sparsity": args.sparsity})
            all_rows.extend(rows)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {args.out}")

    if args.plot:
        for sweep, _ in sweeps:
            sub = [r for r in all_rows if r["sweep"] == sweep]
            if sub:
                base, ext = os.path.splitext(args.plot)
                _plot(sub, sweep, f"{base}_{sweep}{ext or '.png'}")
                print(f"Wrote figure for sweep '{sweep}'")


if __name__ == "__main__":
    main()
