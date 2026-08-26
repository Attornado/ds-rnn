"""Benchmark: FLOPs and wall-clock runtime of DS-LSTM vs a dense LSTM.

Reproduces the data behind the efficiency figure: FLOPs and runtime as a
function of hidden size (fixed sequence length) and of sequence length (fixed
hidden size), at fixed batch size, input size, and sparsity. Times the full
forward pass (routing, top-k selection, and gathering included).

The dense baseline is the repo's eager LSTM (``dsrnn.models.lstm_model.LSTM``):
both it and ``DSLSTM(implementation="gathered")`` are per-timestep loops, so
the gap isolates the routing + gathering overhead.

Example (both sweeps, write CSV + figure):

    python experiments/benchmark.py --sweep both --out _benchmarks/dslstm.csv \\
        --plot _benchmarks/dslstm.png
"""

import argparse
import csv
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn

from dsrnn.models.dslstm_model import DSLSTM
from dsrnn.models.lstm_model import LSTM
from dsrnn.nn.router import RouterConfig


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


def build_models(D: int, H: int, sparsity: float, router_dim: int,
                 device: torch.device) -> Dict[str, nn.Module]:
    return {
        "dense_lstm": LSTM(input_size=D, hidden_size=H).to(device).eval(),
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
