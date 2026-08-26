"""Continual learning experiments (Avalanche-managed training loops).

Ported from the old ``avalanche_training.py`` to the new ``dsrnn`` API.
Argparse-first: every setting is a CLI flag. An optional ``--config`` YAML
(flat ``key: value`` pairs matching the flag names) pre-loads defaults; any
explicitly-passed CLI flag overrides the YAML.

Example (SplitMNIST, DS-LSTM, GEM):

    python experiments/continual_learning.py \\
        --benchmark splitmnist --model_type dslstm --hidden_size 128 \\
        --sparsity 0.9 --strategy gem --epochs 2
"""

import argparse
import math
import os
import sys

import lightning as lit
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import SGD, AdamW

from dsrnn.nn.router import RouterConfig
from dsrnn.training.continual import build_cl_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Config file (optional; CLI flags override it)
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML with flat key: value defaults.")

    # Logging
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="DS-RNN-CL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--plot_save_dir", type=str, default="_plots/cl_plots")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to save the model checkpoint")

    # Model
    parser.add_argument("--model_type", type=str, default="dslstm",
                        choices=["lstm", "gru", "dslstm", "dsgru"])
    parser.add_argument("--input_size", type=int, default=28)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--output_size", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--implementation", type=str, default="gathered",
                        choices=["gathered", "dense"])
    # DS router (only used for ds* model types)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--context_dim", type=int, default=4)
    parser.add_argument("--router_backward", type=str, default="selected_only",
                        choices=["selected_only", "ste"])

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs per experience to use.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)

    # Strategy
    parser.add_argument("--strategy", type=str, default=["gem"], nargs="+",
                        choices=["none", "gem", "ewc", "replay", "reservoir"])
    parser.add_argument("--gem_patterns", type=int, default=256)
    parser.add_argument("--gem_strength", type=float, default=0.5)
    parser.add_argument("--ewc_lambda", type=float, default=0.4)
    parser.add_argument("--ewc_mode", type=str, default="separate",
                        choices=["separate", "online", "onlineweightedsum"])
    parser.add_argument("--ewc_decay_factor", type=float, default=None)
    parser.add_argument("--ewc_keep_importance_data", action="store_true",
                        default=False)
    parser.add_argument("--replay_mem_size", type=int, default=1000)
    parser.add_argument("--use_task_embedding", action="store_true", default=False)

    # Benchmark
    parser.add_argument("--n_experiences", type=int, default=5)
    parser.add_argument("--benchmark", type=str, default="splitmnist",
                        choices=["splitmnist", "permmnist", "splitfmnist",
                                 "splitcifar10", "splitcifar10gray", "rotmnist",
                                 "splithar", "splitsyntcon", "splitdigits"])

    args = parser.parse_args()

    # Optional YAML defaults; any explicitly-set CLI flag wins.
    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        explicitly_set = {t[2:].split("=")[0] for t in sys.argv[1:]
                          if t.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    return args


def make_benchmark(args):
    """Construct the Avalanche benchmark and the input-shaping settings.

    May adjust ``args.input_size`` for the CIFAR benchmarks, matching the
    original script.
    """
    from avalanche.benchmarks import nc_benchmark
    from avalanche.benchmarks.classic import (
        PermutedMNIST, RotatedMNIST, SplitCIFAR10, SplitFMNIST, SplitMNIST)
    from torch.utils.data import TensorDataset
    from torchvision import transforms

    patch_dim, patch_and_project, project, in_channels = None, False, False, 1

    if args.benchmark == "splitmnist":
        benchmark = SplitMNIST(n_experiences=args.n_experiences, seed=args.seed)
    elif args.benchmark == "permmnist":
        benchmark = PermutedMNIST(n_experiences=args.n_experiences, seed=args.seed)
    elif args.benchmark == "splitfmnist":
        benchmark = SplitFMNIST(n_experiences=args.n_experiences, seed=args.seed)
    elif args.benchmark == "rotmnist":
        benchmark = RotatedMNIST(n_experiences=args.n_experiences, seed=args.seed)
    elif args.benchmark == "splitcifar10":
        patch_and_project, patch_dim, in_channels = True, 16, 3
        # patch-and-project maps each patch to the model width
        args.input_size = args.hidden_size
        benchmark = SplitCIFAR10(n_experiences=args.n_experiences, seed=args.seed)
    elif args.benchmark == "splitcifar10gray":
        mean_gs_cifar10 = 0.48086121678352356
        std_gs_cifar10 = 0.23919078707695007
        args.input_size = 32
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((mean_gs_cifar10,), (std_gs_cifar10,)),
        ])
        eval_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((mean_gs_cifar10,), (std_gs_cifar10,)),
        ])
        benchmark = SplitCIFAR10(n_experiences=args.n_experiences, seed=args.seed,
                                 train_transform=train_transform,
                                 eval_transform=eval_transform)
    elif args.benchmark == "splithar":
        from dsrnn.data.classification import UCIHAR
        in_channels, project = 9, True
        train_set = UCIHAR(root="_data/UCI_HAR_Dataset", split="train")
        train_set = TensorDataset(train_set.x, train_set.y)
        test_set = UCIHAR(root="_data/UCI_HAR_Dataset", split="test")
        test_set = TensorDataset(test_set.x, test_set.y)
        benchmark = nc_benchmark(train_dataset=train_set, test_dataset=test_set,
                                 n_experiences=3, seed=args.seed, task_labels=False)
    elif args.benchmark == "splitsyntcon":
        from dsrnn.data.classification import SyntheticControl
        in_channels, project = 1, True
        train_set = SyntheticControl(root="_data/SyntheticControl", train=True)
        train_set = TensorDataset(train_set.x.transpose(-1, -2), train_set.y)
        test_set = SyntheticControl(root="_data/SyntheticControl", train=False)
        test_set = TensorDataset(test_set.x.transpose(-1, -2), test_set.y)
        benchmark = nc_benchmark(train_dataset=train_set, test_dataset=test_set,
                                 n_experiences=3, seed=args.seed, task_labels=False)
    elif args.benchmark == "splitdigits":
        from dsrnn.data.classification import PenDigits
        in_channels, project = 2, True
        test_frac = 0.2
        train_set = PenDigits(train=True, normalize=True, test_frac=test_frac)
        train_set = TensorDataset(train_set.x, train_set.y)
        test_set = PenDigits(train=False, normalize=True, test_frac=test_frac)
        test_set = TensorDataset(test_set.x, test_set.y)
        benchmark = nc_benchmark(train_dataset=train_set, test_dataset=test_set,
                                 n_experiences=5, seed=args.seed, task_labels=False)
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}.")

    return benchmark, patch_and_project, patch_dim, project, in_channels


def make_strategy(args, model, optimizer, criterion, eval_plugin, device):
    from avalanche.training.plugins import EWCPlugin, GEMPlugin, ReplayPlugin
    from avalanche.training.storage_policy import ReservoirSamplingBuffer
    from avalanche.training.supervised import EWC, GEM, Naive, Replay

    common = dict(model=model, optimizer=optimizer, criterion=criterion,
                  train_mb_size=args.batch_size, train_epochs=args.epochs,
                  eval_mb_size=args.batch_size, device=device,
                  evaluator=eval_plugin, eval_every=-1)

    if len(args.strategy) == 1:
        name = args.strategy[0]
        if name == "none":
            return Naive(**common)
        if name == "gem":
            return GEM(patterns_per_exp=args.gem_patterns,
                       memory_strength=args.gem_strength, **common)
        if name == "ewc":
            return EWC(ewc_lambda=args.ewc_lambda, mode=args.ewc_mode,
                       decay_factor=args.ewc_decay_factor,
                       keep_importance_data=args.ewc_keep_importance_data, **common)
        if name == "replay":
            return Replay(mem_size=args.replay_mem_size, **common)
        if name == "reservoir":
            storage = ReservoirSamplingBuffer(max_size=args.replay_mem_size)
            plugin = ReplayPlugin(mem_size=args.replay_mem_size, storage_policy=storage)
            return Naive(plugins=[plugin], **common)
        raise ValueError(f"Unsupported strategy: {name}.")

    if len(args.strategy) > 1:
        plugins = []
        if "replay" in args.strategy:
            if "reservoir" in args.strategy:
                raise ValueError(
                    "Invalid strategy mixing: cannot use reservoir with replay.")
            plugins.append(ReplayPlugin(mem_size=args.replay_mem_size))
        elif "reservoir" in args.strategy:
            storage = ReservoirSamplingBuffer(max_size=args.replay_mem_size)
            plugins.append(ReplayPlugin(mem_size=args.replay_mem_size,
                                        storage_policy=storage))
        if "gem" in args.strategy:
            plugins.append(GEMPlugin(patterns_per_experience=args.gem_patterns,
                                     memory_strength=args.gem_strength))
        if "ewc" in args.strategy:
            plugins.append(EWCPlugin(
                ewc_lambda=args.ewc_lambda, mode=args.ewc_mode,
                decay_factor=args.ewc_decay_factor,
                keep_importance_data=args.ewc_keep_importance_data))
        return Naive(plugins=plugins, **common)

    raise ValueError(f"Please provide a valid strategy/strategy list, {args.strategy} given.")


def _metric_series(results_list, substrs):
    """Extract a stream metric series from Avalanche result dicts.

    Picks the key containing any of ``substrs`` (case-insensitive), preferring
    keys present in every dict and containing 'stream'.
    """
    def matching(d):
        return [k for k in d.keys()
                if any(s in k.lower() for s in substrs)]

    per_dict = [matching(d) for d in results_list]
    common = set(per_dict[0])
    for keys in per_dict[1:]:
        common &= set(keys)

    stream_key = None
    if common:
        for k in common:
            if "stream" in k.lower():
                stream_key = k
                break
        if stream_key is None:
            stream_key = sorted(common)[0]
    else:
        all_keys = []
        for keys in per_dict:
            for k in keys:
                if k not in all_keys:
                    all_keys.append(k)
        for k in all_keys:
            if "stream" in k.lower():
                stream_key = k
                break
        if stream_key is None:
            if not per_dict[0]:
                raise ValueError("No matching keys found in the first result dict.")
            stream_key = per_dict[0][0]

    values = []
    for d in results_list:
        val = d.get(stream_key, math.nan)
        try:
            val = float(val)
        except Exception:
            val = math.nan
        values.append(val)

    per_exp_keys = []
    seen = set()
    for d in results_list:
        for k in d.keys():
            if any(s in k.lower() for s in substrs) and k != stream_key and k not in seen:
                per_exp_keys.append(k)
                seen.add(k)

    T, E = len(results_list), len(per_exp_keys)
    matrix = np.full((T, E), np.nan, dtype=float)
    for t, d in enumerate(results_list):
        for col, key in enumerate(per_exp_keys):
            if key in d:
                try:
                    matrix[t, col] = float(d[key])
                except Exception:
                    matrix[t, col] = np.nan

    return stream_key, values, per_exp_keys, matrix


def _plot_metric_pair(results_list, substrs, ylabel, title_prefix,
                      model_type, strategy_label, save_folder):
    """Plot stream + per-experience series for the metric matching ``substrs``."""
    try:
        _, values, per_exp_keys, matrix = _metric_series(results_list, substrs)
    except ValueError as e:
        print(f"No {ylabel} metric found in results ({e}).")
        return

    T = len(results_list)
    x = np.arange(1, T + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x, values, marker="o", linestyle="-")
    ax1.set_xlabel("Experience (stream step)")
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"Stream {title_prefix}\n({model_type} + {strategy_label})")
    ax1.grid(True)
    ax1.set_xticks(x)
    ax1.set_xlim(1 - 0.5, T + 0.5)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    labels = []
    for e_idx, key in enumerate(per_exp_keys):
        label = f"Exp {e_idx} - {key}"
        labels.append(label)
        ax2.plot(x, matrix[:, e_idx], marker="o", linestyle="-", label=label)
    ax2.set_xlabel("Experience (stream step)")
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"Per-Experience {title_prefix} Over Time\n({model_type} + {strategy_label})")
    ax2.grid(True)
    ax2.set_xticks(x)
    ax2.set_xlim(1 - 0.5, T + 0.5)
    if labels:
        ax2.legend(loc="best", fontsize="small")
    plt.tight_layout()

    if save_folder is not None:
        strategy_str = "_".join(strategy_label.split("+ "))
        stream_path = os.path.join(
            save_folder, f"{strategy_str}_{model_type}_stream_{ylabel.lower()}.png")
        perexp_path = os.path.join(
            save_folder, f"{strategy_str}_{model_type}_per_task_{ylabel.lower()}.png")
        fig1.savefig(stream_path, dpi=200)
        fig2.savefig(perexp_path, dpi=200)
        print(f"Saved Stream {title_prefix} plot to: {stream_path}")
        print(f"Saved Per-Experience {title_prefix} plot to: {perexp_path}")
        plt.close(fig1)
        plt.close(fig2)


def plot_results(results_list, args, save_folder="_plots/cl_plots"):
    """Plot stream accuracy/loss, per-task accuracy, forgetting, BWT, FWT."""
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        print(f"Plots will be saved in: {save_folder}")

    experiences = range(1, len(results_list) + 1)
    strategy_label = args.strategy[0] if len(args.strategy) == 1 else "+ ".join(args.strategy)
    model_type = args.model_type

    # --- 1. Stream accuracy ---
    acc_stream_key = [k for k in results_list[0].keys() if "Top1_Acc_Stream" in k][0]
    accuracies = [r[acc_stream_key] for r in results_list]
    plt.figure(figsize=(10, 6))
    plt.plot(experiences, accuracies, marker="o", linewidth=2, color="tab:blue",
             label="Stream Accuracy")
    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy")
    plt.title(f"Avalanche Results - Stream Accuracy\n({model_type} + {strategy_label})")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if save_folder:
        filename = f"{strategy_label}_{model_type}_stream_acc.png"
        plt.savefig(os.path.join(save_folder, filename), dpi=300)
        print(f"Saved: {filename}")
    else:
        plt.show()
    plt.close()

    # --- 2. Stream loss ---
    loss_stream_key = [k for k in results_list[0].keys() if "Loss_Stream" in k]
    if loss_stream_key:
        losses = [r[loss_stream_key[0]] for r in results_list]
        plt.figure(figsize=(10, 6))
        plt.plot(experiences, losses, marker="x", linewidth=2, color="tab:red",
                 label="Stream Loss")
        plt.xlabel("Training Experience")
        plt.ylabel("Loss")
        plt.title(f"Avalanche Results - Stream Loss\n({model_type} + {strategy_label})")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        if save_folder:
            filename = f"{'_'.join(args.strategy)}_{model_type}_stream_loss.png"
            plt.savefig(os.path.join(save_folder, filename), dpi=300)
            print(f"Saved: {filename}")
        else:
            plt.show()
        plt.close()

    # --- 3. Per-task accuracy (forgetting analysis) ---
    plt.figure(figsize=(12, 7))
    last_res = results_list[-1]
    exp_keys = sorted(k for k in last_res.keys() if "Top1_Acc_Exp" in k)
    for i, k in enumerate(exp_keys):
        task_acc = [r.get(k, None) for r in results_list]
        plt.plot(experiences, task_acc, marker=".", label=f"Task {i}")
    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy")
    plt.title(f"Per-Task Accuracy (Forgetting Analysis)\n({model_type} + {strategy_label})")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    if save_folder:
        filename = f"{'_'.join(args.strategy)}_{model_type}_per_task_acc.png"
        plt.savefig(os.path.join(save_folder, filename), dpi=300)
        print(f"Saved: {filename}")
    else:
        plt.show()
    plt.close()

    # --- 4/5/6. Forgetting, BWT, FWT (stream + per-experience) ---
    _plot_metric_pair(results_list, ["forget"], "Forgetting", "Forgetting",
                      model_type, strategy_label, save_folder)
    _plot_metric_pair(results_list, ["bwt"], "BWT", "BWT",
                      model_type, strategy_label, save_folder)
    _plot_metric_pair(results_list, ["forward", "fwt"], "FWT", "Forward Transfer (FWT)",
                      model_type, strategy_label, save_folder)


def main() -> None:
    args = parse_args()

    lit.seed_everything(seed=args.seed)
    device = torch.device(args.device)

    benchmark, patch_and_project, patch_dim, project, in_channels = make_benchmark(args)

    router_config = None
    if args.model_type in ("dslstm", "dsgru"):
        router_config = RouterConfig(
            num_channels=args.hidden_size,
            input_dim=args.input_size,
            context_dim=args.context_dim,
            sparsity=args.sparsity,
            backward=args.router_backward,
        )

    model = build_cl_model(
        args.model_type,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
        router_config=router_config,
        implementation=args.implementation,
        use_task_embedding=args.use_task_embedding,
        n_tasks=args.n_experiences,
        patch_and_project=patch_and_project,
        project=project,
        patch_dim=16 if patch_dim is None else patch_dim,
        in_channels=in_channels,
    ).to(device)

    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}.")
    criterion = nn.CrossEntropyLoss()

    # Metrics/Logger
    from avalanche.evaluation.metrics import (
        accuracy_metrics, bwt_metrics, forgetting_metrics, loss_metrics)
    from avalanche.logging import InteractiveLogger
    from avalanche.training.plugins import EvaluationPlugin

    loggers: list = [InteractiveLogger()]
    if args.use_wandb:
        from avalanche.logging import WandBLogger
        sparsity_string = f"-sp{args.sparsity}" if args.model_type in ("dslstm", "dsgru") else ""
        model_str = f"{args.model_type}({args.hidden_size}){sparsity_string}"
        strategy_str = "-".join(args.strategy)
        task_emb_str = "-task_emb" if args.use_task_embedding else ""
        run_name = f"{model_str}-{strategy_str}-{args.benchmark}{task_emb_str}({args.n_experiences})"
        loggers.append(WandBLogger(
            project_name=args.wandb_project, run_name=run_name,
            log_artifacts=False, path="_tmp/_avalanchecheckpoints", save_code=True,
            params={"entity": args.wandb_entity, "config": vars(args)}))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=loggers,
    )

    cl_strategy = make_strategy(args, model, optimizer, criterion, eval_plugin, device)

    # Training loop
    from avalanche.checkpointing import save_checkpoint

    print("Training loop start...")
    results = []
    for experience in benchmark.train_stream:
        print(f"\n--- Start of experience: {experience.current_experience} ---")

        cl_strategy.train(experience)
        print("Training completed.")

        print("Computing accuracy on test set...")
        res = cl_strategy.eval(benchmark.test_stream)
        results.append(res)

        if args.ckpt_path:
            save_checkpoint(strategy=cl_strategy, fname=args.ckpt_path + ".avalanche.ckpt")
            torch.save(
                model.state_dict(),
                os.path.join(os.path.dirname(args.ckpt_path),
                             os.path.basename(args.ckpt_path) + "_model.ckpt"),
            )

    if len(results) > 0:
        plot_results(results, args, save_folder=args.plot_save_dir)


if __name__ == "__main__":
    main()
