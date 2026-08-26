"""Continual learning experiments (Avalanche-managed training loops).

Ported from the old ``avalanche_training.py`` to the new ``dsrnn`` API.
Argparse-first; the model is built from CLI flags (an optional ``--config``
YAML pre-loads defaults, overridden by any flag).

Example (SplitMNIST, DS-LSTM, GEM):

    python experiments/continual_learning.py \\
        --benchmark splitmnist --model_type dslstm --hidden_size 128 \\
        --sparsity 0.9 --strategy gem --epochs 2
"""

import argparse

import torch
import yaml
import lightning as lit
from torch import nn
from torch.optim import AdamW, SGD

from dsrnn.nn.router import RouterConfig
from dsrnn.training.continual import build_cl_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override it.")

    # Logging
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="DS-RNN-CL")
    parser.add_argument("--wandb_entity", type=str, default=None)
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
    # DS router
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--context_dim", type=int, default=4)
    parser.add_argument("--router_backward", type=str, default="selected_only",
                        choices=["selected_only", "ste"])

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs per experience.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)

    # CL strategy
    parser.add_argument("--strategy", type=str, default=["gem"], nargs="+",
                        choices=["none", "gem", "ewc", "replay", "reservoir"])
    parser.add_argument("--gem_patterns", type=int, default=256)
    parser.add_argument("--gem_strength", type=float, default=0.5)
    parser.add_argument("--ewc_lambda", type=float, default=0.4)
    parser.add_argument("--ewc_mode", type=str, default="separate",
                        choices=["separate", "online", "onlineweightedsum"])
    parser.add_argument("--ewc_decay_factor", type=float, default=None)
    parser.add_argument("--ewc_keep_importance_data", action="store_true", default=False)
    parser.add_argument("--replay_mem_size", type=int, default=1000)
    parser.add_argument("--use_task_embedding", action="store_true", default=False)

    # Benchmark
    parser.add_argument("--n_experiences", type=int, default=5)
    parser.add_argument("--benchmark", type=str, default="splitmnist",
                        choices=["splitmnist", "permmnist", "splitfmnist",
                                 "splitcifar10", "splitcifar10gray", "rotmnist",
                                 "splithar", "splitsyntcon", "splitdigits"])

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        import sys
        explicitly_set = {t[2:] for t in sys.argv[1:] if t.startswith("--")}
        for key, value in file_cfg.items():
            if hasattr(args, key) and key not in explicitly_set:
                setattr(args, key, value)

    return args


def make_benchmark(args):
    """Construct the Avalanche benchmark and the input-shaping settings."""
    from avalanche.benchmarks.classic import (
        SplitMNIST, SplitCIFAR10, PermutedMNIST, SplitFMNIST, RotatedMNIST)
    from avalanche.benchmarks import nc_benchmark
    from torch.utils.data import TensorDataset
    from torchvision import transforms

    patch_dim, patch_and_project, project, in_channels = None, False, False, 1
    input_size = args.input_size

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
        input_size = args.hidden_size  # patch-and-project maps to model width
        benchmark = SplitCIFAR10(n_experiences=args.n_experiences, seed=args.seed)
    elif args.benchmark == "splitcifar10gray":
        mean_gs, std_gs = 0.48086121678352356, 0.23919078707695007
        input_size = 32
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((mean_gs,), (std_gs,)),
        ])
        eval_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((mean_gs,), (std_gs,)),
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

    return benchmark, input_size, patch_and_project, patch_dim, project, in_channels


def make_strategy(args, model, optimizer, criterion, eval_plugin, device):
    from avalanche.training.supervised import GEM, Naive, EWC, Replay
    from avalanche.training.plugins import ReplayPlugin, EWCPlugin, GEMPlugin
    from avalanche.training.storage_policy import ReservoirSamplingBuffer

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

    # Mixed strategies -> Naive with plugins
    plugins = []
    if "replay" in args.strategy:
        if "reservoir" in args.strategy:
            raise ValueError("Cannot mix reservoir with replay strategy.")
        plugins.append(ReplayPlugin(mem_size=args.replay_mem_size))
    elif "reservoir" in args.strategy:
        storage = ReservoirSamplingBuffer(max_size=args.replay_mem_size)
        plugins.append(ReplayPlugin(mem_size=args.replay_mem_size, storage_policy=storage))
    if "gem" in args.strategy:
        plugins.append(GEMPlugin(patterns_per_experience=args.gem_patterns,
                                 memory_strength=args.gem_strength))
    if "ewc" in args.strategy:
        plugins.append(EWCPlugin(ewc_lambda=args.ewc_lambda, mode=args.ewc_mode,
                                 decay_factor=args.ewc_decay_factor,
                                 keep_importance_data=args.ewc_keep_importance_data))
    return Naive(plugins=plugins, **common)


def main() -> None:
    args = parse_args()

    lit.seed_everything(seed=args.seed)
    device = torch.device(args.device)

    benchmark, input_size, patch_and_project, patch_dim, project, in_channels = \
        make_benchmark(args)

    router_config = None
    if args.model_type in ("dslstm", "dsgru"):
        router_config = RouterConfig(
            num_channels=args.hidden_size,
            input_dim=input_size,
            context_dim=args.context_dim,
            sparsity=args.sparsity,
            backward=args.router_backward,
        )

    model = build_cl_model(
        args.model_type, input_size=input_size, hidden_size=args.hidden_size,
        output_size=args.output_size, num_layers=args.num_layers,
        router_config=router_config, implementation=args.implementation,
        use_task_embedding=args.use_task_embedding, n_tasks=args.n_experiences,
        patch_and_project=patch_and_project, project=project,
        patch_dim=16 if patch_dim is None else patch_dim, in_channels=in_channels,
    ).to(device)

    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}.")
    criterion = nn.CrossEntropyLoss()

    # Loggers / evaluator
    from avalanche.logging import InteractiveLogger
    from avalanche.evaluation.metrics import (
        accuracy_metrics, loss_metrics, forgetting_metrics, bwt_metrics)
    from avalanche.training.plugins import EvaluationPlugin

    loggers = [InteractiveLogger()]
    if args.use_wandb:
        from avalanche.logging import WandBLogger
        sparsity_str = f"-sp{args.sparsity}" if args.model_type in ("dslstm", "dsgru") else ""
        task_emb_str = "-task_emb" if args.use_task_embedding else ""
        run_name = (f"{args.model_type}({args.hidden_size}){sparsity_str}-"
                    f"{'-'.join(args.strategy)}-{args.benchmark}{task_emb_str}"
                    f"({args.n_experiences})")
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

    # Training loop over the experience stream
    results = []
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        results.append(cl_strategy.eval(benchmark.test_stream))

    if args.ckpt_path is not None:
        from avalanche.checkpointing import save_checkpoint
        save_checkpoint(model, optimizer, args.ckpt_path)
        print(f"Checkpoint saved to {args.ckpt_path}")


if __name__ == "__main__":
    main()
