"""
Wrappers adapting sequence models to Avalanche for continual learning.
"""
import torch
from torch import nn
from einops.layers.torch import Rearrange
from dsrnn.training.sequence import build_backbone

try:
    from avalanche.models import MultiTaskModule
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "dsrnn.training.continual requires the 'avalanche-lib' package; "
        "install it with `pip install ds-rnn[cl]`."
    ) from e


class AvalancheWrapper(nn.Module):
    """Adapt a sequence model returning ``(output, state)`` to Avalanche."""

    def __init__(self, model: nn.Module, hidden_size: int, output_size: int,
                 patch_and_project: bool = False, project: bool = False,
                 patch_dim: int = 16, in_channels: int = 1):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(hidden_size, output_size)
        self.patch_and_project = patch_and_project
        self.project = project
        self.patch_dim = patch_dim
        self.in_channels = in_channels

        self.proj_layer = None
        if project and not patch_and_project:
            self.proj_layer = nn.Linear(in_channels, model.input_size)
        self.patch_and_project_layer = None
        if patch_and_project:
            assert hasattr(model, "input_size") or hasattr(model, "input_dim"), \
                f"Model class {type(model)} has no attribute 'input_size' or 'input_dim'."
            self.patch_and_project_layer = nn.Sequential(
                Rearrange(pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                          p1=patch_dim, p2=patch_dim),
                nn.Linear(patch_dim * patch_dim * in_channels, model.input_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and not self.patch_and_project:
            x = x.squeeze(1)
        elif x.dim() == 4 and self.patch_and_project:
            x = self.patch_and_project_layer(x)
        elif x.dim() == 2:
            if x.shape[-1] % 28 == 0:
                x = x.view(-1, x.shape[-1] // 28, 28)
            elif x.shape[-1] % 32 == 0:
                x = x.view(-1, x.shape[-1] // 32, 32)
            else:
                raise ValueError(
                    f"Unsupported dimension: {x.shape[-1]} (should be divisible "
                    "by 32 or 28 when passing flattened inputs).")
        elif self.project:
            x = self.proj_layer(x)

        out, _ = self.model(x)
        last_out = out[:, -1, :]
        return self.classifier(last_out)


class MultiTaskAvalancheWrapper(MultiTaskModule):
    """Avalanche multi-task variant with a learned per-task input embedding."""

    def __init__(self, model: nn.Module, hidden_size: int, output_size: int,
                 n_tasks: int, patch_and_project: bool = False,
                 project: bool = False, patch_dim: int = 16, in_channels: int = 1):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(hidden_size, output_size)
        self.patch_and_project = patch_and_project
        self.project = project
        self.patch_dim = patch_dim
        self.in_channels = in_channels
        self.task_embedding = nn.Embedding(n_tasks, model.input_size)

        self.proj_layer = None
        if project and not patch_and_project:
            self.proj_layer = nn.Linear(in_channels, model.input_size)
        self.patch_and_project_layer = None
        if patch_and_project:
            assert hasattr(model, "input_size") or hasattr(model, "input_dim"), \
                f"Model class {type(model)} has no attribute 'input_size' or 'input_dim'."
            self.patch_and_project_layer = nn.Sequential(
                Rearrange(pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                          p1=patch_dim, p2=patch_dim),
                nn.Linear(patch_dim * patch_dim * in_channels, model.input_size),
            )

    def forward(self, x: torch.Tensor, task_label) -> torch.Tensor:
        if x.dim() == 4 and not self.patch_and_project:
            x = x.squeeze(1)
        elif x.dim() == 4 and self.patch_and_project:
            x = self.patch_and_project_layer(x)
        elif x.dim() == 2:
            if x.shape[-1] % 28 == 0:
                x = x.view(-1, x.shape[-1] // 28, 28)
            elif x.shape[-1] % 32 == 0:
                x = x.view(-1, x.shape[-1] // 32, 32)
            else:
                raise ValueError(
                    f"Unsupported dimension: {x.shape[-1]} (should be divisible "
                    "by 32 or 28 when passing flattened inputs).")
        elif self.project:
            x = self.proj_layer(x)

        if isinstance(task_label, torch.Tensor):
            task_label = task_label.to(x.device)
        task_emb = self.task_embedding(task_label)
        task_emb = task_emb.unsqueeze(1)
        x = x + task_emb

        out, _ = self.model(x)
        last_out = out[:, -1, :]
        return self.classifier(last_out)


def build_cl_model(model_type: str,
                   input_size: int,
                   hidden_size: int,
                   output_size: int,
                   *,
                   num_layers: int = 1,
                   router_config=None,
                   implementation: str = "gathered",
                   use_task_embedding: bool = False,
                   n_tasks: int = -1,
                   patch_and_project: bool = False,
                   project: bool = False,
                   patch_dim: int = 16,
                   in_channels: int = 1) -> nn.Module:
    """Build a CL-ready model (base backbone + Avalanche wrapper).

    The base model is constructed through the shared ``build_backbone``
    registry (headless, ``(out, state)`` interface), so dense and
    dynamic-sparse models share one path. For DS models, ``router_config`` is
    a RouterConfig built by the caller.
    """
    model_type = model_type.lower()
    if model_type in ("lstm", "gru"):
        model_kwargs = {"num_layers": num_layers, "dropout": 0.0}
    elif model_type in ("dslstm", "dsgru"):
        if router_config is None:
            raise ValueError(f"{model_type} requires a router_config.")
        model_kwargs = {"router_config": router_config}
    else:
        raise ValueError(f"Unsupported model_type: {model_type}.")

    base_model = build_backbone(
        model_type, input_size=input_size, hidden_size=hidden_size,
        batch_first=True, implementation=implementation,
        model_kwargs=model_kwargs,
    )

    if use_task_embedding:
        return MultiTaskAvalancheWrapper(
            base_model, hidden_size, output_size, n_tasks=n_tasks,
            patch_and_project=patch_and_project, project=project,
            patch_dim=patch_dim, in_channels=in_channels)
    return AvalancheWrapper(
        base_model, hidden_size, output_size,
        patch_and_project=patch_and_project, project=project,
        patch_dim=patch_dim, in_channels=in_channels)
