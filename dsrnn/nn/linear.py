import math
from typing import Optional

import torch
from opt_einsum import contract

from dsrnn.nn.activations import get_initializer, resolve_activation2


def linear(x: torch.Tensor,
           weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute x @ weight.T + bias.

    Args:
        x: Input of shape (*, in_features).
        weight: Weight of shape (out_features, in_features).
        bias: Optional bias of shape (out_features,).

    Returns:
        Output of shape (*, out_features).
    """
    out = torch.einsum('b...i,oi->b...o', x, weight)
    if bias is not None:
        out = out + bias
    return out


def batched_linear(x: torch.Tensor,
                   weight: torch.Tensor,
                   bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute a linear map where each batch sample uses its own weight/bias.

    Args:
        x: Input of shape (B, *, in_features).
        weight: Weight of shape (B, *, out_features, in_features).
        bias: Optional bias of shape (out_features,) or (B, out_features).

    Returns:
        Output of shape (B, *, out_features).
    """
    out = torch.einsum('b...i,b...oi->b...o', x, weight)
    if bias is not None:
        if bias.ndim == 1:
            bias = bias.view(*([1] * (out.ndim - 1)), -1)
        elif bias.ndim == 2 and bias.shape[0] == out.shape[0]:
            bias = bias.view(bias.shape[0], *([1] * (out.ndim - 2)), bias.shape[1])
        out = out + bias
    return out


class TransposedLinear(torch.nn.Module):
    """Linear module on the second-to-last dimension. Assumes shape (B, D, L)."""

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(d_output, d_input))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            torch.nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        num_axis = len(x.shape[2:])
        y = contract('b u ..., v u -> b v ...', x, self.weight) + self.bias.view(-1, *[1] * num_axis)
        return y


def LinearActivation(d_input, d_output, bias=True,
                     zero_bias_init=False,
                     transposed=False,
                     initializer=None,
                     activation=None,
                     activate=False,
                     weight_norm=False,
                     **kwargs):
    """Return a linear module with control over axes order, init, and activation."""
    linear_cls = TransposedLinear if transposed else torch.nn.Linear
    if activation == 'glu':
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    if bias and zero_bias_init:
        torch.nn.init.zeros_(linear.bias)

    if weight_norm:
        linear = torch.nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = resolve_activation2(activation, d_output, dim=1 if transposed else -1)
        linear = torch.nn.Sequential(linear, activation)
    return linear
