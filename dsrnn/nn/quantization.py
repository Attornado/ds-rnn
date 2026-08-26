"""Fixed-point quantization helpers for the DeltaRNN cells."""

import torch


class GradPreserveFloor(torch.autograd.Function):
    """Floor with a straight-through (identity) gradient."""

    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GradPreserveRound(torch.autograd.Function):
    """Round with a straight-through (identity) gradient."""

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_tensor(x: torch.Tensor, qi: int, qf: int, q_type: int = 0) -> torch.Tensor:
    """Quantize a tensor to fixed-point precision.

    Args:
        x: Input tensor.
        qi: Number of integer bits before the binary point.
        qf: Number of fraction bits after the binary point.
        q_type: 0 = none (identity), 1 = round, 2 = floor.
    """
    if q_type == 0:
        return x
    power = float(2.0 ** qf)
    clip_val = float(2.0 ** (qi + qf - 1))
    if q_type == 2:
        value = GradPreserveFloor.apply(x * power)
    else:
        value = GradPreserveRound.apply(x * power)
    value = torch.clamp(value, -clip_val, clip_val - 1)
    return value / power
