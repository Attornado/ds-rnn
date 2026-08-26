"""Fixed-point quantization helpers used by the DeltaRNN cells.

Ported from the relevant part of ``connectomencps.nn.quantization``:
``quantize_tensor`` plus its straight-through floor/round estimators. During
training (``q_type=0``) quantization is the identity; fixed-point quantization
is applied at inference. (The unrelated simhash/binary-dot utilities in the
original file are not part of the DeltaRNN path and were not ported.)
"""

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

    Returns:
        The (fixed-point) quantized tensor.
    """
    if q_type == 0:
        return x
    power = float(2.0 ** qf)
    clip_val = float(2.0 ** (qi + qf - 1))
    if q_type == 2:
        value = GradPreserveFloor.apply(x * power)
    else:
        value = GradPreserveRound.apply(x * power)
    value = torch.clamp(value, -clip_val, clip_val - 1)  # saturation arithmetic
    return value / power
