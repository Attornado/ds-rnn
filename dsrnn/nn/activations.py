import math
from typing import Final, Dict, Optional
from functools import partial
import torch
from einops import rearrange



class ScaledSigmoid(torch.nn.Module):
    """
    Scaled sigmoid with learnable sharpness and threshold. Computes sigmoid(tau * (x - mu)). tau is stored in log-space
    to keep it positive during optimization.
    """

    def __init__(self,
                 tau_init: float = 1.0,
                 mu_init: float = 0.0,
                 learnable: bool = True):
        """
        Initialize the scaled sigmoid.

        Args:
            tau_init: Initial sharpness. Must be positive.
            mu_init: Initial threshold.
            learnable: Whether tau and mu are learnable parameters.
        """
        super().__init__()
        if tau_init <= 0:
            raise ValueError(f"tau_init must be positive, got {tau_init}.")
        self.tau_init = tau_init
        self.mu_init = mu_init
        self.log_tau = torch.nn.Parameter(torch.log(torch.tensor(tau_init)), requires_grad=learnable)
        self.mu = torch.nn.Parameter(torch.tensor(mu_init), requires_grad=learnable)

    @property
    def tau(self) -> torch.Tensor:
        """Return the current sharpness."""
        return self.log_tau.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the scaled sigmoid.

        Args:
            x: Input tensor.

        Returns:
            Tensor of the same shape with values in (0, 1).
        """
        return torch.sigmoid(self.tau * (x - self.mu))

    def reset_parameters(self) -> None:
        """Reset tau and mu to their initial values."""
        with torch.no_grad():
            self.log_tau.fill_(math.log(self.tau_init))
            self.mu.fill_(self.mu_init)


class SteepSigmoid(torch.nn.Module):
    """Steep sigmoid with optional learnable threshold and sharpness."""

    def __init__(self, q: float = 0.0, m: float = 5.0, learnable: bool = True):
        super().__init__()
        self.q_init = q
        self.m_init = m
        self.learnable = learnable
        self.q = torch.tensor(q)
        self.m = torch.log(torch.tensor(m))

        if learnable:
            self.q = torch.nn.Parameter(self.q)
            self.m = torch.nn.Parameter(self.m)

    def reset_parameters(self):
        if self.learnable:
            self.q = torch.nn.Parameter(torch.tensor(self.q_init))
            self.m = torch.nn.Parameter(torch.tensor(self.m_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.exp(self.m) * (x - self.q))


class DropoutNd(torch.nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)."""
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1. - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


class TransposedLN(torch.nn.Module):
    """LayerNorm module over second dimension. Assumes shape (B, D, L)."""

    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = torch.nn.Parameter(torch.zeros(1))
            self.s = torch.nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = torch.nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s / s) * (x - m + self.m)
        else:
            _x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(_x, 'b ... d -> b d ...')
        return y


def hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """Element-wise hard sigmoid of x."""
    x = (0.25 * x) + 0.5
    x = torch.nn.functional.threshold(-x, -1.0, -1.0)
    x = torch.nn.functional.threshold(-x, 0.0, 0.0)
    return x


class HardSigmoid(torch.nn.Module):
    """Element-wise hard sigmoid."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hardsigmoid(x)


class SquaredReLU(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu(x) ** 2


class ModReLU(torch.nn.Module):
    def __init__(self, features):
        super(ModReLU, self).__init__()
        self.features = features
        self.b = torch.nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = torch.nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)
        return phase * magnitude


class Modrelu(ModReLU):
    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)


class LeCun(torch.nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


def get_initializer(name, activation=None):
    if activation in [None, 'id', 'identity', 'linear', 'modrelu']:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu'
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer


_ACTIVATIONS: Final[Dict[str, torch.nn.Module]] = {
    "linear": torch.nn.Identity(),
    "relu": torch.nn.ReLU(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "rrelu": torch.nn.RReLU(),
    "relu6": torch.nn.ReLU6(),
    "gelu": torch.nn.GELU(),
    "elu": torch.nn.ELU(),
    "celu": torch.nn.CELU(),
    "glu": torch.nn.GLU(),
    "selu": torch.nn.SELU(),
    "prelu": torch.nn.PReLU(),
    "silu": torch.nn.SiLU(),
    "hardswish": torch.nn.Hardswish(),
    "tanh": torch.nn.Tanh(),
    "lecun": LeCun(),
    "sigmoid": torch.nn.Sigmoid(),
    "log_sigmoid": torch.nn.LogSigmoid(),
    "softmax": torch.nn.Softmax(dim=-1),
    "hardtanh": torch.nn.Hardtanh(),
    "hardsigmoid": HardSigmoid(),
}


def resolve_activation(activation: str) -> torch.nn.Module:
    return _ACTIVATIONS[activation]


def resolve_activation2(activation: Optional[str] = None, size: Optional[int] = None, dim: int = -1) -> torch.nn.Module:
    if activation in [None, 'id', 'identity', 'linear']:
        return torch.nn.Identity()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'gelu':
        return torch.nn.GELU()
    elif activation in ['swish', 'silu']:
        return torch.nn.SiLU()
    elif activation == 'glu':
        return torch.nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'modrelu':
        return Modrelu(size)
    elif activation == 'sqrelu':
        return SquaredReLU()
    elif activation == 'ln':
        return TransposedLN(dim)
    else:
        try:
            return resolve_activation(activation)
        except KeyError as e:
            raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))
