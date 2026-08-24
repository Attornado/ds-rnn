"""SSM convolution kernels.

SSKernelNPLR is the S4 kernel, implementing the 'normal plus low-rank' algorithm from the original S4 paper. This stores parameters A, B, C, dt, and calling it creates the SSM convolution kernel bar{K}.

A much simpler version SSKernelSlow is included for illustration purposes: it has the same output, but uses the naive algorithm which is much slower. This module is meant for testing and exposition, to understand what the State Space Kernel actually does.

SSKernelDiag is the S4D kernel, a simpler algorithm for computing the kernel for the case of diagonal state matrices A.

SSKernel wraps these with common options and handles the initialization.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from opt_einsum import contract, contract_expression

import dsrnn.nn.rnn.ssm.functional.dplr as dplr
import dsrnn.nn.rnn.ssm.functional.hippo as hippo
from dsrnn.nn.rnn.ssm.functional.krylov import krylov, power
from dsrnn.utils.logging import get_logger

log = get_logger(__name__)

try:  # Try CUDA extension
    from extensions.cauchy.cauchy import cauchy_mult
    has_cauchy_extension = True
    log.info("CUDA extension for Cauchy multiplication found.")
except:
    log.warn(
        "CUDA extension for Cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    )
    has_cauchy_extension = False

try:
    import pykeops
    from dsrnn.nn.rnn.ssm.functional.cauchy import cauchy_conj
    from dsrnn.nn.rnn.ssm.functional.vandermonde import log_vandermonde, log_vandermonde_transpose

    has_pykeops = True
    log.info("Pykeops installation found.")
except ImportError:
    has_pykeops = False
    from dsrnn.nn.rnn.ssm.functional.cauchy import cauchy_naive
    from dsrnn.nn.rnn.ssm.functional.vandermonde import log_vandermonde_naive as log_vandermonde
    from dsrnn.nn.rnn.ssm.functional.vandermonde import log_vandermonde_transpose_naive as log_vandermonde_transpose
    if not has_cauchy_extension:
        log.error(
            "Falling back on slow Cauchy kernel. Install at least one of pykeops or the CUDA extension for memory efficiency."
        )
    log.error(
        "Falling back on slow Vandermonde kernel. Install pykeops for improved memory efficiency."
    )

_isnan = lambda x: torch.isnan(x).any()
_isinf = lambda x: torch.isinf(x).any()

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters."""

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class SSKernelNPLR(OptimModule):
    """Stores a representation of and computes the SSKernel function K_L(dt, A, B, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)."""

    @torch.no_grad()
    def _setup_C(self, L):
        """Construct C~ from C."""
        if self.L.item() == 0:
            if self.verbose:
                log.info(f"S4: Initializing kernel to length {L}")
            double_length = False
        elif L > self.L.item():
            if self.verbose:
                log.info(f"S4: Doubling length from L = {self.L.item()} to {2 * self.L.item()}")
            double_length = True
            L = self.L.item()
        else:
            return

        C = _r2c(self.C)
        dA, _ = self._setup_state()
        dA_L = power(L, dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length:
            prod = -prod  # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N]  # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        self.L = 2 * self.L if double_length else self.L + L

    def _omega(self, L, dtype, device, cache=True):
        """Calculate (and cache) FFT nodes and their "unprocessed" version with the bilinear transform."""
        if cache and hasattr(self, 'omega') and self.omega.size(-1) == L // 2 + 1:
            return self.omega, self.z

        omega = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )  # \\omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)

        if cache:
            self.omega = omega
            self.z = z
        return omega, z

    def __init__(
            self,
            w, P, B, C, log_dt,
            L=None,
            lr=None,
            verbose=False,
            keops=False,
            real_type='exp',
            real_tolerance=1e-3,
            bandlimit=None,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        A is represented by diag(w) - PP^*
        w: (S, N) diagonal part
        P: (R, S, N) low-rank part
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature
        lr: [dict | float | None] hook to set lr of special parameters (A, B, dt)
        """
        super().__init__()
        self.verbose = verbose
        self.keops = keops
        self.bandlimit = bandlimit
        self.real_type = real_type
        self.real_tolerance = real_tolerance

        # Rank of low-rank correction
        self.rank = P.shape[-3]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)

        # Check different SSM inits
        assert w.size(-2) == P.size(-2) == B.size(-2)  # n_ssm
        assert self.H % w.size(0) == 0
        self.n_ssm = w.size(0)
        self.broadcast = self.H // w.size(0)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))  # (C, H, N)
        B = B.unsqueeze(0)  # (1, 1, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None
        self.register("log_dt", log_dt, lr_dict.get('dt', lr))
        self.register("B", _c2r(B), lr_dict.get('B', lr))
        self.register("P", _c2r(P), lr_dict.get('A', lr))
        self.register("inv_w_real", self._w_init(w.real), lr_dict.get('A', lr))
        self.register("w_imag", w.imag, lr_dict.get('A', lr))

        self.l_max = L
        self.register_buffer('L', torch.tensor(0))  # Internal length

    def _w_init(self, w_real):
        w_real = torch.clamp(w_real, max=-self.real_tolerance)
        if self.real_type == 'none':
            return -w_real
        elif self.real_type == 'exp':
            return torch.log(-w_real)
        elif self.real_type == 'relu':
            return -w_real
        elif self.real_type == 'sigmoid':
            return torch.logit(-w_real)
        elif self.real_type == 'softplus':
            return torch.log(torch.exp(-w_real) - 1)
        else:
            raise NotImplementedError

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.real_type == 'none':
            w_real = -self.inv_w_real
        elif self.real_type == 'exp':
            w_real = -torch.exp(self.inv_w_real)
        elif self.real_type == 'relu':
            w_real = -F.relu(self.inv_w_real)
        elif self.real_type == 'sigmoid':
            w_real = -F.sigmoid(self.inv_w_real)
        elif self.real_type == 'softplus':
            w_real = -F.softplus(self.inv_w_real)
        else:
            raise NotImplementedError
        w = w_real + 1j * self.w_imag
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """
        # Initialize C~ if necessary (done in forward pass so it's on the correct device)
        if self.L.item() == 0 and self.l_max is not None and self.l_max > 0:
            self._setup_C(self.l_max)

        # Handle sampling rate logic
        if L is None:
            L = round(self.L.item() / rate)

        # Increase the internal length if needed
        continuous_L = round(rate * L)
        while continuous_L > self.L.item():
            self._setup_C(continuous_L)
        discrete_L = round(self.L.item() / rate)

        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj()
        w = self._w()  # (S, N) where S=n_ssm

        # Address bandlimiting
        if self.bandlimit is not None:
            freqs = w.imag.abs() / (2 * math.pi)  # (H, N)
            freqs = dt[:, None] / rate * freqs  # (H, N)
            mask = torch.where(freqs < self.bandlimit * .5, 1, 0)
            C = C * mask

        # Get FFT nodes of right length
        omega, z = self._omega(discrete_L, dtype=w.dtype, device=w.device, cache=(rate == 1.0))

        # Broadcast parameters to same hidden features H
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.broadcast)
        P = repeat(P, 'r t n -> r (v t) n', v=self.broadcast)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.broadcast)
        w = repeat(w, 't n -> (v t) n', v=self.broadcast)

        # Augment B
        if state is not None:
            s = _conj(state) if state.size(-1) == self.N else state  # (B H N)
            sA = (
                s * _conj(w)  # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., :self.N]

            B = torch.cat([s, B], dim=-3)  # (B+1, H, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3)  # (B+1+R, H, N)
        C = torch.cat([C, Q], dim=-3)  # (C+R, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (B+1+R, C+R, H, N)

        # Calculate resolvent at omega
        if has_cauchy_extension and z.dtype == torch.cfloat and not self.keops:
            r = cauchy_mult(v, z, w, symmetric=True)
        elif has_pykeops:
            r = cauchy_conj(v, z, w)
        else:
            r = cauchy_naive(v, z, w)
        r = r * dt[None, None, :, None]  # (B+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, : -self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=discrete_L)  # (B+1, C, H, L)

        # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (B, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :]  # (C H L)

        return k_B, k_state

    @torch.no_grad()
    def double_length(self):
        self._setup_C(2 * self.L)

    @torch.no_grad()
    def _check(self):
        """Check if A, B, C parameters and vanilla SSKernel construction can be recovered."""
        K = self.forward(L=self.l_max)[0]

        self._setup_step()
        K_ = krylov(self.l_max, self.dA, self.dB, self.dC)

        diff = K - K_
        print("checking DPLR Kernel construction", torch.sum(diff ** 2))

    @torch.no_grad()
    def _setup_linear(self):
        """Create parameters that allow fast linear stepping of state."""
        w = self._w()
        B = _r2c(self.B)  # (H N)
        P = _r2c(self.P)
        Q = P.conj()

        # Repeat w shape properly
        B = repeat(B, '1 t n -> 1 (v t) n', v=self.broadcast)
        P = repeat(P, 'r t n -> r (v t) n', v=self.broadcast)
        Q = repeat(Q, 'r t n -> r (v t) n', v=self.broadcast)
        w = repeat(w, 't n -> (v t) n', v=self.broadcast)

        # Prepare Linear stepping
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        R = (torch.eye(self.rank, dtype=w.dtype, device=w.device) + 2 * contract('r h n, h n, s h n -> h r s', Q, D, P).real)  # (H R R)
        Q_D = rearrange(Q * D, 'r h n -> h r n')
        try:
            R = torch.linalg.solve(R, Q_D)  # (H R N)
        except:
            R = torch.tensor(np.linalg.solve(R.to(Q_D).contiguous().detach().cpu(), Q_D.contiguous().detach().cpu())).to(Q_D)
        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D,
            "R": R,
            "P": P,
            "Q": Q,
            "B": B,
            "E": 2.0 / dt.unsqueeze(-1) + w,
        }

    def _step_state_linear(self, u=None, state=None):
        """Version of the step function that has time O(N) instead of O(N^2) per step."""
        C = _r2c(self.C)

        if u is None:
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N:
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N]
        else:
            assert state.size(-1) == 2 * self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y)
        D = step_params["D"]
        E = step_params["E"]
        R = step_params["R"]
        P = step_params["P"]
        Q = step_params["Q"]
        B = step_params["B"]

        new_state = E * state - contract_fn(P, Q, state)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """Construct dA and dB for discretized state equation."""
        self._setup_linear()
        C = _r2c(self.C)

        state = torch.eye(2 * self.N, dtype=C.dtype, device=C.device).unsqueeze(-2)  # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        dB = rearrange(dB, '1 h n -> h n')  # (H N)
        return dA, dB

    def _step_state(self, u, state):
        """Must be called after self.default_state() is used to construct an initial state."""
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state

    def _setup_step(self, mode='dense'):
        """Set up dA, dB, dC discretized parameters for stepping."""
        self.dA, self.dB = self._setup_state()

        # Calculate original C
        C = _conj(_r2c(self.C))  # (H C N)
        if self.L.item() == 0:
            dC = C
        else:
            dA_L = power(self.L.item(), self.dA)
            I = torch.eye(self.dA.size(-1)).to(dA_L)

            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2),
                C.unsqueeze(-1),
            ).squeeze(-1)
        self.dC = dC

        self._step_mode = mode
        if mode == 'linear':
            self.dC = 2 * self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            if self.verbose:
                print("Diagonalization error:", torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA))

            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else:
            raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        step_mode = getattr(self, "_step_mode", "dense")
        if step_mode != 'linear':
            N *= 2

            if step_mode == 'diagonal':
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N),
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N),
            batch_shape + (H, N),
        )

        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """Must have called self._setup_step() and created state with self.default_state() before calling this."""
        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y.real, new_state


class SSKernelSlow(OptimModule):
    """Slow version of SSKernel function for illustration and benchmarking."""

    @staticmethod
    def bilinear(dt, A, B=None):
        """
        dt: (...) timescales
        A: (... N N)
        B: (... N)
        """
        N = A.shape[-1]
        I = torch.eye(N).to(A)
        A_backwards = I - dt[:, None, None] / 2 * A
        A_forwards = I + dt[:, None, None] / 2 * A

        if B is None:
            dB = None
        else:
            dB = dt[..., None] * torch.linalg.solve(
                A_backwards, B.unsqueeze(-1)
            ).squeeze(-1)

        dA = torch.linalg.solve(A_backwards, A_forwards)  # (... N N)
        return dA, dB

    def __init__(self, A, B, C, log_dt, L=None, lr=None):
        super().__init__()
        self.L = L
        self.N = A.size(-1)
        self.H = log_dt.size(-1)

        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))  # (C, H, N)

        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None
        self.register("log_dt", log_dt, lr_dict.get('dt', lr))
        self.register("A", _c2r(A), lr_dict.get('A', lr))
        self.register("B", _c2r(B), lr_dict.get('B', lr))
        self.C = nn.Parameter(_resolve_conj(C))

        self.trainable = lr_dict.get('dt', lr) > 0.0 or lr_dict.get('A', lr) > 0.0 or lr_dict.get('B', lr) > 0.0
        self.K = None

    def forward(self, state=None, rate=1.0, L=None):
        if L is None:
            L = self.L
        assert rate == 1.0 and L is not None

        if self.trainable:
            dA, dB = SSKernelSlow.bilinear(torch.exp(self.log_dt), self.A, self.B)
            k = krylov(L, dA, dB, self.C)  # (H L)
        else:
            if self.K is None:
                dA, dB = SSKernelSlow.bilinear(torch.exp(self.log_dt), self.A, self.B)
                self.K = krylov(L, dA, dB)  # (H N L)
            k = contract('hnl,chn->chl', self.K[..., :L], self.C)
        k = k.float()

        if state is not None:
            state = state.to(self.dA)
            state = contract("... n m, ... m -> ... n", self.dA, state)
            k_state = krylov(L, self.dA, state.unsqueeze(-3), self.C)
            k_state = k_state.float()
        else:
            k_state = None
        return k, k_state

    def default_state(self, *batch_shape):
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=self.C.dtype, device=self.C.device)
        return state

    def _setup_state(self):
        dA, dB = SSKernelSlow.bilinear(torch.exp(self.log_dt), self.A, self.B)
        return dA, dB

    def _setup_step(self):
        self.dA, self.dB = self._setup_state()
        self.dC = self.C

    def step(self, u, state):
        next_state = contract("h m n, b h n -> b h m", self.dA, state) \
                     + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return y.real, next_state


class SSKernelDiag(OptimModule):
    """Version using (complex) diagonal state matrix (S4D)."""

    def __init__(
            self,
            A, B, C, log_dt,
            L=None,
            disc='bilinear',
            real_type='exp',
            lr=None,
            bandlimit=None,
    ):
        super().__init__()
        self.L = L
        self.disc = disc
        self.bandlimit = bandlimit
        self.real_type = real_type

        assert A.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = A.size(-1)
        assert A.size(-2) == B.size(-2)
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)

        self.channels = C.shape[0]
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None

        self.register("log_dt", log_dt, lr_dict.get('dt', lr))
        self.register("A", _c2r(A), lr_dict.get('A', lr))
        self.register("B", _c2r(B), lr_dict.get('B', lr))
        self.register("inv_A_real", self._A_init(A.real), lr_dict.get('A', lr))
        self.register("A_imag", A.imag, lr_dict.get('A', lr))

    def _A_init(self, A_real):
        A_real = torch.clamp(A_real, max=-1e-4)
        if self.real_type == 'none':
            return -A_real
        elif self.real_type == 'exp':
            return torch.log(-A_real)
        elif self.real_type == 'relu':
            return -A_real
        elif self.real_type == 'sigmoid':
            return torch.logit(-A_real)
        elif self.real_type == 'softplus':
            return torch.log(torch.exp(-A_real) - 1)
        else:
            raise NotImplementedError

    def _A(self):
        if self.real_type == 'none':
            A_real = -self.inv_A_real
        elif self.real_type == 'exp':
            A_real = -torch.exp(self.inv_A_real)
        elif self.real_type == 'relu':
            A_real = -F.relu(self.inv_A_real) - 1e-4
        elif self.real_type == 'sigmoid':
            A_real = -F.sigmoid(self.inv_A_real)
        elif self.real_type == 'softplus':
            A_real = -F.softplus(self.inv_A_real)
        else:
            raise NotImplementedError
        A = A_real + 1j * self.A_imag
        return A

    def forward(self, L, state=None, rate=1.0, u=None):
        """
        state: (B, H, N) initial state
        rate: sampling rate factor
        L: target length

        returns:
        (C, H, L) convolution kernel (generally C=1)
        (B, H, L) output from initial state
        """
        dt = torch.exp(self.log_dt) * rate  # (H)
        C = _r2c(self.C)  # (C H N)
        A = self._A()  # (H N)

        B = _r2c(self.B)
        B = repeat(B, 't n -> 1 (v t) n', v=self.repeat)

        if self.bandlimit is not None:
            freqs = dt[:, None] / rate * A.imag.abs() / (2 * math.pi)  # (H, N)
            mask = torch.where(freqs < self.bandlimit * .5, 1, 0)
            C = C * mask

        A = repeat(A, 't n -> (v t) n', v=self.repeat)
        dtA = A * dt.unsqueeze(-1)  # (H N)

        # Augment B with state
        if state is not None:
            s = state / dt.unsqueeze(-1)
            if self.disc == 'bilinear':
                s = s * (1. + dtA / 2)
            elif self.disc == 'zoh':
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.)
            B = torch.cat([s, B], dim=-3)  # (1+B H N)

        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)
        if self.disc == 'zoh':
            C = C * (torch.exp(dtA) - 1.) / A
            K = log_vandermonde(C, dtA, L)  # (H L)
        elif self.disc == 'bilinear':
            C = C * (1. - dtA / 2).reciprocal() * dt.unsqueeze(-1)
            dA = (1. + dtA / 2) / (1. - dtA / 2)
            K = log_vandermonde(C, dA.log(), L)
        elif self.disc == 'dss':
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device)  # [H N L]
            A_gt_0 = A.real > 0  # [N]
            if A_gt_0.any():
                with torch.no_grad():
                    P_max = dtA * (A_gt_0 * (L - 1))  # [H N]
                P = P - P_max.unsqueeze(-1)  # [H N L]
            S = P.exp()  # [H N L]

            dtA_neg = dtA * (1 - 2 * A_gt_0)  # [H N]
            num = dtA_neg.exp() - 1  # [H N]
            den = (dtA_neg * L).exp() - 1  # [H N]

            x = den * A
            x_conj = _resolve_conj(x)
            r = x_conj / (x * x_conj + 1e-7)

            C = C * num * r  # [C H N]
            K = contract('chn,hnl->chl', C, S).float()
        else:
            assert False, f"{self.disc} not supported"

        K = K.view(-1, self.channels, self.H, L)  # (1+B C H L)
        if state is not None:
            K_state = K[:-1, :, :, :]  # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :]  # (C H L)
        return K, K_state

    def _setup_step(self):
        dt = torch.exp(self.log_dt)  # (H)
        B = _r2c(self.B)  # (H N)
        C = _r2c(self.C)  # (C H N)
        self.dC = C
        A = self._A()  # (H N)

        dtA = A * dt.unsqueeze(-1)  # (H N)
        if self.disc == 'zoh':
            self.dA = torch.exp(dtA)  # (H N)
            self.dB = B * (torch.exp(dtA) - 1.) / A  # (C H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA / 2) / (1. - dtA / 2)
            self.dB = B * (1. - dtA / 2).reciprocal() * dt.unsqueeze(-1)

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                     + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2 * y.real, next_state

    def forward_state(self, u, state):
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous()  # (B H L)
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state


class SSKernel(nn.Module):
    """Wrapper around SSKernel parameterizations."""

    def __init__(
            self,
            H,
            N=64,
            L=None,
            measure="legs",
            rank=1,
            channels=1,
            dt_min=0.001,
            dt_max=0.1,
            deterministic=False,
            lr=None,
            mode="nplr",
            n_ssm=None,
            verbose=False,
            measure_args={},
            **kernel_args,
    ):
        """State Space Kernel which computes the convolution kernel \\bar{K}."""
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), H))
        else:
            log_dt = torch.rand(self.H, dtype=dtype) * (
                    math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)

        # Compute the preprocessed representation
        if mode == "real":
            A, B = hippo.transition(measure, self.N)
            A = torch.as_tensor(A, dtype=dtype)
            B = torch.as_tensor(B, dtype=dtype)[:, 0]

            if deterministic:
                C = torch.zeros(channels, self.H, self.N, dtype=dtype)
                C[..., :1] = 1.0
            else:
                C = torch.randn(channels, self.H, self.N, dtype=dtype)

            self.kernel = SSKernelSlow(
                A, B, C, log_dt, L=L,
                lr=lr,
            )
        else:
            w, P, B, V = dplr.combination(measure, self.N, rank, self.n_ssm, **measure_args)

            if deterministic:
                C = torch.zeros(channels, self.H, self.N, dtype=cdtype)
                C[:, :, :1] = 1.
                C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C)  # V^* C
            else:
                C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

            assert self.n_ssm % B.size(-2) == 0 \
                   and self.n_ssm % P.size(-2) == 0 \
                   and self.n_ssm % w.size(-2) == 0
            B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
            P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
            w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()
            C = C.contiguous()

            if mode == "nplr":
                self.kernel = SSKernelNPLR(
                    w, P, B, C,
                    log_dt, L=L,
                    lr=lr,
                    verbose=verbose,
                    **kernel_args,
                )
            elif mode == "diag":
                C = C * repeat(B, 't n -> (v t) n', v=H // self.n_ssm)
                self.kernel = SSKernelDiag(
                    w, B, C, log_dt, L=L,
                    lr=lr,
                    **kernel_args,
                )
            elif mode == "slow":
                A = torch.diag_embed(_conj(w)) \
                      - contract("... r p, ... r q -> ... p q", _conj(P), _conj(P).conj())
                self.kernel = SSKernelSlow(
                    A, _conj(B), _conj(C), log_dt, L=L,
                    lr=lr,
                )
            else:
                raise NotImplementedError(f"{mode=} is not valid")
        self.B = B
        self.C = C
        self.w = w
        self.log_dt = log_dt

    def forward(self, state=None, L=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence."""
        if hasattr(self.kernel, "forward_state"):
            return self.kernel.forward_state(u, state)

        dA, dB = self.kernel._setup_state()

        conj = state.size(-1) != dA.size(-1)
        if conj:
            state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1))
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj:
            next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, u, state, **kwargs):
        y, state = self.kernel.step(u, state, **kwargs)
        return y, state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)
