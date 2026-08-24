"""Dynamic-sparse S4 layer."""

import itertools
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import opt_einsum as oe
from einops import rearrange

from dsrnn.nn.activations import DropoutNd, LeCun, resolve_activation2
from dsrnn.nn.linear import LinearActivation
from dsrnn.nn.router import Router, RouterConfig, RouterOutput
from dsrnn.nn.rnn.ssm.kernel import SSKernel, _conj
from dsrnn.utils.logging import get_logger

contract = oe.contract


class DSS4(nn.Module):
    """S4 layer with input-dependent sparse gating over hidden channels.

    A sequence-level router selects which hidden dimensions are active for each
    input sequence; the mask is applied to the S4 convolution kernel (in the
    frequency domain by default), or the active channels are gathered for a
    reduced computation.
    """

    def __init__(self,
                 d_model: int,
                 d_state: int = 64,
                 l_max: Optional[int] = None,
                 channels: int = 1,
                 bidirectional: bool = False,
                 activation: str = "gelu",
                 postact: str = "glu",
                 initializer: Optional[str] = None,
                 weight_norm: bool = False,
                 hyper_act: Optional[str] = None,
                 dropout: float = 0.0,
                 tie_dropout: bool = False,
                 bottleneck: Optional[int] = None,
                 gate: Optional[int] = None,
                 transposed: bool = True,
                 verbose: bool = False,
                 shift: bool = False,
                 linear: bool = False,
                 liquid_kernel: Optional[str] = None,
                 liquid_degree: int = 2,
                 allcombs: bool = True,
                 lcontract: Optional[str] = None,
                 router_config: Optional[RouterConfig] = None,
                 sparsity_target: str = "k_f",
                 sparsity_dim: str = "H",
                 use_timewise_mask: bool = False,
                 use_sparse_implementation: bool = False,
                 **kernel_args):
        """Initialize the layer.

        Args:
            d_model: Model dimension H.
            d_state: SSM state dimension N.
            l_max: Maximum kernel length L. If None, always use a global kernel.
            channels: Number of channels/heads C of the SSM.
            bidirectional: If True, the convolution kernel is two-sided.
            activation: Activation between SS and FF.
            postact: Activation after FF.
            initializer: Optional initializer on FF.
            weight_norm: Whether to use weight normalization on FF.
            hyper_act: Optional multiplicative modulation activation.
            dropout: Dropout rate. tie_dropout ties the mask across the sequence.
            bottleneck: Optional SSM dimension reduction (GSS).
            gate: Optional gated activation (GSS).
            transposed: If True, use (B, H, L) axis ordering, else (B, L, H).
            verbose: Whether to log the kernel setup.
            shift: Experimental off-by-one correction option.
            linear: If True, remove the pointwise components.
            liquid_kernel: Liquid-S4 kernel type, "polyb" or "kb", or None.
            liquid_degree: Degree of the liquid kernel approximation.
            allcombs: Whether to use all combinations in the liquid kernel.
            lcontract: Optional contraction nonlinearity for the liquid kernel.
            router_config: Router configuration. If provided, enables dynamic
                sparsity. num_channels and input_dim must match the masked
                dimension.
            sparsity_target: "k_f" to mask the frequency-domain kernel, "k" for
                the time-domain kernel.
            sparsity_dim: "H" to mask hidden dimensions, "C" to mask channels.
            use_timewise_mask: If True, compute a mask per timestep instead of
                one per sequence.
            use_sparse_implementation: If True, gather the active channels
                instead of applying a dense mask.
            **kernel_args: Additional SSKernel arguments.
        """
        super().__init__()
        log = get_logger(__name__)
        if verbose:
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")
        if liquid_degree <= 1:
            raise ValueError(f"Illegal argument for liquid_degree ({liquid_degree}). Valid options are >= 2")
        if liquid_kernel not in [None, "polyb", "kb"]:
            raise ValueError(f"Invalid argument for liquid_kernel ('{liquid_kernel}'). Use 'polyb', 'kb'")
        self.liquid_kernel = liquid_kernel
        self.liquid_degree = liquid_degree

        self.d_model = d_model
        self.H = d_model
        self.N = d_state
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.shift = shift
        self.linear = linear
        if lcontract == "lecun":
            self.lcontract = LeCun()
        elif lcontract == "tanh":
            self.lcontract = nn.Tanh()
        else:
            self.lcontract = nn.Identity()

        self.allcombs = allcombs
        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.H = self.H // bottleneck
            self.input_linear = LinearActivation(self.d_model, self.H, transposed=self.transposed,
                                                 initializer=initializer, activation=activation,
                                                 activate=True, weight_norm=weight_norm)

        if gate is not None:
            self.input_gate = LinearActivation(self.d_model, self.d_model * gate, transposed=self.transposed,
                                               initializer=initializer, activation=activation, activate=True,
                                               weight_norm=weight_norm)
            self.output_gate = LinearActivation(self.d_model * gate, self.d_model, transposed=self.transposed,
                                                initializer=initializer, activation=None, activate=False,
                                                weight_norm=weight_norm)

        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = resolve_activation2(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=channels, verbose=verbose, **kernel_args)
        if not self.linear:
            self.activation = resolve_activation2(activation)
            dropout_fn = DropoutNd if tie_dropout else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            self.output_linear = LinearActivation(self.H * self.channels,
                                                  self.d_model * (1 if self.gate is None else self.gate),
                                                  transposed=self.transposed, initializer=initializer,
                                                  activation=postact, activate=True, weight_norm=weight_norm)
        self._allcombs_index_cache = None
        self._allcombs_cache_L = None

        # Dynamic sparsity
        if sparsity_target not in ("k", "k_f"):
            raise ValueError(f"Invalid sparsity_target '{sparsity_target}'. Use 'k' or 'k_f'.")
        if sparsity_dim not in ("H", "C"):
            raise ValueError(f"Invalid sparsity_dim '{sparsity_dim}'. Use 'H' or 'C'.")
        self.sparsity_target = sparsity_target
        self.sparsity_dim = sparsity_dim
        self.use_timewise_mask = use_timewise_mask
        self.use_sparse_implementation = use_sparse_implementation
        self.router = None
        if router_config is not None:
            num_masked = self.H if sparsity_dim == "H" else self.channels
            if router_config.num_channels != num_masked:
                raise ValueError(f"router_config.num_channels ({router_config.num_channels}) must equal "
                                 f"the masked dimension ({num_masked}).")
            if router_config.input_dim != self.H:
                raise ValueError(f"router_config.input_dim ({router_config.input_dim}) must equal "
                                 f"d_model ({self.H}).")
            self.router = Router(router_config)
        self.last_dynamic_mask: Optional[torch.Tensor] = None
        self.last_dynamic_idx: Optional[torch.Tensor] = None

    def get_combs_cache(self, seq_len, i):
        if self._allcombs_cache_L != seq_len:
            self._allcombs_index_cache = []
            self._allcombs_cache_L = seq_len
            for p in range(2, self.liquid_degree + 1):
                selected_count = 1
                for n in range(2, seq_len):
                    count = math.comb(n, p)
                    if count >= seq_len:
                        selected_count = n
                        break
                indices = range(seq_len - selected_count, seq_len)
                indices = list(itertools.combinations(indices, p))
                if len(indices) != seq_len:
                    indices = indices[-seq_len:]
                indices = torch.LongTensor(indices)
                self._allcombs_index_cache.append((p, indices))
        return self._allcombs_index_cache[i]

    def upgrade_degree(self, us, u, i):
        seq_len = u.size(-1)
        if self.allcombs:
            p, indices = self.get_combs_cache(seq_len, i)
            us = u[..., indices[:, 0]]
            for j in range(1, p):
                us = us * u[..., indices[:, j]]
            if us.size(-1) != u.size(-1):
                us = nnf.pad(us, (0, u.size(-1) - us.size(-1)))
        else:
            us_shift = torch.nn.functional.pad(us[..., :-1], (1, 0), "constant", 0)
            us = us * us_shift
        return us

    def _gathered_path_supported(self) -> bool:
        return (self.router is not None
                and not self.use_timewise_mask
                and self.sparsity_dim == "H"
                and not self.bidirectional
                and self.liquid_kernel is None)

    def compute_dynamic_mask(self, u: torch.Tensor) -> RouterOutput:
        """Compute the channel mask for a batch of input sequences.

        Args:
            u: Input of shape (B, H, L) (transposed layout).

        Returns:
            The RouterOutput for the batch.
        """
        if self.use_timewise_mask:
            ctx = u.transpose(-2, -1)  # (B, L, H)
            batch, length = ctx.size(0), ctx.size(1)
            routing = self.router(ctx.reshape(batch * length, self.H))
            routing = RouterOutput(
                mask=routing.mask.view(batch, length, -1),
                idx=routing.idx.view(batch, length, -1),
                scores=routing.scores.view(batch, length, -1),
                logits=routing.logits.view(batch, length, -1),
            )
        else:
            ctx = u.mean(dim=-1)  # (B, H): temporal mean pooling
            routing = self.router(ctx)
        self.last_dynamic_mask = routing.mask.detach()
        self.last_dynamic_idx = routing.idx
        return routing

    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs):
        """Compute the S4 convolution with the dynamic channel mask.

        Args:
            u: Input of shape (B, H, L) if transposed, else (B, L, H).
            state: Optional initial state.
            rate: Sampling rate factor.
            lengths: Optional lengths for padding masks.

        Returns:
            A (output, next_state) pair with the same layout as u.
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1., 0.)
            u = u * mask

        if self.gate is not None:
            v = self.input_gate(u)
        if self.bottleneck is not None:
            u = self.input_linear(u)

        u_f = None
        if self.sparsity_target == "k_f":
            if self.shift:
                u_f = torch.fft.rfft(nnf.pad(u.flip(-1), (L, 0)), n=2 * L)
            else:
                L_kernel = L if self.L is None else min(L, round(self.L / rate))
                u_f = torch.fft.rfft(u, n=L_kernel + L)

        # Dynamic sparsity mask
        routing = None
        dynamic_mask = None
        if self.router is not None:
            # For frequency-domain masking, route from the frequency magnitude.
            u_target = u_f.abs().to(u) if self.sparsity_target == "k_f" else u
            routing = self.compute_dynamic_mask(u_target)
            mask_flat = routing.mask
            if self.sparsity_dim == "H":
                if self.use_timewise_mask:
                    dynamic_mask = mask_flat.transpose(-1, -2).unsqueeze(1)  # (B, 1, H, L)
                else:
                    dynamic_mask = mask_flat.unsqueeze(1).unsqueeze(-1)  # (B, 1, H, 1)
            else:  # "C"
                if self.use_timewise_mask:
                    dynamic_mask = mask_flat.transpose(-1, -2).unsqueeze(-2)  # (B, C, 1, L)
                else:
                    dynamic_mask = mask_flat.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        if (self._gathered_path_supported() and self.use_sparse_implementation
                and self.sparsity_target == "k_f" and state is None and routing is not None):
            idx = routing.idx  # (B, k_H)
            idx_u = idx.unsqueeze(-1).expand(-1, -1, u.size(-1))
            u_gathered = torch.gather(u, dim=1, index=idx_u)  # (B, k_H, L)
            u_f = torch.fft.rfft(u_gathered, n=L_kernel + L)

            k, k_state = self.kernel(L=L_kernel, rate=rate, state=None)  # (C, H, L)
            k_f = torch.fft.rfft(k, n=L_kernel + L)  # (C, H, L_freq)

            B, H = u.size(0), u.size(1)
            C, L_freq = k_f.size(0), k_f.size(-1)
            k_f_exp = k_f.unsqueeze(0).expand(B, C, H, L_freq)
            idx_k = idx.unsqueeze(1).unsqueeze(-1).expand(B, C, idx.size(1), L_freq)
            k_f_gathered = torch.gather(k_f_exp, dim=2, index=idx_k)  # (B, C, k_H, L_freq)

            # Apply the gathered mask values, matching the dense path.
            mask_g = routing.mask.gather(1, idx).unsqueeze(1).unsqueeze(-1)  # (B, 1, k_H, 1)
            k_f_gathered = k_f_gathered * mask_g

            y_f = contract("bhl,bchl->bchl", u_f, k_f_gathered)
            y_reduced = torch.fft.irfft(y_f, n=L_kernel + L)[..., :L]

            y = u.new_zeros(B, self.channels, H, L)
            idx_scatter = idx.unsqueeze(1).unsqueeze(-1).expand(B, self.channels, idx.size(1), L)
            y = y.scatter(dim=2, index=idx_scatter, src=y_reduced)

            y = y + contract("bhl,ch->bchl", u, self.D)
            next_state = None
        else:
            L_kernel = L if self.L is None else min(L, round(self.L / rate))
            k, k_state = self.kernel(L=L_kernel, rate=rate, state=state)  # (C H L) (B C H L)

            if self.bidirectional:
                k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
                k = nnf.pad(k0, (0, L)) + nnf.pad(k1.flip(-1), (L, 0))

            if self.sparsity_target == "k" and dynamic_mask is not None:
                k = k.unsqueeze(0) * dynamic_mask  # (B, C, H, L)

            if self.shift:
                k_padded = nnf.pad(k.flip(-1), (L, 0))
                k_f = torch.fft.rfft(k_padded, n=2 * L)
                if u_f is None:
                    u_f = torch.fft.rfft(nnf.pad(u.flip(-1), (L, 0)), n=2 * L)
                if self.sparsity_target == "k_f" and dynamic_mask is not None:
                    k_f = k_f.unsqueeze(0) * dynamic_mask
                if k_f.ndim == 4:
                    y_f = contract("bhl,bchl->bchl", u_f, k_f)
                else:
                    y_f = contract("bhl,chl->bchl", u_f, k_f)
                y = torch.fft.irfft(y_f, n=L_kernel + L)[..., L:].flip(-1)
                if self.liquid_kernel == "kb":
                    raise NotImplementedError()
            else:
                k_f = torch.fft.rfft(k, n=L_kernel + L)

                if u_f is None:
                    u_f = torch.fft.rfft(u, n=L_kernel + L)

                if self.sparsity_target == "k_f" and dynamic_mask is not None:
                    k_f = k_f.unsqueeze(0) * dynamic_mask

                if k_f.ndim == 4:
                    y_f = contract("bhl,bchl->bchl", u_f, k_f)
                else:
                    y_f = contract("bhl,chl->bchl", u_f, k_f)

                y_sum = y_f
                if self.liquid_kernel == "kb":
                    k_b = k
                    us = u
                    dt = torch.exp(self.kernel.log_dt.to(u.device))
                    B_ = _conj(self.kernel.B).to(u.device)
                    w = _conj(self.kernel.w).to(u.device)
                    dB = torch.diag_embed(1.0 / (1.0 - 0.5 * dt[:, None] * w))
                    dB = dt[:, None] * contract("dab,db->da", dB, B_)
                    dB = dB.unsqueeze(0).unsqueeze(-1)

                    for i in range(self.liquid_degree - 1):
                        us = self.upgrade_degree(us, u, i)
                        k_b = k_b.unsqueeze(2)
                        k_b = contract("abcd,abcd->abd", k_b, dB)
                        u_corr = self.lcontract(us)
                        u_corr = u_corr.flip(dims=[-1])
                        k_b_f = torch.fft.fft(k_b, n=(L_kernel + L) // 2 + 1)
                        u_corr_f = torch.fft.rfft(u_corr, n=L_kernel + L)
                        y_corr_f = contract("bhl,chl->bchl", u_corr_f, k_b_f)
                        y_sum = y_sum + y_corr_f

                y = torch.fft.irfft(y_sum, n=L_kernel + L)[..., :L]

            y = y + contract("bhl,ch->bchl", u, self.D)

        if self.liquid_kernel == "polyb":
            dt = torch.exp(self.kernel.log_dt.to(u.device))
            B_ = _conj(self.kernel.B).to(u.device)
            dC = _conj(self.kernel.C).to(u.device)
            w = _conj(self.kernel.w).to(u.device)
            dB = torch.diag_embed(1.0 / (1.0 - 0.5 * dt[:, None] * w))
            dB = dt[:, None] * contract("dab,db->da", dB, B_)
            us = u
            for i in range(self.liquid_degree - 1):
                us = self.upgrade_degree(us, u, i)
                u_corr = self.lcontract(us)
                us_corr = torch.flip(u_corr, [-1])
                dB1 = dB.unsqueeze(2)
                dB2 = dB.unsqueeze(1)
                dB = (dB1 * dB2).sum(2)
                dCB = contract("abc,bc->ab", dC, dB).unsqueeze(2)
                if self.bidirectional:
                    fwd, bwd = dCB.unbind(0)
                    fwd, bwd = fwd.unsqueeze(0), bwd.unsqueeze(0)
                    y = (y + (us_corr * fwd).unsqueeze(1).float()
                         + (us_corr.flip(2) * bwd).unsqueeze(1).float())
                else:
                    y = y + (us_corr * dCB).unsqueeze(1).float()

        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None

        if self.hyper:
            y, yh = rearrange(y, "b (s c) h l -> s b c h l", s=2)
            y = self.hyper_activation(yh) * y

        y = rearrange(y, "... c h l -> ... (c h) l")

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        if not self.linear:
            y = self.output_linear(y)

        if self.gate is not None:
            y = self.output_gate(y * v)

        return y, next_state

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, u, state):
        """Step one time step as a recurrent model. Intended for validation."""
        assert not self.training
        y, next_state = self.kernel.step(u, state)  # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, "b c h -> b (c h)")
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.H * self.N

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return lambda state: rearrange("... h n -> ... (h n)", state)
