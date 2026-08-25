from typing import Optional, Sequence, List, Union, Tuple, Literal

import torch


class RandomChannelMask:
    """Randomly masks out a fraction of channels along channel_dim."""

    def __init__(self,
                 p: float,
                 channel_dim: int = -1,
                 non_broadcast_dims: Optional[Sequence[int]] = None,
                 mask_value: float = 0.0):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0 and 1")
        self.p = float(p)
        self.channel_dim = int(channel_dim)
        self.non_broadcast_dims = None if non_broadcast_dims is None else list(non_broadcast_dims)
        self.mask_value = float(mask_value)

    def __call__(self,
                 batch: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, y = batch
        if not torch.is_tensor(x):
            raise TypeError("Input must be a torch.Tensor")

        nd = x.ndim
        ch = self.channel_dim % nd

        if self.non_broadcast_dims is None:
            non_bdims: List[int] = []
        else:
            non_bdims = sorted({int(d) % nd for d in self.non_broadcast_dims})

        if ch in non_bdims:
            raise ValueError("channel_dim cannot be in non_broadcast_dims")

        num_channels = x.size(ch)
        num_mask = int(round(num_channels * self.p))

        if num_mask == 0:
            return x
        if num_mask >= num_channels:
            if self.mask_value == 0.0:
                return x * 0.0, y
            mask_shape = [1] * nd
            mask_shape[ch] = num_channels
            for d in non_bdims:
                mask_shape[d] = x.size(d)
            mask_full = torch.full(mask_shape, self.mask_value, dtype=x.dtype, device=x.device)
            return x * mask_full, y

        mask_shape = [1] * nd
        mask_shape[ch] = num_channels
        for d in non_bdims:
            mask_shape[d] = x.size(d)

        scores = torch.rand(tuple(mask_shape), device=x.device, dtype=torch.float32)

        topk_vals = torch.topk(scores, num_mask, dim=ch).values
        kth_val = topk_vals.select(dim=ch, index=num_mask - 1).unsqueeze(ch)

        keep = scores < kth_val

        if self.mask_value == 0.0:
            mul_mask = keep.to(dtype=x.dtype)
        else:
            mul_mask = keep.to(dtype=x.dtype) + (~keep).to(dtype=x.dtype) * float(self.mask_value)

        return x * mul_mask, y

    def __repr__(self):
        cls = self.__class__.__name__
        return (f"{cls}(p={self.p}, "
                f"channel_dim={self.channel_dim}, "
                f"non_broadcast_dims={self.non_broadcast_dims})")


class GaussianNoiseTransform:
    """Transform that adds Gaussian noise along the last dimension or element-wise."""

    def __init__(self,
                 mean: Union[float, Tuple[float, float]] = 0.0,
                 std: Union[float, Tuple[float, float]] = 1.0,
                 p: float = 1.0,
                 mode: Literal['element', 'feature', 'sample_feature'] = 'element',
                 clamp: Optional[Tuple[float, float]] = None,
                 generator: Optional[torch.Generator] = None,
                 seed: Optional[int] = None,):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0,1]")
        if mode not in ('element', 'feature', 'sample_feature'):
            raise ValueError("mode must be 'element','feature' or 'sample_feature'")

        self.mean = mean
        self.std = std
        self.p = p
        self.mode = mode
        self.clamp = clamp

        self._user_generator = generator
        self._seed = seed
        self._internal_generator = None

    def _get_generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self._user_generator is not None:
            return self._user_generator
        if self._seed is None:
            return None
        if self._internal_generator is None:
            try:
                gen = torch.Generator(device=device)
                gen.manual_seed(self._seed)
            except TypeError:
                gen = torch.Generator()
                gen.manual_seed(self._seed)
            self._internal_generator = gen
        return self._internal_generator

    def _sample_param(self, param: Union[float, Tuple[float, float]],
                      gen: Optional[torch.Generator], device: torch.device) -> float:
        if isinstance(param, (tuple, list)):
            low, high = float(param[0]), float(param[1])
            if gen is not None:
                r = torch.rand(1, generator=gen, device=device).item()
            else:
                r = torch.rand(1, device=device).item()
            return low + (high - low) * r
        else:
            return float(param)

    def __call__(self,
                 batch: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the transform to tensor x."""
        x, y = batch
        if not torch.is_tensor(x):
            raise TypeError("Input must be a torch.Tensor")
        if not x.dtype.is_floating_point:
            raise TypeError("Input tensor must be floating dtype for Gaussian noise")

        if self.p == 0.0:
            return x

        device = x.device
        gen = self._get_generator(device)

        if self.p < 1.0:
            if gen is not None:
                apply_noise = torch.rand(1, generator=gen, device=device).item() < self.p
            else:
                apply_noise = torch.rand(1, device=device).item() < self.p
            if not apply_noise:
                return x

        mean_val = self._sample_param(self.mean, gen, device)
        std_val = self._sample_param(self.std, gen, device)

        if std_val == 0.0 and mean_val == 0.0:
            return x, y

        last_dim = x.shape[-1]

        if self.mode == 'element':
            shape = x.shape
            if gen is not None:
                noise = torch.randn(shape, generator=gen, device=device, dtype=x.dtype) * std_val + mean_val
            else:
                noise = torch.randn(shape, device=device, dtype=x.dtype) * std_val + mean_val

        elif self.mode == 'feature':
            vec_shape = (last_dim,)
            if gen is not None:
                vec = torch.randn(vec_shape, generator=gen, device=device, dtype=x.dtype) * std_val + mean_val
            else:
                vec = torch.randn(vec_shape, device=device, dtype=x.dtype) * std_val + mean_val
            view_shape = [1] * (x.ndim - 1) + [last_dim]
            noise = vec.view(*view_shape).expand_as(x)

        else:  # sample_feature
            if x.ndim == 1:
                vec_shape = (last_dim,)
                if gen is not None:
                    vec = torch.randn(vec_shape, generator=gen, device=device, dtype=x.dtype) * std_val + mean_val
                else:
                    vec = torch.randn(vec_shape, device=device, dtype=x.dtype) * std_val + mean_val
                view_shape = [1] * (x.ndim - 1) + [last_dim]
                noise = vec.view(*view_shape).expand_as(x)
            else:
                batch = x.shape[0]
                mat_shape = (batch, last_dim)
                if gen is not None:
                    mat = torch.randn(mat_shape, generator=gen, device=device, dtype=x.dtype) * std_val + mean_val
                else:
                    mat = torch.randn(mat_shape, device=device, dtype=x.dtype) * std_val + mean_val
                view_shape = [batch] + [1] * (x.ndim - 2) + [last_dim]
                noise = mat.view(*view_shape).expand_as(x)

        out = x + noise

        if self.clamp is not None:
            mn, mx = self.clamp
            out = out.clamp(min=mn, max=mx)

        return out, y

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p}, mode='{self.mode}', "
                f"clamp={self.clamp}, seed={'set' if (self._seed or self._user_generator) else 'None'})")
