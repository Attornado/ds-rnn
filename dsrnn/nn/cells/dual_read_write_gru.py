"""Reference and gathered GRU cells with independent READ and WRITE masks."""

from dataclasses import replace
from decimal import Decimal
from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from dsrnn.nn.router import Router, RouterConfig, RouterOutput
from dsrnn.nn.functional import BackwardRule
from dsrnn.nn.cells.dual_read_write import _prepare_sparse_routing


class DualReadWriteGRUCell(nn.Module):
    """Dense reference GRU with independent READ and WRITE masks.

    ``read_mask`` is applied to the previous state before every recurrent GRU
    path, including the gated residual in the GRU proposal. ``write_mask`` is
    then applied to that proposal, so an unwritten coordinate is preserved
    exactly. With both masks equal to one this matches the repository's
    reset-before ``GRUCell`` (not PyTorch's reset-after GRU). All three gates
    share each routing mask; reset gates multiply sources before the content
    recurrent affine transform.

    Args:
        input_size: Number of input features; must be positive.
        hidden_size: Number of hidden-state coordinates; must be positive.
        bias: Whether to include input and recurrent biases for each gate.
        read_router_config: Optional READ router configuration matching the
            input and hidden dimensions. Requires ``write_router_config``.
        write_router_config: Optional independent WRITE router configuration.
            When both configs are omitted, explicit masks are required.

    Attributes:
        weight_ih: Input weights with shape ``(3 * hidden_size, input_size)``.
        weight_hh: Recurrent weights with shape
            ``(3 * hidden_size, hidden_size)``. Gate order is reset, update,
            content for weights and biases.
        read_mask: READ mask from the latest forward call, or None initially.
        write_mask: WRITE mask from the latest forward call, or None initially.
        last_read_routing: Latest internal READ router output, or None when
            an external mask was used.
        last_write_routing: Latest internal WRITE router output, or None when
            an external mask was used.

    Raises:
        ValueError: If dimensions are nonpositive, only one router config is
            supplied, or router dimensions do not match the cell.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 *,
                 read_router_config: Optional[RouterConfig] = None,
                 write_router_config: Optional[RouterConfig] = None):
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        if (read_router_config is None) != (write_router_config is None):
            raise ValueError("Provide both read_router_config and write_router_config, or neither.")
        self.read_router = None
        self.write_router = None
        for name, config in (("read_router", read_router_config),
                             ("write_router", write_router_config)):
            if config is not None:
                if config.input_dim != input_size or config.num_channels != hidden_size:
                    raise ValueError(f"{name} config must have input_dim={input_size} "
                                     f"and num_channels={hidden_size}.")
                k = config.k
                if k is None:
                    k = max(1, ceil((Decimal(1) - Decimal(str(config.sparsity)))
                                    * hidden_size))
                setattr(self, name, Router(replace(
                    config, k=k, binary=True, budget_rounding="ceil")))
        self.last_read_routing: Optional[RouterOutput] = None
        self.last_write_routing: Optional[RouterOutput] = None
        self.read_mask: Optional[torch.Tensor] = None
        self.write_mask: Optional[torch.Tensor] = None
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the GRU and router parameters."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        if self.bias_ih is not None:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
        for router in (self.read_router, self.write_router):
            if router is not None:
                router.reset_parameters()

    def compute_routing(self, x: torch.Tensor) -> Tuple[RouterOutput, RouterOutput]:
        """Compute independent input-conditioned routing without changing state.

        Args:
            x: Input tensor with shape ``(batch, input_size)``.

        Returns:
            Pair of READ and WRITE router outputs, including masks and indices.

        Raises:
            ValueError: If internal router configurations were not supplied.
        """
        if self.read_router is None or self.write_router is None:
            raise ValueError("Router configurations are required to compute routing.")
        return self.read_router(x), self.write_router(x)

    def forward(self,
                x: torch.Tensor,
                hx: Optional[torch.Tensor] = None,
                *,
                read_mask: Optional[torch.Tensor] = None,
                write_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute one reset-before GRU step and apply independent WRITE gating.

        Args:
            x: Input tensor with shape ``(batch, input_size)``.
            hx: Previous state with shape ``(batch, hidden_size)``; defaults
                to zeros with the input's device and dtype.
            read_mask: READ multipliers with shape ``(batch, hidden_size)``.
                Uses the internal READ router when omitted. Fractional values
                are supported by this reference implementation.
            write_mask: WRITE multipliers with shape ``(batch, hidden_size)``.
                Uses the internal WRITE router when omitted. Zero preserves
                the corresponding state exactly, including when its unused
                proposal is nonfinite.

        Returns:
            Pair ``(output, new_state)`` referring to the same tensor with
            shape ``(batch, hidden_size)``.

        Raises:
            ValueError: If shapes are invalid or a required mask is unavailable.
        """
        if x.ndim != 2 or x.size(1) != self.input_size:
            raise ValueError(f"x must have shape (B, {self.input_size}).")
        state_shape = (x.size(0), self.hidden_size)
        h = x.new_zeros(state_shape) if hx is None else hx
        self.last_read_routing = None
        self.last_write_routing = None
        if read_mask is None:
            if self.read_router is None:
                raise ValueError("read_mask is required without read_router_config.")
            self.last_read_routing = self.read_router(x)
            read_mask = self.last_read_routing.mask
        if write_mask is None:
            if self.write_router is None:
                raise ValueError("write_mask is required without write_router_config.")
            self.last_write_routing = self.write_router(x)
            write_mask = self.last_write_routing.mask
        for name, tensor in (("hx", h), ("read_mask", read_mask),
                             ("write_mask", write_mask)):
            if tensor.shape != state_shape:
                raise ValueError(f"{name} must have shape {state_shape}.")
        self.read_mask = read_mask
        self.write_mask = write_mask

        h_read = read_mask * h
        w_ih = self.weight_ih.view(3, self.hidden_size, self.input_size)
        w_hh = self.weight_hh.view(3, self.hidden_size, self.hidden_size)
        bias_ih = ([None] * 3 if self.bias_ih is None
                   else self.bias_ih.view(3, self.hidden_size).unbind())
        bias_hh = ([None] * 3 if self.bias_hh is None
                   else self.bias_hh.view(3, self.hidden_size).unbind())

        reset = torch.sigmoid(F.linear(x, w_ih[0], bias_ih[0])
                              + F.linear(h_read, w_hh[0], bias_hh[0]))
        update = torch.sigmoid(F.linear(x, w_ih[1], bias_ih[1])
                               + F.linear(h_read, w_hh[1], bias_hh[1]))
        content = torch.tanh(F.linear(x, w_ih[2], bias_ih[2])
                             + F.linear(reset * h_read, w_hh[2], bias_hh[2]))
        proposal = (1.0 - update) * content + update * h_read
        proposal = torch.where((write_mask == 0) & ~torch.isfinite(proposal),
                               torch.zeros_like(proposal), proposal)
        h_new = (1.0 - write_mask) * h + write_mask * proposal
        return h_new, h_new


def _gru_affine(x, values, wh, wx, bi, bh, cols, rows, gate):
    """Compute one gate's affine transform at selected rows and columns."""
    active = wh[gate][rows.unsqueeze(2), cols.unsqueeze(1)]
    result = (torch.bmm(active, values.unsqueeze(2)).squeeze(2)
              + torch.bmm(wx[gate][rows], x.unsqueeze(2)).squeeze(2))
    if bi is not None:
        result = result + bi[gate][rows] + bh[gate][rows]
    return result


def _gru_written_proposal(x, q, reset, wh, wx, bi, bh, ri, rows, q_rows):
    """Compute update/content gates and the GRU residual at destination rows."""
    update = torch.sigmoid(_gru_affine(x, q, wh, wx, bi, bh, ri, rows, 1))
    content = torch.tanh(_gru_affine(x, reset * q, wh, wx, bi, bh, ri, rows, 2))
    return update, content, (1 - update) * content + update * q_rows


class _GatheredGRUStep(torch.autograd.Function):
    """Gathered reset-before GRU with reference-equivalent first derivatives."""

    @staticmethod
    def forward(ctx, x, h, read, write, weight_hh, weight_ih, bias_ih, bias_hh,
                ri, wi, full_read_grad, full_write_grad):
        hidden = h.size(1)
        wh = weight_hh.view(3, hidden, hidden)
        wx = weight_ih.view(3, hidden, x.size(1))
        bi = None if bias_ih is None else bias_ih.view(3, hidden)
        bh = None if bias_hh is None else bias_hh.view(3, hidden)
        q = h.gather(1, ri)
        q_rows = (read * h).gather(1, wi)
        reset = torch.sigmoid(_gru_affine(x, q, wh, wx, bi, bh, ri, ri, 0))
        update, content, proposal = _gru_written_proposal(
            x, q, reset, wh, wx, bi, bh, ri, wi, q_rows)
        ctx.has_bias = bias_ih is not None
        ctx.full_read_grad = full_read_grad
        ctx.full_write_grad = full_write_grad
        ctx.save_for_backward(x, h, read, write, weight_hh, weight_ih,
                              bias_ih if bias_ih is not None else x.new_empty(0),
                              bias_hh if bias_hh is not None else x.new_empty(0),
                              ri, wi, reset, update, content, proposal)
        return h.scatter(1, wi, proposal)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, h, read, write, weight_hh, weight_ih, bias_ih, bias_hh, ri, wi, reset, update, content, proposal = ctx.saved_tensors
        hidden = h.size(1)
        wh = weight_hh.view(3, hidden, hidden)
        wx = weight_ih.view(3, hidden, x.size(1))
        bi = bias_ih.view(3, hidden) if ctx.has_bias else None
        bh = bias_hh.view(3, hidden) if ctx.has_bias else None
        q = h.gather(1, ri)
        q_rows = (read * h).gather(1, wi)
        upstream = grad_output.gather(1, wi)
        delta_n = upstream * (1 - update) * (1 - content.square())
        delta_z = upstream * (q_rows - content) * update * (1 - update)
        active_n = wh[2][wi.unsqueeze(2), ri.unsqueeze(1)]
        grad_v = torch.bmm(active_n.transpose(1, 2), delta_n.unsqueeze(2)).squeeze(2)
        delta_r = grad_v * q * reset * (1 - reset)

        # Only an STE READ mask needs derivatives at unread coordinates.
        full_read = ctx.full_read_grad and ctx.needs_input_grad[2]
        cols = (torch.arange(hidden, device=h.device).expand(h.size(0), -1)
                if full_read else ri)
        reset_cols = (torch.sigmoid(_gru_affine(x, q, wh, wx, bi, bh, ri, cols, 0))
                      if full_read else reset)
        grad_q = torch.bmm(
            wh[2][wi.unsqueeze(2), cols.unsqueeze(1)].transpose(1, 2),
            delta_n.unsqueeze(2)).squeeze(2) * reset_cols
        for gate, rows, delta in ((0, ri, delta_r), (1, wi, delta_z)):
            grad_q = grad_q + torch.bmm(
                wh[gate][rows.unsqueeze(2), cols.unsqueeze(1)].transpose(1, 2),
                delta.unsqueeze(2)).squeeze(2)
        residual = torch.zeros_like(h).scatter(1, wi, upstream * update)
        grad_h = grad_output.scatter(1, wi, 0.) + residual * read
        if full_read:
            grad_h = grad_h + grad_q * read
            grad_read = (grad_q + residual) * h
        else:
            grad_h = grad_h.scatter_add(1, ri, grad_q)
            grad_read = (torch.zeros_like(read).scatter(1, ri, grad_q * q)
                         + residual * read * h) if ctx.needs_input_grad[2] else None

        # Scatter parameter gradients from the same small blocks used forward.
        grad_x = torch.zeros_like(x)
        grad_wh, grad_wx, grad_bias = [], [], []
        for gate, rows, delta, values in ((0, ri, delta_r, q),
                                          (1, wi, delta_z, q),
                                          (2, wi, delta_n, reset * q)):
            grad_x = grad_x + torch.bmm(
                wx[gate][rows].transpose(1, 2), delta.unsqueeze(2)).squeeze(2)
            offsets = rows.unsqueeze(2) * hidden + ri.unsqueeze(1)
            grad_wh.append(torch.zeros_like(wh[gate]).reshape(-1).index_add(
                0, offsets.reshape(-1),
                (delta.unsqueeze(2) * values.unsqueeze(1)).reshape(-1)).reshape(hidden, hidden))
            grad_wx.append(torch.zeros_like(wx[gate]).index_add(
                0, rows.reshape(-1),
                (delta.unsqueeze(2) * x.unsqueeze(1)).reshape(-1, x.size(1))))
            if ctx.has_bias:
                grad_bias.append(torch.zeros_like(bi[gate]).index_add(
                    0, rows.reshape(-1), delta.reshape(-1)))
        grad_write = None
        if ctx.needs_input_grad[3]:
            if ctx.full_write_grad:
                all_rows = torch.arange(hidden, device=h.device).expand_as(read)
                other_rows = all_rows[write == 0].reshape(h.size(0), hidden - wi.size(1))
                _, _, other = _gru_written_proposal(
                    x, q, reset, wh, wx, bi, bh, ri, other_rows,
                    (read * h).gather(1, other_rows))
                other = torch.where(torch.isfinite(other), other, torch.zeros_like(other))
                full_proposal = torch.zeros_like(h).scatter(1, wi, proposal).scatter(1, other_rows, other)
                grad_write = grad_output * (full_proposal - h)
            else:
                grad_write = torch.zeros_like(write).scatter(
                    1, wi, upstream * (proposal - h.gather(1, wi)))
        bias_grad = torch.stack(grad_bias).reshape_as(bias_ih) if ctx.has_bias else None
        return (grad_x, grad_h, grad_read, grad_write,
                torch.stack(grad_wh).reshape_as(weight_hh),
                torch.stack(grad_wx).reshape_as(weight_ih), bias_grad, bias_grad,
                None, None, None, None)


class SparseDualReadWriteGRUCell(DualReadWriteGRUCell):
    """Reset-before GRU with gathered READ sources and WRITE destinations.

    Reset gates are evaluated on READ coordinates because they gate recurrent
    sources. Update and content gates are evaluated on WRITE coordinates.
    Forward recurrent products therefore use one ``k_read x k_read`` block
    and two ``k_write x k_read`` blocks. Full STE mask gradients require
    additional gathered products in backward. No runtime speedup is implied.

    Parameters and state-dict keys match ``DualReadWriteGRUCell`` exactly.
    Only first derivatives are supported; autocast and fractional masks are
    unsupported. Unwritten state coordinates are copied exactly.

    Args:
        input_size: Number of input features.
        hidden_size: Number of hidden-state coordinates.
        bias: Whether to include input and recurrent biases for all gates.
        read_router_config: Optional READ router configuration. Both router
            configurations must be supplied together, or both omitted.
        write_router_config: Optional independent WRITE router configuration.

    Raises:
        ValueError: If dimensions or router configurations are invalid.
    """

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None, *,
                read_mask: Optional[torch.Tensor] = None,
                write_mask: Optional[torch.Tensor] = None,
                read_backward: Optional[BackwardRule] = None,
                write_backward: Optional[BackwardRule] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute a gathered GRU step with binary routing.

        Args:
            x: Input with shape ``(batch, input_size)`` and nonempty batch.
            hx: Previous state with shape ``(batch, hidden_size)``; defaults
                to zeros on the input device and with its dtype.
            read_mask: Binary READ mask with shape ``(batch, hidden_size)``.
                Each example must have the same budget, which may be zero.
                Uses the internal READ router if omitted.
            write_mask: Binary WRITE mask with the same shape restrictions.
                Its budget is independent of READ. Uses its router if omitted.
            read_backward: External mask gradient rule: ``ste`` (the default)
                supplies reference gradients at all coordinates;
                ``selected_only`` supplies them only at selected coordinates.
                Internal masks use their router's configured rule.
            write_backward: Corresponding gradient rule for the WRITE mask.

        Returns:
            Pair ``(output, new_state)`` referring to the same tensor with
            shape ``(batch, hidden_size)``.

        Raises:
            ValueError: If autocast is enabled, masks are unavailable or
                nonbinary, budgets differ within a batch, a gradient rule is
                invalid, or shapes, devices, or dtypes are incompatible.
        """
        h, masks, indices, backward_modes = _prepare_sparse_routing(
            self, x, hx, read_mask, write_mask, read_backward, write_backward)
        h_new = _GatheredGRUStep.apply(
            x, h, *masks, self.weight_hh, self.weight_ih, self.bias_ih, self.bias_hh,
            *indices, backward_modes[0] == "ste", backward_modes[1] == "ste")
        return h_new, h_new
