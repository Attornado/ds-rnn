"""Two-sided gathered dual READ/WRITE recurrence with reference mask gradients."""

from typing import Optional, Tuple

import torch
from torch.autograd.function import once_differentiable

from dsrnn.nn.cells.dual_read_write import DualReadWriteRNNCell
from dsrnn.nn.functional import BackwardRule


def _candidate(x, h_read, weight_hh, weight_ih, bias, read_idx, rows):
    """Compute candidate values for selected output rows.

    Args:
        x: Input tensor of shape ``(batch_size, input_size)``.
        h_read: Selected hidden values of shape ``(batch_size, read_budget)``.
        weight_hh: Recurrent weight matrix.
        weight_ih: Input weight matrix.
        bias: Candidate-state bias, or ``None``.
        read_idx: Selected hidden indices.
        rows: Selected output-row indices.

    Returns:
        Candidate values for the selected rows.
    """
    active = weight_hh[rows.unsqueeze(2), read_idx.unsqueeze(1)]
    recurrent = torch.bmm(active, h_read.unsqueeze(2)).squeeze(2)
    input_term = torch.bmm(weight_ih[rows], x.unsqueeze(2)).squeeze(2)
    if bias is not None:
        input_term = input_term + bias[rows]
    return torch.tanh(recurrent + input_term)


def _standard_gathered_step(x, h, read, write, weight_hh, weight_ih, bias,
                            read_idx, write_idx):
    """Apply a gathered step using ordinary autograd.

    This path is sufficient for fixed masks and selected-only routing because
    only selected mask entries need derivatives. It avoids the surrogate
    recomputation required to give STE gradients to unselected entries.
    """
    h_selected = h.gather(1, read_idx)
    h_read = read.gather(1, read_idx) * h_selected
    candidate = _candidate(x, h_read, weight_hh, weight_ih, bias,
                           read_idx, write_idx)
    previous_write = h.gather(1, write_idx)
    write_selected = write.gather(1, write_idx)
    written = ((1.0 - write_selected) * previous_write
               + write_selected * candidate)
    return h.scatter(1, write_idx, written)


class _GatheredDualStep(torch.autograd.Function):
    """Apply a gathered dual read/write step with custom first derivatives."""

    @staticmethod
    def forward(ctx, x, h, read, write, weight_hh, weight_ih, bias, read_idx,
                write_idx, full_read_grad, full_write_grad):
        """Compute a gathered recurrent step.

        Args:
            ctx: Autograd context.
            x: Input tensor.
            h: Previous hidden state.
            read: Binary read mask.
            write: Binary write mask.
            weight_hh: Recurrent weight matrix.
            weight_ih: Input weight matrix.
            bias: Candidate-state bias, or ``None``.
            read_idx: Selected read indices.
            write_idx: Selected write indices.

        Returns:
            The updated hidden state.
        """
        h_read = h.gather(1, read_idx)
        candidate = _candidate(x, h_read, weight_hh, weight_ih, bias, read_idx, write_idx)
        ctx.has_bias = bias is not None
        ctx.full_read_grad = full_read_grad
        ctx.full_write_grad = full_write_grad
        ctx.save_for_backward(x, h, read, write, weight_hh, weight_ih,
                              bias if bias is not None else x.new_empty(0),
                              read_idx, write_idx, candidate)
        # Scatter replacement copies every unwritten coordinate exactly.
        return h.scatter(1, write_idx, candidate)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Compute first derivatives for the gathered recurrent step.

        Args:
            ctx: Autograd context.
            grad_output: Gradient of the updated hidden state.

        Returns:
            Gradients for each argument accepted by :meth:`forward`.
        """
        x, h, read, write, wh, wx, bias, ri, wi, candidate = ctx.saved_tensors
        delta = grad_output.gather(1, wi) * (1 - candidate.square())
        active = wh[wi.unsqueeze(2), ri.unsqueeze(1)]
        h_read = h.gather(1, ri)

        grad_h = grad_output.scatter(1, wi, 0.)
        grad_h = grad_h.scatter_add(1, ri, torch.bmm(
            active.transpose(1, 2), delta.unsqueeze(2)).squeeze(2))
        grad_x = torch.bmm(wx[wi].transpose(1, 2), delta.unsqueeze(2)).squeeze(2)

        offsets = wi.unsqueeze(2) * wh.size(1) + ri.unsqueeze(1)
        grad_wh = torch.zeros_like(wh).reshape(-1).index_add(
            0, offsets.reshape(-1), (delta.unsqueeze(2) * h_read.unsqueeze(1)).reshape(-1))
        grad_wh = grad_wh.reshape_as(wh)
        grad_wx = torch.zeros_like(wx).index_add(
            0, wi.reshape(-1), (delta.unsqueeze(2) * x.unsqueeze(1)).reshape(-1, x.size(1)))
        grad_bias = (torch.zeros_like(bias).index_add(0, wi.reshape(-1), delta.reshape(-1))
                     if ctx.has_bias else None)

        grad_read = grad_write = None
        if ctx.needs_input_grad[2]:
            if ctx.full_read_grad:
                # STE requires derivatives for unread coordinates too.
                gamma = torch.bmm(delta.unsqueeze(1), wh[wi]).squeeze(1)
                grad_read = gamma * h
            else:
                selected = torch.bmm(active.transpose(1, 2),
                                     delta.unsqueeze(2)).squeeze(2) * h_read
                grad_read = torch.zeros_like(read).scatter(1, ri, selected)
        if ctx.needs_input_grad[3]:
            if ctx.full_write_grad:
                # STE requires candidates for unwritten coordinates too.
                all_rows = torch.arange(h.size(1), device=h.device).expand_as(h)
                remaining = all_rows[write == 0].reshape(h.size(0), h.size(1) - wi.size(1))
                other = _candidate(x, h_read, wh, wx, bias if ctx.has_bias else None,
                                   ri, remaining)
                # Match reference protection for non-finite unwritten candidates.
                other = torch.where(torch.isfinite(other), other, torch.zeros_like(other))
                full_candidate = (torch.zeros_like(h).scatter(1, wi, candidate)
                                  .scatter(1, remaining, other))
                grad_write = grad_output * (full_candidate - h)
            else:
                selected = (grad_output.gather(1, wi)
                            * (candidate - h.gather(1, wi)))
                grad_write = torch.zeros_like(write).scatter(1, wi, selected)
        return (grad_x, grad_h, grad_read, grad_write, grad_wh, grad_wx,
                grad_bias, None, None, None, None)


def _binary_indices(mask: torch.Tensor, name: str) -> torch.Tensor:
    """Extract indices from a batched binary mask.

    Args:
        mask: Binary mask of shape ``(batch_size, hidden_size)``.
        name: Mask name used in validation errors.

    Returns:
        Selected indices of shape ``(batch_size, budget)``.

    Raises:
        ValueError: If the mask is nonbinary, empty, or has unequal per-example
            budgets.
    """
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must be binary for gathered execution; use the reference for fractional masks.")
    counts = mask.count_nonzero(dim=1)
    if counts.numel() == 0 or not torch.all(counts == counts[0]):
        raise ValueError(f"{name} requires a nonempty batch with equal per-example budgets.")
    return mask.nonzero(as_tuple=False)[:, 1].reshape(mask.size(0), int(counts[0]))


class SparseDualReadWriteRNNCell(DualReadWriteRNNCell):
    """Dual read/write cell using gathered recurrent computation.

    Args:
        input_size: Number of input features.
        hidden_size: Number of recurrent-state features.
        bias: Whether to include a candidate-state bias.
        read_router_config: Configuration for the read router.
        write_router_config: Configuration for the write router.
    """

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None, *,
                read_mask: Optional[torch.Tensor] = None,
                write_mask: Optional[torch.Tensor] = None,
                read_backward: Optional[BackwardRule] = None,
                write_backward: Optional[BackwardRule] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute one gathered recurrent step.

        Args:
            x: Input tensor of shape ``(batch_size, input_size)``.
            hx: Previous state of shape ``(batch_size, hidden_size)``. Defaults
                to zeros.
            read_mask: Binary read mask with an equal budget per example. Uses
                the read router when omitted.
            write_mask: Binary write mask with an equal budget per example. Uses
                the write router when omitted.
            read_backward: Surrogate rule for an externally supplied read mask.
                Differentiable overrides default to full STE-compatible gradients.
            write_backward: Surrogate rule for an externally supplied write mask.
                Differentiable overrides default to full STE-compatible gradients.

        Returns:
            A pair containing the output and new hidden state.

        Raises:
            ValueError: If autocast is enabled, an input is invalid, or a
                required mask is unavailable.
        """
        if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
            raise ValueError("Gathered dual recurrence does not support autocast.")
        if x.ndim != 2 or x.size(1) != self.input_size or x.size(0) == 0:
            raise ValueError(f"x must have shape (B, {self.input_size}) with B > 0.")
        state_shape = (x.size(0), self.hidden_size)
        h = x.new_zeros(state_shape) if hx is None else hx
        self.last_read_routing = self.last_write_routing = None
        indices = []
        masks = []
        backward_modes = []
        for name, mask, router, requested_backward in (
                ("read", read_mask, self.read_router, read_backward),
                ("write", write_mask, self.write_router, write_backward)):
            if requested_backward not in (None, "ste", "selected_only"):
                raise ValueError(f"Unknown {name}_backward rule {requested_backward!r}.")
            if mask is None:
                if router is None:
                    raise ValueError(f"{name}_mask is required without {name}_router_config.")
                result = router(x)
                setattr(self, f"last_{name}_routing", result)
                mask, idx = result.mask, result.idx
                backward_mode = router.config.backward
            else:
                if mask.shape != state_shape:
                    raise ValueError(f"{name}_mask must have shape {state_shape}.")
                idx = _binary_indices(mask, name + "_mask")
                # Differentiable overrides historically matched the reference
                # mask gradients. Callers can request selected-only explicitly.
                backward_mode = requested_backward or "ste"
            if not mask.requires_grad:
                backward_mode = None
            masks.append(mask)
            indices.append(idx)
            backward_modes.append(backward_mode)
        for name, tensor in (("hx", h), ("read_mask", masks[0]), ("write_mask", masks[1])):
            if tensor.shape != state_shape:
                raise ValueError(f"{name} must have shape {state_shape}.")
            if tensor.device != x.device or tensor.dtype != x.dtype:
                raise ValueError(f"{name} must match x dtype and device.")
        self.read_mask, self.write_mask = masks
        full_read_grad = backward_modes[0] == "ste"
        full_write_grad = backward_modes[1] == "ste"
        if full_read_grad or full_write_grad:
            h_new = _GatheredDualStep.apply(
                x, h, *masks, self.weight_hh, self.weight_ih, self.bias,
                *indices, full_read_grad, full_write_grad)
        else:
            h_new = _standard_gathered_step(
                x, h, *masks, self.weight_hh, self.weight_ih, self.bias, *indices)
        return h_new, h_new
