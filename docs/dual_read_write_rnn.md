# Dual Sparse Read/Write RNN

## 1. Goal

The project investigates independent, input-dependent sparse routing for:

- **READ access**: which components of the previous recurrent state may
  participate in candidate-state computation.
- **WRITE access**: which components of the recurrent state may be overwritten.

The key idea is to make these two decisions distinct.

This document is the source of truth for the intended architecture.

---

## 2. Vanilla formulation

Let:

- `h_{t-1} in R^H` be the previous hidden state.
- `x_t in R^D` be the current input.
- `r_t in [0,1]^H` be the read mask.
- `w_t in [0,1]^H` be the write mask.

Read selection:

\[
h_t^{read} = r_t \odot h_{t-1}
\]

Candidate state:

\[
\hat h_t =
\phi(W_h h_t^{read} + W_x x_t + b)
\]

Selective write:

\[
h_t =
(1-w_t)\odot h_{t-1}
+
w_t\odot \hat h_t
\]

Interpretation per hidden coordinate:

| Read | Write | Interpretation |
|---|---|---|
| 0 | 0 | Dormant state; preserved and not consulted |
| 1 | 0 | Read-only / protected memory |
| 0 | 1 | Writable location whose previous value is ignored |
| 1 | 1 | Normal active recurrent state |

The distinction between `(read=1, write=0)` and `(read=0, write=1)` is central
to the project.

---

## 3. Routing

Initially use two separate router heads:

\[
z_t^R = R_R(c_t)
\]

\[
z_t^W = R_W(c_t)
\]

where `c_t` should initially match the context used by the existing DS-RNN
implementation whenever practical.

Recommended initial choice:

\[
c_t = x_t
\]

or the same low-dimensional input projection already used in the repository.

Do **not** initially condition the router on `h_{t-1}` unless required by the
existing implementation, because doing so changes both the computation graph
and the clean Jacobian interpretation.

### Hard top-k budgets

Let:

\[
k_R = \lceil \rho_R H \rceil
\]

\[
k_W = \lceil \rho_W H \rceil
\]

Read and write budgets are independent.

The initial implementation should reuse the same hard-top-k / STE machinery
already used by DS-RNN if available.

The code should expose, at minimum:

- `read_mask`
- `write_mask`
- `read_indices`
- `write_indices`
- router scores/logits when analysis mode is enabled

---

## 4. Reference implementation

The first implementation should prioritize semantic correctness, not speed.

It may compute the full candidate vector:

\[
\hat h_t =
\phi(W_h(r_t\odot h_{t-1}) + W_xx_t + b)
\]

and then apply the write mask:

\[
h_t =
(1-w_t)\odot h_{t-1}
+
w_t\odot\hat h_t
\]

This reference implementation is intentionally not compute efficient.

Its purpose is to serve as a numerical oracle for the optimized
implementation.

### 4.1 GRU reference extension

For long-range experiments, the same routing semantics may wrap a GRU. Define:

\[
h_t^{read} = r_t \odot h_{t-1}
\]

and compute the reset gate, update gate, and new content using
`h_t^{read}` everywhere that the repository's dense GRU uses `h_{t-1}`. The
GRU proposal is:

\[
g_t = (1-z_t) \odot n_t + z_t \odot h_t^{read}
\]

The independently routed write is then:

\[
h_t = (1-w_t) \odot h_{t-1} + w_t \odot g_t
\]

Masking the GRU residual with READ is intentional: an unread source coordinate
must not bypass the recurrent affine transformations through the update-gate
skip path. With all-one masks this equation is exactly the existing dense GRU;
with `write=0`, the previous state is preserved exactly. The first GRU version
is a dense reference implementation. An optimized gathered GRU requires its
own equivalence tests before any efficiency claim.

---

## 5. Efficient two-sided recurrent computation

Define diagonal mask matrices:

\[
D_R = \operatorname{diag}(r_t),
\qquad
D_W = \operatorname{diag}(w_t)
\]

Only written outputs need candidate values, and those candidate values only
need readable previous-state coordinates.

The recurrent interaction of interest is therefore:

\[
D_W W_h D_R
\]

If:

- `read_idx` has `k_R` indices
- `write_idx` has `k_W` indices

then the active recurrent matrix per example is:

\[
W_{active}
=
W_h[\text{write\_idx}, \text{read\_idx}]
\]

with shape:

\[
k_W \times k_R
\]

The active recurrent multiplication is:

\[
\hat h_{write}
=
W_{active}
h_{read}
\]

followed by scattering candidate values back into the full state and
preserving unwritten coordinates exactly.

Ignoring router, gather/scatter, bias and activation overhead, recurrent
matrix multiplication scales as:

\[
O(k_W k_R)
\]

rather than:

\[
O(H^2)
\]

Equivalently:

\[
O(\rho_W \rho_R H^2)
\]

This reduction should only be claimed when the implementation actually avoids
the dense `H x H` recurrent multiply.

---

## 6. Numerical equivalence requirement

For identical parameters, inputs and masks:

\[
\text{optimized\_forward}
\approx
\text{reference\_forward}
\]

within normal floating-point tolerance.

Tests should cover:

- batch size 1
- batch size >1
- mixed read/write patterns
- edge cases for all four `(R,W)` combinations
- multiple timesteps

Gradients with respect to recurrent weights and hidden states should agree
with the intended reference semantics.

Router-gradient behavior under hard top-k / STE should be tested separately.

---

## 7. Jacobian structure

Ignoring derivatives through the router, the recurrent-state Jacobian has the
form:

\[
J_t =
I - D_W
+
D_W J_{cand} D_R
\]

where `J_cand` is the Jacobian of the candidate-state transformation with
respect to `h_{t-1}`.

Interpretation:

- `I - D_W` gives an exact identity path for non-written state.
- `D_W J_{cand} D_R` confines nonlinear recurrent interaction to selected
  write and read dimensions.

This is a research hypothesis / analytical starting point, not a substitute
for empirical validation.

---

## 8. Instantaneous recurrent-weight gradient

For a vanilla recurrent map, ignoring router derivatives, the recurrent
weight gradient has the structural form:

\[
G_t
=
(D_W \delta_t)
(D_R h_{t-1})^\top
\]

For two tasks or examples A and B:

\[
\langle G_A, G_B\rangle
=
\big((D_W^A \delta_A)^\top(D_W^B \delta_B)\big)
\big((D_R^A h_A)^\top(D_R^B h_B)\big)
\]

This motivates the hypothesis that gradient interference may depend on both
read-side and write-side overlap.

Do not treat a product-form expected interference bound as established until
it is derived rigorously and tested empirically.

---

## 9. LSTM / GRU extension policy

Do not begin by changing all architectures.

First establish the idea on the simplest recurrent cell compatible with the
repository.

For GRU/LSTM, explicitly decide:

- what constitutes READ access
- what state is WRITE-protected
- whether masks are shared across gates
- whether cell state and hidden state require separate write semantics
- how the design differs from existing reset/update/input/forget gates

These choices should be documented before implementation.

---

## 10. Non-goals for the initial implementation

Do not initially:

- add custom CUDA kernels
- rewrite the training framework
- redesign every recurrent architecture
- introduce multiple router families
- change existing baseline defaults
- claim wall-clock gains from FLOP estimates alone
- optimize away the reference implementation
