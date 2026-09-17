# AGENTS.md

## Project

This repository contains research code for Dynamic-Sparse recurrent neural networks
and related sequence models.

The current research branch investigates independent sparse READ and WRITE routing
for recurrent hidden state.

The mathematical specification is:

    docs/dual_read_write_rnn.md

Read that document before modifying recurrent-model code.

## Research principles

1. Preserve all existing dense and DS-RNN baselines.
2. Do not silently change existing experiment defaults.
3. Prefer adding a new model/configuration over changing baseline behavior.
4. Mathematical correctness takes priority over optimization.
5. Do not introduce an optimized sparse implementation until an equivalent
   reference implementation exists and is tested.
6. Keep read and write routing conceptually and programmatically distinct.
7. Any claimed FLOP reduction must correspond to computation that is actually
   skipped, not merely multiplied by zeros.
8. Do not change experiment methodology, evaluation metrics, dataset splits,
   or baseline hyperparameters without explicitly reporting the change.
9. Avoid introducing unnecessary dependencies.
10. Follow existing repository conventions and avoid unrelated refactors.

## Required semantics

READ mask `r_t`:
controls which components of `h_{t-1}` may contribute to candidate-state
computation.

WRITE mask `w_t`:
controls which components of the recurrent state may be overwritten.

For a vanilla recurrent cell the intended semantics are:

    h_read = r_t * h_prev

    h_candidate =
        phi(W_h h_read + W_x x_t + b)

    h_new =
        (1 - w_t) * h_prev
        + w_t * h_candidate

Therefore:

    write=0 => the corresponding state must be preserved exactly.

Read and write masks MUST NOT be collapsed into one gate.

## Implementation policy

Implement incrementally:

1. reference dual-read-write cell
2. independent read/write routers
3. hard top-k routing
4. semantic and gradient tests
5. optimized two-sided gathered matmul
6. optimized/reference equivalence tests
7. benchmarks and profiling
8. mechanistic analyses

Do not skip directly to optimized sparse computation.

## Verification

Before declaring an implementation complete:

- run relevant existing tests
- run new model unit tests
- run a forward/backward smoke test
- verify tensor shapes for batch sizes greater than 1
- verify CPU/GPU device handling
- verify gradients reach both routers
- verify requested top-k budgets exactly
- verify `write=0` preserves state exactly
- compare optimized implementation numerically against the reference
- check that the optimized sparse path does not secretly execute the full HxH
  recurrent matmul
- report any tests that could not be run and why

## Source of truth

Architecture:

    docs/dual_read_write_rnn.md

Experimental hypotheses and baselines:

    docs/experiment_plan.md

Independent review procedure:

    docs/review_checklist.md

Codex workflow prompts:

    docs/codex_prompts.md
