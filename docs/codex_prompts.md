# Codex Workflow Prompts

These prompts are intended to be used in separate Codex sessions inside the IDE.

---

## Prompt 1 — Repository mapping only

Read `AGENTS.md` and `docs/dual_read_write_rnn.md`.

Study this repository carefully.

Do not modify any files yet.

I want you to determine how the dual sparse read/write recurrent architecture
described in the design document should fit into this codebase.

Find:

- existing dense RNN/LSTM/GRU implementations
- DS-RNN implementation
- router implementation
- hard top-k and STE implementation
- gathered sparse matmul implementation
- model/config factory
- experiment entry points
- relevant tests

Then propose the smallest implementation plan.

For every proposed change, name the exact file, class/function and reason.

Also identify any places where the existing implementation conflicts with the
mathematical design document.

Do not implement anything yet.

---

## Prompt 2 — Reference implementation only

Implement only Phase 1 of the approved plan:

the mathematically correct reference dual-read-write recurrent cell.

Do not implement gathered sparse matrix multiplication yet.

Requirements:

- preserve every existing baseline unchanged
- expose read and write masks independently
- `write=0` must preserve the corresponding hidden state exactly
- add unit tests for the semantics in `docs/dual_read_write_rnn.md`
- add a forward/backward smoke test
- keep the implementation deliberately straightforward

After making the changes, run the relevant tests.

Then report:

1. files changed
2. tests run
3. test results
4. any unresolved concerns

Do not proceed to router optimization or sparse kernels.

---

## Prompt 3 — Add independent routers

Implement the next phase only:

add independent READ and WRITE router heads to the reference model.

Requirements:

- reuse existing DS-RNN routing utilities where appropriate
- preserve separate read and write budgets
- support hard top-k
- reuse existing STE / selected-only behavior if compatible
- expose masks, selected indices and router scores for analysis
- do not implement two-sided sparse matmul yet
- do not change baseline defaults

Add tests that verify:

- exact top-k counts
- read/write masks can differ
- both router heads receive gradients
- behavior is stable for batch size >1

Run tests and report changed files, results and unresolved concerns.

---

## Prompt 4 — Adversarial second pass

START THIS IN A FRESH CODEX CONVERSATION.

Read:

- `AGENTS.md`
- `docs/dual_read_write_rnn.md`
- `docs/review_checklist.md`

Act as an independent skeptical reviewer.

Review the current branch relative to its parent branch.

DO NOT modify files yet.

Assume the implementation may contain subtle mathematical, autograd, batching
or sparsity errors.

Inspect the actual code and tests.

Return findings grouped as:

CRITICAL
MAJOR
MINOR
MISSING TESTS

For every finding, cite the exact file and relevant function/class.

Try to falsify the implementation rather than justify it.

Only after I review your findings will we decide what should be changed.

---

## Prompt 5 — Fix selected review findings

Fix only the review findings I explicitly list below.

For each fix:

- make the smallest necessary change
- add a regression test that would have failed before the fix
- preserve existing baselines and defaults

Findings to fix:

[PASTE SELECTED FINDINGS HERE]

Afterward run the relevant tests and report exactly what changed.

---

## Prompt 6 — Optimized two-sided gathered recurrent matmul

Implement the optimized two-sided sparse recurrent path described in
`docs/dual_read_write_rnn.md`.

Important requirements:

- keep the reference implementation intact
- gather both WRITE rows and READ columns
- the active recurrent matrix per example must be `k_write x k_read`
- do not compute the full `H x H` recurrent matmul and mask afterward
- preserve exact unwritten-state identity semantics
- preserve router-gradient behavior

Before changing code, identify the exact optimized tensor shapes you intend to
use.

Then implement the optimized path and add equivalence tests against the
reference implementation for:

- outputs
- hidden states over multiple timesteps
- recurrent-weight gradients
- previous-state gradients

Use multiple batch sizes and sparsity levels.

Run tests and report results.

---

## Prompt 7 — Sparse-compute audit

START THIS IN A FRESH CODEX CONVERSATION.

Audit the optimized dual read/write implementation specifically for compute
claims.

Do not modify files.

Determine from the actual operations whether the recurrent path avoids the full
dense `H x H` matrix multiplication.

Check:

- gathered tensor shapes
- whether both row and column sparsity are realized
- whether any hidden dense fallback remains
- whether large per-example weight tensors are materialized
- router/gather/scatter overhead
- whether FLOP accounting matches implementation

Return:

1. actual asymptotic recurrent-matmul cost
2. likely memory bottlenecks
3. likely GPU bottlenecks
4. any misleading compute claims
5. recommended profiling experiments

---

## Prompt 8 — Synthetic selective-memory benchmark

Read `docs/experiment_plan.md`.

Implement only the synthetic selective-memory experiment.

The benchmark must contain cases where the model needs to consult one memory
while modifying another.

Compare:

- dense recurrent baseline
- read-only DS baseline
- write-only selective-update baseline
- dual read/write model

Log enough information to measure:

- read-mask activity
- write-mask activity
- frequencies of the four `(R,W)` states
- alignment of those states with STORE / QUERY / UPDATE operations

Do not modify existing benchmark methodology.

Provide a minimal smoke-test configuration first before adding large sweeps.

---

## Prompt 9 — Final research-code review

START THIS IN A FRESH CODEX CONVERSATION.

Review the full branch as a research artifact.

Check:

- architecture correctness
- gradient correctness
- baseline integrity
- experiment fairness
- reproducibility
- logging sufficiency
- sparse-compute validity
- claims supported by code/results
- unnecessary complexity

Do not modify files.

Return:

- blocking issues
- non-blocking issues
- experiments still needed before making claims
- code that should be simplified before merging
