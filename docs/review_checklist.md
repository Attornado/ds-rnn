# Independent Implementation Review Checklist

Review the current branch as if the implementation had been written by
another researcher.

Do not assume comments, variable names or documentation are correct.
Check behavior from the code itself.

Do not modify code during the first review pass.

---

## 1. Semantic correctness

Check:

- Are read and write masks genuinely independent?
- Does `write=0` preserve the corresponding state exactly?
- Can a `read=0` hidden coordinate still influence the candidate accidentally?
- Is masking applied at the intended mathematical location?
- Are batch dimensions handled correctly?
- Are recurrent semantics consistent across timesteps?
- Are returned/logged masks the actual masks used in computation?
- Does the implementation match `docs/dual_read_write_rnn.md`?

Explicitly test all four states:

- `read=0, write=0`
- `read=1, write=0`
- `read=0, write=1`
- `read=1, write=1`

including mixtures of these within the same batch.

---

## 2. Gradient correctness

Check:

- Do recurrent weights receive the intended gradients?
- Do input weights receive the intended gradients?
- Does the previous hidden state receive gradients through the correct paths?
- Do both router heads receive gradients?
- Does hard top-k block gradients unexpectedly?
- Is STE behavior consistent with its claimed semantics?
- Does gather/scatter accidentally detach tensors?
- Are unselected router logits handled as intended?
- Does the optimized backward path agree with the reference behavior where it
  should?

Construct minimal counterexamples rather than relying only on end-to-end loss.

---

## 3. Sparse-computation correctness

Check:

- Does the optimized path actually avoid an `H x H` recurrent matmul?
- Are both rows and columns gathered?
- Is the active recurrent matrix actually `k_write x k_read`?
- Is a dense candidate secretly computed and masked afterward?
- Is the implementation accidentally materializing a larger tensor than needed?
- Are gather/scatter costs included in runtime measurements?
- Are router costs included in runtime measurements?
- Are FLOP calculations consistent with the actual executed operations?

---

## 4. Numerical equivalence

Compare optimized and reference implementations with identical:

- parameters
- inputs
- masks
- random seeds

Check:

- outputs
- hidden states across multiple timesteps
- recurrent-weight gradients
- input-weight gradients
- previous-state gradients

Use multiple:

- batch sizes
- hidden sizes
- read/write sparsity levels
- dtypes where supported

---

## 5. Baseline integrity

Check:

- Were existing dense baselines changed?
- Were existing DS-RNN baselines changed?
- Were config defaults changed?
- Were dataset splits or preprocessing changed?
- Were training schedules changed?
- Are comparisons using equivalent hidden sizes and router budgets?
- Is any performance improvement partly due to extra parameters rather than
  routing?

---

## 6. Research-claim hygiene

Check that the code/results do not overclaim:

- FLOP savings as wall-clock speedups
- correlation as causal proof
- synthetic-task success as broad sequence-model superiority
- the product-overlap interference story as proven theory unless a rigorous
  derivation exists
- read/write gating as conceptually novel without accounting for prior gated
  recurrent architectures

---

## 7. Missing tests

Try to identify tests that should exist but do not.

Especially look for:

- all-zero read mask
- all-zero write mask
- all-one read mask
- all-one write mask
- `k_read=1`
- `k_write=1`
- batch size 1
- batch size >1
- different masks per batch element
- long multi-step recurrence
- CPU/GPU consistency
- mixed precision if the repository uses it

---

## 8. Review output format

Return findings grouped as:

### CRITICAL
Errors that invalidate the architecture, gradients, or claimed sparse compute.

### MAJOR
Issues that materially affect experiments, fairness, reproducibility or
performance claims.

### MINOR
Code quality, logging, maintainability or small test gaps.

### MISSING TESTS
Specific tests that should be added.

For each finding include:

- exact file
- class/function
- why it matters
- a minimal reproducer or proposed regression test when possible
