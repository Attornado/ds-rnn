# Dual Read/Write Routing: Experiment Plan

## 1. Main research question

Does independently controlling recurrent READ and WRITE access provide
capabilities, optimization behavior, or efficiency benefits that are not
captured by:

- dense recurrence
- read-only DS-style routing
- write-only selective updating

The experiments should distinguish architectural capability from mere
regularization or parameter-count effects.

---

## 2. Core baselines

At minimum compare:

1. Dense recurrent model
2. Read-only DS model
3. Write-only selective-update model
4. Dual read/write model

Wherever possible, match:

- hidden size
- training schedule
- optimizer
- batch size
- seeds
- data splits
- router capacity
- active compute budget

Any unavoidable mismatch must be reported.

---

## 3. Experiment A: synthetic selective memory

### Purpose

Test whether independent read and write access enables behaviors that a single
mask conflates.

### Example sequence operations

- `STORE A value`
- `STORE B value`
- `QUERY A`
- `UPDATE B value`
- `QUERY A`
- `QUERY B`

Construct cases where one memory must be consulted while another is modified.

### Primary behavioral hypothesis

The dual model should learn meaningful use of:

\[
R=1,\;W=0
\]

for protected-but-readable memory.

It may also use:

\[
R=0,\;W=1
\]

for scratch / overwrite locations whose previous value is irrelevant.

### Measurements

Log per timestep:

- read mask
- write mask
- read indices
- write indices
- router scores
- task loss
- optional hidden-state summaries

Measure the empirical frequency of:

- `(R=0, W=0)`
- `(R=1, W=0)`
- `(R=0, W=1)`
- `(R=1, W=1)`

Also measure how these states align with symbolic operations such as STORE,
QUERY and UPDATE.

---

## 4. Experiment B: correctness-to-efficiency transition

### Purpose

Verify that the optimized implementation reproduces the reference model while
actually reducing recurrent matrix work.

### Compare

- dense reference dual-routing implementation
- optimized two-sided gathered implementation

### Sweep

- hidden size
- sequence length
- batch size
- read sparsity
- write sparsity

Suggested initial hidden sizes:

- 64
- 128
- 256
- 512

### Metrics

- output numerical error vs reference
- gradient numerical error vs reference
- theoretical recurrent FLOPs
- measured wall-clock runtime
- peak memory
- router overhead
- gather/scatter overhead

Do not equate theoretical FLOP reduction with runtime speedup.

---

## 5. Experiment C: read/write ablation

Test:

- read-only routing
- write-only routing
- shared single mask for read and write
- independent read/write masks

This is crucial: if the independent-mask model does not outperform or exhibit
different learned behavior from a shared-mask control, then the core claim is
weaker.

---

## 6. Experiment D: multi-task interference

### Purpose

Test whether read and write overlap explain interference differently.

Measure per task pair:

- read-mask overlap
- write-mask overlap
- product of read and write overlap
- recurrent-weight gradient cosine similarity
- hidden-state error-signal cosine similarity
- task performance

Compare which routing statistic best predicts interference:

1. read overlap alone
2. write overlap alone
3. product of read and write overlap

This tests the structural hypothesis suggested by the dual-sided gradient
form.

---

## 7. Experiment E: long-range memory

### Purpose

Test the write-protection mechanism directly.

Use tasks with long periods in which stored information should remain
unchanged while unrelated computation continues.

Possible controlled tasks:

- delayed copy
- selective copy
- associative recall
- synthetic register-machine tasks

Measure:

- task performance vs sequence length
- number of writes per hidden coordinate
- average duration of protected state
- gradient norm as a function of effective write count

---

## 8. Experiment F: compatibility with DS-RNN benchmarks

Only after the controlled experiments work, evaluate on existing repository
benchmarks.

The goal here is comparison, not discovery.

Preserve baseline methodology and report:

- dense
- DS read-only
- dual read/write
- optional write-only baseline

At matched sparsity / compute budgets where meaningful.

---

## 9. Statistical reporting

Use the same seed and confidence-interval conventions as the repository unless
there is a documented reason to change them.

Never select sparsity values using the test set.

Report failed runs rather than silently excluding them.

---

## 10. Stop conditions

Do not proceed to broad benchmark scaling if:

- the reference and optimized implementations disagree
- write protection is not exact
- gradients do not reach both router heads
- the synthetic selective-memory task does not show meaningful independent
  read/write behavior
- the optimized implementation still performs the full dense recurrent matmul
