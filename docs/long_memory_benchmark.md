# Long-delay selective memory

This implements the next controlled stage of Experiments A/C/E. The six-step
task and its entry point are unchanged. All five recurrent variants and their
full post-write-state readout retain their existing semantics. No readout mask
or operation-dependent WRITE budget is introduced.

Every program initializes all registers in shuffled order, then performs random
UPDATE, QUERY, and joint QUERY A + UPDATE B operations with distinct addresses.
A random register receives a new value at a random time and is queried after
exactly d intervening operations. That register is never overwritten during
this interval. The other registers continue to receive updates and supervised
queries. Standalone distractor UPDATEs make the target write indistinguishable
from ordinary UPDATEs in the input. Target flags and ages are labels only.
One protected interval occurs per program; this is still a controlled task,
not evidence of general register-machine competence.

Training uses a balanced mixture of delays 0, 8, 32, 128 in sequences of length
154 (8 initialization steps + horizon 128 + 18). Source and query positions
vary within this common horizon; all timesteps are real operations, not padding.
Held-out evaluation reports each delay separately. Delay 256 uses length 282;
an additional delay-0 condition at length 282 controls for a longer context.
All models receive the same data and loader ordering within a seed. Generated
train/validation/test streams have separate seeds; changing one split's size
does not change the others. The seed varies both data and initialization.

## Explicit methodology changes

- Objective: protected-target MSE only, averaged over target queries and value
  dimensions. Distractor operations remain in the input, but their query outputs
  receive no direct supervision. Gradients still flow through intervening
  recurrent steps from the target query. `distractor_mse`, `query_mse`, and the
  legacy `balanced_mse` are diagnostic metrics only.
- Checkpoint selection and plateau scheduling both monitor `val/target_mse`.
  This explicitly replaces the earlier 50/50 target/distractor objective for all
  long-memory configurations. Use fresh run IDs; previous results used a
  different objective. The manifest/config hash records `loss: target_mse`,
  and comparison rows now report `best_validation_target_mse`.
- Train 50 epochs with no early stopping, 2048 sequences per epoch, batch 32.
  This is 64 updates and 315392 input tokens between validation checks, and
  15769600 tokens / 3200 updates per model in total. Plateau patience is 5
  validation checks. All variants use this budget; best checkpoints can occur
  at different steps. The inherited early_stopping_patience option is unused.
- These budgets are not matched to the old six-step experiment. For future
  horizon sweeps, choose n_train and epochs to preserve total tokens and tokens
  between validations, and report changes in optimizer steps.
- Metrics include target/distractor error, delay-resolved target error, pure
  versus joint target-query error, protected-readable frequency, mean writes
  per coordinate, and mean longest contiguous protected run per coordinate.
  Conditional error/frequency metrics are weighted by their query counts.
- Full-test metrics are computed over every sample. Only the first test batch
  per condition is saved as a routing trace to bound memory/disk consumption.
  Source-write/read/protection overlap for the target uses indexed source
  masks and cumulative write counts, avoiding the older cubic-time diagnostic.
  Trace metadata includes latest source timestep and actual query age for
  subsequent analysis. No causal memory-encoding claim follows from masks.
- Single-device execution only. No new dependencies. No sparse speedup claims;
  the primary configuration uses the reference implementation. Router counts
  and active recurrent-interaction budgets still differ across controls.
- Fixed train/validation programs are materialized once and reused by all model
  variants. The runner prints this preparation stage because long-horizon data
  construction can take around a minute before Lightning starts training.
  Held-out condition datasets are likewise reused across variants.
- Gradient-vs-write-count analysis from Experiment E remains a later analysis;
  this stage tests behavioral retention first.

## GRU development comparison

The GRU extension uses the same five routing variants and the same full
post-write-state readout. READ masks every recurrent source path in the GRU,
including its gated residual, and WRITE controls whether each destination
coordinate commits the GRU proposal. Dense masks therefore reproduce the
existing dense GRU exactly, while `write=0` still preserves state exactly.

`long_memory_gru_reference.yaml` changes only the recurrent cell family and
the result directory relative to `long_memory_reference.yaml`. It deliberately
keeps the already inspected seed and uses `--skip_test` for development. The
GRU currently has only a dense reference implementation; no sparse speed or
FLOP claim applies.

Because that reference run contains 537.6 million recurrent timesteps across
five models, `long_memory_gru_pilot.yaml` is the practical first check. It keeps
the 128/256/512/1024 delays and hidden size 128, but compares only dense and
dual for 10 epochs on 512 training programs. This smaller budget is an explicit
development-only methodology change. If neither model beats the trivial
baseline, do not spend the full reference budget yet.

## Launch

From the repository root, with the project environment active:

```bash
python -m experiments.train_long_memory --config experiments/configs/long_memory_smoke.yaml
python -m experiments.train_long_memory --config experiments/configs/long_memory_reference.yaml --seed 100 --skip_test --run_id development_seed100
python -m experiments.train_long_memory --config experiments/configs/long_memory_gru_pilot.yaml --skip_test --run_id development_seed100
python -m experiments.train_long_memory --config experiments/configs/long_memory_gru_reference.yaml --skip_test --run_id development_seed100
```

Inspect development validation curves first. Freeze methodology before running
fresh confirmatory seeds (101–105 proposed; do not reuse previously inspected
seeds 42–47). The runner evaluates best checkpoints automatically:

```bash
for seed in 101 102 103 104 105; do
  python -m experiments.train_long_memory \
    --config experiments/configs/long_memory_reference.yaml \
    --seed "$seed" --run_id "long_reference_seed${seed}" || break
done
```

Results live under `_results/long_memory/reference/<run_id>/comparison.json`,
with per-model `evaluation` entries keyed by delay and horizon. The manifest
records the objective, token budget, configuration, and runtime. Completed
variants are saved incrementally; `complete: false` means an incomplete run.
Report all five seeds including failures, seed-level variation, and paired
differences against shared and dense. Do not pool this task with the old task
or tune the protocol using test outcomes.
