# CogScale memory benchmarks

Separate entry point: `python -m experiments.train_cogscale`. Uses the installed
[CogScale library](https://github.com/Naowak/CogScale), tested with 0.2.0, for
simple copy, selective copy, induction heads, and associative recall. Task
presets and generators come from the library; no local reimplementation.

All four tasks use classification: cross-entropy on designated targets only.
A timestep is scored if any target channel differs from -100, matching
`cogscale.compute_score`. Target classes are obtained by argmax (some library
tasks leave -100 in the other class channels). No auxiliary distractor loss.
Validation target cross-entropy controls checkpoint selection and scheduling
(patience 5). Evaluation reports target error rate and the library's official
score (lower is better). Results are separate for each task; no pooled ranking.

The installed library generates fixed arrays per build_task call, not a fresh
stream each epoch. All variants share identical arrays and shuffle ordering.
`split_seed` controls data; `seed` controls model initialization. The library
uses a single sequential RNG stream for its splits, so changing split sizes
changes subsequent splits. `--skip_test` prevents evaluation, although the
library still constructs test arrays as part of build_task. No test scores
select checkpoints. Each task/model receives an independent fresh model.

This differs from the old custom register task: classification replaces MSE,
the library defines the scored targets and sequence structure, and the shipped
config uses medium difficulty, 1000/200/1000 samples and 20 epochs. Hidden size
is matched across routing variants; router parameter counts and compute differ.
GRU is reference-only, with no sparse speed claim. Full post-write state feeds
the output layer as in existing experiments. Old entry points remain available.

Install if needed in the active project environment:

```bash
pip install 'cogscale==0.2.0'
```

Development (four tasks, all five routing variants):

```bash
python -m experiments.train_cogscale --config experiments/configs/cogscale_gru.yaml --skip_test --run_id development_seed100
```

Quick pipeline check (not a learning experiment):

```bash
python -m experiments.train_cogscale --difficulty small --models dense dual --epochs 1 --n_train 16 --n_val 8 --n_test 8 --hidden_size 16 --skip_test --run_id smoke
```

After freezing methodology, omit `--skip_test` for fresh confirmatory seeds and
unique run IDs. Inspect per-task CSV curves, checkpoints, manifest, and
comparison.json. Manifests include the library version, effective task
parameters, actual input/target shapes, objective, and runtime/source metadata.
Each comparison row includes `validation` with target cross-entropy, error rate,
and accuracy evaluated at the checkpoint selected by validation cross-entropy.
This is populated even with `--skip_test`; `evaluation` remains reserved for
held-out test results. These are metrics of the same checkpoint, not independent
best values from different epochs. Validation is rerun once after each fit.
CogScale copy sequence_length parameters may differ from returned total length;
use recorded shapes for interpreting sequence length.

## Optional task sizes

The following CLI flags (or YAML keys) override CogScale's chosen difficulty
preset. All default to `None`; omitted/null values retain the library preset,
independently for each task. Explicit CLI values override YAML values.

| Flag | Applies to | Meaning |
|---|---|---|
| `--sequence_length` | All four tasks | CogScale sequence length parameter |
| `--n_symbols` | All four tasks | Vocabulary size |
| `--delay` | Simple/selective copy | Delay before output trigger |
| `--n_markers` | Selective copy | Number of marked tokens to reproduce |
| `--num_pairs` | Associative recall | Number of stored key-value pairs |

Overrides apply to every compatible selected task. A task-specific flag is
ignored for other tasks in the same run; a flag that applies to none of the
selected tasks is rejected. Combined sizes are validated before creating the
run directory. The console and manifest record effective task sizes.

For simple copy, the total sequence is `2*sequence_length + delay + 1`.
For selective copy it is `sequence_length + delay + 1 + n_markers`.
For induction heads, sequence_length must be even and at least 4.
Associative recall requires `sequence_length >= 2*num_pairs + 2` and
`n_symbols >= num_pairs + 2` (the vocabulary includes two reserved tokens).
Selective copy requires `1 <= n_markers <= sequence_length`.

A short copy experiment keeping all other medium preset parameters:

```bash
python -m experiments.train_cogscale \
  --config experiments/configs/cogscale_gru.yaml \
  --tasks simple_copy --sequence_length 4 --n_symbols 3 --delay 2 \
  --models dense dual --skip_test --run_id short_copy_seed100
```

For all four tasks at custom sizes:

```bash
python -m experiments.train_cogscale \
  --config experiments/configs/cogscale_gru.yaml \
  --sequence_length 8 --n_symbols 6 --delay 2 --n_markers 2 --num_pairs 3 \
  --skip_test --run_id custom_sizes_seed100
```

Split-size settings (`n_train`, `n_val`, `n_test`) keep their existing runner/config
defaults; these new options control task structure, not dataset sample counts.
