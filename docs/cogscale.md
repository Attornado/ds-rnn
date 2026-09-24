# CogScale experiments

## Adding tasks to an existing run

Reuse the same `--log_dir` and `--run_id` with a different `--tasks` selection:

```bash
python -m experiments.train_cogscale \
  --config experiments/configs/cogscale_gru.yaml \
  --run_id development_seed100 --tasks associative_recall --skip_test
```

Keep the original experiment settings, including seed, difficulty, model list,
size overrides, epochs, and `--skip_test`. Changes to those settings require
a new run ID. W&B logging options may change.

Completed task/model pairs are skipped. New results are appended to the shared
`comparison.json`; the task list and completion flag cover the cumulative run.
Previous configurations, hashes, and runtime metadata are retained under
`invocations`. Existing runs created before this extension are supported.
If a pair has files but no completed result row, it is trained again from
scratch in a new `attempts/` subdirectory, preserving the interrupted files.
This does not resume optimizer state or add test evaluation to completed rows.
Concurrent writers to the same run are rejected. Omit `--run_id` for a fresh
automatically named run.

`experiments/train_cogscale.py` orchestrates runs, CLI/YAML configuration,
checkpoint selection, and reporting. `dsrnn/data/cogscale.py` owns data
generation, preset resolution, and override validation;
`dsrnn/training/cogscale.py` owns the model and Lightning training wrapper.

The data extraction preserves the library splits, seeds, target mask, and
preset values. Size options default to `None`, retaining the chosen library
preset. An override is applied only to compatible tasks and must apply to at
least one selected task. `--skip_test` skips test evaluation; the library still
generates the same test split as before.

## Optional W&B logging

CSV metrics and `comparison.json` are always written. W&B is disabled by
default. Install the existing optional extra with `pip install '.[wandb]'`,
authenticate with `wandb login`, then run:

```bash
python -m experiments.train_cogscale \
  --config experiments/configs/cogscale_gru.yaml \
  --skip_test --wandb --wandb_project dsrnn-cogscale
```

Use `--wandb_mode offline` to write local W&B logs without uploading or
authenticating. `--wandb_entity` selects an account/team, and `--wandb_group`
overrides the default group (the local run ID). All flags also work as YAML
keys. `--no-wandb` overrides `wandb: true` in YAML.

Each task/model pair has a separate W&B run. Config contains the effective
experiment parameters, seeds, library version, configuration hash, runtime
and Git metadata, evaluation protocol, and resolved task parameters/shapes.
Lightning sends epoch metrics and learning rates to both loggers. W&B summary
`result` contains the same row written to `comparison.json`, including best
validation metrics, parameter count, routing budgets, optimizer steps, and
optional test scores. Summary `complete` refers to that task/model pair.
Runs finish independently, including on exceptions. Checkpoints stay local.

Logging options are excluded from the scientific configuration hash so that
enabling W&B does not change experiment identity. They are recorded in the
W&B config under `logging`.
