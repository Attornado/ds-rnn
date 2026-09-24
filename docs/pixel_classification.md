# Pixel classification

Run one task and one model per invocation, from the repository root:

```bash
python -m experiments.train_pixels --task sMNIST --model dense
python -m experiments.train_pixels --task psMNIST --model dual --recurrent_cell gru
python -m experiments.train_pixels --task sCIFAR --model dual --implementation sparse
```

Tasks use one scalar pixel per timestep: sMNIST and psMNIST have 784 steps;
sCIFAR uses the existing grayscale CIFAR-10 datamodule with 1,024 steps.
All inputs are scaled to [0, 1], without augmentation. psMNIST uses one fixed
permutation for every image and every split, controlled by `--permutation_seed`.
The datamodule's former random-order implementation attached transforms to
`Subset` objects, leaving training/validation unpermuted; this is now corrected.

The official training set is split into 90% training and 10% validation by
default (`--val_split`, `--split_seed`). The official test set is reserved for
evaluation. Splits persist across repeated Lightning setup calls. Existing
datamodule constructor defaults, including MNIST's 28-pixel chunks, remain unchanged;
this runner explicitly selects one-pixel chunks and seeded splits.

Models are defined in `experiments/train_pixels.py`. Routing choices match
CogScale: `dense`, `read_only`, `write_only`, `shared`, and `dual`, with `rnn` or
`gru` cells and `reference` or `sparse` implementations. The classifier projects
the complete final hidden state to ten logits. Cross-entropy is applied once per
image. READ and WRITE budgets, router gradient rule, and context width use the
same flags as `train_cogscale`.

Training uses AdamW, ReduceLROnPlateau on validation loss, and a best-validation-loss
checkpoint, without early stopping. Validation and optional test metrics are
computed from that checkpoint. Use `--skip_test` to omit testing. `--config`
loads YAML settings; explicit CLI flags take precedence. Use singular `task`
and `model` keys in YAML.

CSV metrics, checkpoints, configuration/runtime metadata (`manifest.json`), and
final metrics (`results.json`) are saved under `--log_dir` (default `_results/pixels`).
A run trains exactly one model; an existing `--run_id` is rejected to preserve
previous outputs. Optional W&B logging uses `--wandb` and the same logging flags
as CogScale. Both runners share helpers in `dsrnn/utils/experiment.py`.
