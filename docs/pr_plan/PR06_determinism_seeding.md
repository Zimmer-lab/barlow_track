# PR06 - Determinism and Reproducibility

Suggested branch: `feature/determinism-seeding`

## Goal

Make runs reproducible when a seed is supplied and avoid silent nondeterminism from Python `random`, NNDescent, UMAP, and DataLoader workers.

## Files Likely Touched

- `barlow_track/utils/utils_label_propagation.py`
- `barlow_track/utils/utils_tracking.py`
- `barlow_track/utils/volume_data.py`
- `barlow_track/scripts/eval_accuracy.py`
- possibly config/template files if UMAP options are exposed

## Background

The pipeline uses several random sources:

- Python `random.shuffle` for choosing label-propagation seed times;
- `NNDescent` for kNN graph construction;
- UMAP, if used;
- torchio augmentations;
- DataLoader workers, which may fork the same RNG state.

Current `--seed` arguments set NumPy and PyTorch seeds but do not cover all sources.

## Bugs / Tasks

- [ ] Seed Python's built-in `random` in entry points that use it.
  - Location: `eval_accuracy.py` and any script/notebook runner that seeds only `np.random` and `torch`.
  - Fix: `random.seed(seed)` or, better, pass an explicit RNG into pipeline functions.
  - Acceptance: same seed gives same seed-time selection.

- [ ] Give NNDescent a deterministic `random_state`.
  - Location: `utils_label_propagation.py` kNN graph construction.
  - Fix: add `random_state=seed` or a config-controlled default.
  - Acceptance: same input and seed produce identical neighbor graphs.

- [ ] Seed UMAP where used.
  - Location: tracking config dictionaries and label-propagation tracker.
  - Fix: add `random_state` to UMAP options or expose a global seed.
  - Acceptance: repeated runs with same config produce identical embeddings/labels.

- [ ] Decide on a single seed-propagation strategy.
  - Prefer passing an explicit `random_state`/`rng` into functions instead of relying on global `random.seed`.
  - If global seeding is kept, document it in the entry point.

- [ ] Handle DataLoader worker seeding for augmentation.
  - Location: `volume_data.py`.
  - Current risk: one `RandomState(seed)` shared across epochs/workers can repeat or drift unpredictably.
  - Fix options:
    - reseed per epoch;
    - derive per-worker seeds from base seed, epoch, and worker id;
    - or explicitly state that augmentation is not reproducible across worker counts.
  - Acceptance: documented behavior and, if practical, testable determinism for fixed worker count.

## Suggested Test Coverage

- [ ] Same seed, same synthetic embedding matrix: label-propagation outputs identical.
- [ ] Same seed, same config: seed-time list identical.
- [ ] Same seed, same synthetic project: eval accuracy JSON records identical predictions or labels.
- [ ] Different seeds still produce different results.

## Out of Scope

- Changing clustering algorithm defaults.
- Improving augmentation diversity.
