# PR02 - Label Propagation and Spectral Relabeling Reliability

Suggested branch: `bugfix/label-propagation-reliability`

## Goal

Make multi-seed label propagation and cross-run label alignment robust and predictable on edge cases.

## Files Likely Touched

- `barlow_track/utils/utils_label_propagation.py`
- `barlow_track/utils/utils_spectral_relabeling.py`
- possibly `barlow_track/utils/utils_tracking.py` only where it passes parameters into these functions

## Background

`utils_label_propagation.py` propagates seed labels through a kNN graph over embeddings and aligns multiple seed-run labelings. `utils_spectral_relabeling.py` then synchronizes labels across seed runs and produces a final consensus labeling.

The review found several correctness traps. Some are currently latent because default arguments avoid them, but they are likely to be encountered as soon as experiments vary thresholds, seeds, or graph connectivity.

## Bugs / Tasks

- [ ] Return thresholded predictions from label propagation.
  - Location: `utils_label_propagation.py`, near the end of the second `run_label_propagation`.
  - Current code pattern:
    ```python
    pred_labels[max_probs < prob_thresh] = -1
    pred = out.argmax(dim=-1)
    return pred, max_probs
    ```
  - Fix: return `pred_labels`.
  - Acceptance: low-confidence nodes are `-1` when `prob_thresh` is active.

- [ ] Remove ghost label 0 from clamped label propagation.
  - Location: seed construction and one-hot matrix construction in `utils_label_propagation.py`.
  - Current risk: seed labels are 1-based, but `num_classes = y.max() + 1` creates an unused class 0. Disconnected nodes can argmax to 0.
  - Downstream: `utils_spectral_relabeling.py` assumes 1-based labels and may raise on label 0.
  - Fix: use `num_classes = y.max()` and map seed labels to zero-based columns internally, or explicitly prevent class 0.
  - Acceptance: disconnected nodes never receive label 0.

- [ ] Do not force zero-evidence label matches in `align_pair`.
  - Location: `utils_label_propagation.py`, `align_pair`.
  - Current risk: Hungarian matching is complete and may map a new cluster to an unrelated old label when the confusion-matrix entry is zero.
  - Fix: drop matched pairs with `cm[r, c] == 0` or below an evidence threshold, and route them through the unmatched-new path.
  - Acceptance: a zero-overlap new label gets a fresh global ID instead of stealing an existing ID.

- [ ] Consider new labels that never co-occur with the reference labeling.
  - Location: `utils_label_propagation.py`, `align_pair`.
  - Current code computes `labels_new` only from positions where both reference and new labels are valid.
  - Fix: compute new labels from all valid new-label entries and assign fresh IDs to labels missing from the mapping.
  - Acceptance: clusters existing only where reference is `-1` are retained.

- [ ] Make `align_all` handle empty and single-labeling inputs.
  - Current risk: single labeling can hit `NameError` for `ref_confidence`; empty labelings can return wrong arity.
  - Fix: initialize defaults and always return a consistent 3-tuple.
  - Acceptance: `align_all([lab])` and `align_all([])` do not crash.

- [ ] Guard empty time points in `fuse_labels_per_time`.
  - Current risk: `np.vectorize` on empty input can raise.
  - Fix: skip empty time points or pass `otypes=[np.int64]`.
  - Acceptance: time point with all `-1` labels is handled.

- [ ] Do not discard all but the top-1 label in spectral relabeling when top-k data exists.
  - Location: `utils_spectral_relabeling.py`, near `row_labels = labels_topk[rows, 0]`.
  - Fix: flatten `(object, topk_candidate)` entries and choose best still-free label per time point.
  - Acceptance: if top-1 label is taken but top-2 is free, the object can be labeled with top-2.

- [ ] Allow single-object time points to be labeled.
  - Location: `utils_spectral_relabeling.py`, guard `if len(object_indices) <= 1: continue`.
  - Fix: skip only for zero objects. For one object, no matching conflict exists.
  - Acceptance: single-object frames can receive labels when probabilities pass thresholds.

- [ ] Clean duplicate/dead `run_label_propagation` definitions.
  - Current risk: first definition is overwritten by second; API is confusing.
  - Fix: delete the old definition.
  - Acceptance: grep shows one active `def run_label_propagation`.

- [ ] Document or fix null-cost units in spectral consensus.
  - Current risk: null/dummy costs compare against unnormalized consensus values, not probability-of-runs.
  - Fix: normalize before Hungarian padding or scale threshold by `R`.
  - Acceptance: threshold semantics are tested or documented.

## Suggested Test Coverage

- [ ] Synthetic graph with disconnected nodes: no label 0 leaks.
- [ ] Two seed runs with a genuinely new cluster in one run: new ID retained.
- [ ] Low-confidence propagation returns `-1`.
- [ ] `align_all` one-labeling and empty-labeling cases.
- [ ] Top-1 stolen but top-2 valid.
- [ ] Single object time point can be labeled.

## Out of Scope

- Global UMAP/HDBSCAN clustering choices.
- Seeding/reproducibility unless required to make tests deterministic.
