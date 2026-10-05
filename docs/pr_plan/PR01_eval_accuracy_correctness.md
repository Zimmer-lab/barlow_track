# PR01 - Eval Accuracy Correctness

Suggested branch: `bugfix/eval-accuracy-correctness`

## Goal

Make accuracy evaluation measure the intended time range, avoid crashes on NaN/duplicate labels, and make CPU/GPU/device paths comparable.

## Files Likely Touched

- `barlow_track/scripts/eval_accuracy.py`
- `barlow_track/utils/utils_ground_truth.py`
- possibly `barlow_track/scripts/var_test.py` or stored accuracy utilities if they reuse the same matching pattern

## Background

The eval pipeline compares predicted candidate tracks to ground-truth tracks by:

1. loading or embedding predictions;
2. building `df_pred` with columns as predicted neurons;
3. loading ground truth `df_gt`;
4. renaming predicted columns to GT names using bipartite matching;
5. computing accuracy as correct / valid GT detections.

The current subagent review found multiple high-impact reporting bugs. These can make quick evaluation runs report much lower accuracy than intended or crash on ordinary NaN gaps.

## Bugs / Tasks

- [ ] Truncate ground truth to the predicted time range when `--max_frames` is used.
  - Location: `barlow_track/scripts/eval_accuracy.py`, around the `n_frames` and `max_len = max(len(df_gt), len(df_pred))` logic.
  - Current risk: predictions cover only `0..n_frames-1`, but GT spans the whole video; predictions are padded with NaN for all unevaluated frames.
  - Effect: `--max_frames 50` can produce accuracy approximately `(50 / total_frames) * true_accuracy`.
  - Fix: restrict GT to the union/range actually evaluated by predictions. Prefer intersecting on the time index rather than padding by row count.
  - Acceptance: with `--max_frames N`, no GT row with time `>= N` should count as a miss.

- [ ] Avoid NaN costs in bipartite matching.
  - Location: `barlow_track/scripts/eval_accuracy.py` around `rename_columns_using_matching(...)` calls.
  - Current risk: padded/real NaN rows become NaN costs in `cdist`, causing `linear_sum_assignment` invalid-entry errors.
  - Fix: pass `try_to_fix_inf=True` where supported, or strip/replace NaN rows before matching.
  - Acceptance: synthetic GT/pred with NaN gaps completes matching.

- [ ] Handle unmatched prediction columns without duplicate `unmatched_neuron` labels.
  - Location: `barlow_track/scripts/eval_accuracy.py` and/or `wbfm/utils/candidate_matches/utils_candidate_matches.py` call site.
  - Current risk: when predictions have fewer columns than GT, several predicted columns may be renamed to `unmatched_neuron`, causing `df_pred.reindex(...)` to raise `cannot reindex from a duplicate axis`.
  - Fix: drop unmatched predicted columns explicitly before reindexing, or suffix each unmatched column uniquely.
  - Acceptance: fewer predicted columns than GT no longer crashes.

- [ ] Make CPU/GPU crop intensity normalization consistent.
  - Location: `barlow_track/scripts/eval_accuracy.py` GPU helper `_rescale_gpu` vs CPU crop normalization.
  - Current risk: GPU path reportedly computes percentiles over the whole volume while CPU/canonical path uses crop stack. This changes embeddings and metrics.
  - Fix: use one canonical percentile source for both paths. Recommend crop-stack percentiles to match training/inference.
  - Acceptance: on a small synthetic tensor, CPU and GPU normalization match within numerical tolerance.

- [ ] Make project-source and NWB-source GT filters comparable.
  - Location: `barlow_track/scripts/eval_accuracy.py` GT loading branches.
  - Current risk: project path filters only finished neurons, NWB path may use full `final_tracks`.
  - Fix: apply the same finished/valid neuron filter in both paths or explicitly log the GT neuron set.
  - Acceptance: project and NWB runs on equivalent data have the same denominator semantics.

- [ ] Avoid conflating raw array index and raw segmentation ID in fallback paths.
  - Location: `barlow_track/scripts/eval_accuracy.py` around fallback `raw_ind = seg`.
  - Current risk: downstream metadata join interprets a mask id as an array index.
  - Fix: emit `np.nan`/skip row, or derive the correct array index; do not substitute a different ID space.

- [ ] Rename duplicate accuracy stats keys.
  - Location: `barlow_track/utils/utils_ground_truth.py`, `calculate_accuracy(...)`.
  - Current risk: `"misses"` and `"mismatches"` are first scalar totals, then overwritten by DataFrames.
  - Fix: use distinct keys such as `total_misses`, `misses_df`.
  - Acceptance: stats keys have stable types.

- [ ] Make `build_accuracy_dict` append per-trial records atomically.
  - Location: `barlow_track/utils/utils_ground_truth.py`.
  - Current risk: exception branches append some fields but not all, creating ragged columns or misaligned rows.
  - Fix: build a per-trial dict with all keys and append once.
  - Acceptance: missing `project_config.yaml` still yields equal-length columns.

## Suggested Test Coverage

- [ ] Unit test `--max_frames` behavior with fake GT/pred DataFrames.
- [ ] Unit test matching with NaN gaps.
- [ ] Unit test prediction fewer columns than GT.
- [ ] Unit test CPU/GPU normalization helper on tiny arrays.
- [ ] Unit test `calculate_accuracy` key types.
- [ ] Unit test `build_accuracy_dict` ragged-list avoidance.

## Out of Scope

- Improving matching quality beyond correctness.
- Label propagation clustering behavior.
- Model architecture changes.
