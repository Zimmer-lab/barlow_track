# PR03 - Tracking Windows, Resume, and Final Frame Handling

Suggested branch: `bugfix/tracking-windows-resume`

## Goal

Fix tracking window coverage, remove the apparent final-frame omission, and make resume/legacy tracking paths fail predictably.

## Files Likely Touched

- `barlow_track/utils/utils_tracking.py`
- `barlow_track/utils/track_using_barlow.py`
- possibly `barlow_track/tests/test_pipeline_position.py` or new tracker-focused unit tests

## Background

The default scripted path uses `track_using_barlow_from_config` and either global clustering or label propagation. There is also an overlapping-window mode used by notebooks/legacy workflows.

The review found both active result-affecting bugs and latent traps in resume/legacy paths.

## Bugs / Tasks

- [ ] Decide and fix final-frame handling: `num_frames = project_data.num_frames - 1`.
  - Location: `track_using_barlow.py`, around the `num_frames` setup.
  - Current code comment itself asks: `# Why am I subtracting 1?`
  - Risk: frame `N-1` may be omitted from embeddings, tracker metadata, and evaluation.
  - Fix: likely use `project_data.num_frames`; if the `-1` is intentional for this data format, document why.
  - Acceptance: embeddings/tracks cover the expected frame range for a known project.

- [ ] Fix `all_start_volumes` off-by-one in overlapping windows.
  - Location: `utils_tracking.py`:
    ```python
    all_start_volumes = list(np.arange(0, self.num_frames - self.n_volumes_per_window, step=self.tracker_stride))
    all_start_volumes.append(self.num_frames - self.n_volumes_per_window - 1)
    ```
  - Risk: valid final start `num_frames - n_volumes_per_window` is excluded; appended value may be `-1`.
  - Fix:
    ```python
    np.arange(0, num_frames - n_volumes_per_window + 1, stride)
    ```
    then deduplicate.
  - Acceptance: max covered time reaches `num_frames - 1`.

- [ ] Include the first window in `track_using_overlapping_windows`.
  - Location: `utils_tracking.py`, loop:
    ```python
    for df in tqdm(all_dfs[1:], leave=False):
    ```
  - Risk: `all_dfs[0]` usually covers frames `0..n_volumes_per_window-1` and is dropped from combination.
  - Fix: iterate `all_dfs`, unless a test proves the first window is intentionally duplicated.
  - Acceptance: early-frame coverage is not worse than the global window.

- [ ] Fix resume pickle naming mismatch.
  - Location: `track_using_barlow.py` resume branches load `linear_ind_to_raw_neuron_ind.pickle`, while normal save writes `linear_ind_to_t_and_seg_id.pickle` and `linear_ind_to_gt_ind.pickle`.
  - Risk: resume path can raise `FileNotFoundError` or load metadata from a different legacy format.
  - Fix: save/load one consistent set of metadata names and version them if needed.
  - Acceptance: an integration or unit-style test writes and reads back the resume artifacts.

- [ ] Apply the same feature preprocessing on resume.
  - Current risk: fresh path may apply SVD, resume path may load raw embedding and skip SVD.
  - Fix: apply `_robust_svd` or explicitly document that resume expects already-reduced embeddings.
  - Acceptance: resume and fresh feature shapes/normalization match.

- [ ] Fix or delete broken latent branches.
  - `track_using_label_propagation_clusterer`: when `umap_projection=True`, `X` can be unbound.
  - `cluster_single_window`: non-SVD branch can return unbound `db_svd` / `Y_tsne_svd`.
  - Suggested fix: bind `X = X_umap`, and cluster/return window-local projected features.

- [ ] Re-indent `BarlowProject.embed_data` executor block if the class is still supported.
  - Location: `track_using_barlow.py`.
  - Risk: executor submit appears outside the frame loop, so only the last frame is embedded.
  - Fix: move `futures = ...` and collection logic into the loop, mirroring `embed_using_barlow`.
  - Acceptance: multi-frame test produces entries for every frame.

- [ ] Add logging for zero-detection frames.
  - Location: embedding loops that silently `continue` when a frame has no detections.
  - Fix: log frame index and counts.
  - Acceptance: skipped frames are visible.

## Suggested Test Coverage

- [ ] Unit test `all_start_volumes` for several `num_frames`, window sizes, and strides.
- [ ] Unit test overlapping-window combination includes first and last windows.
- [ ] Unit test resume pickle roundtrip.
- [ ] Unit test `num_frames` includes last frame using a mock project with known `num_frames`.
- [ ] Static test or small unit test that `umap_projection=True` path reaches `multi_seed_propagation` without `NameError`.

## Out of Scope

- Label propagation algorithm internals.
- Eval metric denominator fixes.
