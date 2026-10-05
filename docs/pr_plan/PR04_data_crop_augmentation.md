# PR04 - Data, Crop, and Augmentation Pipeline

Suggested branch: `bugfix/data-crop-augmentation`

## Goal

Make crop extraction and global augmentation robust at volume boundaries and fix tests that assume constant object counts.

## Files Likely Touched

- `barlow_track/utils/data_loading.py`
- `barlow_track/utils/volume_data.py`
- `barlow_track/tests/test_step1_global_augment.py`
- possibly `barlow_track/utils/barlow_lightning.py`

## Already Completed

- Int-vs-string `tracked_segs` bug in `data_loading.get_bbox_data_for_volume_with_label` was fixed in commit `9fbeda5`.

This PR should preserve that fix and address related follow-up issues.

## Bugs / Tasks

- [ ] Clamp crop windows near volume boundaries instead of always padding at the beginning.
  - Location: `data_loading.get_3d_crop_using_bbox_or_centroid`.
  - Current behavior: if the requested crop extends past the tail edge, missing voxels are padded at the beginning after slicing is clipped.
  - Risk: object centroid/keypoint may no longer correspond to the center of the crop.
  - Fix: clamp the window:
    ```python
    z0 = max(0, min(z0, Z - target_z))
    x0 = max(0, min(x0, X - target_x))
    y0 = max(0, min(y0, Y - target_y))
    ```
    then set `z1 = z0 + target_z`, etc. Only pad when the entire volume is smaller than the target crop.
  - Acceptance: synthetic tests for head-edge, tail-edge, and interior centroids produce centered crops.

- [ ] Skip and log all-zero or empty-slice crops.
  - Current risk: out-of-range centroids can become clipped to an empty slice and padded into a full-size all-zero crop.
  - Fix: detect `z1 <= z0`, `x1 <= x0`, or `y1 <= y0`; return `None` or skip with warning.
  - Acceptance: no silent full-size zero crops for invalid centroids.

- [ ] Fix `VolumeCoordsDataModule` split handling for `train_fraction == 1.0`.
  - Current code:
    ```python
    n_train = int(n * self.train_fraction) if self.train_fraction < 1.0 else int(self.train_fraction)
    ```
  - Risk: `train_fraction=1.0` yields a 1-volume train split.
  - Fix: treat `tf == 1` as all volumes; treat `tf > 1` as an absolute count if that is the intended API.
  - Acceptance: `train_fraction=1.0` gives full train and empty val, with an explicit warning or error for empty val if desired.

- [ ] Update `test_dataset_getitem_shapes` for variable N after augmentation.
  - Current failing expectation:
    ```python
    n = ds.num_objects(0)
    assert y1.shape == (n, 1, 4, 32, 32)
    assert y2.shape == y1.shape
    ```
  - Correct behavior: global augmentation can drop out-of-bounds objects, and dropout can further reduce N.
  - Fix: use per-view indices and assert bounds instead of exact equality:
    ```python
    n1, n2 = len(i1), len(i2)
    assert y1.shape == (n1, 1, 4, 32, 32)
    assert y2.shape == (n2, 1, 4, 32, 32)
    assert 0 < n1 <= n
    assert 0 < n2 <= n
    ```
  - Acceptance: baseline test suite passes or this test is rewritten with clearer invariants.

- [ ] Add drop-count logging for bounds filtering and dropout.
  - Location: `volume_data.py` `_augmented_view`.
  - Current risk: objects can silently disappear from a view.
  - Fix: optional debug/info log with raw count, in-bounds count, dropped count, and after-dropout count.

- [ ] Make untracked naming robust.
  - Location: `data_loading.py`, where `untracked_time_*` names are constructed.
  - Risk: `mask_index_to_i_in_array(...)` may return `None`; duplicate raw segmentation IDs can create duplicate names.
  - Fix: fall back to row index when mapping is missing, warn on duplicate keys, and make names unique if necessary.
  - Acceptance: untracked crop construction does not raise or silently overwrite.

## Suggested Test Coverage

- [ ] Synthetic volume with object near start edge, center, and tail edge.
- [ ] Object outside range produces skip/warning rather than all-zero crop.
- [ ] `train_fraction=0.5`, `1.0`, and absolute-count cases.
- [ ] `test_dataset_getitem_shapes` accepts different N for two views.
- [ ] Duplicate/unmappable untracked labels do not crash.

## Out of Scope

- Determinism/RNG worker seeding unless a tiny seed utility is needed for tests.
- Full redesign of augmentation operators.
