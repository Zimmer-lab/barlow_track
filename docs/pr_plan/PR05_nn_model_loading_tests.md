# PR05 - Neural Network Architecture, Model Loading, and Tests

Suggested branch: `bugfix/nn-model-loading-tests`

## Goal

Make architecture defaults, checkpoint loading, and position-fusion behavior less surprising, and fix the failing architecture tests.

## Files Likely Touched

- `barlow_track/utils/barlow.py`
- `barlow_track/utils/barlow_superglue.py`
- `barlow_track/utils/siamese.py`
- `barlow_track/tests/test_barlow_position.py`
- possibly `barlow_track/scripts/train_barlow_clusterer.py`

## Bugs / Tasks

- [ ] Fix `test_encode_position_single_detection_falls_back`.
  - Current assertion expects:
    ```python
    assert torch.allclose(fused, model.backbone(y1[:1]))
    ```
  - Actual behavior with default `fusion_norm='layernorm'`:
    ```python
    fused = model.norm_visual(model.backbone(...)) + learned position fallback
    ```
  - Fix: either build the test model with `fusion_norm='none'`, or assert against `model.norm_visual(model.backbone(...))`.
  - Acceptance: test passes and still verifies that single-detection positions do not introduce a real position encoding.

- [ ] Fix `test_old_batchnorm_checkpoint_loads_leniently`.
  - Current test creates a fake checkpoint from a current LayerNorm model but pickles args without `fusion_norm`, forcing loader to legacy `none`.
  - Fix: add `_a.fusion_norm = 'layernorm'` to the fake args, or construct the fake checkpoint with `fusion_norm='none'`.
  - Acceptance: test passes for intended old-BN compatibility behavior.

- [ ] Make checkpoint loading infer `fusion_norm` when missing.
  - Location: `barlow.load_barlow_model`.
  - Risk: a checkpoint containing `norm_visual.weight` / `norm_pos.weight` can fail if its pickled args lack `fusion_norm`.
  - Fix: inspect `state_dict` keys:
    - if `norm_visual.weight` or `norm_pos.weight` exists, default to `layernorm` when missing;
    - otherwise keep legacy `none`.
  - Acceptance: synthetic checkpoint load works for both old and new model variants.

- [ ] Make single-detection fallback truly visual-only when no position exists.
  - Location: `barlow_superglue.py`, `encode_position` or `fused_descriptors`.
  - Current risk: position encoding returns zeros, but LayerNorm of a zero vector can become a learned constant offset after training.
  - Fix: return early and skip `norm_pos` when `N < 2`, or explicitly zero out the position branch after normalization.
  - Acceptance: for `N=1`, fused output is independent of trained `norm_pos.bias`.

- [ ] Replace hardcoded encoder projection size.
  - Location: `siamese.py`, projection using `crop_sz.prod()/8`.
  - Risk: only valid for `num_levels=2`; non-default levels produce cryptic shape errors.
  - Fix: compute flatten size dynamically with a dummy forward, or assert `num_levels == 2` with a clear message.
  - Acceptance: non-default `num_levels` either works or raises a clear configuration error.

- [ ] Optional numerical nit: standardize Barlow correlation normalization.
  - Location: Barlow loss implementation.
  - Current risk: `std(0)` is unbiased while correlation divides by `N`.
  - Fix: use `std(..., unbiased=False)` or divide by `N-1`.
  - Acceptance: synthetic loss test matches expected Pearson correlation normalization.

## Suggested Test Coverage

- [ ] Unit tests for old/new checkpoint variants with missing `fusion_norm`.
- [ ] Unit test single-detection fallback with non-zero LayerNorm bias.
- [ ] Unit test encoder projection for `num_levels=2` and invalid/unsupported `num_levels`.
- [ ] Existing `test_barlow_position.py` passes.

## Out of Scope

- Training hyperparameter changes.
- Architecture performance rewrites.
