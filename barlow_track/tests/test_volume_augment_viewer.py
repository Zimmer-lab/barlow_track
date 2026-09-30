"""Tests for the napari augmentation viewer (Step 1 global path).

Synthetic tests run anywhere. Project tests need a real wbfm project:
    set BARLOW_TEST_PROJECT=C:\\Users\\Charlie\\Documents\\ZIM2165_Gcamp7b_worm1-2022_11_28
(or rely on the default below). GUI (Qt) tests additionally need a display
or QT_QPA_PLATFORM=offscreen.

Run:
    python -m pytest barlow_track/tests/test_volume_augment_viewer.py -q
"""
import os
from pathlib import Path

import numpy as np
import pytest

from barlow_track.utils.volume_augment_viewer import (
    ViewerParams,
    ViewerState,
    augment_frame_for_viewer,
    box_edges_from_bbox,
    load_project_for_viewer,
)

TEST_PROJECT_CANDIDATES = [
    r"C:\Users\Charlie\Documents\ZIM2165_Gcamp7b_worm1-2022_11_28",
    "/home/charles/Current_work/test_projects/barlow/worm4-2-2025-03-09/project_config.yaml",
]


def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


PROJECT_PATH = os.environ.get("BARLOW_TEST_PROJECT") or _first_existing(TEST_PROJECT_CANDIDATES)
requires_project = pytest.mark.skipif(PROJECT_PATH is None, reason="No test project found")


@pytest.fixture(scope="module")
def project_data():
    assert PROJECT_PATH is not None
    return load_project_for_viewer(PROJECT_PATH)


def _synthetic_volume_and_points():
    rng = np.random.RandomState(42)
    vol = (rng.rand(10, 40, 50) * 500).astype(np.float32)
    vol[5, 20, 25] = 2000.0  # bright peak so crops are non-trivial
    pts = np.array([[5.0, 20.0, 25.0], [2.0, 10.0, 10.0]])
    return vol, pts


def _identity_params(**overrides):
    kw = dict(use_affine=True, p_global_affine=0.0,
              use_blur=False, p_blur=0.0, use_noise=False, p_noise=0.0)
    kw.update(overrides)
    return ViewerParams(**kw)


# ------------------------------------------------------------ synthetic
def test_box_edges_shape_and_connectivity():
    edges = box_edges_from_bbox([2, 4, 6, 5, 10, 14])
    assert edges.shape == (12, 2, 3)
    # Every edge connects corners differing in exactly one coordinate ...
    for a, b in edges:
        assert int(np.sum(a != b)) == 1
    # ... and all 8 corners are covered
    corners = {tuple(c) for e in edges for c in e}
    assert len(corners) == 8


def test_augment_identity_reproduces_volume():
    vol, pts = _synthetic_volume_and_points()
    res = augment_frame_for_viewer(vol, pts, _identity_params(), target_sz=(4, 16, 16))
    assert np.allclose(res.vol_aug, vol)
    assert np.allclose(res.pts_aug, pts)
    assert res.crop_raw.shape == (4, 16, 16)
    # No photometric enabled: augmented crop == normalized raw crop
    assert np.allclose(res.crop_aug, res.crop_raw, atol=1e-5)
    assert np.isfinite(res.crop_aug).all()


def test_disable_affine_flag_forces_identity():
    vol, pts = _synthetic_volume_and_points()
    params = _identity_params(use_affine=False, p_global_affine=1.0,
                              max_degrees_z=180.0, p_flip=1.0)
    res = augment_frame_for_viewer(vol, pts, params, target_sz=(4, 16, 16))
    assert np.allclose(res.vol_aug, vol)
    assert np.allclose(res.pts_aug, pts)


def test_enabled_affine_changes_volume_and_points_together():
    vol, pts = _synthetic_volume_and_points()
    params = ViewerParams(p_global_affine=1.0, max_degrees_z=180.0,
                          use_blur=False, use_noise=False, seed=0)
    res = augment_frame_for_viewer(vol, pts, params, target_sz=(4, 16, 16))
    assert not np.allclose(res.vol_aug, vol)
    assert not np.allclose(res.pts_aug, pts)
    # Peak voxel and its point move to the same place (within interpolation)
    moved_peak = np.unravel_index(np.argmax(res.vol_aug), res.vol_aug.shape)
    assert np.allclose(moved_peak, res.pts_aug[0], atol=2.0)


def test_photometric_builder_shared_with_training():
    import torchio as tio
    from barlow_track.utils.volume_data import build_photometric_transform
    t = build_photometric_transform(dict(p_blur=0.0, p_noise=0.0))
    assert isinstance(t, tio.Compose) and len(t) == 3


# ------------------------------------------------------------ real project
@requires_project
def test_viewer_state_on_real_project(project_data):
    state = ViewerState(project_data, target_sz=(4, 32, 32),
                        params=_identity_params())
    assert state.num_points > 1
    assert state.volume.ndim == 3
    assert state.result.vol_aug.shape == state.volume.shape
    assert state.result.pts_aug.shape == state.points.shape
    assert np.isfinite(state.result.crop_aug).all()
    # Identity global args: points untouched, crop boxes identical
    assert np.allclose(state.result.pts_aug, state.points)
    assert state.result.bbox_raw == state.result.bbox_aug


@requires_project
def test_viewer_state_seed_changes_augmentation(project_data):
    params = ViewerParams(p_global_affine=1.0, max_degrees_z=180.0,
                          use_blur=False, use_noise=False, seed=0)
    s0 = ViewerState(project_data, target_sz=(4, 32, 32), params=params)
    first = s0.result.vol_aug.copy()
    s0.params.seed = 1
    s0.reaugment()
    assert not np.allclose(first, s0.result.vol_aug)


@requires_project
def test_viewer_state_crop_selection_moves_box(project_data):
    state = ViewerState(project_data, target_sz=(4, 32, 32),
                        params=_identity_params())
    state.params.crop_idx = 0
    state.reaugment()
    box0 = state.result.bbox_raw
    state.params.crop_idx = 1
    state.reaugment()
    assert state.result.bbox_raw != box0
    assert state.result.crop_raw.shape == (4, 32, 32)
