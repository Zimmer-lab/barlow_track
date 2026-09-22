"""Step 1 tests: global-before-crop augmentation + VolumeCoordsDataset.

Needs the same test project as Step 0 (BARLOW_TEST_PROJECT or defaults).
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest barlow_track/tests/test_step1_global_augment.py -q --assert=plain
"""
import numpy as np
import pytest
import torch
import torchio as tio

from test_step0_embed_and_augment import PROJECT_PATH, requires_project

from barlow_track.utils.volume_data import (
    VolumeCoordsDataset,
    apply_global_affine,
    get_centroids_for_volume,
    load_volume,
    sample_global_affine,
)


@pytest.fixture(scope="module")
def project_data():
    from wbfm.utils.projects.finished_project_data import ProjectData

    assert PROJECT_PATH is not None
    return ProjectData.load_final_project_data(PROJECT_PATH, allow_hybrid_loading=True)


# ---------------------------------------------------------- affine math (synthetic, fast)
def test_affine_identity_is_noop():
    rng = np.random.RandomState(0)
    vol = rng.rand(6, 20, 30).astype(np.float32)
    pts = np.array([[2.5, 10.0, 15.0], [0.0, 0.0, 0.0]])
    R, t = sample_global_affine(rng, p_global_affine=0.0)  # forced identity
    vol_aug, pts_aug = apply_global_affine(vol, pts, R, t)
    assert np.allclose(vol_aug, vol)
    assert np.allclose(pts_aug, pts)


def test_affine_translation_consistent():
    # Pure translation: points shift by t, voxels shift identically
    vol = np.zeros((8, 20, 20), dtype=np.float32)
    vol[4, 10, 10] = 1.0
    pts = np.array([[4.0, 10.0, 10.0]])
    R = np.eye(3)
    t = np.array([1.0, -2.0, 3.0])
    vol_aug, pts_aug = apply_global_affine(vol, pts, R, t)
    assert np.allclose(pts_aug, pts + t)
    assert vol_aug[5, 8, 13] == pytest.approx(1.0)  # peak moved with the points


def test_affine_rotation_about_z_keeps_z():
    rng = np.random.RandomState(1)
    R, t = sample_global_affine(rng, p_global_affine=1.0, max_degrees_z=180.0,
                                scale_jitter=0.0, max_translation=(0, 0, 0), p_flip=0.0)
    assert R[0, 1] == 0 and R[0, 2] == 0 and R[0, 0] == pytest.approx(1.0)
    assert np.allclose(t, 0)
    pts = np.array([[3.0, 5.0, 7.0]])
    _, pts_aug = apply_global_affine(np.zeros((8, 20, 20), dtype=np.float32), pts, R, t)
    assert pts_aug[0, 0] == pytest.approx(3.0)  # z untouched by in-plane rotation


# ---------------------------------------------------------- dataset on real project
@requires_project
def test_centroids_available(project_data):
    zxy, seg = get_centroids_for_volume(project_data, 0)
    assert zxy.shape[1] == 3 and len(zxy) > 1
    assert np.isfinite(zxy).all()


@requires_project
def test_dataset_getitem_shapes(project_data):
    ds = VolumeCoordsDataset(project_data, [0, 1, 2], target_sz=(4, 32, 32), seed=0)
    y1, y2, k1, k2 = ds[0]
    n = ds.num_objects(0)
    assert y1.shape == (n, 1, 4, 32, 32) and y2.shape == y1.shape
    assert k1.shape == (n, 3) and k2.shape == (n, 3)
    assert torch.isfinite(y1).all() and torch.isfinite(k1).all()
    assert not torch.allclose(y1, y2)  # two stochastic global views differ
    assert not torch.allclose(k1, k2)


@requires_project
def test_identity_global_matches_legacy_crop_path(project_data):
    """With identity geometry, new plumbing == legacy crop + same intensity norm."""
    from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid

    target_sz = (4, 32, 32)
    ds = VolumeCoordsDataset(project_data, [0], target_sz=target_sz,
                             global_args=dict(p_global_affine=0.0),
                             photometric_args=dict(p_blur=0.0, p_noise=0.0), seed=0)
    y1, _, k1, _ = ds[0]

    vol = load_volume(project_data, 0)
    zxy, _ = get_centroids_for_volume(project_data, 0)
    sz = np.array([1, *vol.shape])
    legacy = np.stack([get_3d_crop_using_bbox_or_centroid(p, sz, np.array(target_sz), vol)[0]
                       for p in zxy])
    norm = tio.RescaleIntensity(percentiles=(5, 100))
    expected = norm(torch.from_numpy(legacy)).float().unsqueeze(1)
    assert torch.allclose(y1, expected, atol=1e-5)

    # Identity also leaves voxel coordinates untouched (up to normalization)
    assert torch.allclose(k1, ds._normalize(torch.from_numpy(zxy.astype(np.float32)),
                                            vol.shape), atol=1e-5)


@requires_project
def test_kpts_normalized_range(project_data):
    ds = VolumeCoordsDataset(project_data, [0], target_sz=(4, 32, 32), seed=0)
    _, _, k1, _ = ds[0]
    assert (k1.abs() <= 1.0).all()
