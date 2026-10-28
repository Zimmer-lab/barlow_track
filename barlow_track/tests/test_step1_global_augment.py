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
    DEFAULT_GLOBAL_ARGS,
    apply_global_affine,
    get_centroids_for_volume,
    load_volume,
    sample_elastic_field,
    sample_global_affine,
    split_global_args,
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


# ---------------------------------------------------------- elastic field (synthetic, fast)
def test_elastic_off_draws_nothing():
    # Default (elastic_std 0) must leave existing configs' RNG streams untouched
    rng = np.random.RandomState(0)
    state = rng.get_state()[1].copy()
    _, elastic = split_global_args(DEFAULT_GLOBAL_ARGS)
    assert sample_elastic_field(rng, (8, 20, 20), **elastic) is None
    assert np.array_equal(rng.get_state()[1], state)


def test_elastic_scale_zero_is_off_and_scales_the_field():
    rng = np.random.RandomState(0)
    state = rng.get_state()[1].copy()
    assert sample_elastic_field(rng, (8, 20, 20), elastic_std=(1, 4, 4), elastic_scale=0.0) is None
    assert np.array_equal(rng.get_state()[1], state)
    full = sample_elastic_field(np.random.RandomState(1), (8, 20, 20), elastic_std=(1, 4, 4), elastic_length=10)
    half = sample_elastic_field(np.random.RandomState(1), (8, 20, 20), elastic_std=(1, 4, 4), elastic_length=10,
                                elastic_scale=0.5)
    assert np.allclose(half.coarse, 0.5 * full.coarse, rtol=1e-4, atol=1e-6)


def test_elastic_field_std_and_correlation_length():
    rng = np.random.RandomState(0)
    std, length = (1.0, 3.0, 5.0), 10.0
    field = sample_elastic_field(rng, (32, 96, 96), elastic_std=std, elastic_length=length)
    full = field.at(np.indices((32, 96, 96)))
    assert full.shape == (3, 32, 96, 96)
    assert np.allclose(full.reshape(3, -1).std(1), std, rtol=0.05)
    # Correlation exp(-d^2 / L^2): ~1/e at d = L along x, near 0 at 2L
    f = full[2] / full[2].std()
    corr = lambda d: np.mean(f[:, :-d] * f[:, d:])
    assert 0.2 < corr(int(length)) < 0.55
    assert abs(corr(2 * int(length))) < 0.15


def test_elastic_moves_volume_with_points():
    # Blobs at the input points must land at the output points (affine + field)
    shape = (16, 64, 64)
    pts = np.array([[8.0, 20.0, 20.0], [6.0, 40.0, 30.0], [10.0, 30.0, 45.0]])
    zz, xx, yy = np.indices(shape)
    vol = np.zeros(shape, dtype=np.float32)
    for p in pts:
        vol += np.exp(-((zz - p[0]) ** 2 + (xx - p[1]) ** 2 + (yy - p[2]) ** 2) / (2 * 1.5 ** 2))
    rng = np.random.RandomState(3)
    R, t = sample_global_affine(rng, max_degrees_z=30.0, scale_jitter=0.05, max_translation=(1, 3, 3))
    field = sample_elastic_field(rng, shape, elastic_std=(1.0, 3.0, 3.0), elastic_length=15.0)
    vol_aug, pts_aug = apply_global_affine(vol, pts, R, t, field=field)
    _, pts_rigid = apply_global_affine(vol, pts, R, t)
    assert np.linalg.norm(pts_aug - pts_rigid, axis=1).max() > 1.0  # the field did move them
    for q in pts_aug:
        lo = np.maximum(np.round(q).astype(int) - 4, 0)
        win = vol_aug[lo[0]:lo[0] + 9, lo[1]:lo[1] + 9, lo[2]:lo[2] + 9]
        peak = lo + np.array(np.unravel_index(np.argmax(win), win.shape))
        assert np.linalg.norm(peak - q) <= 1.5


def test_elastic_forward_points_inverts_backward_map():
    rng = np.random.RandomState(1)
    # Moderate field (no folding, max |grad v| < 1); strong fields fold and the
    # fixed point cannot converge everywhere
    field = sample_elastic_field(rng, (16, 64, 64), elastic_std=(0.5, 1.5, 1.5), elastic_length=15.0)
    p = np.random.RandomState(2).uniform([2, 8, 8], [13, 55, 55], size=(200, 3))
    y = field.forward_points(p)
    assert np.abs(y + field.at(y.T).T - p).max() < 0.05


def test_crop_rotation_off_by_default_and_about_crop_z_axis():
    from barlow_track.utils.volume_data import build_photometric_transform
    off = build_photometric_transform(dict(p_blur=0.0, p_noise=0.0))
    assert not any(isinstance(t, tio.RandomAffine) for t in off.transforms)
    rot = build_photometric_transform(dict(p_blur=0.0, p_noise=0.0, crop_degrees_z=180.0))
    # Two crops (N, Z, X, Y): blob at the crop centre, blob off-centre in x, both at z = 2
    zz, xx, yy = np.indices((6, 33, 33))
    blob = lambda x: np.exp(-((zz - 2) ** 2 + (xx - x) ** 2 + (yy - 16) ** 2) / 2.0)
    crops = torch.from_numpy(np.stack([blob(16), blob(26)]).astype(np.float32))
    torch.manual_seed(5)
    out = rot(crops).numpy()
    # Centre of the clipped top (RescaleIntensity saturates the 99.5th percentile, so argmax ties)
    from scipy import ndimage
    peaks = [np.array(ndimage.center_of_mass(c >= 0.99 * c.max())) for c in out]
    assert np.allclose(peaks[0], [2, 16, 16], atol=0.5)  # centre stays put
    assert peaks[1][0] == pytest.approx(2, abs=0.5)  # z untouched
    assert abs(np.hypot(peaks[1][1] - 16, peaks[1][2] - 16) - 10) <= 1.5  # rotated in-plane


def test_loader_repeats_views_until_set_epoch(monkeypatch):
    # train_barlow_clusterer builds the loader once and drives epochs itself (no
    # Lightning trainer): views repeat every epoch unless set_epoch advances them
    import barlow_track.utils.volume_data as vd
    rng = np.random.RandomState(0)
    vol = rng.rand(8, 40, 40).astype(np.float32)
    pts = np.array([[4.0, 12.0, 12.0], [4.0, 20.0, 25.0], [3.0, 28.0, 15.0]])
    monkeypatch.setattr(vd, 'load_volume', lambda project_data, t: vol)
    monkeypatch.setattr(vd, 'get_centroids_for_volume', lambda project_data, t: (pts, np.arange(1, 4)))
    project = type('P', (), {'num_frames': 3})()
    dm = vd.VolumeCoordsDataModule(project_data=project, num_frames=3, train_fraction=1.0,
                                   val_fraction=0.0, target_sz=(4, 16, 16), seed=0)
    dm.setup()
    loader = dm.train_dataloader()
    first = [b[0] for b in loader]
    again = [b[0] for b in loader]
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    dm.set_epoch(1)
    later = [b[0] for b in loader]
    assert not any(a.shape == b.shape and torch.equal(a, b) for a, b in zip(first, later))


def test_rounded_bounds_match_the_crop_rounding():
    from barlow_track.utils.volume_data import extract_crops, in_bounds_mask
    shape = (8, 40, 30)
    pts = np.array([[4.0, 20.0, 29.6], [4.0, 20.0, 29.4], [-0.4, 20.0, 15.0], [7.6, 20.0, 15.0]])
    assert in_bounds_mask(pts, shape).tolist() == [True, True, False, True]
    rounded = in_bounds_mask(pts, shape, rounded=True)
    assert rounded.tolist() == [False, True, True, False]
    # Exactly the points rounded keeps get a real (non-placeholder) crop.
    vol = np.ones(shape, dtype=np.float32)
    crops = extract_crops(vol, pts, (4, 16, 16))
    assert (crops.reshape(len(pts), -1).max(1) > 0).tolist() == rounded.tolist()


def test_dataset_round_bounds_leaves_no_placeholder_crops(monkeypatch):
    import barlow_track.utils.volume_data as vd
    vol = 1.0 + np.random.RandomState(0).rand(8, 40, 30).astype(np.float32)  # no real crop is blank
    pts = np.array([[4.0, 12.0, 12.0], [4.0, 20.0, 29.6], [3.0, 28.0, 15.0]])
    monkeypatch.setattr(vd, 'load_volume', lambda project_data, t: vol)
    monkeypatch.setattr(vd, 'get_centroids_for_volume', lambda project_data, t: (pts, np.arange(1, 4)))
    no_motion = dict(p_global_affine=0.0)
    for round_bounds, n_kept in ((False, 3), (True, 2)):
        ds = vd.VolumeCoordsDataset(None, [0], target_sz=(4, 16, 16), seed=0, global_args=no_motion,
                                    photometric_args=dict(p_noise=0.0), round_bounds=round_bounds)
        y1 = ds[0][0]
        assert len(y1) == n_kept
        n_blank = int((y1.reshape(len(y1), -1).std(1) == 0).sum())
        assert n_blank == (0 if round_bounds else 1)


def _synthetic_dataset(monkeypatch, **kwargs):
    import barlow_track.utils.volume_data as vd
    vol = 1.0 + np.random.RandomState(0).rand(8, 40, 40).astype(np.float32)
    pts = np.array([[4.0, 12.0, 12.0], [4.0, 20.0, 28.0], [3.0, 28.0, 15.0], [4.0, 30.0, 30.0]])
    monkeypatch.setattr(vd, 'load_volume', lambda project_data, t: vol)
    monkeypatch.setattr(vd, 'get_centroids_for_volume', lambda project_data, t: (pts, np.arange(1, 5)))
    return vd, vol, pts, vd.VolumeCoordsDataset(None, [0], target_sz=(4, 16, 16), seed=0, **kwargs)


def test_keypoint_only_global_augment_leaves_crops_at_the_original_centroids(monkeypatch):
    rotate = dict(max_degrees_z=180.0, scale_jitter=0.1, max_translation=(1, 4, 4), apply_to_volume=False)
    vd, vol, pts, ds = _synthetic_dataset(monkeypatch, global_args=rotate,
                                          photometric_args=dict(p_noise=0.0),
                                          position_args=dict(jitter_std=1.0))
    y1, y2, k1, k2, i1, i2 = ds[0]
    # every object kept, crops identical across views and equal to untransformed crops
    assert i1.tolist() == i2.tolist() == [0, 1, 2, 3]
    expected = ds.crop_transform(torch.from_numpy(vd.extract_crops(vol, pts, (4, 16, 16)))).float()
    assert torch.allclose(y1[:, 0], expected) and torch.allclose(y2[:, 0], expected)
    # keypoints still move, independently per view
    k0 = vd.VolumeCoordsDataset._normalize(torch.from_numpy(pts.astype(np.float32)), vol.shape)
    assert not torch.allclose(k1, k0, atol=1e-3) and not torch.allclose(k1, k2, atol=1e-3)


def test_keypoints_only_skips_volumes_and_keeps_the_views(monkeypatch):
    import barlow_track.utils.volume_data as vd
    vol = 1.0 + np.random.RandomState(0).rand(8, 40, 40).astype(np.float32)
    pts = np.array([[4.0, 12.0, 12.0], [4.0, 20.0, 28.0], [3.0, 28.0, 15.0], [4.0, 30.0, 30.0]])
    monkeypatch.setattr(vd, 'get_centroids_for_volume', lambda project_data, t: (pts, np.arange(1, 5)))
    project = type('P', (), {'red_data': np.zeros((5, *vol.shape), dtype=np.float32)})()
    aug = dict(global_args=dict(max_degrees_z=180.0, scale_jitter=0.1, apply_to_volume=False,
                                elastic_std=[0.5, 2.0, 2.0], elastic_length=10.0),
               position_args=dict(jitter_std=1.0, dropout_p=0.3))
    monkeypatch.setattr(vd, 'load_volume', lambda project_data, t: vol)
    full = vd.VolumeCoordsDataset(project, [0], target_sz=(4, 16, 16), seed=0, **aug)[0]

    def no_volume(project_data, t):
        raise AssertionError("keypoints_only must not load volumes")
    monkeypatch.setattr(vd, 'load_volume', no_volume)
    fast = vd.VolumeCoordsDataset(project, [0], target_sz=(4, 16, 16), seed=0,
                                  keypoints_only=True, **aug)[0]
    # same keypoints and kept objects as the crop path; placeholder crops with the right N
    for a, b in zip(full[2:], fast[2:]):
        assert torch.equal(a, b)
    assert fast[0].shape == (len(fast[4]), 1, 1, 1, 1) and fast[1].shape == (len(fast[5]), 1, 1, 1, 1)
    with pytest.raises(ValueError, match="apply_to_volume"):
        vd.VolumeCoordsDataset(project, [0], target_sz=(4, 16, 16), keypoints_only=True)


def test_keypoints_only_is_refused_for_models_that_read_crops():
    from types import SimpleNamespace
    from barlow_track.scripts.train_barlow_clusterer import _keypoints_only
    assert _keypoints_only(SimpleNamespace(fusion='add')) is False
    assert _keypoints_only(SimpleNamespace(keypoints_only=True, fusion='position_only')) is True
    with pytest.raises(ValueError, match="position_only"):
        _keypoints_only(SimpleNamespace(keypoints_only=True, fusion='add'))


def test_keypoint_only_global_augment_moves_points_like_the_volume_path():
    import barlow_track.utils.volume_data as vd
    vol = np.zeros((8, 40, 30), dtype=np.float32)
    pts = np.array([[4.0, 12.0, 12.0], [3.0, 28.0, 15.0]])
    R, t = vd.sample_global_affine(np.random.RandomState(3), max_degrees_z=90.0)
    field = vd.sample_elastic_field(np.random.RandomState(4), vol.shape, elastic_std=1.0, elastic_length=10.0)
    _, moved = vd.apply_global_affine(vol, pts, R, t, field=field)
    assert np.allclose(vd.transform_points(pts, vol.shape, R, t, field=field), moved)
    # the flag never reaches sample_global_affine
    affine, _ = vd.split_global_args(dict(max_degrees_z=10.0, apply_to_volume=False))
    assert affine == dict(max_degrees_z=10.0)


def test_legacy_crop_augment_gives_the_two_legacy_views(monkeypatch):
    legacy = dict(p_RandomAffine_base=0.0, p_RandomBlur_base=0.0, p_RandomAffine=0.0,
                  p_RandomAffine_flip=0.0, p_RandomElasticDeformation=0.0, p_RandomNoise=1.0,
                  std_RandomNoise=0.25, global_augment=dict(max_degrees_z=5.0))
    vd, vol, pts, ds = _synthetic_dataset(monkeypatch, global_args=dict(p_global_affine=0.0),
                                          crop_augment='legacy', legacy_transform_args=legacy)
    y1, y2 = ds[0][:2]
    # view 1 (legacy.transform) is only normalization with these probabilities;
    # view 2 (legacy.transform_prime) always adds noise
    from barlow_track.utils.barlow import Transform
    plain = Transform().final_normalization(torch.from_numpy(vd.extract_crops(vol, pts, (4, 16, 16)))).float()
    assert torch.allclose(y1[:, 0], plain)
    assert not torch.allclose(y2[:, 0], plain)
    # reproducible per (seed, epoch, index)
    assert torch.equal(ds[0][1], y2)
    with pytest.raises(ValueError, match="crop_augment"):
        _synthetic_dataset(monkeypatch, crop_augment='bogus')


def test_legacy_crop_path_rejects_keypoint_only_global_augment():
    from barlow_track.utils.barlow import Transform
    tr = Transform(dict(global_augment=dict(apply_to_volume=False)))
    with pytest.raises(ValueError, match="apply_to_volume"):
        tr(torch.rand(3, 4, 16, 16))


def test_legacy_crop_path_rejects_elastic():
    from barlow_track.utils.barlow import Transform
    tr = Transform(dict(global_augment=dict(elastic_std=2.0)))
    with pytest.raises(ValueError, match="elastic_std"):
        tr(torch.rand(3, 4, 16, 16))


# ---------------------------------------------------------- dataset on real project
@requires_project
def test_centroids_available(project_data):
    zxy, seg = get_centroids_for_volume(project_data, 0)
    assert zxy.shape[1] == 3 and len(zxy) > 1
    assert np.isfinite(zxy).all()


@requires_project
def test_dataset_getitem_shapes(project_data):
    ds = VolumeCoordsDataset(project_data, [0, 1, 2], target_sz=(4, 32, 32), seed=0)
    y1, y2, k1, k2, i1, i2 = ds[0]
    n = ds.num_objects(0)
    n1, n2 = len(i1), len(i2)
    assert y1.shape == (n1, 1, 4, 32, 32)
    assert y2.shape == (n2, 1, 4, 32, 32)
    assert k1.shape == (n1, 3) and k2.shape == (n2, 3)
    assert 0 < n1 <= n
    assert 0 < n2 <= n
    assert torch.isfinite(y1).all() and torch.isfinite(k1).all()
    if n1 == n2:
        assert not torch.allclose(y1, y2)  # two stochastic global views differ
        assert not torch.allclose(k1, k2)
    else:
        # Different surviving N already proves the views diverged
        assert True


@requires_project
def test_identity_global_matches_legacy_crop_path(project_data):
    """With identity geometry, new plumbing == legacy crop + same intensity norm."""
    from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid

    target_sz = (4, 32, 32)
    ds = VolumeCoordsDataset(project_data, [0], target_sz=target_sz,
                             global_args=dict(p_global_affine=0.0),
                             photometric_args=dict(p_blur=0.0, p_noise=0.0), seed=0)
    y1, _, k1, _ = ds[0][:4]

    vol = load_volume(project_data, 0)
    zxy, _ = get_centroids_for_volume(project_data, 0)
    sz = np.array([1, *vol.shape])
    legacy = np.stack([get_3d_crop_using_bbox_or_centroid(p, sz, np.array(target_sz), vol)[0]
                       for p in zxy])
    norm = tio.RescaleIntensity(percentiles=(5, 99.5))  # in sync with training
    expected = norm(torch.from_numpy(legacy)).float().unsqueeze(1)
    assert torch.allclose(y1, expected, atol=1e-5)

    # Identity also leaves voxel coordinates untouched (up to normalization)
    assert torch.allclose(k1, ds._normalize(torch.from_numpy(zxy.astype(np.float32)),
                                            vol.shape), atol=1e-5)


@requires_project
def test_kpts_normalized_range(project_data):
    ds = VolumeCoordsDataset(project_data, [0], target_sz=(4, 32, 32), seed=0)
    _, _, k1, _ = ds[0][:4]
    assert (k1.abs() <= 1.0).all()


# ---------------------------------------------------------- position-only augments (synthetic, fast)
def test_position_jitter_is_noop_at_zero():
    from barlow_track.utils.volume_data import apply_position_jitter
    rng = np.random.RandomState(0)
    pts = np.array([[4.0, 10.0, 10.0], [4.0, 12.0, 12.0]])
    assert np.allclose(apply_position_jitter(pts, rng, 0.0), pts)


def test_position_jitter_moves_points():
    from barlow_track.utils.volume_data import apply_position_jitter
    pts = np.array([[4.0, 10.0, 10.0], [4.0, 12.0, 12.0]])
    out = apply_position_jitter(pts, np.random.RandomState(0), 1.0)
    assert out.shape == pts.shape and not np.allclose(out, pts)


def test_dropout_keep_guarantees_min_keep():
    from barlow_track.utils.volume_data import sample_dropout_keep
    _, idx = sample_dropout_keep(6, np.random.RandomState(1), dropout_p=0.5, min_keep=2)
    assert len(idx) >= 2
    _, idx = sample_dropout_keep(1, np.random.RandomState(0), dropout_p=0.9, min_keep=2)
    assert len(idx) == 1  # tiny volumes keep everything


def test_intersection_gather_pairs_views():
    import torch
    from barlow_track.utils.barlow_superglue import intersection_gather
    s1, s2 = intersection_gather(torch.tensor([0, 1, 3, 4]), torch.tensor([1, 2, 3]))
    assert s1.tolist() == [1, 2] and s2.tolist() == [0, 2]
