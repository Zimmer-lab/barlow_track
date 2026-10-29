"""Synthetic segmentation errors (spurious points, merges, splits) in the position views."""
import numpy as np
import pytest
import torch

import barlow_track.utils.volume_data as vd
from barlow_track.utils.barlow_superglue import intersection_gather
from barlow_track.utils.volume_data import apply_detection_errors

# Two well-separated pairs: each point's nearest neighbour is its partner
PAIRS = np.array([[2.0, 10.0, 10.0], [2.0, 10.0, 14.0], [6.0, 40.0, 40.0], [6.0, 44.0, 40.0]])


def test_off_returns_the_input_and_draws_nothing():
    rng = np.random.RandomState(0)
    state = rng.get_state()[1].copy()
    pts, ids = apply_detection_errors(PAIRS, rng)
    assert np.array_equal(pts, PAIRS) and ids.tolist() == [0, 1, 2, 3]
    assert np.array_equal(rng.get_state()[1], state)


def test_merges_replace_nearest_neighbour_pairs_with_their_midpoint():
    pts, ids = apply_detection_errors(PAIRS, np.random.RandomState(0), merge_p=1.0)
    assert ids.tolist() == [-1, -1]
    want = np.array([[2.0, 10.0, 12.0], [6.0, 42.0, 40.0]])
    assert np.allclose(sorted(map(tuple, pts)), sorted(map(tuple, want)))


def test_splits_replace_a_point_by_two_fragments_split_dist_apart():
    pts, ids = apply_detection_errors(PAIRS, np.random.RandomState(0), split_p=1.0, split_dist=3.0)
    assert len(pts) == 8 and (ids == -1).all()
    frag = pts.reshape(4, 2, 3)
    assert np.allclose(np.linalg.norm(frag[:, 0] - frag[:, 1], axis=1), 3.0)
    assert np.allclose(frag.mean(1), PAIRS)


def test_spurious_points_are_appended_without_identity():
    pts, ids = apply_detection_errors(PAIRS, np.random.RandomState(0), spurious_p=1.0, spurious_std=0.5)
    n_extra = len(pts) - 4
    assert n_extra > 0 and ids[:4].tolist() == [0, 1, 2, 3] and (ids[4:] == -1).all()
    assert np.array_equal(pts[:4], PAIRS)
    # each spurious point sits near some real object
    d = np.linalg.norm(pts[4:, None] - PAIRS[None], axis=-1).min(1)
    assert (d < 3.0).all()


def test_intersection_gather_never_pairs_synthetic_detections():
    s1, s2 = intersection_gather(torch.tensor([0, 2, -1, -1]), torch.tensor([-1, 2, 0]))
    assert s1.tolist() == [0, 1] and s2.tolist() == [2, 1]


def _dataset(monkeypatch, **position_args):
    monkeypatch.setattr(vd, 'get_centroids_for_volume', lambda project_data, t: (PAIRS, np.arange(1, 5)))
    project = type('P', (), {'red_data': np.zeros((1, 8, 50, 50), dtype=np.float32)})()
    return vd.VolumeCoordsDataset(project, [0], target_sz=(4, 16, 16), seed=0, keypoints_only=True,
                                  global_args=dict(apply_to_volume=False, max_degrees_z=30.0),
                                  position_args=dict(jitter_std=0.5, **position_args))


def test_default_views_are_unchanged_by_the_new_keys(monkeypatch):
    base = _dataset(monkeypatch)[0]
    zeros = _dataset(monkeypatch, spurious_p=0.0, merge_p=0.0, split_p=0.0)[0]
    for a, b in zip(base, zeros):
        assert torch.equal(a, b)


@pytest.mark.parametrize('kw', [dict(spurious_p=0.5), dict(merge_p=1.0), dict(split_p=0.5)])
def test_views_carry_synthetic_rows_as_minus_one(monkeypatch, kw):
    y1, y2, k1, k2, i1, i2 = _dataset(monkeypatch, **kw)[0]
    assert len(k1) == len(i1) == len(y1) and len(k2) == len(i2) == len(y2)
    assert (i1 == -1).any() or (i2 == -1).any()
    real = [v for v in i1.tolist() if v >= 0]
    assert len(real) == len(set(real)) and set(real) <= {0, 1, 2, 3}


def test_auto_jitter_recovers_anisotropic_centroid_noise(monkeypatch):
    # A grid of objects moving rigidly between frames plus (z, x, y) noise (0.2, 1, 1): the
    # GT-free estimate must find the anisotropy without any track labels
    g = np.stack(np.meshgrid(np.arange(4) * 6.0, np.arange(10) * 12.0, np.arange(10) * 12.0, indexing='ij'), -1)
    base = g.reshape(-1, 3) + [2.0, 10.0, 10.0]
    rng = np.random.RandomState(0)
    frames = {t: base + [0, 3.0 * t, 0] + rng.normal(size=base.shape) * [0.2, 1.0, 1.0] for t in range(12)}
    monkeypatch.setattr(vd, 'get_centroids_for_volume', lambda project_data, t: (frames[t], np.arange(len(base))))
    project = type('P', (), {'red_data': np.zeros((12, 30, 140, 140), dtype=np.float32), 'num_frames': 12})()
    ds = vd.VolumeCoordsDataset(project, list(range(11)), target_sz=(4, 16, 16), keypoints_only=True,
                                global_args=dict(apply_to_volume=False), position_args=dict(jitter_std='auto'))
    z, x, y = ds.position_args['jitter_std']
    assert 0.1 < z < 0.4 and 0.6 < x < 1.6 and 0.6 < y < 1.6
    with pytest.raises(ValueError, match="auto"):
        vd.VolumeCoordsDataset(project, [0], target_sz=(4, 16, 16), keypoints_only=True,
                               global_args=dict(apply_to_volume=False), position_args=dict(jitter_std='fit'))
