"""SVD space (scaled U*S vs whitened U) and the track-to-GT column matching cost."""
import numpy as np
import pandas as pd

from barlow_track.utils.track_using_barlow import _robust_svd
from barlow_track.utils.utils_ground_truth import calculate_accuracy, match_columns_to_gt


def _same_up_to_sign(a, b):
    signs = np.sign((a * b).sum(0))
    return np.allclose(a, b * signs, atol=1e-6)


def test_robust_svd_default_keeps_the_size_dependent_space_and_whiten_forces_one():
    rng = np.random.RandomState(0)
    small = rng.randn(500, 8) * np.arange(1, 9)
    scaled = _robust_svd(small, 4, random_state=0)
    assert np.allclose(np.linalg.norm(scaled, axis=0), np.linalg.svd(small, compute_uv=False)[:4])
    white = _robust_svd(small, 4, random_state=0, whiten=True)
    assert np.allclose(np.linalg.norm(white, axis=0), 1.0)

    tall = rng.randn(10001, 8) * np.arange(1, 9)  # the dask branch: every real video
    u = _robust_svd(tall, 4)
    assert np.allclose(np.linalg.norm(u, axis=0), 1.0)  # legacy default there: whitened
    assert _same_up_to_sign(_robust_svd(tall, 4, whiten=False),
                            u * np.linalg.svd(tall, compute_uv=False)[:4])


def _tracks(cols):
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples([(n, 'raw_segmentation_id') for n in df.columns])
    return df


def test_gt_valid_matching_does_not_penalize_tracking_through_frames_the_gt_dropped():
    nan = np.nan
    # GT neuron_A (id 1) is annotated only in frames 0-4; neuron_B (id 2) in none of p's
    gt = _tracks({'neuron_A': [1] * 5 + [nan] * 5, 'neuron_B': [2] * 10})
    pred = _tracks({'p1': [1.0] * 10,                    # follows id 1 the whole video
                    'p2': [1, 1] + [nan] * 8})           # loses it after two frames
    col = lambda df: df.droplevel(1, axis=1)
    acc = {m: calculate_accuracy(col(gt), col(match_columns_to_gt(gt, pred, 'raw_segmentation_id', m)))
           for m in ('nandist', 'gt_valid')}
    # nandist charges p1 for frames 5-9 where neuron_A is NaN, so hands neuron_A to p2
    assert np.isclose(acc['nandist']['accuracy'], 2 / 15)
    assert np.isclose(acc['gt_valid']['accuracy'], 5 / 15)
