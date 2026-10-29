"""GT-free statistics of a recording's detections, for sizing position augmentation.

Everything here uses only the segmentation (per-frame centroids), so it is available on a
new recording before any annotation. Settings derived from it generalize the way the method
must; nothing here may read ground-truth tracks.
"""
import numpy as np
from scipy.spatial import cKDTree


def _mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v, 0)), 0)


def affine_icp(a, b, iters=10):
    """Points a moved onto b by an affine refitted on mutual nearest neighbours."""
    tb, moved = cKDTree(b), np.asarray(a, dtype=float).copy()
    for _ in range(iters):
        _, j = tb.query(moved)
        _, back = cKDTree(moved).query(b)
        m = back[j] == np.arange(len(a))
        if m.sum() < 8:
            break
        A, *_ = np.linalg.lstsq(np.c_[a[m], np.ones(m.sum())], b[j[m]], rcond=None)
        moved = np.c_[a, np.ones(len(a))] @ A
    return moved


def lag1_matches(a, b, icp_iters=10):
    """(a aligned onto b, index into b per row of a, mutual-match mask) for consecutive frames.

    Matches are mutual nearest neighbours after an affine ICP, closer than half the
    within-frame nearest-neighbour distance of the row in a.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d_in = cKDTree(a).query(a, 2)[0][:, 1]
    a_al = affine_icp(a, b, icp_iters) if icp_iters else a
    dab, iab = cKDTree(b).query(a_al)
    _, iba = cKDTree(a_al).query(b)
    mutual = (iba[iab] == np.arange(len(a))) & (dab < 0.5 * d_in)
    return a_al, iab, mutual


def centroid_noise_from_pairs(pairs, k=6, icp_iters=10):
    """Per-axis centroid noise (z, x, y), robust std / sqrt(2), from consecutive-frame centroid pairs.

    Each match's displacement minus the mean displacement of its k nearest matched
    neighbours removes locally rigid motion; what is left is both frames' centroid noise
    plus local non-rigid motion, so this is an upper bound. On Zimmer, Flavell and Samuel it
    is within ~2x of the same estimate from GT tracks and keeps their anisotropy
    (scratch/dataset_stats.py, scratch/centroid_noise.py). None with too few matches.
    """
    residuals = []
    for a, b in pairs:
        if len(a) <= k or len(b) <= k:
            continue
        a_al, iab, mutual = lag1_matches(a, b, icp_iters)
        am, bm = a_al[mutual], np.asarray(b, dtype=float)[iab[mutual]]
        if len(am) <= k:
            continue
        disp = bm - am
        _, nbr = cKDTree(am).query(am, k + 1)
        residuals.append(disp - disp[nbr[:, 1:]].mean(1))
    if not residuals:
        return None
    return _mad(np.concatenate(residuals)) / np.sqrt(2)
