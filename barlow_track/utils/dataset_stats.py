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


def nonrigid_matches(a, b, iters=4, k=8, resid=2.0):
    """Row of b per row of a (-1: unmatched) for consecutive frames, without GT.

    Affine ICP, then `iters` rounds moving each point by the median displacement of its k
    nearest matched points (locally rigid motion), keeping mutual nearest neighbours closer
    than half the within-frame NN distance. Finally drops matches whose displacement is
    more than `resid` voxels from their neighbours' median (None: keep all). On Zimmer
    lag 1 the residual filter takes precision from 0.84 to 0.96 at ~47 matches per frame
    (scratch/pair_pseudolabels.py)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    none = np.full(len(a), -1)
    if len(a) <= k or len(b) <= k:
        return none
    d_in = cKDTree(a).query(a, 2)[0][:, 1]
    moved, tb = affine_icp(a, b), cKDTree(b)
    for it in range(iters + 1):
        dab, iab = tb.query(moved)
        _, iba = cKDTree(moved).query(b)
        mutual = (iba[iab] == np.arange(len(a))) & (dab < 0.5 * d_in)
        am = np.where(mutual)[0]
        if len(am) <= k:
            return none
        disp = b[iab[am]] - a[am]
        if it == iters:
            break
        _, nb = cKDTree(a[am]).query(a, k)
        moved = a + np.median(disp[nb], 1)
    if resid is not None:
        _, nb = cKDTree(a[am]).query(a[am], k + 1)
        bad = np.linalg.norm(disp - np.median(disp[nb[:, 1:]], 1), axis=1) >= resid
        mutual[am[bad]] = False
    return np.where(mutual, iab, -1)


def chained_matches(frames, **match_kw):
    """Row of frames[-1] per row of frames[0] (-1: unmatched) by chaining nonrigid_matches.

    A row survives only if every consecutive pair matches it, so recall drops with the lag."""
    idx = np.arange(len(frames[0]))
    for a, b in zip(frames[:-1], frames[1:]):
        nxt = nonrigid_matches(a, b, **match_kw)
        idx = np.where(idx >= 0, nxt[np.maximum(idx, 0)], -1)
    return idx


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
