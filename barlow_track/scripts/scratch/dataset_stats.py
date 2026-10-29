"""GT-free dataset statistics for sizing position augmentation (segmentation only, no tracks).

Everything here is computable on a new recording before any annotation:
- detections per frame (mean, coefficient of variation);
- nearest-neighbour distance per axis (z, x, y voxels) within a frame;
- unlabelled lag-1 matching: frame t is aligned to t+1 by an affine ICP (mutual nearest
  neighbours, refitted `icp_iters` times), then mutual nearest neighbours closer than half the
  within-frame nearest-neighbour distance are matches. Lag-1 motion is comparable to the
  neighbour spacing on Zimmer and Flavell, so unaligned nearest neighbours are mostly wrong.
  The fraction of detections without a partner bounds how often detections appear, vanish,
  merge or split between frames;
- centroid noise per axis from those matches: each match's displacement minus the mean
  displacement of its k nearest matched neighbours (removes locally rigid motion), robust
  std / sqrt(2), as centroid_noise.py does with GT tracks. A GT-free counterpart of it.

Usage: python barlow_track/scripts/scratch/dataset_stats.py [zimmer flavell samuel]
"""
import logging
import sys

import numpy as np
from scipy.spatial import cKDTree

from centroid_noise import PROJECTS


def _mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v, 0)), 0)


def _affine_icp(a, b, iters=10):
    """a moved onto b by an affine refitted on mutual nearest neighbours; returns moved a."""
    tb, moved = cKDTree(b), a.copy()
    for _ in range(iters):
        d, j = tb.query(moved)
        _, back = cKDTree(moved).query(b)
        m = back[j] == np.arange(len(a))
        if m.sum() < 8:
            break
        X = np.c_[a[m], np.ones(m.sum())]
        A, *_ = np.linalg.lstsq(X, b[j[m]], rcond=None)
        moved = np.c_[a, np.ones(len(a))] @ A
    return moved


def frame_stats(project_data, every=5, k=6, max_frames=None, icp_iters=10):
    from barlow_track.utils.volume_data import get_centroids_for_volume
    n = project_data.num_frames if max_frames is None else min(max_frames, project_data.num_frames)
    counts, nn_axis, unmatched, residuals = [], [], [], []
    for t in range(0, n - 1, every):
        a = get_centroids_for_volume(project_data, t)[0]
        b = get_centroids_for_volume(project_data, t + 1)[0]
        counts.append(len(a))
        if len(a) <= k or len(b) <= k:
            continue
        d_in, i_in = cKDTree(a).query(a, 2)
        nn_axis.append(np.abs(a - a[i_in[:, 1]]))
        a_al = _affine_icp(a, b, icp_iters) if icp_iters else a
        ta, tb = cKDTree(a_al), cKDTree(b)
        dab, iab = tb.query(a_al)
        _, iba = ta.query(b)
        mutual = (iba[iab] == np.arange(len(a))) & (dab < 0.5 * d_in[:, 1])
        unmatched.append(1 - mutual.mean())
        am, bm = a_al[mutual], b[iab[mutual]]
        if len(am) <= k:
            continue
        disp = bm - am
        _, nbr = cKDTree(am).query(am, k + 1)
        residuals.append(disp - disp[nbr[:, 1:]].mean(1))
    nn = np.concatenate(nn_axis)
    return dict(detections_mean=float(np.mean(counts)), detections_cv=float(np.std(counts) / np.mean(counts)),
                nn_median_zxy=np.median(nn, 0).round(2).tolist(),
                unmatched_lag1=float(np.mean(unmatched)),
                noise_zxy=(_mad(np.concatenate(residuals)) / np.sqrt(2)).round(2).tolist())


def main(labs):
    logging.disable(logging.CRITICAL)
    from wbfm.utils.projects.finished_project_data import ProjectData
    for lab in labs:
        p = ProjectData.load_final_project_data(PROJECTS[lab], allow_hybrid_loading=True)
        s = frame_stats(p)
        print(f"{lab}: volume {tuple(p.red_data.shape[1:])}, " + ", ".join(f"{k} {v}" for k, v in s.items()), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or list(PROJECTS))
