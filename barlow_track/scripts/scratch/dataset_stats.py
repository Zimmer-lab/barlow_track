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

from barlow_track.utils.dataset_stats import centroid_noise_from_pairs, lag1_matches


def frame_stats(project_data, every=5, k=6, max_frames=None, icp_iters=10):
    from barlow_track.utils.volume_data import get_centroids_for_volume
    n = project_data.num_frames if max_frames is None else min(max_frames, project_data.num_frames)
    counts, nn_axis, unmatched, pairs = [], [], [], []
    for t in range(0, n - 1, every):
        a = get_centroids_for_volume(project_data, t)[0]
        b = get_centroids_for_volume(project_data, t + 1)[0]
        counts.append(len(a))
        if len(a) <= k or len(b) <= k:
            continue
        nn_axis.append(np.abs(a - a[cKDTree(a).query(a, 2)[1][:, 1]]))
        unmatched.append(1 - lag1_matches(a, b, icp_iters)[2].mean())
        pairs.append((a, b))
    nn = np.concatenate(nn_axis)
    return dict(detections_mean=float(np.mean(counts)), detections_cv=float(np.std(counts) / np.mean(counts)),
                nn_median_zxy=np.median(nn, 0).round(2).tolist(),
                unmatched_lag1=float(np.mean(unmatched)),
                noise_zxy=np.round(centroid_noise_from_pairs(pairs, k, icp_iters), 2).tolist())


def main(labs):
    logging.disable(logging.CRITICAL)
    from wbfm.utils.projects.finished_project_data import ProjectData
    for lab in labs:
        p = ProjectData.load_final_project_data(PROJECTS[lab], allow_hybrid_loading=True)
        s = frame_stats(p)
        print(f"{lab}: volume {tuple(p.red_data.shape[1:])}, " + ", ".join(f"{k} {v}" for k, v in s.items()), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or list(PROJECTS))
