"""Per-axis centroid noise of tracked neurons, to size position_augment.jitter_std.

For each GT neuron, the lag-1 displacement minus the mean lag-1 displacement of its 6
nearest neighbours removes the locally rigid part of the motion; what is left is
centroid noise of both frames plus local non-rigid motion. Robust std / sqrt(2) is
then an upper bound on one frame's centroid noise, in voxels (z, x, y). Training
jitter is drawn independently per view, so a matched jitter_std equals this value.

Usage: python barlow_track/scripts/scratch/centroid_noise.py [zimmer flavell samuel]
"""
import logging
import sys

import numpy as np

PROJECTS = {
    'zimmer': '/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/'
              'ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format/project_config.yaml',
    'flavell': '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/'
               '2025_07_01/project_config.yaml',
    'samuel': '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data/projects/'
              '153/project_config.yaml',
}


def centroid_noise(tracks, every=5, k=6):
    names = sorted(set(tracks.columns.get_level_values(0)))
    X = np.stack([tracks.loc[:, [(n, c) for n in names]].to_numpy(dtype=float) for c in ('z', 'x', 'y')], -1)
    res = []
    for t in range(0, len(X) - 1, every):
        a, b = X[t], X[t + 1]
        ok = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
        a, b = a[ok], b[ok]
        if len(a) <= k:
            continue
        d = b - a
        dist = np.linalg.norm(a[:, None] - a[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        nn = np.argsort(dist, 1)[:, :k]
        res.append(d - d[nn].mean(1))
    r = np.concatenate(res)
    mad = 1.4826 * np.median(np.abs(r - np.median(r, 0)), 0)
    return mad / np.sqrt(2), len(r)


def main(labs):
    logging.disable(logging.CRITICAL)
    from wbfm.utils.projects.finished_project_data import ProjectData
    for lab in labs:
        p = ProjectData.load_final_project_data(PROJECTS[lab], allow_hybrid_loading=True)
        tracks = p.final_tracks
        if tracks is None:  # samuel 153: GT only as finished neurons, as eval_accuracy reads it
            tracks = p.get_final_tracks_only_finished_neurons()[0]
        noise, n = centroid_noise(tracks)
        print(f"{lab}: centroid noise (z, x, y) voxels {np.round(noise, 2).tolist()} from {n} displacements, "
              f"volume {tuple(p.red_data.shape[1:])}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or list(PROJECTS))
