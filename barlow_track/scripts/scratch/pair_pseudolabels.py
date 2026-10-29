"""How often GT-free frame-pair matches (dataset_stats.chained_matches) name the same neuron.

Diagnostic for position_augment real-frame pairs: the pairs and their identities come from
the segmentation alone; GT tracks are read here only to score them. For lags 1, 2, 3, 5
and residual filters (None: off) over frames spread through the video: matches per frame,
the fraction of matches between two GT-labelled detections, precision among those, and recall of GT neurons present in
both frames.

Usage: python barlow_track/scripts/scratch/pair_pseudolabels.py [zimmer flavell samuel]
"""
import logging
import sys

import numpy as np

from barlow_track.scripts.scratch.centroid_noise import PROJECTS
from barlow_track.utils.dataset_stats import chained_matches
from barlow_track.utils.volume_data import get_centroids_for_volume


def gt_names(tracks, t, seg):
    """GT neuron name per detection (by raw segmentation id), None where unlabelled."""
    row = tracks.xs('raw_segmentation_id', axis=1, level=1).iloc[t]
    by_seg = {int(v): n for n, v in row.items() if v == v}
    return [by_seg.get(int(s)) for s in seg]


def main(labs, lags=(1, 2, 3, 5), n_starts=40, resid=(None, 2.0, 1.5)):
    logging.disable(logging.CRITICAL)
    from wbfm.utils.projects.finished_project_data import ProjectData
    for lab in labs:
        p = ProjectData.load_final_project_data(PROJECTS[lab], allow_hybrid_loading=True)
        tracks = p.final_tracks
        if tracks is None:
            tracks = p.get_final_tracks_only_finished_neurons()[0]
        T = min(p.num_frames, len(tracks))
        starts = np.linspace(0, T - max(lags) - 1, n_starts).astype(int)
        cache = {}

        def frame(t):
            if t not in cache:
                cache[t] = get_centroids_for_volume(p, int(t))
            return cache[t]
        for lag, r in [(lag, r) for r in resid for lag in lags]:
            n_match, n_lab, n_ok, n_gt, n_gt_ok = 0, 0, 0, 0, 0
            for t in starts:
                frames, segs = zip(*[frame(t + k) for k in range(lag + 1)])
                idx = chained_matches(frames, resid=r)
                na, nb = gt_names(tracks, t, segs[0]), gt_names(tracks, t + lag, segs[-1])
                for i, j in enumerate(idx):
                    if j < 0:
                        continue
                    n_match += 1
                    if na[i] is not None and nb[j] is not None:
                        n_lab += 1
                        n_ok += na[i] == nb[j]
                both = set(n for n in na if n) & set(n for n in nb if n)
                n_gt += len(both)
                n_gt_ok += sum(1 for i, j in enumerate(idx) if j >= 0 and na[i] in both and na[i] == nb[j])
            print(f"{lab} lag {lag:2d} resid {r}: {n_match / len(starts):6.1f} matches/frame, "
                  f"{n_lab / max(n_match, 1):.2f} both GT-labelled, precision {n_ok / max(n_lab, 1):.3f}, "
                  f"GT recall {n_gt_ok / max(n_gt, 1):.3f}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or list(PROJECTS))
