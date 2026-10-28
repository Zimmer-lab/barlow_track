"""Which ViSNet position-encoder settings can tell neurons apart before any training?

Trained Flavell position branches carry about 2 effective dimensions, the same as at
random initialization (position_branch_matching.py). This probes the architecture
itself: for a grid of cutoff (normalized keypoint units; Flavell's nearest-neighbour
distance is about 0.034), max neighbours and canonicalize_keypoints, build a randomly
initialized ViSNetPositionEncoder + LayerNorm (seeds 0-2) and measure on Flavell frames
  - effective rank (participation ratio) of the per-frame standardized descriptors,
  - self-matching under 1 voxel of jitter (the training jitter),
  - matching between real frames at lag 1 and lag 100 (GT neurons by name),
with the Hungarian assignment of position_branch_matching.py.

Usage (wbfm env):
  python visnet_capacity_probe.py --frames 30 --out probe.jsonl
"""
import argparse
import itertools
import json
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np
import torch
import torch.nn.functional as F

from barlow_track.scripts.scratch.position_branch_matching import PROJECT, match_accuracy


class _Branch(torch.nn.Module):
    def __init__(self, cutoff, neighbors, canonicalize, layers, hidden=64, dim=1024):
        super().__init__()
        from barlow_track.utils.barlow_superglue import ViSNetPositionEncoder
        self.enc = ViSNetPositionEncoder(dim, hidden_channels=hidden, num_layers=layers, cutoff=cutoff,
                                         max_num_neighbors=neighbors)
        self.norm = torch.nn.LayerNorm(dim)
        self.canonicalize = canonicalize

    def forward(self, kpts):
        from barlow_track.utils.barlow_superglue import canonicalize_keypoints
        if self.canonicalize:
            kpts = canonicalize_keypoints(kpts)
        return self.norm(self.enc(kpts))


def embed(branch, points, shape, device):
    from barlow_track.utils.volume_data import VolumeCoordsDataset
    kpts = VolumeCoordsDataset._normalize(torch.as_tensor(points, dtype=torch.float32), shape).to(device)
    with torch.no_grad():
        d = branch(kpts)
    d = (d - d.mean(0, keepdim=True)) / d.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return F.normalize(d, dim=1).cpu().numpy()


def eff_rank(d):
    sv = np.linalg.svd(d, compute_uv=False)
    return float((sv ** 2).sum() ** 2 / (sv ** 4).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default=PROJECT)
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--cutoffs', type=float, nargs='+', default=[0.04, 0.07, 0.1, 0.15, 0.25, 0.35])
    ap.add_argument('--neighbors', type=int, nargs='+', default=[8, 32])
    ap.add_argument('--layers', type=int, nargs='+', default=[3])
    ap.add_argument('--canonicalize', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--init_seeds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_project_gt
    from barlow_track.utils.volume_data import get_centroids_for_volume

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    project_data = ProjectData.load_final_project_data(args.project, allow_hybrid_loading=True, verbose=0)
    shape = tuple(project_data.red_data.shape[1:])
    df = _load_project_gt(args.project)
    seg = df.loc[:, (slice(None), 'raw_segmentation_id')]
    frames = [int(t) for t in df.index]
    rng = np.random.RandomState(args.seed)
    starts = [int(t) for t in rng.choice([t for t in frames if t + 100 in set(frames)], args.frames, replace=False)]
    needed = sorted(set(starts) | {t + 1 for t in starts} | {t + 100 for t in starts})
    dets = {t: get_centroids_for_volume(project_data, t) for t in needed}

    def names(t):
        row = seg.loc[t]
        to_name = {int(v): n for (n, _), v in row.items() if np.isfinite(v)}
        return [to_name.get(int(s), ('det', t, int(s))) for s in dets[t][1]], set(to_name.values())

    for cutoff, neighbors, layers, canon in itertools.product(args.cutoffs, args.neighbors, args.layers,
                                                               args.canonicalize):
        stats = dict(rank=[], jitter=[], lag1=[], lag100=[])
        for init in range(args.init_seeds):
            torch.manual_seed(init)
            branch = _Branch(cutoff, neighbors, bool(canon), layers).to(device).eval()
            jrng = np.random.RandomState(init)
            for t in starts:
                z, s = dets[t]
                d = embed(branch, z, shape, device)
                stats['rank'].append(eff_rank(d))
                dj = embed(branch, z + jrng.normal(0, 1.0, z.shape), shape, device)
                stats['jitter'].append(match_accuracy(d, dj, list(s), list(s), set(s))[0])
                n0, g0 = names(t)
                for lag in (1, 100):
                    n1, g1 = names(t + lag)
                    d1 = embed(branch, dets[t + lag][0], shape, device)
                    stats[f'lag{lag}'].append(match_accuracy(d, d1, n0, n1, g0 & g1)[0])
        rec = dict(cutoff=cutoff, neighbors=neighbors, layers=layers, canonicalize=bool(canon),
                   **{k: float(np.nanmean(v)) for k, v in stats.items()})
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
