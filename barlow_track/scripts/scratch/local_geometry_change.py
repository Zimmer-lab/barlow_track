"""How much does local geometry (neighbor distances) change between frames vs under augmentation?

ViSNet position features see relative geometry: distances and angles to neighbors.
Real frames: for neuron pairs closer than d_max at frame t, the log ratio of their
distance at t + lag. Augmentation: the same pairs in one GT frame, before vs after
the training position augmentation (scale jitter, per-point jitter, and optionally a
global elastic field of a given strength). Reported as the std of log(distance ratio),
binned by the initial distance, plus the fraction of near-folding pairs (ratio < 0.5).

Usage (wbfm env):
  python local_geometry_change.py --elastic_std 0 0.5,1,1 1,2,2 1,3,3
"""
import argparse
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np

PAPER = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data'
BINS = np.array([0, 8, 12, 20, 30, 50, 70, 140])


def pair_log_ratios(a, b, d_max=BINS[-1]):
    """log(|b_i - b_j| / |a_i - a_j|) for pairs with |a_i - a_j| < d_max, with the initial distance."""
    da = np.linalg.norm(a[:, None] - a[None], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None], axis=-1)
    iu = np.triu_indices(len(a), 1)
    keep = da[iu] < d_max
    return np.log(db[iu][keep] / da[iu][keep]), da[iu][keep]


def summarize(label, lr, d0):
    idx = np.digitize(d0, BINS)
    cells = [f'{BINS[k - 1]}-{BINS[k]}: {lr[idx == k].std():.3f}' for k in range(1, len(BINS))]
    print(f'{label:34s} std log ratio by distance  ' + ', '.join(cells) +
          f';  ratio<0.5 {np.mean(lr < np.log(0.5)):.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=f'{PAPER}/images_for_charlie/flavell_data.nwb')
    ap.add_argument('--shape', type=int, nargs=3, default=[64, 284, 120])
    ap.add_argument('--lags', type=int, nargs='+', default=[1, 10, 100])
    ap.add_argument('--elastic_std', nargs='+', default=['0', '0.5,1,1', '1,2,2', '1,3,3'])
    ap.add_argument('--elastic_length', type=float, default=15.0)
    ap.add_argument('--jitter_std', type=float, default=1.0)
    ap.add_argument('--scale_jitter', type=float, default=0.1)
    ap.add_argument('--pairs', type=int, default=100)
    args = ap.parse_args()
    from barlow_track.scripts.eval_accuracy import _load_project_gt
    from barlow_track.utils.volume_data import (apply_global_affine, apply_position_jitter,
                                                sample_elastic_field, sample_global_affine)

    df = _load_project_gt(args.gt)
    names = list(df.columns.get_level_values(0).unique())
    pos = np.stack([df.loc[:, [(n, c) for n in names]].values.astype(float) for c in ['z', 'x', 'y']], -1)
    T = pos.shape[0]
    rng = np.random.default_rng(0)

    for lag in args.lags:
        lrs, d0s = [], []
        for t in rng.integers(0, T - lag, size=args.pairs):
            ok = np.isfinite(pos[t]).all(1) & np.isfinite(pos[t + lag]).all(1)
            lr, d0 = pair_log_ratios(pos[t][ok], pos[t + lag][ok])
            lrs.append(lr)
            d0s.append(d0)
        summarize(f'real frames, lag {lag}', np.concatenate(lrs), np.concatenate(d0s))

    vol = np.zeros(args.shape, dtype=np.float32)  # only the points are used
    for es in args.elastic_std:
        std = [float(v) for v in es.split(',')]
        std = std[0] if len(std) == 1 else std
        r = np.random.RandomState(0)
        lrs, d0s = [], []
        for t in rng.integers(0, T, size=args.pairs // 2):
            p = pos[t][np.isfinite(pos[t]).all(1)]
            views = []
            for _ in range(2):  # two training views, as in VolumeCoordsDataset
                R, tv = sample_global_affine(r, scale_jitter=args.scale_jitter)
                field = sample_elastic_field(r, vol.shape, elastic_std=std, elastic_length=args.elastic_length)
                if field is None:
                    _, q = apply_global_affine(vol, p, R, tv)
                else:
                    q = field.forward_points((p - (np.array(vol.shape) - 1) / 2) @ R.T + (np.array(vol.shape) - 1) / 2 + tv)
                views.append(apply_position_jitter(q, r, args.jitter_std))
            lr, d0 = pair_log_ratios(views[0], views[1])
            lrs.append(lr)
            d0s.append(d0)
        summarize(f'augmentation, elastic_std {es}', np.concatenate(lrs), np.concatenate(d0s))


if __name__ == '__main__':
    main()
