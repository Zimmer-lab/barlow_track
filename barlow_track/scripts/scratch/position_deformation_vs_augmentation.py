"""How much do neurons move between frames beyond a similarity transform, vs the position augmentation?

ViSNet position features are invariant to rotation and translation, so between the two
training views they only see the uniform scale jitter, per-point jitter (jitter_std) and
dropout. Between real frames neurons also move non-rigidly. For pairs of GT frames
(t, t + lag), fit the best similarity transform (rotation, translation, uniform scale;
Umeyama) on the shared neurons and report the residual displacement per neuron, in voxels,
next to the nearest-neighbor distance and the training jitter. Positions are (z, x, y)
voxel indices, the same units as the augmentation.

Usage (wbfm env):
  python position_deformation_vs_augmentation.py --gt flavell_data.nwb --lags 1 10 100 500
"""
import argparse
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np

PAPER = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data'


def umeyama_residuals(a, b):
    """Residuals of b vs the best similarity transform of a onto b (rows matched)."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / A.var(0).sum()
    fit = scale * A @ R.T + mb
    return np.linalg.norm(b - fit, axis=1), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=f'{PAPER}/images_for_charlie/flavell_data.nwb')
    ap.add_argument('--lags', type=int, nargs='+', default=[1, 10, 100, 500])
    ap.add_argument('--pairs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    from barlow_track.scripts.eval_accuracy import _load_project_gt

    df = _load_project_gt(args.gt)
    names = list(df.columns.get_level_values(0).unique())
    pos = np.stack([df.loc[:, [(n, c) for n in names]].values.astype(float) for c in ['z', 'x', 'y']], -1)
    T = pos.shape[0]
    print(f'{len(names)} neurons x {T} frames')

    nn = []
    for t in np.linspace(0, T - 1, 20).astype(int):
        p = pos[t][np.isfinite(pos[t]).all(1)]
        dist = np.linalg.norm(p[:, None] - p[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        nn.append(dist.min(1))
    nn = np.concatenate(nn)
    print(f'nearest-neighbor distance (voxels): median {np.median(nn):.2f}, p10 {np.percentile(nn, 10):.2f}')
    print('training position augmentation: jitter_std 1.0 voxel per axis (|d| median ~1.5), '
          'scale_jitter 0.1, dropout 0.2; nothing non-rigid')

    rng = np.random.default_rng(args.seed)
    for lag in args.lags:
        res, scales = [], []
        for t in rng.integers(0, T - lag, size=args.pairs):
            a, b = pos[t], pos[t + lag]
            ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
            if ok.sum() < 10:
                continue
            r, s = umeyama_residuals(a[ok], b[ok])
            res.append(r)
            scales.append(s)
        res = np.concatenate(res)
        print(f'lag {lag:4d}: residual after similarity fit median {np.median(res):.2f}, '
              f'p90 {np.percentile(res, 90):.2f} voxels; '
              f'fraction > half NN distance {np.mean(res > np.median(nn) / 2):.2f}; '
              f'fitted scale p5-p95 {np.percentile(scales, 5):.3f}-{np.percentile(scales, 95):.3f}')


if __name__ == '__main__':
    main()
