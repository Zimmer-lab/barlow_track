"""How much does local geometry (neighbor distances) change between frames vs under augmentation?

ViSNet position features see relative geometry: distances and angles to neighbors within
a cutoff of 0.35 in normalized keypoint units (normalize_keypoints divides by 0.7 x the
largest volume dimension), i.e. 0.245 x max(volume shape) voxels.

Per dataset:
  - nearest-neighbor (NN) distance, ViSNet cutoff, and the residual motion after the best
    similarity transform (rotation, translation, uniform scale) between frames;
  - real frames: for neuron pairs within the cutoff at frame t, the std of log(distance at
    t + lag / distance at t), binned by initial distance in units of the NN distance;
  - augmentation: the same for the two training views of one GT frame (global affine with
    scale jitter, optional global elastic field, per-point jitter), for the current
    settings and a grid of elastic settings. The grid is set in NN units: elastic xy std
    a x NN, z std scaled by the worm's z/x extent, length b x NN. The best match to real
    frames at --target_lag (RMS log difference over the bins) is reported.
Distances are in voxels (the augmentation's units), so z anisotropy is as the network sees it.

Usage (wbfm env):
  python local_geometry_change.py --datasets flavell zimmer_1128 leifer samuel
"""
import argparse
import itertools
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np

_P = '/lisc/data/scratch/neurobiology/zimmer/fieseler'
DATASETS = {
    'zimmer_1128': f'{_P}/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format',
    'flavell': f'{_P}/barlow_track_paper/flavell_data/images_for_charlie/flavell_data.nwb',
    'leifer': f'{_P}/barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb',
    'samuel': f'{_P}/barlow_track_paper/samuel_data/153.nwb',
}
BINS_NN = np.array([0, 1.5, 2.5, 4, 6, 10, 1e9])  # in NN units; last bin ends at the cutoff
CUTOFF_FRACTION = 0.35 * 0.7  # ViSNet cutoff / max volume dimension


def load_positions(path):
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_project_gt

    df = _load_project_gt(path)
    names = list(df.columns.get_level_values(0).unique())
    pos = np.stack([df.loc[:, [(n, c) for n in names]].values.astype(float) for c in ['z', 'x', 'y']], -1)
    shape = ProjectData.load_final_project_data(path, allow_hybrid_loading=True, verbose=0).red_data.shape[1:]
    return pos, tuple(int(s) for s in shape)


def umeyama_residuals(a, b):
    """Residual vectors of b vs the best similarity transform of a onto b, and the scale."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / A.var(0).sum()
    return b - (scale * A @ R.T + mb), scale


def pair_log_ratios(a, b, d_max):
    """log(|b_i - b_j| / |a_i - a_j|) for pairs with |a_i - a_j| < d_max, with the initial distance."""
    da = np.linalg.norm(a[:, None] - a[None], axis=-1)
    db = np.linalg.norm(b[:, None] - b[None], axis=-1)
    iu = np.triu_indices(len(a), 1)
    keep = (da[iu] < d_max) & (da[iu] > 0)
    return np.log(db[iu][keep] / da[iu][keep]), da[iu][keep]


def binned_std(lr, d0, bins):
    idx = np.digitize(d0, bins)
    return np.array([lr[idx == k].std() if np.sum(idx == k) > 20 else np.nan for k in range(1, len(bins))])


def augmented_views(p, shape, rng, elastic_std, elastic_length, jitter_std, scale_jitter):
    from barlow_track.utils.volume_data import apply_position_jitter, sample_elastic_field, sample_global_affine

    center = (np.array(shape) - 1) / 2
    views = []
    for _ in range(2):  # two training views, as in VolumeCoordsDataset
        R, tv = sample_global_affine(rng, scale_jitter=scale_jitter)
        q = (p - center) @ R.T + center + tv
        field = sample_elastic_field(rng, shape, elastic_std=elastic_std, elastic_length=elastic_length)
        if field is not None:
            q = field.forward_points(q)
        views.append(apply_position_jitter(q, rng, jitter_std))
    return views


def analyze(label, path, args):
    pos, shape = load_positions(path)
    T = pos.shape[0]
    frames = np.linspace(0, T - 1, 20).astype(int)
    nn = []
    for t in frames:
        q = pos[t][np.isfinite(pos[t]).all(1)]
        if len(q) < 10:
            continue
        d = np.linalg.norm(q[:, None] - q[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn.append(d.min(1))
    nn = float(np.median(np.concatenate(nn)))
    cutoff = CUTOFF_FRACTION * max(shape)
    extent = np.nanpercentile(pos.reshape(-1, 3), 95, axis=0) - np.nanpercentile(pos.reshape(-1, 3), 5, axis=0)
    bins = BINS_NN * nn
    bins[-1] = cutoff
    print(f'\n=== {label}: volume {shape}, {pos.shape[1]} neurons x {T} frames; NN distance {nn:.1f} voxels; '
          f'ViSNet cutoff {cutoff:.0f} voxels = {cutoff / nn:.1f} NN; worm extent p5-p95 (z, x, y) {np.round(extent, 0)}')
    head = ', '.join(f'{BINS_NN[k]:g}-{BINS_NN[k + 1]:g}' if k < len(BINS_NN) - 2 else f'{BINS_NN[k]:g}-cutoff'
                     for k in range(len(BINS_NN) - 1))
    print(f'  std log distance ratio by initial distance (NN units: {head})')

    rng = np.random.default_rng(0)
    real = {}
    for lag in args.lags:
        lrs, d0s, mags, scales = [], [], [], []
        for t in rng.integers(0, T - lag, size=args.pairs):
            ok = np.isfinite(pos[t]).all(1) & np.isfinite(pos[t + lag]).all(1)
            if ok.sum() < 10:
                continue
            lr, d0 = pair_log_ratios(pos[t][ok], pos[t + lag][ok], cutoff)
            lrs.append(lr)
            d0s.append(d0)
            r, s = umeyama_residuals(pos[t][ok], pos[t + lag][ok])
            mags.append(np.linalg.norm(r, axis=1))
            scales.append(s)
        real[lag] = binned_std(np.concatenate(lrs), np.concatenate(d0s), bins)
        mags = np.concatenate(mags)
        print(f'  real lag {lag:4d}: {np.round(real[lag], 3)}; residual after similarity fit median '
              f'{np.median(mags) / nn:.2f} NN; scale p5-p95 {np.percentile(scales, 5):.3f}-{np.percentile(scales, 95):.3f}')

    target = real[args.target_lag]
    sample_frames = rng.integers(0, T, size=args.pairs // 2)

    def aug_profile(elastic_std, elastic_length, jitter_std):
        r = np.random.RandomState(0)
        lrs, d0s = [], []
        for t in sample_frames:
            p = pos[t][np.isfinite(pos[t]).all(1)]
            if len(p) < 10:
                continue
            v1, v2 = augmented_views(p, shape, r, elastic_std, elastic_length, jitter_std, args.scale_jitter)
            lr, d0 = pair_log_ratios(v1, v2, cutoff)
            lrs.append(lr)
            d0s.append(d0)
        prof = binned_std(np.concatenate(lrs), np.concatenate(d0s), bins)
        ok = np.isfinite(prof) & np.isfinite(target)
        return prof, float(np.sqrt(np.mean(np.log(prof[ok] / target[ok]) ** 2)))

    cur, cur_err = aug_profile(0.0, 40.0, args.current_jitter)
    print(f'  current augmentation (jitter {args.current_jitter} voxel, no elastic): {np.round(cur, 3)}; '
          f'RMS log mismatch vs lag {args.target_lag}: {cur_err:.3f}')
    z_ratio = extent[0] / extent[1]
    results = []
    for a, b, j in itertools.product(args.grid_std_nn, args.grid_length_nn, args.grid_jitter_nn):
        xy = a * nn
        std = (xy * z_ratio, xy, xy)
        prof, err = aug_profile(std, b * nn, j * nn)
        results.append((err, a, b, j, std, b * nn, j * nn, prof))
    results.sort(key=lambda r: r[0])
    for err, a, b, j, std, length, jit, prof in results[:3]:
        print(f'  match: elastic_std {np.round(std, 1)} (xy {a} NN), elastic_length {length:.0f} ({b} NN), '
              f'jitter_std {jit:.2f} ({j} NN): {np.round(prof, 3)}; RMS log mismatch {err:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--lags', type=int, nargs='+', default=[1, 10, 100])
    ap.add_argument('--target_lag', type=int, default=100)
    ap.add_argument('--pairs', type=int, default=100)
    ap.add_argument('--current_jitter', type=float, default=1.0)
    ap.add_argument('--scale_jitter', type=float, default=0.1)
    ap.add_argument('--grid_std_nn', type=float, nargs='+', default=[0.3, 0.6, 0.9, 1.2])
    ap.add_argument('--grid_length_nn', type=float, nargs='+', default=[3, 6, 10])
    ap.add_argument('--grid_jitter_nn', type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.15])
    args = ap.parse_args()
    for label in args.datasets:
        analyze(label, DATASETS[label], args)


if __name__ == '__main__':
    main()
