"""How far each pipeline's training crops are from the crops the network sees at evaluation.

Geometry only (volume shape + segmentation-metadata centroids; no image data is read),
for a sample of frames, per object and view:

- eval: crop centred on the rounded centroid in the raw volume, window clamped to the
  volume (get_3d_crop_using_bbox_or_centroid), so edge neurons sit off-centre.
- volume (position pipeline): sample_global_affine on the whole volume, objects whose
  moved centroid leaves the volume dropped (in_bounds_mask), crop clamped in the rotated
  volume. Voxels whose source lies outside the raw volume are border-replicated
  (ndimage mode='nearest') in training; they have no counterpart at evaluation.
- legacy: the eval crop, then torchio RandomAffine about the crop centre (z rotation up
  to 180 deg, scale 0.9-1.1); voxels from outside the crop are padded with the minimum,
  and an off-centre neuron moves with the rotation.

Reports per pipeline: fraction of objects kept, neuron offset from the crop centre
(voxels, in-plane), fraction of crop voxels with no source in the raw data (fill).
The elastic field, jitter and photometric transforms are left out (small next to these).

Usage (wbfm env):
  python crop_geometry_by_pipeline.py --project P/project_config.yaml --label flavell \
      [--target_sz 16 64 64] [--n_frames 50] [--views 4] [--seed 0]
"""
import argparse

import numpy as np

from barlow_track.utils.volume_data import (DEFAULT_GLOBAL_ARGS, get_centroids_for_volume, in_bounds_mask,
                                            sample_global_affine)


def clamped_window(center, dims, target):
    """(start (3,), rounded centre (3,)) of the crop window, as get_3d_crop_using_bbox_or_centroid."""
    c = np.round(center).astype(int)
    start = np.empty(3, dtype=int)
    for a in range(3):
        if dims[a] <= target[a]:
            start[a] = 0
        else:
            start[a] = max(0, min(c[a] - target[a] // 2, dims[a] - target[a]))
    return start, c


def crop_grid(start, target):
    """(3, M) voxel coordinates of a crop window starting at `start`."""
    g = np.indices(tuple(target)).reshape(3, -1)
    return g + start[:, None]


def outside(coords, dims):
    """(M,) bool: coordinate outside [0, dim - 1] on any axis."""
    d = np.asarray(dims)[:, None]
    return ((coords < -0.5) | (coords > d - 0.5)).any(0)


def offset_inplane(center, start, target):
    """In-plane distance (voxels) from the neuron to the crop centre."""
    mid = start + (np.asarray(target) - 1) / 2.0
    return float(np.hypot(*(np.asarray(center)[1:] - mid[1:])))


def measure(project_path, target, n_frames, n_views, seed, sub=4):
    from wbfm.utils.projects.finished_project_data import ProjectData
    pd_ = ProjectData.load_final_project_data(project_path)
    dims = np.array(pd_.red_data.shape[1:])
    rng = np.random.RandomState(seed)
    frames = rng.choice(pd_.num_frames, size=min(n_frames, pd_.num_frames), replace=False)
    target = np.asarray(target)
    # Every `sub`-th crop voxel per axis is enough for the fill fractions
    g_local = np.indices(tuple(target))[:, ::sub, ::sub, ::sub].reshape(3, -1)
    crop_mid = (target - 1) / 2.0
    out = {k: dict(n=0, kept=0, off=[], fill=[]) for k in ('eval', 'volume', 'legacy')}
    for t in frames:
        zxy, _ = get_centroids_for_volume(pd_, int(t))
        if len(zxy) == 0:
            continue
        # eval and legacy start from the same clamped crop of the raw volume
        for p in zxy:
            start, _ = clamped_window(p, dims, target)
            out['eval']['n'] += 1
            out['eval']['kept'] += 1
            out['eval']['off'].append(offset_inplane(p, start, target))
            out['eval']['fill'].append(0.0)
            for _ in range(n_views):
                # torchio RandomAffine(degrees=(180, 0, 0)): rotation about the crop's z axis
                theta = np.deg2rad(rng.uniform(-180, 180))
                s = rng.uniform(0.9, 1.1)
                c, si = np.cos(theta), np.sin(theta)
                R = s * np.array([[1, 0, 0], [0, c, -si], [0, si, c]])
                src = np.linalg.inv(R) @ (g_local - crop_mid[:, None]) + crop_mid[:, None]
                out['legacy']['n'] += 1
                out['legacy']['kept'] += 1
                out['legacy']['fill'].append(float(outside(src, target).mean()))
                p_crop = np.asarray(p) - start  # neuron in crop coordinates
                p_new = R @ (p_crop - crop_mid) + crop_mid
                out['legacy']['off'].append(float(np.hypot(*(p_new[1:] - crop_mid[1:]))))
        center = (dims - 1) / 2.0
        for _ in range(n_views):
            R, tv = sample_global_affine(rng, **{k: DEFAULT_GLOBAL_ARGS[k] for k in
                                                  ('p_global_affine', 'max_degrees_z', 'scale_jitter',
                                                   'max_translation', 'p_flip')})
            pts = (zxy - center) @ R.T + center + tv
            keep = in_bounds_mask(pts, dims)
            out['volume']['n'] += len(pts)
            out['volume']['kept'] += int(keep.sum())
            Rinv = np.linalg.inv(R)
            for p in pts[keep]:
                start, _ = clamped_window(p, dims, target)
                y = g_local + start[:, None]
                src = Rinv @ (y - (center + tv)[:, None]) + center[:, None]
                out['volume']['fill'].append(float(outside(src, dims).mean()))
                out['volume']['off'].append(offset_inplane(p, start, target))
    return dims, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--target_sz', type=int, nargs=3, default=[16, 64, 64])
    ap.add_argument('--n_frames', type=int, default=50)
    ap.add_argument('--views', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dims, out = measure(args.project, args.target_sz, args.n_frames, args.views, args.seed)
    print(f"{args.label}: volume (Z, X, Y) {tuple(int(d) for d in dims)}, crop {tuple(args.target_sz)}, "
          f"{args.n_frames} frames, {args.views} views")
    print(f"{'pipeline':8s} {'kept':>6s} {'offset med':>10s} {'offset p90':>10s} {'off>8 vox':>9s} "
          f"{'fill mean':>9s} {'fill>10%':>8s} {'fill>30%':>8s}")
    for k, d in out.items():
        off, fill = np.asarray(d['off']), np.asarray(d['fill'])
        print(f"{k:8s} {d['kept'] / max(d['n'], 1):6.3f} {np.median(off):10.1f} {np.percentile(off, 90):10.1f} "
              f"{(off > 8).mean():9.3f} {fill.mean():9.3f} {(fill > 0.1).mean():8.3f} {(fill > 0.3).mean():8.3f}")


if __name__ == '__main__':
    main()
