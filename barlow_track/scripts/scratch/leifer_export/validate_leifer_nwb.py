"""Validate a re-exported Leifer NWB (export_leifer_nwb.py) against the raw files and the old NWB. Read-only.

Checks:
  pixels   per plane identity of CalciumImageSeries vs Leifer_NeRVE_Worm1.nwb (rule A only)
  gt       NeuronCentroidsManual vs the old CentroidSeries (names neuron_XXX = control + 1)
  seg      label counts, GT match rate, NeuronCentroids == mask regionprops centroid where matched,
           the eval_accuracy _nwb_all_objects mapping (GT within 0.5 voxel of a centroid), mean red
           inside vs outside the masks (and with the mask shifted by one plane)
  load     ProjectData.load_final_project_data(new, allow_hybrid_loading=True)

Usage: python validate_leifer_nwb.py --new new.nwb [--checks pixels gt seg load] [--seg_every 10]
"""
import argparse
import os
import time
from multiprocessing import Pool

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/claude-1000/numba_cache')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/claude-1000/mpl')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import h5py
import numpy as np

OLD = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb'
IMG = 'acquisition/CalciumImageSeries/data'
SEG = 'processing/CalciumActivity/CalciumSeriesSegmentationUntracked/data'
ACT = 'processing/CalciumActivity'
SLAB = 48


def _pixel_slab(args):
    new_path, i0, i1 = args
    with h5py.File(OLD, 'r') as f:
        old = f[IMG][i0:i1, ..., 0]
    with h5py.File(new_path, 'r') as f:
        new = f[IMG][i0:i1, ..., 0]
    nz = min(old.shape[-1], new.shape[-1])
    diff = (old[..., :nz] != new[..., :nz]).sum((1, 2))  # (vol, z) differing pixels
    return i0, diff


def check_pixels(new_path, n_vol, n_workers):
    t0 = time.time()
    diff = np.zeros((n_vol, 32), np.int64)
    jobs = [(new_path, i0, min(i0 + SLAB, n_vol)) for i0 in range(0, n_vol, SLAB)]
    with Pool(n_workers) as pool:
        for i0, d in pool.imap_unordered(_pixel_slab, jobs):
            diff[i0:i0 + len(d)] = d
    bad_vol = np.flatnonzero(diff.any(1))
    print(f'[pixels] {n_vol} volumes in {time.time() - t0:.0f}s: {n_vol - len(bad_vol)} pixel identical, '
          f'{(diff == 0).sum()} / {diff.size} planes identical')
    for t in bad_vol:
        zs = np.flatnonzero(diff[t])
        print(f'  volume {t}: {len(zs)} planes differ (z, n px): {[(int(z), int(diff[t, z])) for z in zs][:8]}')
    return diff


def _series(group):
    names = sorted(k for k in group.keys())
    return names, np.stack([group[n]['data'][:] for n in names], 1)  # (T, N, 3)


def check_gt(new_path):
    with h5py.File(OLD, 'r') as f:
        g = f[f'{ACT}/NeuronCentroids/CentroidSeries']
        c, d, ts = g['control'][:].astype(int), g['data'][:], g['timestamps'][:].astype(int)
    with h5py.File(new_path, 'r') as f:
        names, man = _series(f[f'{ACT}/NeuronCentroidsManual'])
    T = man.shape[0]
    col = {n: j for j, n in enumerate(names)}
    sel = ts < T
    j = np.array([col.get(f'neuron_{k + 1:03d}', -1) for k in c[sel]])
    assert (j >= 0).all(), 'old neuron missing in new file'
    new_at_old = man[ts[sel], j]
    err = np.abs(new_at_old - d[sel])
    same = (err.max(1) < 1e-3)
    n_new = int(np.isfinite(man[..., 0]).sum())
    print(f'[gt] new: {n_new} points, {len(names)} neurons, {int(np.isfinite(man[..., 0]).any(1).sum())} volumes; '
          f'old (t < {T}): {sel.sum()} points, {len(np.unique(c))} ids, {len(np.unique(ts[sel]))} volumes')
    print(f'[gt] old points equal in new (all 3 coords, 1e-3): {same.sum()} / {sel.sum()}; max |err| per axis '
          f'{np.nanmax(err, 0).round(4)}; volumes with differences {sorted(set(ts[sel][~same]))}')
    old_mask = np.zeros(man.shape[:2], bool)
    old_mask[ts[sel], j] = True
    extra = np.isfinite(man[..., 0]) & ~old_mask
    print(f'[gt] points in new but not old: {extra.sum()} in volumes {sorted(set(np.nonzero(extra)[0]))}')


def check_seg(new_path, n_vol, every):
    from skimage.measure import regionprops
    with h5py.File(new_path, 'r') as f:
        vi = f['processing/NeRVEExport/VolumeInfo']
        status = np.array([s.decode() if isinstance(s, bytes) else s for s in vi['segmentation_status'][:]])
        n_lab = vi['n_labels_in_mask'][:]
        n_obj = vi['n_nerve_objects'][:]
        n_out = vi['n_vox_out_of_window'][:]
        names, cen = _series(f[f'{ACT}/NeuronCentroids'])
        _, man = _series(f[f'{ACT}/NeuronCentroidsManual'])
        sid = np.stack([f[f'{ACT}/NeuronSegmentationID/{n}'][:] for n in names], 1)
        ok = status == 'ok'
        print(f'[seg] status: {dict(zip(*np.unique(status, return_counts=True)))}; empty volumes '
              f'{np.flatnonzero(~ok).tolist()}')
        print(f'[seg] labels per ok volume: mean {n_lab[ok].mean():.1f}, median {np.median(n_lab[ok]):.0f}, '
              f'min {n_lab[ok].min()}, max {n_lab[ok].max()}; NeRVE objects dropped by the z window: '
              f'{int((n_obj - n_lab)[ok].sum())}; voxels outside the window {int(n_out.sum())}')
        has_gt = np.isfinite(man[..., 0])
        matched = np.isfinite(sid)
        print(f'[seg] GT matched to a mask: {matched.sum()} / {has_gt.sum()} = {matched.sum() / has_gt.sum():.3f} '
              f'(on volumes with segmentation: {matched[ok].sum() / has_gt[ok].sum():.3f})')
        snap = np.linalg.norm((cen - man)[matched] * [0.3226, 0.3226, 1.5], axis=1)
        print(f'[seg] matched click -> centroid distance (um): median {np.median(snap):.2f}, p90 '
              f'{np.percentile(snap, 90):.2f}, max {snap.max():.2f}')
        img, seg = f[IMG], f[SEG]
        n_cen_ok = n_cen = n_map = n_map_ok = 0
        inside, outside, shifted = [], [], []
        for t in range(0, n_vol, every):
            if not ok[t]:
                continue
            s = seg[t]
            red = img[t, ..., 0].astype(float)
            props = regionprops(np.ascontiguousarray(s.transpose(2, 0, 1)))  # ZXY, as ProjectData
            lab = np.array([p.label for p in props])
            pc = np.array([p.centroid for p in props])  # (z, x, y)
            assert len(lab) == n_lab[t], (t, len(lab), n_lab[t])
            js = np.flatnonzero(matched[t])
            for j in js:
                k = np.flatnonzero(lab == sid[t, j])[0]
                n_cen += 1
                n_cen_ok += np.abs(pc[k] - cen[t, j][[2, 0, 1]]).max() < 1e-3
            g = cen[t][np.isfinite(cen[t, :, 0])][:, [2, 0, 1]]
            d = np.sqrt(((g[:, None] - pc[None]) ** 2).sum(-1))
            n_map += len(g)
            n_map_ok += int((d.min(1) <= 0.5).sum())
            m = s > 0
            inside.append(red[m].mean())
            outside.append(red[~m].mean())
            shifted.append(red[:, :, 1:][m[:, :, :-1]].mean())
        print(f'[seg] {len(inside)} sampled volumes: NeuronCentroids == regionprops centroid of its matched label '
              f'for {n_cen_ok} / {n_cen}; GT positions within 0.5 voxel of a centroid (_nwb_all_objects) '
              f'{n_map_ok} / {n_map}')
        print(f'[seg] mean red inside masks {np.mean(inside):.0f}, outside {np.mean(outside):.0f}, '
              f'mask shifted +1 plane {np.mean(shifted):.0f}')


def check_load(new_path):
    from wbfm.utils.projects.finished_project_data import ProjectData
    t0 = time.time()
    p = ProjectData.load_final_project_data(new_path, allow_hybrid_loading=True)
    print(f'[load] {time.time() - t0:.0f}s; red_data {p.red_data.shape} {p.red_data.dtype}; raw_segmentation '
          f'{None if p.raw_segmentation is None else p.raw_segmentation.shape}; segmentation '
          f'{None if p.segmentation is None else p.segmentation.shape}; final_tracks '
          f'{None if p.final_tracks is None else p.final_tracks.shape}; num_frames {p.num_frames}; '
          f'physical z {p.physical_unit_conversion.zimmer_um_per_pixel_z}')
    lv1 = sorted(set(p.final_tracks.columns.get_level_values(1)))
    print(f'[load] final_tracks level 1: {lv1}; duplicated columns {p.final_tracks.columns.duplicated().sum()}')
    t = int(np.flatnonzero(p.final_tracks.loc[:, (slice(None), 'raw_segmentation_id')].notna().any(axis=1).values)[0])
    vol = np.asarray(p.raw_segmentation[t])
    print(f'[load] raw_segmentation[{t}] {vol.shape} labels {len(np.unique(vol)) - 1}; red_data[{t}] mean '
          f'{float(np.asarray(p.red_data[t]).mean()):.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--checks', nargs='+', default=['pixels', 'gt', 'seg', 'load'])
    ap.add_argument('--n_workers', type=int, default=8)
    ap.add_argument('--seg_every', type=int, default=10)
    args = ap.parse_args()
    with h5py.File(args.new, 'r') as f:
        n_vol = f[IMG].shape[0]
        print(f'{args.new}: images {f[IMG].shape} chunks {f[IMG].chunks} {f[IMG].compression}; seg '
              f'{f[SEG].shape} {f[SEG].dtype} chunks {f[SEG].chunks}; size {os.path.getsize(args.new) / 1e9:.2f} GB')
    if 'pixels' in args.checks:
        check_pixels(args.new, n_vol, args.n_workers)
    if 'gt' in args.checks:
        check_gt(args.new)
    if 'seg' in args.checks:
        check_seg(args.new, n_vol, args.seg_every)
    if 'load' in args.checks:
        check_load(args.new)


if __name__ == '__main__':
    main()
