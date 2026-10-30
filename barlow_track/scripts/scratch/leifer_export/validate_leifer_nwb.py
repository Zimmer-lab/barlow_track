"""Validate a re-exported Leifer NWB (export_leifer_nwb.py) against the raw files and the old NWB. Read-only.

Checks:
  pixels   per plane identity of CalciumImageSeries vs Leifer_NeRVE_Worm1.nwb (rule A only)
  gt       NeuronCentroidsManual vs the old CentroidSeries (names neuron_XXX = control + 1)
  seg      label counts, GT match rate, NeuronCentroids == mask regionprops centroid where matched,
           the eval_accuracy _nwb_all_objects mapping (GT within 0.5 voxel of a centroid), mean red
           inside vs outside the masks (and with the mask shifted by one plane)
  load     ProjectData.load_final_project_data(new, allow_hybrid_loading=True); red_data / green_data
           equal file channels 0 / 1 (and red equals the old NWB)
  ids      (v2) every GT point has a finite NeuronSegmentationID whose label exists in its volume's mask;
           pseudo-mask counts and sizes vs NeRVE masks and the PseudoMasks table
  green    (v2) control-point fit residuals, mean green inside vs outside NeRVE masks, and +-1 / +-0.5 px
           shifts of the warp re-done from the .dat (green in masks, red-green correlation)

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
    if len(bad_vol) and bad_vol[-1] == 1535 and diff[1535].all():
        with h5py.File(OLD, 'r') as f, h5py.File(new_path, 'r') as g:
            old, new = f[IMG][1535, ..., 0], g[IMG][1535, ..., 0]
        for c in range(25, 34):  # new z = k equals old z = c - k?
            ks = [k for k in range(32) if 0 <= c - k < 32]
            n_eq = sum(np.array_equal(new[:, :, k], old[:, :, c - k]) for k in ks)
            if n_eq:
                print(f'  volume 1535: new z = k equals old z = {c} - k on {n_eq} / {len(ks)} planes (a z flip)')
    return diff


def _series(group):
    names = sorted(k for k in group.keys())
    return names, np.stack([group[n]['data'][:] for n in names], 1)  # (T, N, 3)


def _manual_name(f):
    # v1 files (commit 0314776) keep the clicks in NeuronCentroidsManual; v2 in NeuronCentroids
    return 'NeuronCentroidsManual' if 'NeuronCentroidsManual' in f[ACT] else 'NeuronCentroids'


def check_gt(new_path):
    with h5py.File(OLD, 'r') as f:
        g = f[f'{ACT}/NeuronCentroids/CentroidSeries']
        c, d, ts = g['control'][:].astype(int), g['data'][:], g['timestamps'][:].astype(int)
    with h5py.File(new_path, 'r') as f:
        names, man = _series(f[f'{ACT}/{_manual_name(f)}'])
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
    bad_t = sorted(set(ts[sel][~same]))
    print(f'[gt] old points equal in new (all 3 coords, 1e-3): {same.sum()} / {sel.sum()}; max |err| per axis '
          f'{np.nanmax(err, 0).round(4)}; volumes with differences {bad_t}')
    keep = ~np.isin(ts[sel], bad_t)
    print(f'[gt] max |err| outside those volumes: {err[keep].max() if keep.any() else np.nan:.6g} '
          f'(exactly identical points: {(err.max(1) == 0).sum()})')
    for t in bad_t:
        k = ts[sel] == t
        zs = new_at_old[k, 2] + d[sel][k, 2]
        print(f'[gt] volume {t}: {k.sum()} points, max |err| per axis {err[k].max(0).round(4)}; x, y identical: '
              f'{bool((err[k, :2] == 0).all())}; z_new + z_old = {np.unique(zs.round(4))} (one value = a pure '
              f'z flip)')
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
        v1 = 'NeuronCentroidsManual' in f[ACT]
        if not v1:
            print('[seg] v2 file: NeRVE-match checks below use labels < 1000 only; see [ids] for pseudo masks')
        man = _series(f[f'{ACT}/NeuronCentroidsManual'])[1] if v1 else cen
        sid = np.stack([f[f'{ACT}/NeuronSegmentationID/{n}'][:] for n in names], 1)
        sid = np.where(sid < 1000, sid, np.nan)
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
        if v1:
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
            s = np.where(s < 1000, s, 0).astype(s.dtype)
            red = img[t, ..., 0].astype(float)
            props = regionprops(np.ascontiguousarray(s.transpose(2, 0, 1)))  # ZXY, as ProjectData
            lab = np.array([p.label for p in props])
            pc = np.array([p.centroid for p in props])  # (z, x, y)
            assert len(lab) == n_lab[t], (t, len(lab), n_lab[t])
            js = np.flatnonzero(matched[t])
            if v1:
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


def _label_counts(args):
    """Per volume {label: voxels}, and the mask value at given voxels (clicks)."""
    new_path, t0, t1, clicks = args
    out = {}
    with h5py.File(new_path, 'r') as f:
        for t in range(t0, t1):
            s = f[SEG][t]
            c = np.bincount(s.ravel())
            labs = np.flatnonzero(c)
            labs = labs[labs > 0]
            v = s[tuple(clicks[t].T)] if len(clicks.get(t, [])) else np.zeros(0, int)
            out[t] = (dict(zip(labs.tolist(), c[labs].tolist())), v)
    return out


def check_ids(new_path, n_vol, n_workers):
    """v2: every GT point has a finite NeuronSegmentationID whose label exists in that volume's mask;
    pseudo-mask counts and sizes vs NeRVE masks and vs the PseudoMasks table."""
    with h5py.File(new_path, 'r') as f:
        names, cen = _series(f[f'{ACT}/NeuronCentroids'])
        sid = np.stack([f[f'{ACT}/NeuronSegmentationID/{n}'][:] for n in names], 1)
        pm = f['processing/NeRVEExport/PseudoMasks']
        ptab = {k: pm[k][:] for k in pm.keys() if k != 'id'}
    t0 = time.time()
    with h5py.File(new_path, 'r') as f:
        shape = np.array(f[SEG].shape[1:4])
    pj = {t: np.flatnonzero(np.isfinite(sid[t]) & (sid[t] >= 1000)) for t in range(n_vol)}
    clicks = {t: np.clip(np.rint(cen[t, j]).astype(int), 0, shape - 1) for t, j in pj.items() if len(j)}
    counts, at_click = {}, {}
    jobs = [(new_path, a, min(a + 16, n_vol), {t: clicks[t] for t in range(a, min(a + 16, n_vol)) if t in clicks})
            for a in range(0, n_vol, 16)]
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(_label_counts, jobs):
            for t, (c, v) in r.items():
                counts[t], at_click[t] = c, v
    has = np.isfinite(cen[..., 0])
    fin = np.isfinite(sid)
    exists = np.array([int(sid[t, j]) in counts[t] for t, j in np.argwhere(fin)])
    print(f'[ids] ({time.time() - t0:.0f}s) GT points {has.sum()}, finite NeuronSegmentationID {(has & fin).sum()}, '
          f'ids where not annotated {(fin & ~has).sum()}, ids whose label exists in that volume\'s mask '
          f'{exists.sum()} / {len(exists)}; NeRVE ids {(fin & (sid < 1000)).sum()}, pseudo ids '
          f'{(fin & (sid >= 1000)).sum()}')
    ids_t = [set(sid[t][np.isfinite(sid[t])].astype(int).tolist()) for t in range(n_vol)]
    dup = sum(int(np.isfinite(sid[t]).sum()) - len(ids_t[t]) for t in range(n_vol))
    print(f'[ids] GT neurons sharing a label within a volume: {dup}')
    # pseudo-mask clicks: what is at the click voxel?
    n_own = n_bg = n_taken = n_free = 0
    for t, j in pj.items():
        for jj, v in zip(j, at_click.get(t, [])):
            v = int(v)
            if v == int(sid[t, jj]):
                n_own += 1
            elif v == 0:
                n_bg += 1
            elif v < 1000 and v in ids_t[t]:
                n_taken += 1
            else:
                n_free += 1
    print(f'[ids] pseudo-mask clicks: voxel in own pseudo mask {n_own}, background {n_bg}, inside a NeRVE mask '
          f'matched to another GT neuron {n_taken}, inside a NeRVE mask matched to no GT neuron (centroid > cut-off) '
          f'{n_free}')
    nerve = np.array([v for t in range(n_vol) for k, v in counts[t].items() if k < 1000])
    pseudo = np.array([v for t in range(n_vol) for k, v in counts[t].items() if k >= 1000])
    for name, a in [('NeRVE', nerve), ('pseudo', pseudo)]:
        print(f'[ids] {name} masks: n {len(a)}, voxels median {np.median(a):.0f}, p10 {np.percentile(a, 10):.0f}, '
              f'p90 {np.percentile(a, 90):.0f}, max {a.max()}')
    tab = {(int(t), int(l)): int(n) for t, l, n in zip(ptab['volume'], ptab['label'], ptab['n_voxels'])}
    file_p = {(t, k): v for t in range(n_vol) for k, v in counts[t].items() if k >= 1000}
    agree = all(tab.get(k) == v for k, v in file_p.items()) and all(v == 0 or k in file_p for k, v in tab.items())
    print(f'[ids] PseudoMasks table rows {len(tab)} (empty {sum(v == 0 for v in tab.values())}), agrees with the '
          f'mask voxel counts: {agree}; fallback {int(ptab["fallback"].sum())}; clicks inside a NeRVE mask '
          f'{int(ptab["seed_in_nerve_mask"].sum())} (their pseudo voxels: median '
          f'{np.median(ptab["n_voxels"][ptab["seed_in_nerve_mask"].astype(bool)]):.0f}); ellipsoid voxels lost to '
          f'NeRVE masks {int(ptab["n_lost_to_nerve"].sum())} of {int(ptab["n_ellipsoid"].sum())} '
          f'({int(ptab["n_lost_to_nerve"][ptab["seed_in_nerve_mask"].astype(bool)].sum())} for clicks inside a mask)')
    st = np.array([s.decode() if isinstance(s, bytes) else s for s in ptab['nerve_status']])
    print(f'[ids] pseudo masks in volumes without NeRVE masks: {int((st != "ok").sum())}, in volumes with: '
          f'{int((st == "ok").sum())}')


def check_green(new_path, n_vol, every, raw_volumes):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import export_leifer_nwb as E
    from scipy import ndimage as ndi
    tform, res = E.fit_green_to_red()
    rn = np.linalg.norm(res, axis=1)
    loo = []
    from skimage.transform import ProjectiveTransform
    import scipy.io as sio
    a = sio.loadmat(E.RAW / 'alignments.mat')['alignments'][0, 0]['S2AHiRes'][0, 0]
    for i in range(len(a['Aall'])):
        m = np.ones(len(a['Aall']), bool)
        m[i] = False
        tt = ProjectiveTransform()
        tt.estimate(a['Aall'][m], a['Sall'][m])
        loo.append(np.linalg.norm(tt(a['Aall'][i:i + 1]) - a['Sall'][i]))
    print(f'[green] control-point fit: {len(rn)} points, residual rms {np.sqrt((rn ** 2).mean()):.2f} px, median '
          f'{np.median(rn):.2f}, max {rn.max():.2f}; leave-one-out error median {np.median(loo):.2f}, max '
          f'{np.max(loo):.2f}')
    with h5py.File(new_path, 'r') as f:
        img, seg = f[IMG], f[SEG]
        if img.shape[-1] < 2:
            print('[green] no green channel')
            return
        gin, gout, rin = [], [], []
        for t in range(0, n_vol, every):
            s = seg[t]
            m = (s > 0) & (s < 1000)
            if not m.any():
                continue
            g = img[t, ..., 1].astype(float)
            valid = g > 0  # inside the warped green crop
            gin.append(g[m & valid].mean())
            gout.append(g[~m & valid].mean())
        print(f'[green] in-file, {len(gin)} volumes: mean green inside NeRVE masks {np.mean(gin):.1f}, outside '
              f'{np.mean(gout):.1f} (ratio {np.mean(gin) / np.mean(gout):.2f})')
        # warp shifts from the raw .dat: green inside masks and red-green correlation near masks
        vols = E.load_volume_table()
        dat = np.memmap(E.RAW / 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, *E.DAT_SHAPE)
        shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]
        coords = {sh: E.green_sample_coords(tform, sh) for sh in shifts}
        mean_in = {sh: [] for sh in shifts}
        corr = {sh: [] for sh in shifts}
        for t in raw_volumes:
            fm = f['processing/NeRVEExport/VolumeInfo/frame_map'][t]
            s = seg[t]
            m = (s > 0) & (s < 1000)
            if not m.any():
                continue
            region = ndi.binary_dilation(m, iterations=3)
            stored = img[t, ..., 1]
            for k in np.flatnonzero(fm >= 0):
                if not m[:, :, k].any():
                    continue
                frame = np.asarray(dat[fm[k]])
                gp, rp = frame[:, 634:1200].T, frame[:, :632].T.astype(float)
                for sh in shifts:
                    g = E.warp_green(gp, coords[sh]).astype(float)
                    if sh == (0, 0):
                        assert np.array_equal(g, stored[:, :, k]), f'stored green differs from re-warp, t {t} z {k}'
                    mean_in[sh].append((g[m[:, :, k]].sum(), m[:, :, k].sum()))
                    rg = region[:, :, k]
                    corr[sh].append(np.corrcoef(g[rg], rp[rg])[0, 1])
        print(f'[green] warp shift test on volumes {list(raw_volumes)} (stored green == re-warp at 0 shift):')
        for sh in shifts:
            mi = sum(a for a, _ in mean_in[sh]) / sum(b for _, b in mean_in[sh])
            print(f'  shift (dim0, dim1) {sh}: mean green in NeRVE masks {mi:.2f}, red-green corr near masks '
                  f'{np.mean(corr[sh]):.4f}')


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
    with h5py.File(new_path, 'r') as f, h5py.File(OLD, 'r') as fo:
        nc = f[IMG].shape[-1]
        r = np.asarray(p.red_data[t])
        ok_r = np.array_equal(r, f[IMG][t, ..., 0].transpose(2, 0, 1))
        ok_old = np.array_equal(r, fo[IMG][t, ..., 0].transpose(2, 0, 1)) if r.shape[0] == 32 else None
        g = np.asarray(p.green_data[t]) if p.green_data is not None else None
        ok_g = g is not None and np.array_equal(g, f[IMG][t, ..., nc - 1].transpose(2, 0, 1))
    print(f'[load] channels in file {nc}; red_data[{t}] == file channel 0: {ok_r}; == old NWB red: {ok_old}; '
          f'green_data present: {g is not None}, == file channel {nc - 1}: {ok_g}; red != green: '
          f'{g is not None and not np.array_equal(r, g)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--checks', nargs='+', default=['pixels', 'gt', 'seg', 'load'],
                    help='pixels gt seg load ids green')
    ap.add_argument('--green_raw_volumes', type=int, nargs='+', default=[100, 800, 1500])
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
    if 'ids' in args.checks:
        check_ids(args.new, n_vol, args.n_workers)
    if 'green' in args.checks:
        check_green(args.new, n_vol, args.seg_every, [t for t in args.green_raw_volumes if t < n_vol])
    if 'load' in args.checks:
        check_load(args.new)


if __name__ == '__main__':
    main()
