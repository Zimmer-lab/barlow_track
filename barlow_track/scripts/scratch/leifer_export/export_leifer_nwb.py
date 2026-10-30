"""Export the Leifer NeRVE worm1 recording (Nguyen et al. 2017) from its raw files into NWB,
with NeRVE's own segmentation of every detected neuron.

Inputs (read-only): the .dat movie, manualSummary.mat (stackList, manualPoints), hiResData.mat (Z),
alignments.mat (S2AHiRes control points) and outputFiles/CLstraight/pointStatsNNNNN.mat (baseImg +
transformx/y/z) of barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1/.

Output layout (read by wbfm ProjectData.load_final_project_data_from_nwb):
- acquisition/CalciumImageSeries: (T, 632, 600, nz, C) uint16, one volume and channel per gzip chunk.
  Channel 0 = red (tagRFP, .dat frame[:, :632].T), channel 1 = green (GCaMP6s, frame[:, 634:1200].T warped
  into red coordinates with the projective transform refit from alignments.S2AHiRes Aall -> Sall, as NeRVE's
  fitgeotrans(Aall, Sall, 'projective') + imwarp, bilinear, 0 outside). Same order as the Samuel NWBs.
- processing/CalciumActivity/CalciumSeriesSegmentationUntracked: (T, 632, 600, nz) uint16.
  Labels < 1000: NeRVE masks, label = NeRVE pointIdx of that volume (NOT tracked across volumes).
  Labels >= 1000: pseudo masks, 1000 + manualPoints row, one per GT click without a matched NeRVE mask:
  watershed of the smoothed red seeded at the click, restricted to background (never over a NeRVE mask),
  to voxels >= half the seed's height above background, and to an ellipsoid of --pseudo_radius_um.
- processing/CalciumActivity/NeuronCentroids: one SpatialSeries per GT neuron (neuron_XXX, XXX = row of
  manualPoints + 1, the names of the old NWB), the manual clicks (x632, y600, z) in voxels, as the old NWB.
- processing/CalciumActivity/NeuronSegmentationID: per neuron and volume the Hungarian-matched NeRVE label
  (<= --match_um), else the pseudo-mask label; NaN only where the neuron is not annotated.
- processing/NeRVEExport/VolumeInfo (per volume) and PseudoMasks (per pseudo label): bookkeeping.

z rules (one function, frame_map):
- A: the old NWB's (Leifer_NeRVE_Worm1.nwb) 32-plane map, so red images match it pixel for pixel.
- B: only the volume's own frames, nz = 34, zero padded.
Both order z so that z = 0 is the highest piezo position (up sweeps are reversed).

The files Leifer_NeRVE_Worm1_reexport_{A,B}.nwb were written by commit 0314776 of this script (red only, no
pseudo masks, NeuronCentroids snapped to matched mask centroids + NeuronCentroidsManual).

No randomness. Usage (wbfm env, CUDA_VISIBLE_DEVICES=""):
  python export_leifer_nwb.py --z_rule A --out /path/new.nwb [--max_volumes 20] [--n_workers 24]
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/claude-1000/numba_cache')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/claude-1000/mpl')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import scipy.io as sio
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

RAW = Path('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/'
           'Nguyen2017NeuronTracking/worm1')
DAT_SHAPE = (600, 1200)  # one .dat frame; red = frame[:, :632].T
NX, NY = 632, 600
HIRES_OFFSET = 12  # hiResData index = dat frame + 12 (stackList == stackIdx shifted by 12)
GRID_SPACING = (0.3226, 0.3226, 1.5)  # um, as in the old NWB; z unverified (Nguyen 2016: 1.5-1.6 um)
NZ = {'A': 32, 'B': 34}
NGX = 566  # green crop: frame[:, 634:1200].T is (566, 600)
# Volumes whose clicks the old NWB omits (all GT; NeRVE blanked its detections there, >= 200 objects)
OLD_GT_OMITTED = (78, 405, 945, 1516)
PSEUDO_LABEL0 = 1000  # pseudo-mask label = 1000 + manualPoints row (NeRVE labels are < 200)


# ---------------------------------------------------------------- volumes and frame maps

def load_volume_table(raw=RAW, old_last_volume=False):
    """Per volume (0-based t = stack - 1): first .dat frame, number of frames, sweep direction.
    old_last_volume: frame_map gives the last volume the old NWB's (z-flipped) map."""
    sl = sio.loadmat(raw / 'manualSummary.mat')['stackList'].ravel().astype(int)
    z_piezo = sio.loadmat(raw / 'hiResData.mat')['dataAll'][0, 0]['Z'].ravel()
    n_vol = sl.max()
    start, nfr, up = np.zeros(n_vol, int), np.zeros(n_vol, int), np.zeros(n_vol, bool)
    for t in range(n_vol):
        fr = np.flatnonzero(sl == t + 1)
        assert len(fr) and np.all(np.diff(fr) == 1), f'volume {t} frames not contiguous'
        start[t], nfr[t] = fr[0], len(fr)
        up[t] = z_piezo[fr[-1] + HIRES_OFFSET] > z_piezo[fr[0] + HIRES_OFFSET]
    assert np.all(up[1:] != up[:-1]), 'sweeps do not alternate'
    return dict(start=start, nfr=nfr, up=up, stack_list=sl, old_last_volume=old_last_volume)


def frame_map(t, vols, z_rule):
    """(nz,) .dat frame index of each z plane of volume t; -1 = zero (padding) plane.

    A: the old NWB's map (see _rule_a_offsets), nz = 32, z = 0 at the top of the sweep. With
       vols['old_last_volume'] (load_volume_table), the last volume (a down sweep) gets the up-sweep map,
       as the old NWB stores it (z-flipped; image, GT and masks all use this map, so they stay consistent).
    B: all of the volume's own frames, nz = 34, z = 0 at the top of the sweep, zero padded at the end.
    """
    st, n, up = vols['start'][t], vols['nfr'][t], vols['up'][t]
    nz = NZ[z_rule]
    out = np.full(nz, -1, int)
    if z_rule == 'A':
        if vols.get('old_last_volume', False) and t == len(vols['start']) - 1:
            up = True
        offs = _rule_a_offsets(n, up)
    elif z_rule == 'B':
        offs = np.arange(n)[::-1] if up else np.arange(n)
    else:
        raise ValueError(z_rule)
    offs = offs[:nz]
    out[:len(offs)] = st + offs
    return out


# Old-NWB map, measured plane by plane against the .dat for all 1536 volumes (old_map_check.py).
# Up sweeps: z = k is frame start + top - k (k = 0..top); down sweeps: z = k is start + skip + k up to the
# volume's last frame; at most 32 planes, rest zero. Keyed by the volume's number of frames.
_A_UP_TOP = {30: 29, 31: 30, 32: 30, 33: 31, 34: 31}
_A_DOWN_SKIP = {32: 1, 33: 2, 34: 2}


def _rule_a_offsets(n, up):
    """Frame offsets (from the volume's first frame) of z = 0, 1, ... under rule A.

    Reproduces Leifer_NeRVE_Worm1.nwb on every volume but the last (1535, a down sweep that the old file
    stores z-flipped as if it were an up sweep); load_volume_table(old_last_volume=True) reproduces it too.
    """
    if up:
        if n not in _A_UP_TOP:
            raise ValueError(f'rule A undefined for an up sweep of {n} frames')
        return np.arange(_A_UP_TOP[n], -1, -1)
    if n not in _A_DOWN_SKIP:
        raise ValueError(f'rule A undefined for a down sweep of {n} frames')
    return np.arange(_A_DOWN_SKIP[n], n)


def frame_to_z(fmap):
    """Affine map (sign, base) with z = base + sign * frame, valid on fmap's non-padding planes."""
    k = np.flatnonzero(fmap >= 0)
    sign = -1 if fmap[k[-1]] < fmap[k[0]] else 1
    base = k[0] - sign * fmap[k[0]]
    assert np.array_equal(base + sign * fmap[k], k), 'frame map is not affine'
    return sign, base


# ---------------------------------------------------------------- images

def fit_green_to_red(raw=RAW):
    """Projective transform green -> red, refit from alignments.S2AHiRes control points.

    NeRVE: t_concord = fitgeotrans(Aall, Sall, 'projective') (Aall: green/activity, Sall: red/segment, MATLAB
    spatial (x = column, y = row) of the crops fullImage(rect(2)+1:rect(4), rect(1)+1:rect(3))), then
    imwarp(green, t_concord, 'OutputView', Rsegment). With the .dat read as here, MATLAB crop index (row i,
    col j), 1-based, is python red[i, j] / green[i, j], 0-based (the python crops start one pixel earlier), so
    MATLAB (x, y) = python (dim 1, dim 0). Returns (tform, residuals (n, 2) in px).
    """
    from skimage.transform import ProjectiveTransform
    a = sio.loadmat(raw / 'alignments.mat')['alignments'][0, 0]['S2AHiRes'][0, 0]
    tform = ProjectiveTransform()
    assert tform.estimate(a['Aall'], a['Sall'])
    return tform, tform(a['Aall']) - a['Sall']


def green_sample_coords(tform, shift=(0., 0.)):
    """(2, 632, 600) green (dim0, dim1) coordinates sampled for each red pixel; shift in green px (tests)."""
    rr, cc = np.mgrid[0:NX, 0:NY]
    uv = tform.inverse(np.c_[cc.ravel(), rr.ravel()].astype(float))
    return np.stack([uv[:, 1] + shift[0], uv[:, 0] + shift[1]]).reshape(2, NX, NY)


def warp_green(plane, coords):
    """Green plane (566, 600) -> red coordinates, bilinear, 0 outside (imwarp defaults), rounded uint16."""
    g = ndi.map_coordinates(plane.astype(np.float32), coords, order=1, mode='constant', cval=0.)
    return np.clip(np.rint(g), 0, 65535).astype(np.uint16)


_GREEN_COORDS = None  # set in main before the worker pool forks


def build_volume(args):
    """(632, 600, nz, C) uint16 volume t: red [, green warped into red coordinates]."""
    t, fmap, n_ch = args
    dat = np.memmap(RAW / 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, *DAT_SHAPE)
    vol = np.zeros((NX, NY, len(fmap), n_ch), np.uint16)
    ok = np.flatnonzero(fmap >= 0)
    lo, hi = fmap[ok].min(), fmap[ok].max() + 1
    block = np.asarray(dat[lo:hi])  # (frames, 600, 1200)
    for k in ok:
        frame = block[fmap[k] - lo]
        vol[:, :, k, 0] = frame[:, :NX].T
        if n_ch > 1:
            vol[:, :, k, 1] = warp_green(frame[:, 634:1200].T, _GREEN_COORDS)
    return t, vol


def ordered_bounded_map(pool, fn, jobs, window):
    """pool.imap with at most `window` results in flight (results are large)."""
    pending = {}
    for i in range(min(window, len(jobs))):
        pending[i] = pool.apply_async(fn, (jobs[i],))
    nxt = len(pending)
    for i in range(len(jobs)):
        r = pending.pop(i).get()
        if nxt < len(jobs):
            pending[nxt] = pool.apply_async(fn, (jobs[nxt],))
            nxt += 1
        yield r


def iter_volumes(vols, z_rule, n_vol, n_ch, n_workers):
    """Yield (632, 600, nz, C) uint16 volumes in order, built in worker processes."""
    from multiprocessing import get_context
    jobs = [(t, frame_map(t, vols, z_rule), n_ch) for t in range(n_vol)]
    t0 = time.time()
    with get_context('fork').Pool(n_workers) as pool:
        for t, vol in ordered_bounded_map(pool, build_volume, jobs, window=2 * n_workers):
            yield vol
            if t % 100 == 0 or t == n_vol - 1:
                print(f'[images] {t + 1}/{n_vol} volumes, {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- NeRVE segmentation

def nerve_masks(args):
    """Sparse NeRVE masks of volume t in (x632, y600, z) space of the frame map.

    Returns dict(t, idx (flat in (632, 600, nz)), lab, labels, centroid (L, 3), status, stats).
    baseImg is labelled in MATLAB (Fortran) order, which reproduces NeRVE's pointIdx; every labelled
    straightened voxel is scattered to raw (row = transformy, col = transformx, frame = start +
    transformz; 0 = NaN in MATLAB, dropped), then to z through the frame map. Where several labels hit
    one raw voxel, the first in C order over the straightened array wins.
    """
    t, fmap, start, nz = args
    empty = dict(t=t, idx=np.zeros(0, np.int64), lab=np.zeros(0, np.uint16), labels=np.zeros(0, int),
                 centroid=np.zeros((0, 3)), stats={})
    f = RAW / 'outputFiles' / 'CLstraight' / f'pointStats{t + 1:05d}.mat'
    if not f.exists():
        err = f.with_name(f'ERROR{t + 1:05d}.mat')
        return dict(empty, status='error_file' if err.exists() else 'missing')
    try:
        ps = sio.loadmat(f)['pointStats'][0, 0]
    except Exception as e:  # e.g. pointStats00013.mat: corrupt zlib stream in the release
        try:  # one retry in case of a network read error
            ps = sio.loadmat(f)['pointStats'][0, 0]
        except Exception:
            return dict(empty, status='unreadable', stats=dict(error=repr(e)))
    if ps['rawPoints'].size == 0:
        return dict(empty, status='zero_detections')
    lab = np.transpose(ndi.label(np.transpose(ps['baseImg'], (2, 1, 0)))[0], (2, 1, 0))
    n_lab = int(lab.max())
    vol_sizes = np.bincount(lab.ravel(), minlength=n_lab + 1)[1:]
    assert n_lab == len(ps['pointIdx']) and np.array_equal(vol_sizes, ps['Volume'].ravel()), \
        f'volume {t}: Fortran-order labels do not reproduce pointIdx / Volume'
    tx, ty, tz = (ps[k].astype(np.int64) for k in ('transformx', 'transformy', 'transformz'))
    v = (lab > 0) & (tx > 0) & (ty > 0) & (tz > 0)
    rows, cols, frames, labs = ty[v], tx[v], start + tz[v], lab[v]
    inb = (rows < NX) & (cols < NY)
    # raw-space centroid check against NeRVE's rawPoints (x = col, y = row, z = frame offset)
    raw_c = np.stack([np.bincount(labs, w, minlength=n_lab + 1)[1:] for w in (cols, rows, frames - start)], 1)
    cnt_raw = np.bincount(labs, minlength=n_lab + 1)[1:]
    has = cnt_raw > 0
    rp_err = np.abs(raw_c[has] / cnt_raw[has, None] - ps['rawPoints'][has]).max(1)
    # frames -> z
    ok = fmap >= 0
    inv = np.full(max(frames.max(), fmap.max()) + 1, -1, int)
    inv[fmap[ok]] = np.flatnonzero(ok)
    z = inv[frames]
    keep = inb & (z >= 0)
    flat = np.ravel_multi_index((rows[keep], cols[keep], z[keep]), (NX, NY, nz))
    flat_u, first = np.unique(flat, return_index=True)
    lab_u = labs[keep][first].astype(np.uint16)
    labels = np.unique(lab_u).astype(int)
    xyz = np.stack(np.unravel_index(flat_u, (NX, NY, nz)), 1).astype(float)
    cnt = np.bincount(lab_u, minlength=n_lab + 1)
    cen = np.stack([np.bincount(lab_u, xyz[:, i], minlength=n_lab + 1) for i in range(3)], 1)
    centroid = cen[labels] / cnt[labels, None]
    stats = dict(n_labels=n_lab, n_labels_kept=len(labels), n_vox_straight=int((lab > 0).sum()),
                 n_vox_scattered=int(len(flat)), n_vox_raw=int(len(flat_u)),
                 n_vox_out_of_window=int((inb & (z < 0)).sum()), n_vox_out_of_image=int((~inb).sum()),
                 rawpoints_err_px=float(np.median(rp_err)) if len(rp_err) else np.nan)
    return dict(t=t, idx=flat_u, lab=lab_u, labels=labels, centroid=centroid, status='ok', stats=stats)


def compute_all_masks(vols, z_rule, n_vol, n_workers):
    from multiprocessing import get_context
    nz = NZ[z_rule]
    jobs = [(t, frame_map(t, vols, z_rule), vols['start'][t], nz) for t in range(n_vol)]
    out = [None] * n_vol
    t0 = time.time()
    with get_context('fork').Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(nerve_masks, jobs, chunksize=1)):
            out[r['t']] = r
            if i % 100 == 0 or i == n_vol - 1:
                print(f'[masks] {i + 1}/{n_vol} volumes, {time.time() - t0:.0f}s', flush=True)
    return out


def iter_seg_volumes(masks, pseudo, nz):
    """NeRVE labels, then pseudo labels (which were built in background voxels only)."""
    for t, m in enumerate(masks):
        vol = np.zeros(NX * NY * nz, np.uint16)
        vol[m['idx']] = m['lab']
        if pseudo is not None:
            p = pseudo[t]
            assert not vol[p['idx']].any(), f'volume {t}: pseudo mask over a NeRVE mask'
            vol[p['idx']] = p['lab']
        yield vol.reshape(NX, NY, nz)


# ---------------------------------------------------------------- pseudo masks for unmatched GT

def pseudo_masks(args):
    """Pseudo masks for the unmatched GT clicks of volume t (watershed, background only, ellipsoid cap).

    seeds: (K, 4) rows (manualPoints row, x, y, z) in voxels. On a crop around the seeds: red smoothed
    (gaussian sigma (1, 1, 0.5) voxels); per seed the allowed voxels A_j = inside its ellipsoid (radius_um
    in x, y and z, grid spacing GRID_SPACING) & not a NeRVE voxel & smoothed red >= bg + frac * (peak - bg)
    (bg = crop median, peak = max smoothed red in the seed's 3x3x3 neighbourhood). If A_j is empty, A_j =
    ellipsoid & background (fallback). Marker = the A_j voxel nearest the click (the click itself when
    allowed); watershed(-smoothed red, markers, mask = union of A_j), then each region is cut to its own A_j.
    Returns sparse idx / lab in (632, 600, nz) and per-seed stats.
    """
    from skimage.segmentation import watershed
    t, fmap, nerve_idx, seeds, nz, radius_um, frac = args
    stats = []
    empty = dict(t=t, idx=np.zeros(0, np.int64), lab=np.zeros(0, np.uint16), stats=stats)
    if not len(seeds):
        return empty
    sp = np.asarray(GRID_SPACING)
    r_vox = radius_um / sp
    pos = seeds[:, 1:4]
    pad = np.ceil(r_vox).astype(int) + 2
    shape = np.array([NX, NY, nz])
    lo = np.clip(np.floor(pos.min(0)).astype(int) - pad, 0, shape)
    hi = np.clip(np.ceil(pos.max(0)).astype(int) + pad + 1, 0, shape)
    dat = np.memmap(RAW / 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, *DAT_SHAPE)
    red = np.zeros(tuple(hi - lo), np.float32)
    for k in range(lo[2], hi[2]):
        if fmap[k] >= 0:
            red[:, :, k - lo[2]] = np.asarray(dat[fmap[k], lo[1]:hi[1], lo[0]:hi[0]]).T
    nerve = np.zeros(red.shape, bool)
    if len(nerve_idx):
        xyz = np.stack(np.unravel_index(nerve_idx, tuple(shape)), 1)
        inb = np.all((xyz >= lo) & (xyz < hi), 1)
        nerve[tuple((xyz[inb] - lo).T)] = True
    smooth = ndi.gaussian_filter(red, (1., 1., 0.5))
    bg = float(np.median(red))
    union = np.zeros(red.shape, bool)
    markers = np.zeros(red.shape, np.int32)
    allowed = []
    for row, x, y, z in seeds:
        c = np.array([x, y, z]) - lo  # crop coordinates
        w0 = np.clip(np.floor(c - r_vox).astype(int), 0, red.shape)
        w1 = np.clip(np.ceil(c + r_vox).astype(int) + 1, 0, red.shape)
        sl = tuple(slice(a, b) for a, b in zip(w0, w1))
        g = np.ogrid[sl]
        d2 = sum(((gi - ci) / ri) ** 2 for gi, ci, ri in zip(g, c, r_vox))
        ell = d2 <= 1
        ci = np.clip(np.rint(c).astype(int), 0, np.array(red.shape) - 1)
        nb = tuple(slice(max(v - 1, 0), v + 2) for v in ci)
        peak = float(smooth[nb].max())
        thr = bg + frac * (peak - bg)
        bgr = ell & ~nerve[sl]
        a = bgr & (smooth[sl] >= thr)
        fallback = not a.any()
        if fallback:
            a = bgr
        st = dict(row=int(row), label=PSEUDO_LABEL0 + int(row), n_ellipsoid=int(ell.sum()),
                  n_lost_to_nerve=int((ell & nerve[sl]).sum()), seed_in_nerve=bool(nerve[tuple(ci)]),
                  fallback=bool(fallback), n_vox=0)
        if a.any():
            cand = np.argwhere(a)
            dist = ((((cand + w0) - c) * sp) ** 2).sum(1)
            for q in np.argsort(dist, kind='stable'):  # nearest allowed voxel not already a marker
                v = tuple(cand[q] + w0)
                if markers[v] == 0:
                    markers[v] = st['label']
                    break
            union[sl] |= a
        allowed.append((sl, a))
        stats.append(st)
    if not markers.any():
        return empty
    ws = watershed(-smooth, markers, mask=union)
    idx, lab = [], []
    for (sl, a), st in zip(allowed, stats):
        reg = (ws[sl] == st['label']) & a
        st['n_vox'] = int(reg.sum())
        if st['n_vox']:
            vox = np.argwhere(reg) + np.array([s_.start for s_ in sl]) + lo
            idx.append(np.ravel_multi_index(tuple(vox.T), tuple(shape)))
            lab.append(np.full(len(vox), st['label'], np.uint16))
    if not idx:
        return dict(empty, stats=stats)
    idx, lab = np.concatenate(idx), np.concatenate(lab)
    assert len(np.unique(idx)) == len(idx)
    return dict(t=t, idx=idx, lab=lab, stats=stats)


def compute_all_pseudo(vols, z_rule, n_vol, masks, gt, gt_lab, radius_um, frac, n_workers):
    from multiprocessing import get_context
    nz = NZ[z_rule]
    jobs = []
    for t in range(n_vol):
        js = np.flatnonzero(np.isfinite(gt[t, :, 0]) & ~np.isfinite(gt_lab[t]))
        jobs.append((t, frame_map(t, vols, z_rule), masks[t]['idx'], np.c_[js, gt[t, js]], nz, radius_um, frac))
    out = [None] * n_vol
    t0 = time.time()
    with get_context('fork').Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(pseudo_masks, jobs, chunksize=1)):
            out[r['t']] = r
            if i % 200 == 0 or i == n_vol - 1:
                print(f'[pseudo] {i + 1}/{n_vol} volumes, {time.time() - t0:.0f}s', flush=True)
    return out


# ---------------------------------------------------------------- ground truth

def load_manual_gt(vols, z_rule, n_vol, raw=RAW, drop_volumes=()):
    """manualPoints -> (n_vol, n_rows, 3) in (x632, y600, z) voxels; NaN = not annotated.

    As the old NWB: x = manual column 1, y = manual column 0, z = manual .dat frame through the
    volume's frame map (affine, so fractional frames map to fractional z). drop_volumes: left NaN.
    """
    mp = sio.loadmat(raw / 'manualSummary.mat')['manualPoints']
    n_rows = max(mp[t, 0].shape[0] for t in range(mp.shape[0]))
    gt = np.full((n_vol, n_rows, 3), np.nan)
    for t in range(n_vol):
        a = mp[t, 0]
        pts = np.array([[np.nan if np.size(a[i, j]) == 0 else float(np.ravel(a[i, j])[0]) for j in range(3)]
                        for i in range(a.shape[0])]).reshape(-1, 3)
        sign, base = frame_to_z(frame_map(t, vols, z_rule))
        if t not in drop_volumes:
            gt[t, :len(pts)] = np.c_[pts[:, 1], pts[:, 0], base + sign * pts[:, 2]]
    return gt


def match_gt(gt, masks, max_um, spacing=GRID_SPACING):
    """Hungarian match of GT clicks to mask centroids per volume, in um; returns labels (T, N) NaN unmatched
    and the matched centroid (T, N, 3)."""
    T, N, _ = gt.shape
    lab = np.full((T, N), np.nan)
    cen = np.full((T, N, 3), np.nan)
    sp = np.asarray(spacing)
    for t in range(T):
        js = np.flatnonzero(np.isfinite(gt[t, :, 0]))
        m = masks[t]
        if not len(js) or not len(m['labels']):
            continue
        d = np.sqrt((((gt[t, js, None, :] - m['centroid'][None]) * sp) ** 2).sum(-1))
        # pairs beyond the cut-off cost a constant, so the assignment never trades a close pair for a far one
        r, c = linear_sum_assignment(np.where(d <= max_um, d, 1e6))
        ok = d[r, c] <= max_um
        lab[t, js[r[ok]]] = m['labels'][c[ok]]
        cen[t, js[r[ok]]] = m['centroid'][c[ok]]
    return lab, cen


# ---------------------------------------------------------------- NWB

def optical_channels(n_ch):
    """Channel metadata. Known (Nguyen et al. 2016 PNAS): strain AML14 wtfEx4[rab-3::NLS::GCaMP6s;
    rab-3::NLS::tagRFP], 488 and 561 nm excitation, spinning disk, 40x objective, red and green side by side
    on one sCMOS at 200 fps. Emission filters are not given in the main text: emission_lambda is the
    fluorophore's emission peak and emission_range is NaN (unknown)."""
    from ndx_multichannel_volume import OpticalChannelPlus
    chans = [OpticalChannelPlus(
        name='tagRFP', excitation_lambda=561., excitation_range=[561., 561.], emission_lambda=584.,
        emission_range=[np.nan, np.nan],
        description='Red reference channel, rab-3::NLS::tagRFP (nuclear). Excitation 561 nm laser (Nguyen 2016); '
                    'emission_lambda = tagRFP emission peak; emission filter unknown (range NaN). '
                    '.dat frame[:, :632].T (manualSummary RedCrop [1 1 600 632]).')]
    if n_ch > 1:
        chans.append(OpticalChannelPlus(
            name='GCaMP6s', excitation_lambda=488., excitation_range=[488., 488.], emission_lambda=510.,
            emission_range=[np.nan, np.nan],
            description='Green activity channel, rab-3::NLS::GCaMP6s (nuclear). Excitation 488 nm laser '
                        '(Nguyen 2016); emission_lambda = GCaMP6s emission peak; emission filter unknown (range '
                        'NaN). .dat frame[:, 634:1200].T (GreenCrop [1 635 600 1200]) warped into red coordinates.'))
    return chans


def build_nwbfile(args, vols, masks, pseudo, gt, gt_ids, n_vol, nz, mask_status, green_res):
    from dateutil import tz
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from hdmf.common import DynamicTable, VectorData
    from hdmf.data_utils import DataChunkIterator
    from ndx_multichannel_volume import ImagingVolume, MultiChannelVolumeSeries, OpticalChannelReferences
    from pynwb import NWBFile
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.file import Subject

    n_ch = 2 if args.channels == 'red_green' else 1
    last = ('; the last volume (1535, a down sweep) is z-flipped as in the 2025 export' if vols['old_last_volume']
            else '; the last volume (1535) gets the down-sweep map (the 2025 export stores it z-flipped)')
    rule_txt = {'A': "z rule A: the 32-plane frame map of the 2025 export Leifer_NeRVE_Worm1.nwb; z = 0 is the "
                     "top of the piezo sweep, padding planes are zero" + last,
                'B': "z rule B: each volume's own .dat frames only (nz = 34), z = 0 is the top of the piezo "
                     "sweep, zero padded"}[args.z_rule]
    no_seg = [t for t, s in enumerate(mask_status) if s != 'ok']
    nwbfile = NWBFile(
        session_description='Whole-brain imaging in freely moving C. elegans, NeRVE worm1 (Nguyen et al. 2016, '
                            '2017), re-exported from the raw .dat with NeRVE segmentation. ' + rule_txt,
        identifier=f'NeRVE_Worm1_reexport_{args.z_rule}' + ('2' if n_ch > 1 else ''),
        session_start_time=datetime(2015, 6, 16, 10, 2, tzinfo=tz.gettz('America/New_York')),
        experimenter=['Nguyen, Jeffrey P.', 'Leifer, Andrew M.'],
        lab='Leifer Lab', institution='Princeton University',
        related_publications=['https://doi.org/10.1073/pnas.1507110112',
                              'https://doi.org/10.1371/journal.pcbi.1005517'],
        notes=f'Exported by barlow_track/scripts/scratch/leifer_export/export_leifer_nwb.py from {RAW}.',
    )
    nwbfile.subject = Subject(subject_id='worm1', species='Caenorhabditis elegans',
                              description='strain AML14 wtfEx4[rab-3::NLS::GCaMP6s; rab-3::NLS::tagRFP] (Nguyen '
                                          '2016); recording 20150616_1002_NLS_GC6_fluor_worm1')
    device = nwbfile.create_device(
        name='Spinning Disk Confocal',
        description='Leifer lab whole-brain imaging rig (Nguyen 2016): spinning disk confocal, 40x objective scanned '
                    'in z by a piezo (3 Hz triangle wave, ~33 slices per volume, 6 volumes/s), red and green '
                    'recorded side by side on one sCMOS camera at 200 fps')
    chans = optical_channels(n_ch)
    imvol = ImagingVolume(
        name='CalciumImVol', description='Imaging volume used to acquire calcium imaging data',
        optical_channel_plus=chans,
        order_optical_channels=OpticalChannelReferences(name='order_optical_channels',
                                                        channels=[c.name for c in chans]),
        device=device, location='head', grid_spacing=list(GRID_SPACING), grid_spacing_unit='micrometers',
        origin_coords=[0., 0., 0.], origin_coords_unit='micrometers', reference_frame='image coordinates')
    nwbfile.add_imaging_plane(imvol)
    spacing_txt = ('grid_spacing (0.3226, 0.3226, 1.5) um as in the 2025 export; the z value is unverified '
                   '(Nguyen 2016 PNAS: ~33 slices 1.5-1.6 um apart, 48-51 um deep) and the piezo range '
                   'drifts, so a z index is not a fixed depth')
    green_txt = ''
    if n_ch > 1:
        rn = np.linalg.norm(green_res, axis=1)
        green_txt = (' Channel 1 = green (GCaMP6s): frame[:, 634:1200].T warped into red coordinates by the '
                     'projective transform refit (skimage ProjectiveTransform.estimate) from alignments.S2AHiRes '
                     f'Aall -> Sall ({len(rn)} control points, residual rms {np.sqrt((rn ** 2).mean()):.2f} px, '
                     f'max {rn.max():.2f} px), as NeRVE fitgeotrans(..., projective) + imwarp: bilinear, 0 outside '
                     'the green crop, rounded to uint16. No background subtraction.')

    rate = 6.0
    img = H5DataIO(DataChunkIterator(iter_volumes(vols, args.z_rule, n_vol, n_ch, args.n_workers),
                                     maxshape=(n_vol, NX, NY, nz, n_ch), dtype=np.dtype('uint16'), buffer_size=1),
                   chunks=(1, NX, NY, nz, 1), compression='gzip', compression_opts=args.gzip)
    nwbfile.add_acquisition(MultiChannelVolumeSeries(
        name='CalciumImageSeries',
        description=f'Volumes (T, x=632, y=600, z, channel), channels {[c.name for c in chans]}. Channel 0 = red '
                    '(tagRFP) = .dat frame[:, :632].T. Under z rule A the red channel equals the 2025 export pixel '
                    'for pixel except one plane in each of 39 volumes where the 2025 export has one corrupted '
                    '632-px line (repaired here from the .dat; listed by validate_leifer_nwb.py).'
                    + green_txt + ' ' + rule_txt + '. ' + spacing_txt,
        comments='Nguyen et al. 2017 worm1 from sCMOS_Frames_U16_1200x600.dat, volumes from '
                 'manualSummary.stackList', data=img, device=device, unit='raw camera counts', resolution=1.,
        rate=rate, dimension=[NX, NY, nz], imaging_volume=imvol))

    act = nwbfile.create_processing_module(
        name='CalciumActivity', description='NeRVE segmentation (every detection), pseudo masks and manual GT')
    seg = H5DataIO(DataChunkIterator(iter_seg_volumes(masks, pseudo, nz), maxshape=(n_vol, NX, NY, nz),
                                     dtype=np.dtype('uint16'), buffer_size=1),
                   chunks=(1, NX, NY, nz), compression='gzip', compression_opts=args.gzip)
    pseudo_txt = ''
    if pseudo is not None:
        pseudo_txt = (f' Labels >= {PSEUDO_LABEL0} are pseudo masks (label = {PSEUDO_LABEL0} + manualPoints row, '
                      'i.e. neuron_XXX has label 999 + XXX), one per GT click without a matched NeRVE mask: '
                      'watershed of the smoothed red (sigma 1, 1, 0.5 voxels) seeded at the click, restricted to '
                      'background voxels (never over a NeRVE mask), to smoothed red >= background + '
                      f'{args.pseudo_frac} x (seed peak - background), and to an ellipsoid of {args.pseudo_radius_um}'
                      ' um radius in x, y and z. Listed in processing/NeRVEExport/PseudoMasks.')
    act.add(MultiChannelVolumeSeries(
        name='CalciumSeriesSegmentationUntracked',
        description='Masks, (T, x, y, z) uint16. Labels < 1000: NeRVE masks of every detected neuron, label = NeRVE '
                    'pointIdx of that volume (not tracked across volumes). From outputFiles/CLstraight/'
                    'pointStatsNNNNN.mat: baseImg labelled in MATLAB order and scattered to raw (row = transformy, '
                    'col = transformx, frame = stack start + transformz), then to z with the image frame map. '
                    f'Volumes without NeRVE masks: {no_seg}.' + pseudo_txt +
                    ' See processing/NeRVEExport/VolumeInfo.',
        comments=rule_txt, data=seg, device=device, unit='label', resolution=1., rate=rate,
        dimension=[NX, NY, nz], imaging_volume=imvol))

    has = np.flatnonzero(np.isfinite(gt[..., 0]).any(0))
    names = [f'neuron_{i + 1:03d}' for i in has]
    omit_txt = (f' As in the 2025 export, the clicks of volumes {list(OLD_GT_OMITTED)} are omitted (208 clicks '
                'that manualPoints does have).' if args.omit_old_missing_clicks else
                f' Includes the clicks of volumes {list(OLD_GT_OMITTED)} (208), which the 2025 export omits.')
    ts = np.arange(n_vol, dtype=float)
    position = Position(name='NeuronCentroids')
    for i, n in zip(has, names):
        position.add_spatial_series(SpatialSeries(
            name=n, data=gt[:, i].astype(np.float32), unit='voxels',
            reference_frame='image coordinates (x632, y600, z)', timestamps=ts,
            description='Manual GT clicks (manualSummary.manualPoints, mapped as the 2025 export: x = column 1, '
                        'y = column 0, z = .dat frame through the frame map), (x632, y600, z) voxels.' + omit_txt))
    act.add(position)
    dt = DynamicTable(name='NeuronSegmentationID', id=np.arange(n_vol),
                      description='CalciumSeriesSegmentationUntracked label of each GT neuron per volume: the '
                                  f'Hungarian-matched NeRVE mask (centroid <= {args.match_um} um), else its pseudo '
                                  f'mask ({PSEUDO_LABEL0} + manualPoints row); NaN = not annotated')
    for i, n in zip(has, names):
        dt.add_column(name=n, description=f'Raw seg ID for {n}', data=gt_ids[:, i])
    act.add(dt)

    info = nwbfile.create_processing_module(name='NeRVEExport', description='Export bookkeeping')
    st = [m['stats'] for m in masks]
    n_pseudo = [sum(s['n_vox'] > 0 for s in p['stats']) for p in pseudo] if pseudo is not None else [0] * n_vol
    cols = [VectorData(name='frame_map', description='.dat frame of each z plane, -1 = zero padding plane',
                       data=np.stack([frame_map(t, vols, args.z_rule) for t in range(n_vol)])),
            VectorData(name='start_frame', description='first .dat frame of the volume (stackList)',
                       data=vols['start'][:n_vol]),
            VectorData(name='n_frames', description='.dat frames of the volume', data=vols['nfr'][:n_vol]),
            VectorData(name='up_sweep', description='piezo Z increasing over the volume',
                       data=vols['up'][:n_vol]),
            VectorData(name='segmentation_status',
                       description='NeRVE masks: ok | error_file | zero_detections | unreadable | missing',
                       data=list(mask_status)),
            VectorData(name='n_nerve_objects', description='NeRVE detections (pointIdx) in the volume',
                       data=np.array([s.get('n_labels', 0) for s in st])),
            VectorData(name='n_labels_in_mask', description='NeRVE labels left after scattering to the z window',
                       data=np.array([s.get('n_labels_kept', 0) for s in st])),
            VectorData(name='n_vox_out_of_window', description='scattered voxels whose frame is not in the map',
                       data=np.array([s.get('n_vox_out_of_window', 0) for s in st])),
            VectorData(name='n_pseudo_masks', description=f'pseudo masks (labels >= {PSEUDO_LABEL0}) in the volume',
                       data=np.array(n_pseudo))]
    info.add(DynamicTable(name='VolumeInfo', description='Per-volume frame map and segmentation status',
                          id=np.arange(n_vol), columns=cols))
    if pseudo is not None:
        rows = [(p['t'], s) for p in pseudo for s in p['stats']]
        pcols = [('volume', 'volume index t', [t for t, _ in rows]),
                 ('label', 'label in CalciumSeriesSegmentationUntracked', [s['label'] for _, s in rows]),
                 ('neuron', 'GT neuron name', [f"neuron_{s['row'] + 1:03d}" for _, s in rows]),
                 ('n_voxels', 'voxels of the pseudo mask', [s['n_vox'] for _, s in rows]),
                 ('n_ellipsoid', 'voxels of the capping ellipsoid', [s['n_ellipsoid'] for _, s in rows]),
                 ('n_lost_to_nerve', 'ellipsoid voxels that are NeRVE masks', [s['n_lost_to_nerve'] for _, s in rows]),
                 ('seed_in_nerve_mask', 'the click lies inside a NeRVE mask (matched to another GT neuron or not '
                  'matched)', [s['seed_in_nerve'] for _, s in rows]),
                 ('fallback', 'no background voxel above the intensity threshold; ellipsoid background used',
                  [s['fallback'] for _, s in rows]),
                 ('nerve_status', 'segmentation_status of the volume', [mask_status[t] for t, _ in rows])]
        info.add(DynamicTable(name='PseudoMasks', id=np.arange(len(rows)),
                              description='One row per pseudo mask (GT click without a matched NeRVE mask)',
                              columns=[VectorData(name=n, description=d, data=v) for n, d, v in pcols]))
    return nwbfile


def main():
    global _GREEN_COORDS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--z_rule', choices=['A', 'B'], default='A')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_volumes', type=int, default=None, help='only the first N volumes (quick test)')
    ap.add_argument('--n_workers', type=int, default=24, help='processes for masks and volume building')
    ap.add_argument('--match_um', type=float, default=2.6, help='GT <-> mask centroid cut-off (um)')
    ap.add_argument('--channels', choices=['red', 'red_green'], default='red_green')
    ap.add_argument('--pseudo_radius_um', type=float, default=2.0,
                    help='pseudo-mask ellipsoid radius (um); 0 = no pseudo masks (NaN ids for unmatched GT)')
    ap.add_argument('--pseudo_frac', type=float, default=0.5,
                    help='pseudo masks keep smoothed red >= bg + frac * (seed peak - bg)')
    ap.add_argument('--gzip', type=int, default=4)
    ap.add_argument('--old_last_volume', action='store_true',
                    help='rule A: store volume 1535 (a down sweep) z-flipped as the old NWB does, instead of with '
                         'the down-sweep map (default, also in v1 files A/B)')
    ap.add_argument('--omit_old_missing_clicks', action='store_true',
                    help=f'drop the clicks of volumes {OLD_GT_OMITTED}, which the old NWB omits (default: keep)')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    for forbidden in ('barlow_track_paper',):
        assert forbidden not in str(out.resolve()), f'refusing to write under {forbidden}'
    t0 = time.time()
    vols = load_volume_table(old_last_volume=args.z_rule == 'A' and args.old_last_volume)
    n_vol = len(vols['start']) if args.max_volumes is None else min(args.max_volumes, len(vols['start']))
    nz = NZ[args.z_rule]
    print(f'{n_vol} volumes, z rule {args.z_rule} (nz {nz}), {args.channels} -> {out}', flush=True)
    green_res = None
    if args.channels == 'red_green':
        tform, green_res = fit_green_to_red()
        _GREEN_COORDS = green_sample_coords(tform)
        rn = np.linalg.norm(green_res, axis=1)
        print(f'[green] projective fit on {len(rn)} control points: residual rms {np.sqrt((rn ** 2).mean()):.3f} '
              f'px, median {np.median(rn):.3f}, max {rn.max():.3f}; H =\n{tform.params}', flush=True)

    masks = compute_all_masks(vols, args.z_rule, n_vol, args.n_workers)
    status = [m['status'] for m in masks]
    print('[masks] status counts', {s: status.count(s) for s in set(status)},
          'non-ok volumes', [t for t, s in enumerate(status) if s != 'ok'], flush=True)
    st = [m['stats'] for m in masks if m['status'] == 'ok']
    if st:
        tot = {k: sum(s[k] for s in st) for k in ('n_labels', 'n_labels_kept', 'n_vox_scattered', 'n_vox_raw',
                                                  'n_vox_out_of_window', 'n_vox_out_of_image')}
        print('[masks] totals', tot, 'median rawPoints err (px)',
              np.nanmedian([s['rawpoints_err_px'] for s in st]).round(3), flush=True)

    gt = load_manual_gt(vols, args.z_rule, n_vol, drop_volumes=OLD_GT_OMITTED if args.omit_old_missing_clicks else ())
    gt_lab, _ = match_gt(gt, masks, args.match_um)
    n_gt = int(np.isfinite(gt[..., 0]).sum())
    print(f'[gt] {n_gt} points, {int(np.isfinite(gt[..., 0]).any(0).sum())} neurons, '
          f'{int(np.isfinite(gt[..., 0]).any(1).sum())} volumes; matched {int(np.isfinite(gt_lab).sum())} '
          f'({np.isfinite(gt_lab).sum() / max(n_gt, 1):.3f})', flush=True)
    gt_ids = gt_lab.copy()
    pseudo = None
    if args.pseudo_radius_um > 0:
        pseudo = compute_all_pseudo(vols, args.z_rule, n_vol, masks, gt, gt_lab, args.pseudo_radius_um,
                                    args.pseudo_frac, args.n_workers)
        pst = [(p['t'], s) for p in pseudo for s in p['stats']]
        for t, s in pst:
            if s['n_vox'] > 0:
                gt_ids[t, s['row']] = s['label']
        n = np.array([s['n_vox'] for _, s in pst])
        nerve_sizes = np.concatenate([np.bincount(m['lab'])[m['labels']] for m in masks if m['status'] == 'ok'])
        print(f'[pseudo] {len(n)} seeds, {int((n > 0).sum())} masks; voxels median {np.median(n):.0f} p90 '
              f'{np.percentile(n, 90):.0f} max {n.max()} (NeRVE median {np.median(nerve_sizes):.0f} p90 '
              f'{np.percentile(nerve_sizes, 90):.0f} max {nerve_sizes.max()}); empty {int((n == 0).sum())}; '
              f'fallback {sum(s["fallback"] for _, s in pst)}; click inside a NeRVE mask '
              f'{sum(s["seed_in_nerve"] for _, s in pst)}; ellipsoid voxels lost to NeRVE masks '
              f'{sum(s["n_lost_to_nerve"] for _, s in pst)} of {sum(s["n_ellipsoid"] for _, s in pst)}', flush=True)
    print(f'[gt] finite NeuronSegmentationID {int(np.isfinite(gt_ids).sum())} / {n_gt}', flush=True)

    from pynwb import NWBHDF5IO
    nwbfile = build_nwbfile(args, vols, masks, pseudo, gt, gt_ids, n_vol, nz, status, green_res)
    tmp = out.with_name(out.stem + '.partial.nwb')
    with NWBHDF5IO(str(tmp), 'w') as io:
        io.write(nwbfile)
    tmp.rename(out)
    print(f'wrote {out} ({out.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
