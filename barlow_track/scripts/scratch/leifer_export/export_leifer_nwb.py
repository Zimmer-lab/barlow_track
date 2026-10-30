"""Export the Leifer NeRVE worm1 recording (Nguyen et al. 2017) from its raw files into NWB,
with NeRVE's own segmentation of every detected neuron.

Inputs (read-only): the .dat movie, manualSummary.mat (stackList, manualPoints), hiResData.mat (Z)
and outputFiles/CLstraight/pointStatsNNNNN.mat (baseImg + transformx/y/z) of
barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1/.

Output layout (read by wbfm ProjectData.load_final_project_data_from_nwb):
- acquisition/CalciumImageSeries: red channel, (T, 632, 600, nz, 1) uint16, one volume per gzip chunk.
- processing/CalciumActivity/CalciumSeriesSegmentationUntracked: (T, 632, 600, nz) uint16, NeRVE masks;
  label = NeRVE pointIdx of that volume (labels are NOT tracked across volumes).
- processing/CalciumActivity/NeuronCentroids: one SpatialSeries per GT neuron (neuron_XXX, XXX = row of
  manualPoints + 1, the names of the old NWB), (x632, y600, z) in voxels. Where the GT click is
  Hungarian-matched to a mask (<= --match_um) this is the mask's voxel centroid (so a GT position is a
  regionprops centroid of raw_segmentation), else the manual click.
- processing/CalciumActivity/NeuronCentroidsManual: the manual clicks only, same layout.
- processing/CalciumActivity/NeuronSegmentationID: matched mask label per neuron and volume (NaN unmatched).
- processing/NeRVEExport/VolumeInfo: per-volume frame map and segmentation bookkeeping.

z rules (one function, frame_map):
- A: the old NWB's (Leifer_NeRVE_Worm1.nwb) 32-plane map, so images match it pixel for pixel.
- B: only the volume's own frames, nz = 34, zero padded.
Both order z so that z = 0 is the highest piezo position (up sweeps are reversed).

No randomness. Usage (wbfm env, CUDA_VISIBLE_DEVICES=""):
  python export_leifer_nwb.py --z_rule A --out /path/new.nwb [--max_volumes 20] [--n_workers 24]
"""
import argparse
import os
import queue
import threading
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


# ---------------------------------------------------------------- volumes and frame maps

def load_volume_table(raw=RAW):
    """Per volume (0-based t = stack - 1): first .dat frame, number of frames, sweep direction."""
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
    return dict(start=start, nfr=nfr, up=up, stack_list=sl)


def frame_map(t, vols, z_rule):
    """(nz,) .dat frame index of each z plane of volume t; -1 = zero (padding) plane.

    A: the old NWB's map (see _rule_a_offsets), nz = 32, z = 0 at the top of the sweep.
    B: all of the volume's own frames, nz = 34, z = 0 at the top of the sweep, zero padded at the end.
    """
    st, n, up = vols['start'][t], vols['nfr'][t], vols['up'][t]
    nz = NZ[z_rule]
    out = np.full(nz, -1, int)
    if z_rule == 'A':
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
    stores z-flipped as if it were an up sweep; here it gets the down-sweep map).
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

def iter_red_volumes(vols, z_rule, n_vol, prefetch=4):
    """Yield (632, 600, nz, 1) uint16 red volumes in order; reads the .dat in a background thread."""
    dat = np.memmap(RAW / 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, *DAT_SHAPE)
    q = queue.Queue(maxsize=prefetch)

    def reader():
        for t in range(n_vol):
            fmap = frame_map(t, vols, z_rule)
            vol = np.zeros((NX, NY, len(fmap), 1), np.uint16)
            ok = np.flatnonzero(fmap >= 0)
            lo, hi = fmap[ok].min(), fmap[ok].max() + 1
            block = np.asarray(dat[lo:hi])  # (frames, 600, 1200)
            for k in ok:
                vol[:, :, k, 0] = block[fmap[k] - lo, :, :NX].T
            q.put(vol)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    t0 = time.time()
    for t in range(n_vol):
        yield q.get()
        if t % 100 == 0 or t == n_vol - 1:
            print(f'[images] {t + 1}/{n_vol} volumes, {time.time() - t0:.0f}s', flush=True)
    th.join()


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


def iter_seg_volumes(masks, nz):
    for t, m in enumerate(masks):
        vol = np.zeros(NX * NY * nz, np.uint16)
        vol[m['idx']] = m['lab']
        yield vol.reshape(NX, NY, nz)


# ---------------------------------------------------------------- ground truth

def load_manual_gt(vols, z_rule, n_vol, raw=RAW):
    """manualPoints -> (n_vol, n_rows, 3) in (x632, y600, z) voxels; NaN = not annotated.

    As the old NWB: x = manual column 1, y = manual column 0, z = manual .dat frame through the
    volume's frame map (affine, so fractional frames map to fractional z).
    """
    mp = sio.loadmat(raw / 'manualSummary.mat')['manualPoints']
    n_rows = max(mp[t, 0].shape[0] for t in range(mp.shape[0]))
    gt = np.full((n_vol, n_rows, 3), np.nan)
    for t in range(n_vol):
        a = mp[t, 0]
        pts = np.array([[np.nan if np.size(a[i, j]) == 0 else float(np.ravel(a[i, j])[0]) for j in range(3)]
                        for i in range(a.shape[0])]).reshape(-1, 3)
        sign, base = frame_to_z(frame_map(t, vols, z_rule))
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

def build_nwbfile(args, vols, masks, gt, gt_lab, gt_cen, n_vol, nz, mask_status):
    from dateutil import tz
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from hdmf.common import DynamicTable, VectorData
    from hdmf.data_utils import DataChunkIterator
    from ndx_multichannel_volume import (ImagingVolume, MultiChannelVolumeSeries, OpticalChannelPlus,
                                         OpticalChannelReferences)
    from pynwb import NWBFile
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.file import Subject

    rule_txt = {'A': "z rule A: the 32-plane frame map of the 2025 export Leifer_NeRVE_Worm1.nwb (pixel "
                     "identical images); z = 0 is the top of the piezo sweep, padding planes are zero",
                'B': "z rule B: each volume's own .dat frames only (nz = 34), z = 0 is the top of the piezo "
                     "sweep, zero padded"}[args.z_rule]
    no_seg = [t for t, s in enumerate(mask_status) if s != 'ok']
    nwbfile = NWBFile(
        session_description='Whole-brain imaging in freely moving C. elegans, NeRVE worm1 (Nguyen et al. 2016, '
                            '2017), re-exported from the raw .dat with NeRVE segmentation. ' + rule_txt,
        identifier=f'NeRVE_Worm1_reexport_{args.z_rule}',
        session_start_time=datetime(2015, 6, 16, 10, 2, tzinfo=tz.gettz('America/New_York')),
        experimenter=['Nguyen, Jeffrey P.', 'Leifer, Andrew M.'],
        lab='Leifer Lab', institution='Princeton University',
        related_publications=['https://doi.org/10.1073/pnas.1507110112',
                              'https://doi.org/10.1371/journal.pcbi.1005517'],
        notes=f'Exported by barlow_track/scripts/scratch/leifer_export/export_leifer_nwb.py from {RAW}. '
              f'Red channel only.',
    )
    nwbfile.subject = Subject(subject_id='worm1', species='Caenorhabditis elegans',
                              description='20150616_1002_NLS_GC6_fluor_worm1')
    device = nwbfile.create_device(name='Spinning Disk Confocal')
    # Channel metadata copied verbatim from the 2025 export (Leifer_NeRVE_Worm1.nwb)
    chan = OpticalChannelPlus(name='TagRFP', description='Red channel', excitation_lambda=555.,
                              excitation_range=[488., 488.], emission_range=[584., 584.], emission_lambda=584.)
    imvol = ImagingVolume(
        name='CalciumImVol', description='Imaging volume used to acquire calcium imaging data',
        optical_channel_plus=[chan],
        order_optical_channels=OpticalChannelReferences(name='order_optical_channels', channels=['TagRFP']),
        device=device, location='head', grid_spacing=list(GRID_SPACING), grid_spacing_unit='micrometers',
        origin_coords=[0., 0., 0.], origin_coords_unit='micrometers', reference_frame='image coordinates')
    nwbfile.add_imaging_plane(imvol)
    spacing_txt = ('grid_spacing (0.3226, 0.3226, 1.5) um as in the 2025 export; the z value is unverified '
                   '(Nguyen 2016 PNAS: ~33 slices 1.5-1.6 um apart, 48-51 um deep) and the piezo range '
                   'drifts, so a z index is not a fixed depth')

    rate = 6.0
    img = H5DataIO(DataChunkIterator(iter_red_volumes(vols, args.z_rule, n_vol), maxshape=(n_vol, NX, NY, nz, 1),
                                     dtype=np.dtype('uint16'), buffer_size=1),
                   chunks=(1, NX, NY, nz, 1), compression='gzip', compression_opts=args.gzip)
    nwbfile.add_acquisition(MultiChannelVolumeSeries(
        name='CalciumImageSeries',
        description='Red-channel (TagRFP) volumes, (T, x=632, y=600, z, channel). Red = .dat frame[:, :632].T. '
                    + rule_txt + '. ' + spacing_txt,
        comments='Nguyen et al. 2017 worm1 from sCMOS_Frames_U16_1200x600.dat, volumes from '
                 'manualSummary.stackList', data=img, device=device, unit='raw camera counts', resolution=1.,
        rate=rate, dimension=[NX, NY, nz], imaging_volume=imvol))

    act = nwbfile.create_processing_module(
        name='CalciumActivity', description='NeRVE segmentation (every detection) and manual GT centroids')
    seg = H5DataIO(DataChunkIterator(iter_seg_volumes(masks, nz), maxshape=(n_vol, NX, NY, nz),
                                     dtype=np.dtype('uint16'), buffer_size=1),
                   chunks=(1, NX, NY, nz), compression='gzip', compression_opts=args.gzip)
    act.add(MultiChannelVolumeSeries(
        name='CalciumSeriesSegmentationUntracked',
        description='NeRVE masks of every detected neuron, (T, x, y, z) uint16, label = NeRVE pointIdx of that '
                    'volume (not tracked across volumes). From outputFiles/CLstraight/pointStatsNNNNN.mat: baseImg '
                    'labelled in MATLAB order and scattered to raw (row = transformy, col = transformx, frame = '
                    'stack start + transformz), then to z with the image frame map (voxels outside it dropped). '
                    f'Empty volumes (no NeRVE segmentation): {no_seg}. See processing/NeRVEExport/VolumeInfo.',
        comments=rule_txt, data=seg, device=device, unit='label', resolution=1., rate=rate,
        dimension=[NX, NY, nz], imaging_volume=imvol))

    has = np.flatnonzero(np.isfinite(gt[..., 0]).any(0))
    names = [f'neuron_{i + 1:03d}' for i in has]
    ts = np.arange(n_vol, dtype=float)
    main = np.where(np.isfinite(gt_cen), gt_cen, gt).astype(np.float32)
    for pos_name, arr, desc in [
            ('NeuronCentroids', main,
             f'GT neuron positions (x632, y600, z) in voxels. Where the manual click was Hungarian-matched to a '
             f'NeRVE mask (centroid distance <= {args.match_um} um, grid spacing {GRID_SPACING}) this is the '
             f"mask's voxel centroid, else the manual click (NeuronCentroidsManual)."),
            ('NeuronCentroidsManual', gt.astype(np.float32),
             'Manual GT clicks (manualSummary.manualPoints, mapped as the 2025 export: x = column 1, y = column '
             '0, z = .dat frame through the frame map), (x632, y600, z) voxels.')]:
        position = Position(name=pos_name)
        for i, n in zip(has, names):
            position.add_spatial_series(SpatialSeries(
                name=n, data=arr[:, i], unit='voxels', reference_frame='image coordinates (x632, y600, z)',
                timestamps=ts, description=desc))
        act.add(position)
    dt = DynamicTable(name='NeuronSegmentationID', id=np.arange(n_vol),
                      description='CalciumSeriesSegmentationUntracked label of each GT neuron per volume '
                                  f'(Hungarian match, <= {args.match_um} um); NaN = unmatched or not annotated')
    for i, n in zip(has, names):
        dt.add_column(name=n, description=f'Raw seg ID for {n}', data=gt_lab[:, i])
    act.add(dt)

    info = nwbfile.create_processing_module(name='NeRVEExport', description='Export bookkeeping')
    st = [m['stats'] for m in masks]
    cols = [VectorData(name='frame_map', description='.dat frame of each z plane, -1 = zero padding plane',
                       data=np.stack([frame_map(t, vols, args.z_rule) for t in range(n_vol)])),
            VectorData(name='start_frame', description='first .dat frame of the volume (stackList)',
                       data=vols['start'][:n_vol]),
            VectorData(name='n_frames', description='.dat frames of the volume', data=vols['nfr'][:n_vol]),
            VectorData(name='up_sweep', description='piezo Z increasing over the volume',
                       data=vols['up'][:n_vol]),
            VectorData(name='segmentation_status',
                       description='ok | error_file | zero_detections | unreadable | missing',
                       data=list(mask_status)),
            VectorData(name='n_nerve_objects', description='NeRVE detections (pointIdx) in the volume',
                       data=np.array([s.get('n_labels', 0) for s in st])),
            VectorData(name='n_labels_in_mask', description='labels left after scattering to the z window',
                       data=np.array([s.get('n_labels_kept', 0) for s in st])),
            VectorData(name='n_vox_out_of_window', description='scattered voxels whose frame is not in the map',
                       data=np.array([s.get('n_vox_out_of_window', 0) for s in st]))]
    info.add(DynamicTable(name='VolumeInfo', description='Per-volume frame map and segmentation status',
                          id=np.arange(n_vol), columns=cols))
    return nwbfile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--z_rule', choices=['A', 'B'], default='A')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_volumes', type=int, default=None, help='only the first N volumes (quick test)')
    ap.add_argument('--n_workers', type=int, default=24, help='processes for the mask warping')
    ap.add_argument('--match_um', type=float, default=2.6, help='GT <-> mask centroid cut-off (um)')
    ap.add_argument('--gzip', type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    for forbidden in ('barlow_track_paper',):
        assert forbidden not in str(out.resolve()), f'refusing to write under {forbidden}'
    t0 = time.time()
    vols = load_volume_table()
    n_vol = len(vols['start']) if args.max_volumes is None else min(args.max_volumes, len(vols['start']))
    nz = NZ[args.z_rule]
    print(f'{n_vol} volumes, z rule {args.z_rule} (nz {nz}) -> {out}', flush=True)

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

    gt = load_manual_gt(vols, args.z_rule, n_vol)
    gt_lab, gt_cen = match_gt(gt, masks, args.match_um)
    n_gt = int(np.isfinite(gt[..., 0]).sum())
    print(f'[gt] {n_gt} points, {int(np.isfinite(gt[..., 0]).any(0).sum())} neurons, '
          f'{int(np.isfinite(gt[..., 0]).any(1).sum())} volumes; matched {int(np.isfinite(gt_lab).sum())} '
          f'({np.isfinite(gt_lab).sum() / max(n_gt, 1):.3f})', flush=True)

    from pynwb import NWBHDF5IO
    nwbfile = build_nwbfile(args, vols, masks, gt, gt_lab, gt_cen, n_vol, nz, status)
    tmp = out.with_name(out.stem + '.partial.nwb')
    with NWBHDF5IO(str(tmp), 'w') as io:
        io.write(nwbfile)
    tmp.rename(out)
    print(f'wrote {out} ({out.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
