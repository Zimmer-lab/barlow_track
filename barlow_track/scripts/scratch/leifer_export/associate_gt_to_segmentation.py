"""Re-associate the Leifer manual GT clicks with a new segmentation, and write a GT project.

The clicks come from the A2 re-export NWB (processing/CalciumActivity/NeuronCentroids: raw manual
clicks, (x632, y600, z) voxels). The masks come from a wbfm project's segmentation (metadata
centroids, and the masks themselves with --assign contain). Per frame, greedily and without any
Hungarian trade-off:
  1. each click picks a mask: the nearest mask centroid in um (--assign nearest), or the mask
     under the click voxel, falling back to the nearest centroid on background (--assign contain);
  2. the click is dropped (NaN in the GT) if
       no_masks  the frame has no masks at all,
       dup       another click of the frame is closer than --dup_um (two GT neurons on one cell),
       far       its mask centroid is further than --max_um,
       shared    another surviving click picked the same mask (merge or duplicate: drop all of them).
Distances are in um (z and xy scaled separately: Leifer is ~4.6x anisotropic).

Kept cells get the mask's label (raw_segmentation_id), its row in the frame's metadata
(raw_neuron_ind_in_list) and its centroid (z, x, y), in the column format of the Zimmer GT
projects. --out_parent creates a new project next to the segmentation project with
make_project_like (preprocessing and segmentation linked, everything else fresh), writes the GT as
3-tracking/postprocessing/gt_reassociated.h5 and points tracking_config final_3d_tracks_df at it,
so eval_accuracy --source project --gt <new project> scores against it. The segmentation project
itself is only read. Without --out_parent this is a dry run: it only prints the report.

    python barlow_track/scripts/scratch/leifer_export/associate_gt_to_segmentation.py \
        --seg_project <stardist project>/project_config.yaml \
        [--out_parent <folder>] [--suffix _gt] [--assign contain] [--max_um 2.5] [--dup_um 1.5]
"""
import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

NWB = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/leifer_reexport/'
       'Leifer_NeRVE_Worm1_reexport_A2.nwb')
REASONS = ['kept', 'no_masks', 'dup', 'far', 'shared']  # codes 0..4; -1 = not annotated
GT_FNAME = '3-tracking/postprocessing/gt_reassociated.h5'


def load_clicks(nwb):
    """(T, N, 3) clicks in project (z, x, y) voxel order, and the neuron names."""
    with h5py.File(nwb, 'r') as f:
        grp = f['processing/CalciumActivity/NeuronCentroids']
        names = sorted(grp.keys())
        xyz = np.stack([grp[n]['data'][:] for n in names], 1).astype(float)  # (x632, y600, z)
    return xyz[..., [2, 0, 1]], names


def load_segmentation(seg_project):
    from wbfm.utils.projects.project_config_classes import ModularProjectConfig
    cfg = ModularProjectConfig(str(seg_project)).get_segmentation_config()
    meta_fname = cfg.resolve_relative_path_from_config('output_metadata')
    masks_fname = cfg.resolve_relative_path_from_config('output_masks')
    with open(meta_fname, 'rb') as f:
        meta = pickle.load(f)
    return meta, masks_fname


def frame_masks(meta, t):
    """labels (M,), centroids (M, 3) zxy, volume, brightness of frame t, in metadata row order."""
    df = meta.get(t)
    if df is None or len(df) == 0:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    cen = np.array([np.asarray(c, float) for c in df['centroids']])
    return (np.asarray(df.index, int), cen, df['neuron_volume'].to_numpy(float),
            df['total_brightness'].to_numpy(float))


def labels_under_clicks(masks_fname, clicks, workers=8):
    """(T, N) mask label at each rounded click voxel (0 background, -1 not annotated / outside)."""
    import zarr
    z = zarr.open(masks_fname, mode='r')
    T, N, _ = clicks.shape
    shape = np.array(z.shape[1:])

    def one(t):
        out = np.full(N, -1)
        fin = np.isfinite(clicks[t]).all(1)
        if not fin.any():
            return out
        vol = z[t]
        v = np.rint(clicks[t, fin]).astype(int)
        inb = (v >= 0).all(1) & (v < shape).all(1)
        lab = np.full(len(v), -1)
        lab[inb] = vol[tuple(v[inb].T)]
        out[fin] = lab
        return out

    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(one, range(T))))


def associate(clicks, meta, um, max_um, dup_um, under=None):
    """Greedy association; returns reason (T, N), label, row, centroid (T, N, 3), dist, volume, brightness."""
    T, N, _ = clicks.shape
    reason = np.full((T, N), -1)
    label = np.full((T, N), np.nan)
    row = np.full((T, N), np.nan)
    cen = np.full((T, N, 3), np.nan)
    dist = np.full((T, N), np.nan)
    vol = np.full((T, N), np.nan)
    bright = np.full((T, N), np.nan)
    for t in range(T):
        js = np.flatnonzero(np.isfinite(clicks[t]).all(1))
        if not len(js):
            continue
        labs, c, v, b = frame_masks(meta, t)
        if not len(labs):
            reason[t, js] = REASONS.index('no_masks')
            continue
        p = clicks[t, js] * um
        d = np.sqrt(((p[:, None] - (c * um)[None]) ** 2).sum(-1))  # clicks x masks
        pick = d.argmin(1)
        if under is not None:
            lab_of_row = {l: i for i, l in enumerate(labs)}
            for k, j in enumerate(js):
                if under[t, j] > 0 and under[t, j] in lab_of_row:
                    pick[k] = lab_of_row[under[t, j]]
        dk = d[np.arange(len(js)), pick]
        gg = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        np.fill_diagonal(gg, np.inf)
        dup = gg.min(1) < dup_um
        far = dk > max_um
        counts = np.bincount(pick[~far], minlength=len(labs))
        shared = ~far & (counts[pick] > 1)
        r = np.zeros(len(js), int)
        for name, flag in [('shared', shared), ('far', far), ('dup', dup)]:  # last wins: dup > far > shared
            r[flag] = REASONS.index(name)
        reason[t, js] = r
        dist[t, js] = dk
        ok = r == 0
        jk, pk = js[ok], pick[ok]
        label[t, jk] = labs[pk]
        row[t, jk] = pk
        cen[t, jk] = c[pk]
        vol[t, jk] = v[pk]
        bright[t, jk] = b[pk]
    return dict(reason=reason, label=label, row=row, cen=cen, dist=dist, vol=vol, bright=bright)


def to_dataframe(a, names):
    cols = {}
    for j, n in enumerate(names):
        ok = np.isfinite(a['label'][:, j])
        cols[(n, 'z')], cols[(n, 'x')], cols[(n, 'y')] = (a['cen'][:, j, i] for i in range(3))
        cols[(n, 'likelihood')] = np.where(ok, 1.0, np.nan)
        cols[(n, 'raw_segmentation_id')] = a['label'][:, j]
        cols[(n, 'raw_neuron_ind_in_list')] = a['row'][:, j]
        cols[(n, 'volume')] = a['vol'][:, j]
        cols[(n, 'brightness_red')] = a['bright'][:, j]
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=[None, 'coords'])
    return df


def report(a, names, clicks, nwb_ids=None):
    r = a['reason']
    n = int((r >= 0).sum())
    print(f'{n} annotated GT cells, {len(names)} neurons, {r.shape[0]} frames')
    for k, name in enumerate(REASONS):
        c = int((r == k).sum())
        print(f'  {name:>8}: {c:7d} ({c / n:.3f})')
    d = a['dist'][r == 0]
    print(f'kept: click to mask centroid um p50/90/99 {np.percentile(d, [50, 90, 99]).round(2)}')
    per = pd.DataFrame({REASONS[k]: (r == k).sum(0) for k in range(len(REASONS))}, index=names)
    per['kept_frac'] = per['kept'] / (r >= 0).sum(0)
    print('neurons losing the most cells:')
    print(per.sort_values('kept_frac').head(12).round(3).to_string())
    if nwb_ids is not None:  # dry run on the NeRVE project: agreement with the export's matching
        both = (r == 0) & np.isfinite(nwb_ids) & (nwb_ids < 1000)
        print(f'agreement with the NWB NeuronSegmentationID (NeRVE labels): '
              f'{np.mean(a["label"][both] == nwb_ids[both]):.4f} of {int(both.sum())} cells')


def make_gt_project(seg_project, out_parent, suffix):
    from wbfm.utils.projects.project_config_classes import make_project_like
    new_dir = make_project_like(str(seg_project), str(out_parent), target_suffix=suffix,
                                steps_to_keep=['preprocessing', 'segmentation'], verbose=0)
    return Path(new_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg_project', required=True, help='project_config.yaml of the new segmentation')
    ap.add_argument('--nwb', default=NWB, help='NWB with the GT clicks (NeuronCentroids)')
    ap.add_argument('--out_parent', default=None, help='create the GT project in this folder (else dry run)')
    ap.add_argument('--suffix', default='_gt')
    ap.add_argument('--assign', choices=['nearest', 'contain'], default='nearest')
    ap.add_argument('--max_um', type=float, default=2.5)
    ap.add_argument('--dup_um', type=float, default=1.5)
    ap.add_argument('--um_z', type=float, default=1.5)
    ap.add_argument('--um_xy', type=float, default=0.3226)
    ap.add_argument('--compare_nwb_ids', action='store_true',
                    help='report agreement with the NWB NeuronSegmentationID (only meaningful on the '
                         'NeRVE masks, i.e. the A2 project)')
    args = ap.parse_args()

    clicks, names = load_clicks(args.nwb)
    meta, masks_fname = load_segmentation(args.seg_project)
    T = clicks.shape[0]
    n_meta = len(meta)
    assert n_meta == T, f'segmentation has {n_meta} frames, the GT {T}'
    um = np.array([args.um_z, args.um_xy, args.um_xy])
    under = labels_under_clicks(masks_fname, clicks) if args.assign == 'contain' else None
    a = associate(clicks, meta, um, args.max_um, args.dup_um, under)
    print(f'[assign {args.assign}, max {args.max_um} um, dup {args.dup_um} um, voxel {um.tolist()} um] '
          f'{args.seg_project}')
    nwb_ids = None
    if args.compare_nwb_ids:
        with h5py.File(args.nwb, 'r') as f:
            g = f['processing/CalciumActivity/NeuronSegmentationID']
            nwb_ids = np.stack([g[n][:] for n in names], 1)
    report(a, names, clicks, nwb_ids)
    if args.out_parent is None:
        print('dry run: nothing written (pass --out_parent to create the GT project)')
        return

    new_dir = make_gt_project(args.seg_project, args.out_parent, args.suffix)
    from wbfm.utils.projects.project_config_classes import ModularProjectConfig
    cfg = ModularProjectConfig(str(new_dir / 'project_config.yaml'))
    nwb_cfg = cfg.get_nwb_config()
    nwb_file = nwb_cfg.resolve_relative_path_from_config('nwb_filename')
    if nwb_file is not None and os.path.exists(nwb_file):
        print(f'WARNING: the new project sees an NWB ({nwb_file}); hybrid loading would fill missing '
              'fields (e.g. traces) from it')
    out = new_dir / GT_FNAME
    out.parent.mkdir(parents=True, exist_ok=True)
    to_dataframe(a, names).to_hdf(out, key='df_with_missing', mode='w')
    np.savez(out.with_suffix('.npz'), reason=a['reason'], dist=a['dist'], clicks_zxy=clicks,
             names=np.array(names), reasons=np.array(REASONS), assign=args.assign, max_um=args.max_um,
             dup_um=args.dup_um, um=um)
    track_cfg = cfg.get_tracking_config()
    track_cfg.config['final_3d_tracks_df'] = GT_FNAME
    track_cfg.update_self_on_disk()
    print(f'GT project: {new_dir}\n  tracks: {out}\n  per-cell reasons: {out.with_suffix(".npz")}')


if __name__ == '__main__':
    main()
