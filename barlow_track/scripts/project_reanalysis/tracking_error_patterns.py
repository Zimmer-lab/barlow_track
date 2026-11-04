"""Where do a tracker's errors fall, relative to the quality of the GT entries they are scored on?

Reads the tracks eval_accuracy.py --save_tracks wrote (keys gt, pred_matched, pred_all; raw_segmentation_id
values), the project's segmentation metadata (1-segmentation/metadata.pickle: label -> centroid, volume,
mean intensity per volume) and the manual GT clicks (the GT NWB's final_tracks x/y/z). For every valid GT
cell (volume t, neuron n) it records the outcome (correct, miss, mismatch) and GT-only properties:

  click_obj_um    distance from the click to the centroid of the object the GT assigned (um)
  vol_ratio       that object's volume / the neuron's median GT object volume (splits, merges)
  bright_ratio    its mean red intensity / the neuron's median
  crowd_click_um  nearest other neuron's click at t
  crowd_obj_um    nearest other segmented object's centroid to the GT object at t
  n_obj           objects segmented at t

and per error: a mismatch's predicted object (another neuron's GT object at t, i.e. a swap, or an object
no GT neuron holds) and its distance to the GT object; a miss's GT object (held by another neuron's matched
track, by an unmatched track, or by no track). Also how errors cluster in time (runs per neuron, volumes
with many errors) and over neurons, and the accuracy left after dropping GT cells by GT-only criteria
(fixed thresholds, not chosen from the errors).

Without --project (tracks from the NWB path, values = NWB mask labels) only the outcome, error kinds and
their clustering are computed. --pseudo_min_label splits every rate by whether the GT object's label is at
or above it (the Leifer re-export's pseudo masks, label 1000 + GT index, built around clicks NeRVE missed).

    python .../tracking_error_patterns.py --tracks base_centroidgt.h5 --project .../project_config.yaml \
        --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb --out <dir>
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

BINS = dict(click_obj_um=[0, 0.5, 1, 1.5, 2, 3, np.inf],
            vol_ratio=[0, 0.5, 0.75, 1.33, 2, np.inf],
            bright_ratio=[0, 0.5, 0.75, 1.33, 2, np.inf],
            crowd_click_um=[0, 2, 3, 4, 6, np.inf],
            crowd_obj_um=[0, 2, 3, 4, 6, np.inf],
            n_obj=[0, 115, 125, 135, 1000])


def runs(mask):
    """Lengths of the runs of True in a 1-d boolean array."""
    out, r = [], 0
    for m in mask:
        if m:
            r += 1
        elif r:
            out.append(r)
            r = 0
    if r:
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracks', required=True, help='eval_accuracy --save_tracks h5')
    ap.add_argument('--project', default=None, help='project_config.yaml whose segmentation was tracked '
                    '(omit for NWB-path tracks: no per-object properties)')
    ap.add_argument('--pseudo_min_label', type=int, default=None,
                    help='split rates by GT object label >= this (pseudo masks)')
    ap.add_argument('--gt_nwb', required=True, help='NWB whose final_tracks x/y/z are the manual clicks')
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.5, 0.3226, 0.3226])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    scale = np.asarray(args.voxel_um)

    gt = pd.read_hdf(args.tracks, 'gt')
    pm = pd.read_hdf(args.tracks, 'pred_matched').reindex(columns=gt.columns)
    pa = pd.read_hdf(args.tracks, 'pred_all')
    names = list(gt.columns)
    G, PM, PA = gt.values, pm.values, pa.values
    T, N = G.shape
    # Which pred_all track became which GT column (the matched frame is a renamed copy)
    track_to_gt = {}
    for j, n in enumerate(names):
        for k in range(PA.shape[1]):
            if np.array_equal(PA[:, k], PM[:, j], equal_nan=True) and k not in track_to_gt:
                track_to_gt[k] = j
                break
    n_matched_found = len(track_to_gt)

    md = {}
    if args.project:
        with open(os.path.join(os.path.dirname(os.path.abspath(args.project)), '1-segmentation',
                               'metadata.pickle'), 'rb') as f:
            md = pickle.load(f)
    nwb = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(nwb)
    missing = [n for n in names if n not in a['neurons']]
    if missing:
        raise SystemExit(f'GT neurons not in the NWB: {missing[:5]}')
    idx = [a['neurons'].index(n) for n in names]
    C = np.stack([a['gz'][:, idx], a['gx'][:, idx], a['gy'][:, idx]], 2)[:T]  # T, N, 3 voxels

    valid = np.isfinite(G)
    correct = valid & (PM == G)
    miss = valid & ~np.isfinite(PM)
    mism = valid & np.isfinite(PM) & (PM != G)
    feats = {k: np.full((T, N), np.nan) for k in BINS}
    err_kind = {}  # (t, j) -> description
    vol = np.full((T, N), np.nan)
    bri = np.full((T, N), np.nan)
    for t in range(T):
        d = md.get(t)
        if d is not None and len(d):
            labels = d['label'].values.astype(int)
            cen = np.stack(d['centroids'].values).astype(float)
        else:
            labels, cen = np.zeros(0, int), np.zeros((0, 3))
        lab_to_i = {l: i for i, l in enumerate(labels)}
        gt_owner = {int(G[t, j]): j for j in range(N) if valid[t, j]}
        trk_owner = {}
        for k in range(PA.shape[1]):
            if np.isfinite(PA[t, k]):
                trk_owner[int(PA[t, k])] = k
        clicks = C[t]
        for j in np.flatnonzero(valid[t]):
            i = lab_to_i.get(int(G[t, j]))
            if i is None:
                if mism[t, j]:
                    err_kind[(t, j)] = ('mismatch_swap_with_gt' if int(PM[t, j]) in gt_owner
                                        else 'mismatch_unannotated_object', np.nan)
                elif miss[t, j]:
                    k = trk_owner.get(int(G[t, j]))
                    err_kind[(t, j)] = ('miss_object_in_no_track' if k is None else
                                        'miss_object_in_another_neurons_track' if k in track_to_gt else
                                        'miss_object_in_unmatched_track', np.nan)
                continue
            feats['n_obj'][t, j] = len(labels)
            feats['click_obj_um'][t, j] = np.linalg.norm((cen[i] - clicks[j]) * scale)
            vol[t, j] = d['neuron_volume'].values[i]
            bri[t, j] = d['intensity_mean'].values[i]
            do = np.linalg.norm((cen - cen[i]) * scale, axis=1)
            do[i] = np.inf
            feats['crowd_obj_um'][t, j] = do.min() if len(do) > 1 else np.nan
            dc = np.linalg.norm((clicks - clicks[j]) * scale, axis=1)
            dc[j] = np.inf
            feats['crowd_click_um'][t, j] = np.nanmin(dc) if np.isfinite(dc).any() else np.nan
            if mism[t, j]:
                p = int(PM[t, j])
                ip = lab_to_i.get(p)
                dist = float(np.linalg.norm((cen[ip] - cen[i]) * scale)) if ip is not None else np.nan
                err_kind[(t, j)] = ('mismatch_swap_with_gt' if p in gt_owner else 'mismatch_unannotated_object',
                                    dist)
            elif miss[t, j]:
                k = trk_owner.get(int(G[t, j]))
                if k is None:
                    kind = 'miss_object_in_no_track'
                elif k in track_to_gt:
                    kind = 'miss_object_in_another_neurons_track'
                else:
                    kind = 'miss_object_in_unmatched_track'
                err_kind[(t, j)] = (kind, np.nan)
    with np.errstate(invalid='ignore'):
        feats['vol_ratio'] = vol / np.nanmedian(vol, axis=0, keepdims=True)
        feats['bright_ratio'] = bri / np.nanmedian(bri, axis=0, keepdims=True)

    err = miss | mism
    n_valid = int(valid.sum())
    out = dict(args=vars(args), n_gt_cells=n_valid, accuracy=float(1 - err.sum() / n_valid),
               misses=int(miss.sum()), mismatches=int(mism.sum()),
               matched_tracks_identified=n_matched_found, gt_neurons=N, predicted_tracks=PA.shape[1])

    kinds = pd.Series([v[0] for v in err_kind.values()]).value_counts()
    out['error_kinds'] = kinds.to_dict()
    dists = np.array([v[1] for v in err_kind.values() if v[0].startswith('mismatch')], float)
    if len(dists):
        out['mismatch_pred_to_gt_object_um_pct_10_50_90'] = np.nanpercentile(dists, [10, 50, 90]).round(2).tolist()
        out['mismatch_pred_object_within_3um_frac'] = float(np.nanmean(dists < 3))

    # Error rate by GT-only property
    by = {}
    for k, edges in BINS.items():
        x = feats[k]
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = valid & (x >= lo) & (x < hi)
            if m.sum():
                rows.append(dict(bin=f'{lo}-{hi}', cells=int(m.sum()), frac_of_gt=round(float(m.sum() / n_valid), 4),
                                 error_rate=round(float(err[m].mean()), 4),
                                 miss_rate=round(float(miss[m].mean()), 4),
                                 mismatch_rate=round(float(mism[m].mean()), 4)))
        by[k] = rows
    out['error_rate_by'] = by

    # Errors over neurons and time
    per_neuron = pd.Series(err.sum(0) / np.maximum(valid.sum(0), 1), index=names).sort_values(ascending=False)
    out['per_neuron_error_rate_top10'] = per_neuron.head(10).round(3).to_dict()
    out['per_neuron_error_rate_pct_10_50_90'] = np.percentile(per_neuron, [10, 50, 90]).round(4).tolist()
    e_sorted = np.sort(err.sum(0))[::-1]
    out['share_of_errors_in_worst_5_neurons'] = round(float(e_sorted[:5].sum() / max(err.sum(), 1)), 3)
    out['share_of_errors_in_worst_10_neurons'] = round(float(e_sorted[:10].sum() / max(err.sum(), 1)), 3)
    all_runs = [r for j in range(N) for r in runs(err[valid[:, j], j])]
    all_runs = np.array(all_runs) if all_runs else np.zeros(0, int)
    tot = max(all_runs.sum(), 1)
    out['error_runs'] = dict(n_runs=int(len(all_runs)), frac_errors_isolated_1=round(float(all_runs[all_runs == 1].sum() / tot), 3),
                             frac_errors_in_runs_2_9=round(float(all_runs[(all_runs >= 2) & (all_runs < 10)].sum() / tot), 3),
                             frac_errors_in_runs_10_49=round(float(all_runs[(all_runs >= 10) & (all_runs < 50)].sum() / tot), 3),
                             frac_errors_in_runs_50_plus=round(float(all_runs[all_runs >= 50].sum() / tot), 3),
                             longest=int(all_runs.max()) if len(all_runs) else 0)
    per_t = err.sum(1) / np.maximum(valid.sum(1), 1)
    has = valid.sum(1) > 0
    out['per_volume_error_rate_pct_50_90_99'] = np.percentile(per_t[has], [50, 90, 99]).round(4).tolist()
    for thr in (0.2, 0.5):
        bad = has & (per_t > thr)
        out[f'volumes_with_error_rate_over_{thr}'] = dict(n=int(bad.sum()),
                                                           share_of_errors=round(float(err[bad].sum() / err.sum()), 3))

    if args.pseudo_min_label is not None:
        ps = valid & (G >= args.pseudo_min_label)
        pred_ps = mism & (PM >= args.pseudo_min_label)
        out['pseudo_masks'] = dict(
            gt_cells=int(ps.sum()), frac_of_gt=round(float(ps.sum() / n_valid), 4),
            error_rate_on_pseudo=round(float(err[ps].mean()), 4) if ps.any() else None,
            miss_rate_on_pseudo=round(float(miss[ps].mean()), 4) if ps.any() else None,
            mismatch_rate_on_pseudo=round(float(mism[ps].mean()), 4) if ps.any() else None,
            error_rate_on_rest=round(float(err[valid & ~ps].mean()), 4),
            accuracy_without_pseudo_cells=round(float(1 - err[valid & ~ps].sum() / (valid & ~ps).sum()), 4),
            mismatches_onto_a_pseudo_object=int(pred_ps.sum()),
            share_of_errors_on_or_onto_pseudo=round(float(((err & ps) | pred_ps).sum() / err.sum()), 4))
        kinds_ps = pd.Series([v[0] for (t, j), v in err_kind.items() if ps[t, j]]).value_counts()
        out['pseudo_masks']['error_kinds_on_pseudo'] = kinds_ps.to_dict()

    # Accuracy after GT-only filters (fixed thresholds)
    filt = {'click_obj_um > 1.0': feats['click_obj_um'] > 1.0,
            'vol_ratio < 0.5 or > 2': (feats['vol_ratio'] < 0.5) | (feats['vol_ratio'] > 2),
            'bright_ratio < 0.5': feats['bright_ratio'] < 0.5,
            'crowd_click_um < 2': feats['crowd_click_um'] < 2,
            'crowd_obj_um < 2': feats['crowd_obj_um'] < 2}
    filt['any of the above'] = np.logical_or.reduce(list(filt.values()))
    out['accuracy_after_dropping'] = {k: dict(dropped_frac=round(float((valid & m).sum() / n_valid), 4),
                                              errors_dropped_frac=round(float((err & m).sum() / err.sum()), 4),
                                              accuracy=round(float(1 - (err & ~m).sum() / (valid & ~m).sum()), 4))
                                      for k, m in filt.items()}
    np.savez(os.path.join(args.out, 'cells.npz'), valid=valid, miss=miss, mism=mism, names=np.array(names),
             **feats)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
