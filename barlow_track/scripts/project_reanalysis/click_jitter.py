"""How much do manual GT clicks jitter relative to the object they mark?

For a GT neuron tracked in a project (final tracks = raw_segmentation_id per volume), the offset of its
click from its object's centroid, o(t) = click(t) - centroid(t) in um, mixes a constant per-neuron habit
(clicked off-centre) with per-frame annotation jitter and centroid noise. Between consecutive volumes on
which both clicks sit on objects of a stable size, d(t) = o(t+1) - o(t) cancels the habit; if the jitter is
independent per frame with s.d. sigma per axis, d has s.d. sqrt(2) sigma. Reported per axis (z, x, y), as
robust s.d. (1.4826 x MAD) and plain s.d., pooled over neurons and per neuron, plus the per-neuron median
offset (the habit) and the 3-d jitter distance |o(t) - median o| percentiles. Only "good" frames count:
object volume 0.75-1.33x the neuron's median in both frames (no split or merge on either side). The
percentiles of the 3-d jitter give a tolerance for attaching a click outside every mask to the nearest one.

    python .../click_jitter.py --projects stardist=.../project_config.yaml:.../gt.h5 nerve=... \
        --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb --out <dir>
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd


def rsd(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--projects', nargs='+', required=True,
                    help='NAME=project_config.yaml:gt_tracks.h5 (GT h5 with raw_segmentation_id per neuron)')
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.5, 0.3226, 0.3226])
    ap.add_argument('--min_frames', type=int, default=200, help='neurons with fewer good frame pairs are skipped')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    scale = np.asarray(args.voxel_um)
    nwb = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(nwb)
    out = dict(args=vars(args))
    for spec in args.projects:
        name, rest = spec.split('=', 1)
        cfg, gt_h5 = rest.rsplit(':', 1)
        with open(os.path.join(os.path.dirname(os.path.abspath(cfg)), '1-segmentation', 'metadata.pickle'), 'rb') as f:
            md = pickle.load(f)
        gt = pd.read_hdf(gt_h5).loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
        T = min(len(gt), a['gx'].shape[0])
        per, pooled_d, pooled_r = {}, [], []
        lookup = {}
        for t in range(T):
            d = md.get(t)
            if d is not None and len(d):
                lookup[t] = dict(zip(d['label'].values.astype(int),
                                     zip(np.stack(d['centroids'].values).astype(float), d['neuron_volume'].values)))
        for n in gt.columns:
            if n not in a['neurons']:
                continue
            j = a['neurons'].index(n)
            click = np.stack([a['gz'][:T, j], a['gx'][:T, j], a['gy'][:T, j]], 1)
            off = np.full((T, 3), np.nan)
            vol = np.full(T, np.nan)
            ids = gt[n].values[:T]
            for t in np.flatnonzero(np.isfinite(ids)):
                c = lookup.get(t, {}).get(int(ids[t]))
                if c is not None and np.isfinite(click[t]).all():
                    off[t] = (click[t] - c[0]) * scale
                    vol[t] = c[1]
            good = np.isfinite(vol) & (vol / np.nanmedian(vol) > 0.75) & (vol / np.nanmedian(vol) < 1.33)
            pair = good[:-1] & good[1:]
            if pair.sum() < args.min_frames:
                continue
            dd = (off[1:] - off[:-1])[pair]
            habit = np.nanmedian(off[good], axis=0)
            res = np.linalg.norm(off[good] - habit, axis=1)
            pooled_d.append(dd)
            pooled_r.append(res)
            per[n] = dict(pairs=int(pair.sum()), sigma_rsd_zxy=[round(rsd(dd[:, k]) / np.sqrt(2), 3) for k in range(3)],
                          habit_offset_um_zxy=np.round(habit, 2).tolist(),
                          habit_norm_um=round(float(np.linalg.norm(habit)), 2),
                          jitter_3d_um_p50_p90=np.percentile(res, [50, 90]).round(2).tolist())
        D = np.concatenate(pooled_d)
        Rr = np.concatenate(pooled_r)
        sig_rsd = [rsd(D[:, k]) / np.sqrt(2) for k in range(3)]
        sig_sd = [float(np.std(D[:, k])) / np.sqrt(2) for k in range(3)]
        hab = np.array([v['habit_norm_um'] for v in per.values()])
        nsig = np.array([np.linalg.norm(v['sigma_rsd_zxy']) for v in per.values()])
        out[name] = dict(
            neurons=len(per), frame_pairs=int(len(D)),
            sigma_per_axis_um_zxy_robust=np.round(sig_rsd, 3).tolist(),
            sigma_per_axis_um_zxy_sd=np.round(sig_sd, 3).tolist(),
            sigma_3d_um_robust=round(float(np.linalg.norm(sig_rsd)), 3),
            jitter_3d_from_neuron_median_um_p50_p90_p95_p99=np.percentile(Rr, [50, 90, 95, 99]).round(2).tolist(),
            habit_offset_norm_um_pct_10_50_90=np.percentile(hab, [10, 50, 90]).round(2).tolist(),
            per_neuron_sigma_3d_um_pct_10_50_90=np.percentile(nsig, [10, 50, 90]).round(3).tolist(),
            per_neuron=per)
        print(name, json.dumps({k: v for k, v in out[name].items() if k != 'per_neuron'}), flush=True)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
