"""Leifer accuracy with suspect GT neurons removed, and which neurons are suspect from the GT alone.

Per GT neuron of a gt_tracks_from_clicks.py status file, GT-only warning signs (no tracker involved):
  excluded_frac     clicks excluded as coincident or merged
  seg_miss_frac     clicks with no mask (segmentation misses)
  click_offset_um   median distance from the click to its assigned object's centroid
  gt_spikiness      median |x_t - (x_{t-1} + x_{t+1}) / 2| of the GT-tracked green/red ratio (over its median):
                    single-volume jumps, as when the clicks hop between two objects
  suspicion         mean rank over the four (1 = most suspect)
and per tracking run (saved pred_all tracks) the neuron's accuracy over kept clicks, plus consensus_err_frac:
clicks where every run holds the same object and it is not the GT's (a list for manual checks, written with
--nwb_click_list as consensus_error_clicks.csv in NWB voxels).
Per tracking run (saved pred_all tracks) the neuron's accuracy over kept clicks.

Then every run is rescored with sets of neurons removed: the --remove sets given by name, and the top-k neurons by
suspicion. Tracks are matched to the full GT first (gt_valid), then the removed neurons' clicks are dropped, so the
other neurons' matches do not change. Removal chosen from a tracker's errors would be circular; only the GT-only
ranking or a manual check is a reason to drop a neuron.

    python .../neuron_removal_scores.py --status .../sd_strict_status.npz --gt_project .../stardist_strictgt/project_config.yaml \
        --nwb .../reexport_A2.nwb --tracks base=a.h5 t57=b.h5 --remove ambiguous=neuron_023,neuron_058 --out DIR
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def gt_indicators(status, label, names, codes, gt_cfg, nwb):
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    from barlow_track.scripts.project_reanalysis.compare_traces import _traces
    prj = ProjectData.load_final_project_data(gt_cfg, allow_hybrid_loading=True, verbose=0)
    puc = prj.physical_unit_conversion
    um = np.array([puc.zimmer_um_per_pixel_z, puc.zimmer_fluroscence_um_per_pixel_xy,
                   puc.zimmer_fluroscence_um_per_pixel_xy])
    a = _load_nwb_arrays(ProjectData.load_final_project_data(nwb, allow_hybrid_loading=True, verbose=0))
    if [str(n) for n in a['neurons']] != names:
        raise SystemExit('neuron order differs between the NWB and the status file')
    C = np.stack([a['gz'], a['gx'], a['gy']], 2)
    md = prj.segmentation_metadata
    T, N = status.shape
    off = np.full((T, N), np.nan)
    for t in range(T):
        js = np.flatnonzero(status[t] == codes['assigned'])
        if not len(js):
            continue
        m = md.segmentation_metadata[t]
        for j in js:
            off[t, j] = np.linalg.norm((np.asarray(m.loc[int(label[t, j]), 'centroids'], float) - C[t, j]) * um)
    g, r, _ = _traces(os.path.dirname(gt_cfg))
    clicked = status != codes['no_click']
    excl = np.isin(status, [v for k, v in codes.items() if k.startswith('excluded')])
    miss = np.isin(status, [v for k, v in codes.items() if k.startswith('missed')])
    rows = []
    for j, n in enumerate(names):
        sp = np.nan
        if (n, 'area') in g.columns:
            on = np.nan_to_num(g[(n, 'area')].values) > 0
            with np.errstate(all='ignore'):
                x = np.where(on, g[(n, 'intensity_image')].values / r[(n, 'intensity_image')].values, np.nan)
            x = x / np.nanmedian(x)
            ok = np.isfinite(x[1:-1]) & np.isfinite(x[:-2]) & np.isfinite(x[2:])
            sp = float(np.median(np.abs(x[1:-1] - (x[:-2] + x[2:]) / 2)[ok])) if ok.any() else np.nan
        nc = max(int(clicked[:, j].sum()), 1)
        rows.append(dict(neuron=n, clicks=int(clicked[:, j].sum()), excluded_frac=excl[:, j].sum() / nc,
                         seg_miss_frac=miss[:, j].sum() / nc, click_offset_um=float(np.nanmedian(off[:, j])),
                         gt_spikiness=sp))
    df = pd.DataFrame(rows).set_index('neuron')
    ind = ['excluded_frac', 'seg_miss_frac', 'click_offset_um', 'gt_spikiness']
    df['suspicion'] = df[ind].rank(ascending=False).mean(1).rank()
    return df.sort_values('suspicion')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--status', required=True)
    ap.add_argument('--gt_project', required=True, help='GT-tracked project of the same status file (traces, metadata)')
    ap.add_argument('--nwb', required=True, help='NWB with the clicks')
    ap.add_argument('--tracks', nargs='+', required=True, help='NAME=pred_all h5')
    ap.add_argument('--remove', nargs='*', default=[], help='NAME=neuron_a,neuron_b')
    ap.add_argument('--top_k', nargs='*', type=int, default=[1, 2, 3, 5, 8, 10])
    ap.add_argument('--matching', default='gt_valid')
    ap.add_argument('--nwb_click_list', action='store_true',
                    help='write consensus_error_clicks.csv (voxel click and both objects per click) for manual checks')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from barlow_track.scripts.project_reanalysis.tracks_to_project import matched_labels
    z = np.load(args.status)
    codes = json.loads(str(z['codes']))
    status, label, names = z['status'], z['label'], [str(n) for n in z['names']]
    assigned = status == codes['assigned']
    kept = assigned | np.isin(status, [v for k, v in codes.items() if k.startswith('missed')])

    ind = gt_indicators(status, label, names, codes, args.gt_project, args.nwb)
    correct, Ps = {}, []
    for spec in args.tracks:
        name, path = spec.split('=', 1)
        P, G = matched_labels(pd.read_hdf(path, 'pred_all'), status, label, names, codes, args.matching)
        correct[name] = assigned & (P == G)
        Ps.append(P)
        ind[f'acc_{name}'] = pd.Series(correct[name].sum(0) / np.maximum(kept.sum(0), 1), index=names)
    # Consensus errors: every run holds the same object for the neuron, and it is not the GT's object. Different
    # trackers (e.g. image embeddings and fDNC's positions only) agreeing against the GT point at the GT or the
    # segmentation; a list for manual checking, not a removal rule.
    S = np.stack(Ps)
    agree = np.isfinite(S).all(0) & (S == S[0]).all(0)
    cons = assigned & agree & (S[0] != G)
    ind['consensus_err_frac'] = pd.Series(cons.sum(0) / np.maximum(kept.sum(0), 1), index=names)
    ind['any_err_frac'] = pd.Series((assigned & ~np.stack([correct[k] for k in correct]).all(0)).sum(0)
                                    / np.maximum(kept.sum(0), 1), index=names)
    ind.to_csv(os.path.join(args.out, 'neurons.csv'))
    if args.nwb_click_list:
        from wbfm.utils.projects.finished_project_data import ProjectData
        from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
        a = _load_nwb_arrays(ProjectData.load_final_project_data(args.nwb, allow_hybrid_loading=True, verbose=0))
        md = ProjectData.load_final_project_data(args.gt_project, allow_hybrid_loading=True,
                                                 verbose=0).segmentation_metadata
        ts, js = np.nonzero(cons)
        cl = []
        for t, j in zip(ts, js):
            m = md.segmentation_metadata[t]
            cg = np.asarray(m.loc[int(G[t, j]), 'centroids'], float)
            cc = np.asarray(m.loc[int(S[0, t, j]), 'centroids'], float)
            cl.append(dict(neuron=names[j], t=int(t), click_z=a['gz'][t, j], click_x=a['gx'][t, j],
                           click_y=a['gy'][t, j], gt_label=int(G[t, j]), gt_obj_z=cg[0], gt_obj_x=cg[1],
                           gt_obj_y=cg[2], consensus_label=int(S[0, t, j]), cons_obj_z=cc[0], cons_obj_x=cc[1],
                           cons_obj_y=cc[2]))
        pd.DataFrame(cl).to_csv(os.path.join(args.out, 'consensus_error_clicks.csv'), index=False)

    sets = {'none': []}
    for spec in args.remove:
        k, v = spec.split('=', 1)
        sets[k] = [s for s in v.split(',') if s]
    order = list(ind.index)
    cons_order = list(ind.sort_values('consensus_err_frac', ascending=False).index)
    for k in args.top_k:
        sets[f'gt_suspicion_top{k}'] = order[:k]
    for k in args.top_k:
        sets[f'consensus_err_top{k}'] = cons_order[:k]
    rows = []
    for sname, drop in sets.items():
        bad = [n for n in drop if n not in names]
        if bad:
            raise SystemExit(f'{sname}: unknown neurons {bad}')
        keep = np.array([n not in drop for n in names])
        for name, c in correct.items():
            rows.append(dict(removed=sname, neurons_removed=len(drop), run=name,
                             kept_clicks=int(kept[:, keep].sum()),
                             accuracy_assigned=round(float(c[:, keep].sum() / max(assigned[:, keep].sum(), 1)), 4),
                             accuracy_all_kept=round(float(c[:, keep].sum() / max(kept[:, keep].sum(), 1)), 4),
                             neurons=','.join(drop)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, 'removal_scores.csv'), index=False)
    print(ind.head(12).round(3).to_string(), flush=True)
    print(ind.sort_values('consensus_err_frac', ascending=False).head(12).round(3).to_string(), flush=True)
    print('consensus errors / all errors of the first run:', int(cons.sum()), '/',
          int((assigned & ~correct[next(iter(correct))]).sum()), flush=True)
    piv = res.pivot_table(index=['removed', 'neurons_removed'], columns='run', values='accuracy_all_kept', sort=False)
    print('\naccuracy over kept clicks\n', piv.round(4).to_string())
    piv = res.pivot_table(index=['removed', 'neurons_removed'], columns='run', values='accuracy_assigned', sort=False)
    print('\naccuracy over assigned clicks\n', piv.round(4).to_string())
    with open(os.path.join(args.out, 'sets.json'), 'w') as f:
        json.dump(dict(status=os.path.abspath(args.status), tracks=args.tracks, sets=sets), f, indent=1)


if __name__ == '__main__':
    main()
