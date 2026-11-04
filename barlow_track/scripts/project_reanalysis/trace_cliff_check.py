"""Are step changes ("cliffs") in GT-tracked traces tracking errors or trace processing?

For each project (wbfm projects tracked with GT, sharing one recording and one set of GT clicks):
  1. Step score per neuron in paper_traces.h5: max over t of |median(x[t-w:t]) - median(x[t:t+w])| / std(x).
  2. Per neuron in --gt_project: the GT object's centroid vs the GT click (longest run of frames with the click
     farther than --far_um) and 100-frame windows whose median object volume departs from the neuron's median
     (a switch to a merged or neighbouring object). A tracking error shows up here, not only in the trace.
  3. For --neurons: dr_over_r_50 traces recomputed with wbfm's calculate_traces with and without the per-channel
     exponential bleach correction and the background subtraction (paper defaults otherwise), and the raw
     green/red ratio; early (frames 0-600) vs late (900-1500) means, and a figure.

    WBFM_NWB_LAZY_SEGMENTATION=1 python .../trace_cliff_check.py --projects name=.../project_config.yaml ... \
        --gt_project name --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb --neurons neuron_038 neuron_060 --out <dir>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def step_score(x, w):
    best = (0.0, -1)
    s = np.nanstd(x) + 1e-9
    for t in range(w, len(x) - w, 5):
        v = abs(np.nanmedian(x[t - w:t]) - np.nanmedian(x[t:t + w])) / s
        if v > best[0]:
            best = (float(v), t)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--projects', nargs='+', required=True, help='NAME=project_config.yaml')
    ap.add_argument('--gt_project', required=True, help='NAME of the project whose GT objects are checked')
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--neurons', nargs='*', default=[])
    ap.add_argument('--window', type=int, default=150)
    ap.add_argument('--far_um', type=float, default=2.0)
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.5, 0.3226, 0.3226])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    projects = dict(p.split('=', 1) for p in args.projects)
    dirs = {k: os.path.dirname(v) for k, v in projects.items()}
    out = dict(args=vars(args))

    # 1. Step scores
    steps = {}
    for k, d in dirs.items():
        pt = pd.read_hdf(os.path.join(d, '4-traces', 'paper_traces.h5'))
        for n in pt.columns:
            steps.setdefault(n, {})[k] = step_score(pt[n].values.astype(float), args.window)
    ranked = sorted(steps, key=lambda n: -max(v[0] for v in steps[n].values()))
    out['top_step_scores'] = {n: {k: dict(score=round(v[0], 2), t=v[1]) for k, v in steps[n].items()}
                              for n in ranked[:15]}
    # Neurons whose paper trace falls from early to late in one project
    for k, d in dirs.items():
        pt = pd.read_hdf(os.path.join(d, '4-traces', 'paper_traces.h5'))
        drop = (pt.iloc[:600].mean() - pt.iloc[900:1500].mean()).sort_values(ascending=False)
        out.setdefault('early_minus_late_paper_trace_top5', {})[k] = drop.head(5).round(3).to_dict()

    # 2. GT objects vs clicks
    gt_nwb = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(gt_nwb)
    names = list(a['neurons'])
    scale = np.asarray(args.voxel_um)
    gt = pd.read_hdf(os.path.join(dirs[args.gt_project], '3-tracking', 'postprocessing', 'combined_3d_tracks.h5'))
    per = {}
    for j, n in enumerate(names):
        if n not in gt:
            continue
        g = gt[n]
        click = np.stack([a['gz'][:, j], a['gx'][:, j], a['gy'][:, j]], 1)
        pos = np.stack([g['z'].values, g['x'].values, g['y'].values], 1)
        d = np.linalg.norm((pos - click) * scale, axis=1)
        run = best = 0
        for f in d > args.far_um:
            run = run + 1 if f else 0
            best = max(best, run)
        vol = g['volume'].values.astype(float)
        wmed = pd.Series(vol).rolling(100, center=True, min_periods=30).median() / np.nanmedian(vol)
        per[n] = dict(longest_far_run=int(best), median_dist_um=round(float(np.nanmedian(d)), 3),
                      p99_dist_um=round(float(np.nanpercentile(d, 99)), 3),
                      vol_window_max=round(float(np.nanmax(wmed)), 3), vol_window_min=round(float(np.nanmin(wmed)), 3))
    out['gt_objects'] = dict(
        n_neurons=len(per), max_longest_far_run=max(v['longest_far_run'] for v in per.values()),
        neurons_with_volume_window_over_2x_or_under_half=[n for n, v in per.items()
                                                          if v['vol_window_max'] > 2 or v['vol_window_min'] < 0.5],
        per_neuron=per)

    # 3. Processing variants for the chosen neurons
    if args.neurons:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        prj = ProjectData.load_final_project_data(projects[args.gt_project], allow_hybrid_loading=True, verbose=0)
        red = pd.read_hdf(os.path.join(dirs[args.gt_project], '4-traces', 'red_traces.h5'))
        green = pd.read_hdf(os.path.join(dirs[args.gt_project], '4-traces', 'green_traces.h5'))
        variants = [('paper defaults (bleach correct, background)', {}),
                    ('no bleach correction', dict(bleach_correct=False)),
                    ('no background subtraction', dict(background_per_pixel=0)),
                    ('neither', dict(bleach_correct=False, background_per_pixel=0))]
        out['background_per_pixel'] = prj.background_per_pixel
        fig, ax = plt.subplots(len(args.neurons), 2, figsize=(18, 3.2 * len(args.neurons)), squeeze=False)
        res = {}
        for i, n in enumerate(args.neurons):
            ratio = (green[n]['intensity_image'] / red[n]['intensity_image'].replace(0, np.nan)).values
            res[n] = {'raw green/red': (float(np.nanmean(ratio[:600])), float(np.nanmean(ratio[900:1500])))}
            ax[i, 1].plot(red[n]['intensity_image'].replace(0, np.nan).values, lw=.5, label='red')
            ax[i, 1].plot(green[n]['intensity_image'].replace(0, np.nan).values, lw=.5, label='green')
            ax[i, 1].set_title(f'{n}: integrated intensity')
            ax[i, 1].legend(fontsize=7)
            for lab, kw in variants:
                _, y = prj.calculate_traces(channel_mode='dr_over_r_50', calculation_mode='integration',
                                            neuron_name=n, remove_outliers=True, filter_mode='rolling_mean', **kw)
                y = np.asarray(y, float)
                res[n][lab] = (float(np.nanmean(y[:600])), float(np.nanmean(y[900:1500])))
                ax[i, 0].plot(y, lw=.6, label=lab)
            ax[i, 0].set_title(f'{n}: dr_over_r_50')
            ax[i, 0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'processing_variants.png'), dpi=80)
        out['early_late_means'] = {n: {k: [round(v[0], 3), round(v[1], 3)] for k, v in d.items()} for n, d in res.items()}

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'gt_objects'}, indent=1))
    print(json.dumps({k: v for k, v in out['gt_objects'].items() if k != 'per_neuron'}, indent=1))


if __name__ == '__main__':
    main()
