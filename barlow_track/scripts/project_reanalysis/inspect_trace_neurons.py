"""Why do some neurons' traces disagree between two projects of one recording? Tracking errors vs trace sensitivity.

Companion to compare_traces.py (same NAME=ref:test pairs). Per neuron of the pair:

  same_object        fraction of frames both track where they hold the same mask (same segmentation only)
  diff_runs / diff_longest   runs of differing frames: scattered swaps vs one long wrong stretch
  diff_to_other_gt   of differing frames, fraction where the test holds a mask the ref gives another neuron
                     (a swap between annotated neurons) rather than an unannotated object
  diff_dist_um       median distance between the ref and test objects on differing frames
  r_ratio_same / r_ratio_diff   Pearson r of the raw green/red ratio on same-object / differing frames
  snr                ref ratio s.d. over its noise (1.4826 x median |first difference| / sqrt 2): how much
                     real signal a neuron has; low-snr traces decorrelate from any small perturbation
  red_mean, area_mean   ref mean red intensity and mask size

Writes <out>/<name>_neurons.csv, <name>_worst.png (ratio and paper traces of the --worst neurons with differing
frames shaded) and <name>_r_vs_snr.png (every neuron's r_paper against its snr).

    python .../inspect_trace_neurons.py --pairs tracker_on_stardist=<ref yaml>:<test yaml> --out DIR [--worst 6]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from barlow_track.scripts.project_reanalysis.compare_traces import _traces, _r


def _project(cfg):
    from wbfm.utils.projects.finished_project_data import ProjectData
    prj = ProjectData.load_final_project_data(cfg, allow_hybrid_loading=True, verbose=0)
    lab = prj.final_tracks.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
    seg = str(prj.project_config.get_segmentation_config().resolve_relative_path_from_config('output_masks'))
    return prj, lab, seg


def _centroids(md, t, labels):
    m = md.segmentation_metadata[t]
    out = np.full((len(labels), 3), np.nan)
    for i, l in enumerate(labels):
        if np.isfinite(l) and int(l) in m.index:
            out[i] = np.asarray(m.loc[int(l), 'centroids'], float)
    return out


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return e - s


def inspect(ref_cfg, test_cfg):
    ref_dir, test_dir = os.path.dirname(ref_cfg), os.path.dirname(test_cfg)
    g0, r0, p0 = _traces(ref_dir)
    g1, r1, p1 = _traces(test_dir)
    prj0, lab0, seg0 = _project(ref_cfg)
    _, lab1, seg1 = _project(test_cfg)
    same_seg = os.path.realpath(seg0) == os.path.realpath(seg1)
    puc = prj0.physical_unit_conversion
    um = np.array([puc.zimmer_um_per_pixel_z, puc.zimmer_fluroscence_um_per_pixel_xy,
                   puc.zimmer_fluroscence_um_per_pixel_xy])
    md = prj0.segmentation_metadata
    neurons = sorted(set(g0.columns.get_level_values(0)) & set(g1.columns.get_level_values(0)))
    L0 = lab0.reindex(columns=neurons).values.astype(float)
    rows, series = [], {}
    for j, n in enumerate(neurons):
        a0, a1 = np.nan_to_num(g0[(n, 'area')].values) > 0, np.nan_to_num(g1[(n, 'area')].values) > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            q0 = np.where(a0, g0[(n, 'intensity_image')].values / r0[(n, 'intensity_image')].values, np.nan)
            q1 = np.where(a1, g1[(n, 'intensity_image')].values / r1[(n, 'intensity_image')].values, np.nan)
        both = a0 & a1
        x = q0[np.isfinite(q0)]
        noise = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2) if len(x) > 10 else np.nan
        row = dict(neuron=n, frames_both=int(both.sum()),
                   r_ratio=_r(np.where(both, q0, np.nan), np.where(both, q1, np.nan))[0],
                   r_paper=_r(p0[n].values.astype(float), p1[n].values.astype(float))[0]
                   if n in p0.columns and n in p1.columns else np.nan,
                   snr=float(np.std(x) / noise) if noise and np.isfinite(noise) and noise > 0 else np.nan,
                   red_mean=float(np.nanmean(np.where(a0, r0[(n, 'intensity_image')].values
                                                      / g0[(n, 'area')].values, np.nan))),
                   area_mean=float(np.nanmean(np.where(a0, g0[(n, 'area')].values, np.nan))))
        diff = np.zeros(len(q0), bool)
        if same_seg and n in lab0.columns and n in lab1.columns:
            l0, l1 = lab0[n].values.astype(float), lab1[n].values.astype(float)
            m = np.isfinite(l0) & np.isfinite(l1)
            same = m & (l0 == l1)
            diff = m & (l0 != l1)
            row['same_object'] = float(same.sum() / max(m.sum(), 1))
            runs = _runs(diff)
            row['diff_frames'] = int(diff.sum())
            row['diff_runs'] = int(len(runs))
            row['diff_longest'] = int(runs.max()) if len(runs) else 0
            ts = np.flatnonzero(diff)
            other = [np.isin(l1[t], np.delete(L0[t], j)) for t in ts]
            row['diff_to_other_gt'] = float(np.mean(other)) if len(ts) else np.nan
            if len(ts):
                d = [np.linalg.norm((_centroids(md, t, [l0[t]])[0] - _centroids(md, t, [l1[t]])[0]) * um)
                     for t in ts]
                row['diff_dist_um'] = float(np.nanmedian(d))
            row['r_ratio_same'] = _r(np.where(same & both, q0, np.nan), np.where(same & both, q1, np.nan))[0]
            row['r_ratio_diff'] = _r(np.where(diff & both, q0, np.nan), np.where(diff & both, q1, np.nan))[0]
        rows.append(row)
        series[n] = (q0, q1, p0[n].values if n in p0.columns else None, p1[n].values if n in p1.columns else None,
                     diff)
    return pd.DataFrame(rows), series, same_seg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='+', required=True, help='NAME=ref_project_config.yaml:test_project_config.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--worst', type=int, default=6, help='neurons (lowest r_paper) to plot')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = {}
    for spec in args.pairs:
        name, rest = spec.split('=', 1)
        ref, test = rest.split(':', 1)
        df, series, same_seg = inspect(ref, test)
        df.to_csv(os.path.join(args.out, f'{name}_neurons.csv'), index=False)
        worst = df.sort_values('r_paper').neuron.values[:args.worst]
        good = df[df.r_paper >= 0.9]
        bad = df[df.r_paper < 0.5]
        s = dict(ref=ref, test=test, same_segmentation=same_seg, neurons=len(df), worst=list(worst),
                 spearman_r_paper_snr=float(df[['r_paper', 'snr']].corr('spearman').iloc[0, 1]),
                 snr_median_r_ge_0_9=float(good.snr.median()), snr_median_r_lt_0_5=float(bad.snr.median()),
                 n_r_lt_0_5=int(len(bad)))
        if same_seg:
            s['same_object_median_r_ge_0_9'] = float(good.same_object.median())
            s['same_object_median_r_lt_0_5'] = float(bad.same_object.median())
        summary[name] = s
        cols = [c for c in ('neuron', 'r_paper', 'r_ratio', 'r_ratio_same', 'r_ratio_diff', 'same_object', 'diff_frames',
                            'diff_runs', 'diff_longest', 'diff_to_other_gt', 'diff_dist_um', 'snr', 'red_mean',
                            'area_mean') if c in df]
        print('==', name, json.dumps({k: v for k, v in s.items() if k not in ('ref', 'test')}))
        print(df.sort_values('r_paper')[cols].head(args.worst + 4).round(3).to_string(index=False), flush=True)

        fig, axes = plt.subplots(len(worst), 1, figsize=(12, 1.9 * len(worst)), squeeze=False, sharex=True)
        for ax, n in zip(axes[:, 0], worst):
            q0, q1, pp0, pp1, diff = series[n]
            t = np.arange(len(q0))
            for s0 in np.flatnonzero(np.diff(np.r_[0, diff.astype(int)]) == 1):
                e0 = s0 + np.argmin(diff[s0:]) if not diff[s0:].all() else len(diff)
                ax.axvspan(s0, e0, color='0.85', lw=0)
            ax.plot(t, q0 / np.nanmedian(q0), color='k', lw=0.6, label='ref ratio')
            ax.plot(t, q1 / np.nanmedian(q1), color='C3', lw=0.6, alpha=0.8, label='test ratio')
            r = df.set_index('neuron').loc[n]
            ax.set_title(f"{n}  r_paper {r.r_paper:.2f}  r_ratio {r.r_ratio:.2f}  snr {r.snr:.1f}"
                         + (f"  same object {r.same_object:.2f}" if 'same_object' in r else ''), fontsize=8)
            ax.set_ylabel('ratio / median', fontsize=7)
        axes[0, 0].legend(fontsize=7, loc='upper right')
        axes[-1, 0].set_xlabel('volume (grey: frames where the two projects hold different objects)')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f'{name}_worst.png'), dpi=130)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.scatter(df.snr, df.r_paper, s=12, c=df.same_object if 'same_object' in df else 'C0', cmap='viridis',
                   vmin=0.5, vmax=1)
        ax.set_xscale('log')
        ax.set_xlabel('ref trace SNR (s.d. / noise)')
        ax.set_ylabel('r_paper')
        ax.set_title(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f'{name}_r_vs_snr.png'), dpi=130)
        plt.close(fig)
    with open(os.path.join(args.out, 'inspect_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
