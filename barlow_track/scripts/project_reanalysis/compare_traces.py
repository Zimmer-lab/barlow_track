"""Compare wbfm traces of the same neurons between two projects of one recording, neuron by neuron.

For the Leifer traces panel: GT-tracked vs tracker-tracked on one segmentation (the tracker's effect), and
NeRVE vs StarDist masks both GT-tracked (the segmentation's effect). Per pair (reference, test) and neuron:

  frames_ref / frames_test / frames_both  volumes with a trace (mask area > 0) in each and in both
  r_ratio      Pearson r of the raw green/red ratio (summed mask intensities, 4-traces/{green,red}_traces.h5)
               on frames both have
  r_paper      Pearson r of 4-traces/paper_traces.h5 (wbfm's processed traces) on frames both have finite
  same_object  fraction of frames_both where the two projects' tracks hold the same mask label (only when
               both projects share the segmentation; final tracks raw_segmentation_id)

    python .../compare_traces.py --pairs tracker_on_stardist=<strictgt project>:<track_base project> ... \
        --out <dir>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def _traces(project_dir):
    t = os.path.join(project_dir, '4-traces')
    g = pd.read_hdf(os.path.join(t, 'green_traces.h5'))
    r = pd.read_hdf(os.path.join(t, 'red_traces.h5'))
    p = pd.read_hdf(os.path.join(t, 'paper_traces.h5'))
    return g, r, p


def _final_labels(project_config):
    from wbfm.utils.projects.finished_project_data import ProjectData
    prj = ProjectData.load_final_project_data(project_config, allow_hybrid_loading=True, verbose=0)
    df = prj.final_tracks
    return df.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1), \
        str(prj.project_config.get_segmentation_config().resolve_relative_path_from_config('output_masks'))


def _r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def compare(ref_cfg, test_cfg):
    ref_dir, test_dir = os.path.dirname(ref_cfg), os.path.dirname(test_cfg)
    g0, r0, p0 = _traces(ref_dir)
    g1, r1, p1 = _traces(test_dir)
    lab0, seg0 = _final_labels(ref_cfg)
    lab1, seg1 = _final_labels(test_cfg)
    same_seg = os.path.realpath(seg0) == os.path.realpath(seg1)
    neurons = sorted(set(g0.columns.get_level_values(0)) & set(g1.columns.get_level_values(0)))
    rows = []
    for n in neurons:
        a0, a1 = g0[(n, 'area')].values, g1[(n, 'area')].values
        on0, on1 = np.nan_to_num(a0) > 0, np.nan_to_num(a1) > 0
        both = on0 & on1
        with np.errstate(divide='ignore', invalid='ignore'):  # red sum 0 where a volume has no trace
            ratio0 = np.where(on0, g0[(n, 'intensity_image')].values / r0[(n, 'intensity_image')].values, np.nan)
            ratio1 = np.where(on1, g1[(n, 'intensity_image')].values / r1[(n, 'intensity_image')].values, np.nan)
        r_ratio, _ = _r(np.where(both, ratio0, np.nan), np.where(both, ratio1, np.nan))
        if n in p0.columns and n in p1.columns:
            r_paper, n_paper = _r(p0[n].values.astype(float), p1[n].values.astype(float))
        else:
            r_paper, n_paper = np.nan, 0
        row = dict(neuron=n, frames_ref=int(on0.sum()), frames_test=int(on1.sum()), frames_both=int(both.sum()),
                   r_ratio=r_ratio, r_paper=r_paper, frames_paper=n_paper)
        if same_seg and n in lab0.columns and n in lab1.columns:
            l0, l1 = lab0[n].values.astype(float), lab1[n].values.astype(float)
            m = np.isfinite(l0) & np.isfinite(l1)
            row['same_object'] = float((l0[m] == l1[m]).mean()) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows), same_seg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', nargs='+', required=True, help='NAME=ref_project_config.yaml:test_project_config.yaml')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = {}
    fig, axes = plt.subplots(1, len(args.pairs), figsize=(4.2 * len(args.pairs), 3.6), squeeze=False)
    for ax, spec in zip(axes[0], args.pairs):
        name, rest = spec.split('=', 1)
        ref, test = rest.split(':', 1)
        df, same_seg = compare(ref, test)
        df.to_csv(os.path.join(args.out, f'{name}.csv'), index=False)
        s = dict(ref=ref, test=test, same_segmentation=same_seg, neurons=len(df))
        for c in ('r_ratio', 'r_paper', 'same_object'):
            if c in df and df[c].notna().any():
                v = df[c].dropna().values
                s[c] = dict(n=int(len(v)), median=round(float(np.median(v)), 4),
                            p10=round(float(np.percentile(v, 10)), 4), min=round(float(v.min()), 4),
                            below_0_9=int((v < 0.9).sum()), below_0_5=int((v < 0.5).sum()))
        s['frames_both_over_ref'] = round(float(df.frames_both.sum() / max(df.frames_ref.sum(), 1)), 4)
        s['frames_test_over_ref'] = round(float(df.frames_test.sum() / max(df.frames_ref.sum(), 1)), 4)
        summary[name] = s
        print(name, json.dumps({k: v for k, v in s.items() if k not in ('ref', 'test')}), flush=True)
        for c, col in (('r_ratio', 'C0'), ('r_paper', 'C1')):
            v = np.sort(df[c].dropna().values)
            if len(v):
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, label=c)
        ax.set_xlim(-0.2, 1.0)
        ax.set_xlabel('per-neuron Pearson r')
        ax.set_ylabel('fraction of neurons')
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'trace_agreement.png'), dpi=150)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
