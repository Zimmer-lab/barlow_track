"""TEST_PLAN T3 report: per-time exclusivity ablation of label propagation and a fair HDBSCAN (reviewer R2.3).

Reads round_shared_v1's rescore files for trial 59 (all replicates unless --no_replicates):
  exp_results_v2.jsonl                          label propagation v2 (both exclusivity steps on)
  exp_results_v2_noE2.jsonl / _v2_noE3 / _v2_noexcl   per-run step off / final step off / both off
  exp_results_global_mcs0.5_ms0.02.jsonl        HDBSCAN at the tracker defaults (min_cluster_size 0.5, min_samples
                                                0.02), all cells, with duplicate statistics (the older
                                                exp_results_global.jsonl has none; used only if this is missing)
  exp_results_global_mcs<a>_ms<b>.jsonl         HDBSCAN grid (rep 0 cells)
  exp_results_global_excl.jsonl                 HDBSCAN (defaults) + per-time Hungarian on its soft membership
  exp_results_global_excl_mcs<a>_ms<b>.jsonl    the same at a recording's GT-free tuned setting (rep 0 cells)

Tuned HDBSCAN is chosen WITHOUT ground truth, by a rule fixed before the results (2026-11-24): per cell, among the
grid settings (default included) whose number of clusters is within 20% of the mean detections per volume, the one
with the largest fraction of objects that carry a label not repeated in their volume
((n_labelled - n_duplicate_objects) / n_objects); if none qualifies, the setting closest in cluster count. The
best grid setting by accuracy is reported next to it as an upper bound (GT-selected, not a fair comparison).

Prints one table (recording x variant, accuracy mean over replicates and duplicate fraction) and writes it as tsv.

    python barlow_track/scripts/scratch/exclusivity_ablation/t3_report.py [--out t3_table.tsv]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

ROUND = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_shared_v1'
FIXED = ['v2', 'v2_noE2', 'v2_noE3', 'v2_noexcl', 'global', 'global_excl']
DEFAULT = 'global_mcs0.5_ms0.02'


def latest(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('tag'):
                out[r['tag']] = r
    return out


def trial59(records, no_replicates):
    out = {}
    for tag, r in records.items():
        m = re.search(r'_trial59(r\d+)?_', tag)
        if m and r.get('accuracy') is not None and not (no_replicates and m.group(1)):
            out[tag] = r
    return out


def dup(r):
    d = r.get('label_duplicates') or {}
    return d.get('duplicate_fraction', np.nan)


def gt_free_choice(cands):
    """cands: {setting: record} for one cell -> chosen setting (rule in the module docstring)."""
    def stats(r):
        d = r.get('label_duplicates') or {}
        n_obj, n_frames = d.get('n_objects'), r.get('n_frames')
        if not n_obj or not n_frames:
            return None
        per_vol = n_obj / n_frames
        uniq = (d['n_labelled'] - d['n_duplicate_objects']) / n_obj
        return abs(d['n_clusters'] / per_vol - 1), uniq
    s = {k: stats(r) for k, r in cands.items()}
    s = {k: v for k, v in s.items() if v is not None}
    if not s:
        return None
    ok = [k for k, (dev, _) in s.items() if dev <= 0.2]
    return max(ok, key=lambda k: s[k][1]) if ok else min(s, key=lambda k: s[k][0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round_dir', default=ROUND)
    ap.add_argument('--no_replicates', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    recs = {v: trial59(latest(os.path.join(a.round_dir, f'exp_results_{v}.jsonl')), a.no_replicates) for v in FIXED}
    default = trial59(latest(os.path.join(a.round_dir, f'exp_results_{DEFAULT}.jsonl')), a.no_replicates)
    if default:
        recs['global'] = default
    grid = {}
    for f in glob.glob(os.path.join(a.round_dir, 'exp_results_global_mcs*_ms*.jsonl')):
        name = os.path.basename(f)[len('exp_results_'):-len('.jsonl')]
        grid[name] = trial59(latest(f), True)
    # the default setting is a grid point too
    grid_all = dict(grid)
    if DEFAULT not in grid_all:  # the default is a grid point too
        grid_all[DEFAULT] = {t: r for t, r in recs['global'].items() if not re.search(r'_trial59r\d+_', t)}

    rows = defaultdict(dict)
    for v, rs in recs.items():
        by_lab = defaultdict(list)
        for r in rs.values():
            by_lab[r['lab']].append(r)
        for lab, lst in by_lab.items():
            rows[lab][f'{v}_acc'] = np.mean([r['accuracy'] for r in lst])
            rows[lab][f'{v}_n'] = len(lst)
            rows[lab][f'{v}_dup'] = np.nanmean([dup(r) for r in lst])
    cells = defaultdict(dict)  # lab -> setting -> record (rep 0)
    for name, rs in grid_all.items():
        for r in rs.values():
            cells[r['lab']][name] = r
    for lab, cands in cells.items():
        choice = gt_free_choice(cands)
        if choice is not None:
            rows[lab]['hdbscan_tuned_setting'] = choice
            rows[lab]['hdbscan_tuned_acc'] = cands[choice]['accuracy']
            rows[lab]['hdbscan_tuned_dup'] = dup(cands[choice])
            # HDBSCAN + exclusivity at the tuned setting (launch_t3.sh excl_tuned); the default is global_excl
            excl = 'global_excl' if choice == DEFAULT else 'global_excl' + choice[len('global'):]
            ex = [r for r in trial59(latest(os.path.join(a.round_dir, f'exp_results_{excl}.jsonl')), True).values()
                  if r['lab'] == lab]
            if ex:
                rows[lab]['hdbscan_tuned_excl_acc'] = ex[0]['accuracy']
                rows[lab]['hdbscan_tuned_excl_dup'] = dup(ex[0])
        best = max(cands, key=lambda k: cands[k]['accuracy'])
        rows[lab]['hdbscan_gtbest_setting'] = best
        rows[lab]['hdbscan_gtbest_acc'] = cands[best]['accuracy']
        rows[lab]['hdbscan_grid_points'] = len(cands)

    t = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    acc = [c for c in t.columns if c.endswith('_acc')]
    with pd.option_context('display.width', 250, 'display.max_columns', 50):
        print(t[acc].round(4))
        print(t[[c for c in t.columns if c.endswith('_dup')]].round(4))
        if 'hdbscan_tuned_setting' in t:
            print(t[['hdbscan_tuned_setting', 'hdbscan_gtbest_setting', 'hdbscan_grid_points']])
        if 'v2_acc' in t:
            print('mean over recordings, pt vs v2:')
            print((100 * t[acc].sub(t['v2_acc'], axis=0)).mean().round(2))
    if a.out:
        t.to_csv(a.out, sep='\t')


if __name__ == '__main__':
    main()
