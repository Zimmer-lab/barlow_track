"""TEST_PLAN T3 report: per-time exclusivity ablation of label propagation and a fair HDBSCAN (reviewer R2.3).

Reads round_shared_v1's rescore files for trial 59 (all replicates unless --no_replicates):
  exp_results_v2.jsonl                              label propagation v2 (both exclusivity steps on)
  exp_results_v2_noE2 / _v2_noE3 / _v2_noexcl.jsonl per-run step off / final step off / both off
  exp_results_<cluster>_mcs<a>_ms<b>.jsonl          HDBSCAN cells, <cluster> one of the 2 x 2:
      global            hard labels (noise unlabelled), no exclusivity
      global_soft       soft membership for every object (noise can get a cluster), no exclusivity
      global_excl_hard  hard-labelled objects only, with label propagation's per-time Hungarian step
      global_excl       soft membership for every object, with that step
    at the defaults (mcs0.5_ms0.02, all cells) and, for `global`, a grid on the rep-0 cells. Without exclusivity a
    repeated (volume, cluster) keeps its most confident object in every cell, as v2_noE3 does.
The older exp_results_global.jsonl (first object kept, no duplicate statistics) is only a fallback for `global`.

Splitting HDBSCAN + exclusivity into the 2 x 2 separates the two things global_excl changes at once: which objects
get a label (noise included or not) and the exclusivity itself (side-agent note, 2026-11-24).

Tuned HDBSCAN is chosen WITHOUT ground truth, by a rule fixed before the results (2026-11-24): per cell, among the
grid settings (default included) whose number of clusters is within 20% of the mean detections per volume, the one
with the largest fraction of objects that carry a label not repeated in their volume
((n_labelled - n_duplicate_objects) / n_objects); if none qualifies, the setting closest in cluster count. The 2 x 2
is then read at that setting (t3_launch.sh excl_tuned). The best grid setting by accuracy is reported next to it as
an upper bound (GT-selected, not a fair comparison).

Prints the tables (recording x variant: accuracy, mean over replicates; duplicate fraction before resolution) and
writes them as one tsv.

    python barlow_track/scripts/paper_scripts/revision/t3_report.py [--out t3_table.tsv] [--no_replicates]
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
LABELPROP = ['v2', 'v2_noE2', 'v2_noE3', 'v2_noexcl']
GLOBAL = ['global', 'global_soft', 'global_excl_hard', 'global_excl']
DEFAULT = '_mcs0.5_ms0.02'


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


def load(round_dir, name, no_replicates):
    return trial59(latest(os.path.join(round_dir, f'exp_results_{name}.jsonl')), no_replicates)


def dup(r):
    return (r.get('label_duplicates') or {}).get('duplicate_fraction', np.nan)


def gt_free_choice(cands):
    """cands: {setting suffix: record} for one cell -> chosen suffix (rule in the module docstring)."""
    def stats(r):
        d = r.get('label_duplicates') or {}
        n_obj, n_frames = d.get('n_objects'), r.get('n_frames')
        if not n_obj or not n_frames:
            return None
        return abs(d['n_clusters'] / (n_obj / n_frames) - 1), (d['n_labelled'] - d['n_duplicate_objects']) / n_obj
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

    recs = {v: load(a.round_dir, v, a.no_replicates) for v in LABELPROP}
    for c in GLOBAL:
        recs[c] = load(a.round_dir, c + DEFAULT, a.no_replicates)
    if not recs['global']:
        recs['global'] = load(a.round_dir, 'global', a.no_replicates)  # older records, first object kept

    rows = defaultdict(dict)
    for v, rs in recs.items():
        by_lab = defaultdict(list)
        for r in rs.values():
            by_lab[r['lab']].append(r)
        for lab, lst in by_lab.items():
            rows[lab][f'{v}_acc'] = np.mean([r['accuracy'] for r in lst])
            rows[lab][f'{v}_n'] = len(lst)
            rows[lab][f'{v}_dup'] = np.nanmean([dup(r) for r in lst])

    # HDBSCAN grid (rep 0): suffix -> {lab: record}
    grid = defaultdict(dict)
    for f in glob.glob(os.path.join(a.round_dir, 'exp_results_global_mcs*_ms*.jsonl')):
        suffix = os.path.basename(f)[len('exp_results_global'):-len('.jsonl')]
        for r in trial59(latest(f), True).values():
            grid[suffix][r['lab']] = r
    by_lab = defaultdict(dict)
    for suffix, cells in grid.items():
        for lab, r in cells.items():
            by_lab[lab][suffix] = r
    for lab, cands in by_lab.items():
        choice = gt_free_choice(cands)
        best = max(cands, key=lambda k: cands[k]['accuracy'])
        rows[lab].update(hdbscan_grid_points=len(cands), hdbscan_gtbest=best, hdbscan_gtbest_acc=cands[best]['accuracy'])
        if choice is None:
            continue
        rows[lab]['hdbscan_tuned'] = choice
        for c in GLOBAL:
            r = cands[choice] if c == 'global' else next(
                (x for x in load(a.round_dir, c + choice, True).values() if x['lab'] == lab), None)
            if r is not None:
                rows[lab][f'tuned_{c}_acc'] = r['accuracy']
                rows[lab][f'tuned_{c}_dup'] = dup(r)

    t = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    with pd.option_context('display.width', 250, 'display.max_columns', 50):
        for title, cols in (('label propagation (accuracy)', [f'{v}_acc' for v in LABELPROP]),
                            ('HDBSCAN 2 x 2 at defaults (accuracy)', [f'{c}_acc' for c in GLOBAL]),
                            ('HDBSCAN 2 x 2 at the GT-free tuned setting (accuracy)', [f'tuned_{c}_acc' for c in GLOBAL]),
                            ('duplicate fraction before resolution', [c for c in t.columns if c.endswith('_dup')]),
                            ('HDBSCAN settings', ['hdbscan_tuned', 'hdbscan_gtbest', 'hdbscan_gtbest_acc',
                                                  'hdbscan_grid_points'])):
            cols = [c for c in cols if c in t]
            if cols:
                print(f'\n{title}')
                print(t[cols].round(4))
        acc = [c for c in t.columns if c.endswith('_acc')]
        if 'v2_acc' in t:
            print('\nmean over recordings, pt vs v2:')
            print((100 * t[acc].sub(t['v2_acc'], axis=0)).mean().round(2))
    if a.out:
        t.to_csv(a.out, sep='\t')


if __name__ == '__main__':
    main()
