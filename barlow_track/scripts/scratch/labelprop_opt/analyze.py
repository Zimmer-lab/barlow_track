"""Paired comparison of label-propagation variants against 'base' from sweep_labelprop.py JSON lines.

For every (case, seed) the variant's accuracy minus base's accuracy at the same seed is one paired
delta (same SVD, seed volumes and kNN graph wherever the variant does not change them). Reports per
variant: mean delta (pt), its standard error over pairs, wins/pairs, the worst case-mean delta, the
base seed spread per case (noise yardstick), the delta in the GT-free agreement, and time
(graph + prop + sync + post seconds of that record; stages served from the memo count 0, so compare
times on variants run first in their group, or use the speed batch).

Decision rule (PLAN.md section 4): promote from the screen if mean delta >= +0.3 pt and the variant
wins in >= 75% of pairs; adopt after the confirmation batch if mean delta >= +0.3 pt with the paired
t-test p < 0.05 over all confirmation pairs and no case's mean delta below -0.5 pt.

    python analyze.py batch1.jsonl [more.jsonl ...] [--metric accuracy|acc_fast] [--cases substr ...]
"""
import argparse
import collections
import json
import math

import numpy as np


def load(paths):
    recs = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                recs[(x['case'], x['variant'], x['seed'])] = x  # latest wins
    return list(recs.values())


def t_p(d):
    """Two-sided paired t-test p value (normal approximation for n > 30; scipy if available)."""
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or np.allclose(d.std(ddof=1), 0):
        return float('nan')
    try:
        from scipy import stats
        return float(stats.ttest_1samp(d, 0.0).pvalue)
    except Exception:
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        return float(math.erfc(abs(t) / math.sqrt(2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('jsonl', nargs='+')
    ap.add_argument('--metric', default='accuracy')
    ap.add_argument('--cases', nargs='*', default=None, help='keep cases containing any of these')
    ap.add_argument('--per_case', action='store_true')
    args = ap.parse_args()
    recs = load(args.jsonl)
    if args.cases:
        recs = [r for r in recs if any(c in r['case'] for c in args.cases)]
    by = {(r['case'], r['variant'], r['seed']): r for r in recs}
    base = {(c, s): r for (c, v, s), r in by.items() if v == 'base'}
    cases = sorted({c for c, _ in base})
    print(f"{len(recs)} records, {len(cases)} cases with a base")
    print("base per case (mean over seeds, seed range in pt):")
    for c in cases:
        a = [base[(c, s)][args.metric] for (cc, s) in base if cc == c]
        print(f"  {c:70s} {np.mean(a):.4f}  n={len(a)}  range {100 * (max(a) - min(a)):.2f}")
    variants = sorted({v for (_, v, _) in by if v != 'base'})
    rows = []
    for v in variants:
        d, dag, per_case, secs = [], [], collections.defaultdict(list), []
        for (c, vv, s), r in by.items():
            if vv != v or (c, s) not in base:
                continue
            b = base[(c, s)]
            d.append(100 * (r[args.metric] - b[args.metric]))
            per_case[c].append(d[-1])
            ag, bg = (r.get('consistency') or {}).get('agree_mean'), (b.get('consistency') or {}).get('agree_mean')
            if ag is not None and bg is not None:
                dag.append(ag - bg)
            secs.append(sum(r.get(k, 0) or 0 for k in ('graph_s', 'prop_s', 'sync_s', 'post_s')))
        if not d:
            continue
        d = np.array(d)
        worst = min(np.mean(x) for x in per_case.values())
        rows.append((v, d.mean(), d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float('nan'),
                     int((d > 0).sum()), len(d), worst, t_p(d), np.mean(dag) if dag else float('nan'),
                     np.mean(secs), per_case))
    rows.sort(key=lambda r: -r[1])
    print(f"\n{'variant':16s} {'mean_dpt':>8s} {'se':>5s} {'wins':>7s} {'worst_case':>10s} {'p':>6s} "
          f"{'d_agree':>8s} {'sec':>6s}  verdict")
    for v, m, se, w, n, worst, p, dag, sec, per_case in rows:
        verdict = 'promote' if (m >= 0.3 and w >= 0.75 * n) else ('harmful' if m <= -0.3 else '')
        print(f"{v:16s} {m:+8.2f} {se:5.2f} {w:3d}/{n:<3d} {worst:+10.2f} {p:6.3f} {dag:+8.3f} {sec:6.0f}  {verdict}")
        if args.per_case:
            for c, x in sorted(per_case.items()):
                print(f"    {c:70s} {np.mean(x):+.2f} ({', '.join(f'{y:+.2f}' for y in x)})")


if __name__ == '__main__':
    main()
