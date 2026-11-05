"""Spearman correlation of the GT-free self-consistency metrics (consistency_proxy.py output) with accuracy,
per round: over all scored trials, and over the better half by accuracy (where tuning decisions are made,
and where a proxy that only separates broken from working networks stops helping).

    python barlow_track/scripts/scratch/gt_free_proxy/consistency_corr.py consistency.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

METRICS = ['agree_mean', 'agree_p10', 'conf_mean', 'conf_med', 'assigned_frac', 'dup_frac',
           'complete_tracks_50', 'complete_tracks_90', 'ari_seed1']


def main(path):
    by_round = defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by_round[r['round']].append(r)
    for rnd, rows in sorted(by_round.items()):
        rows = sorted(rows, key=lambda r: r['accuracy'])
        acc = np.array([r['accuracy'] for r in rows])
        top = rows[len(rows) // 2:]
        print(f"\n{rnd}: n={len(rows)}, accuracy {acc.min():.3f}-{acc.max():.3f}; "
              f"top half n={len(top)} from {top[0]['accuracy']:.3f}")
        print(f"  {'metric':20s} {'rho all':>8s} {'rho top':>8s}   values (sorted by accuracy)")
        for m in METRICS:
            if not all(m in r and r[m] is not None for r in rows):
                continue
            v = np.array([r[m] for r in rows], dtype=float)
            rho = spearmanr(v, acc).correlation
            vt = v[len(rows) // 2:]
            rho_t = spearmanr(vt, acc[len(rows) // 2:]).correlation if np.ptp(vt) > 0 else np.nan
            print(f"  {m:20s} {rho:8.2f} {rho_t:8.2f}   " + ' '.join(f'{x:.3f}' for x in v))
        print(f"  {'accuracy':20s} {'':8s} {'':8s}   " + ' '.join(f'{x:.3f}' for x in acc))


if __name__ == '__main__':
    main(sys.argv[1])
