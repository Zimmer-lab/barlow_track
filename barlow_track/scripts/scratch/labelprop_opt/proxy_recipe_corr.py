"""Does the GT-free consistency proxy still rank networks by accuracy under a new label propagation recipe?

Reads the harness records of launch_proxy.sh (batch2_proxy.jsonl: variants base and a new recipe, seed 0, on
16 trials per round at evenly spaced accuracy ranks). Per recording and recipe, Spearman rho between
consistency[metric] and accuracy over all trials and over the better half (by that recipe's accuracy), and
the pick: accuracy lost by taking the trial with the highest metric instead of the most accurate one.
Cross rows: the old proxy (base agreement) against the new recipe's accuracy, i.e. whether a sweep that keeps
scoring agreement with the old tracker still selects the networks that are best under the new one.

    python proxy_recipe_corr.py $WORK/batch2_proxy.jsonl --new c_all_gh_s150_plain
"""
import argparse
import collections
import json

import numpy as np
from scipy.stats import spearmanr


def rho(x, y):
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(spearmanr(x, y).correlation)


def stats(prox, acc):
    """rho all, rho better half (by acc), pick loss in pt."""
    prox, acc = np.asarray(prox), np.asarray(acc)
    top = acc >= np.median(acc)
    return rho(prox, acc), rho(prox[top], acc[top]), 100 * (acc.max() - acc[np.argmax(prox)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl')
    ap.add_argument('--base', default='base')
    ap.add_argument('--new', default='c_all_gh_s150_plain')
    ap.add_argument('--metric', default='agree_mean')
    ap.add_argument('--out', default=None, help='write the per-round table as JSON')
    args = ap.parse_args()
    rec = collections.defaultdict(dict)  # (label, case) -> variant -> record (latest wins)
    for line in open(args.jsonl):
        r = json.loads(line)
        if r.get('seed') == 0 and r['variant'] in (args.base, args.new):
            rec[(r['label'], r['case'])][r['variant']] = r
    by_label = collections.defaultdict(list)
    for (label, case), v in rec.items():
        if args.base in v and args.new in v:
            by_label[label].append(v)
    hdr = (f"{'recording':24s} {'n':>3s}  {'acc base':>15s} {'acc new':>15s}  "
           f"{'base: rho all/half pick':>24s}  {'new: rho all/half pick':>24s}  {'old proxy->new acc':>20s}  "
           f"{'acc base~new':>12s}")
    print(f'metric {args.metric}; rho all / better half / pick loss (pt)')
    print(hdr)
    table = {}
    for label in sorted(by_label):
        vs = by_label[label]
        ab = [v[args.base]['accuracy'] for v in vs]
        an = [v[args.new]['accuracy'] for v in vs]
        pb = [v[args.base]['consistency'][args.metric] for v in vs]
        pn = [v[args.new]['consistency'][args.metric] for v in vs]
        sb, sn, sx = stats(pb, ab), stats(pn, an), stats(pb, an)
        rab = rho(ab, an)
        table[label] = dict(n=len(vs), acc_base=[min(ab), max(ab)], acc_new=[min(an), max(an)],
                            base=sb, new=sn, old_proxy_new_acc=sx, rho_acc_base_new=rab,
                            mean_gain_pt=100 * float(np.mean(np.subtract(an, ab))))
        f = lambda s: f'{s[0]:+.2f}/{s[1]:+.2f} {s[2]:5.2f}'
        print(f'{label:24s} {len(vs):3d}  {min(ab):.3f}-{max(ab):.3f}    {min(an):.3f}-{max(an):.3f}    '
              f'{f(sb):>24s}  {f(sn):>24s}  {f(sx):>20s}  {rab:+12.2f}')
    if args.out:
        json.dump(table, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
