"""Absolute, GT-free accuracy estimate from the agreement between independently trained networks.

agree_mean (utils_consistency) compares label-propagation runs on ONE embedding, so it cannot see an error
every run makes alike and its level differs between recordings. Here the tracks of several networks trained
on the same recording (other seeds, other recipes, other input modalities) are compared object by object:

  a_ij = fraction of objects on which networks i and j agree, after matching their track labels one to one
         (Hungarian on the contingency table); an object tracked by only one of the two counts as a
         disagreement.

If the networks' errors are independent and two wrong labels rarely coincide, a_ij = p_i * p_j with p_i the
accuracy of network i, so with three or more networks the p_i follow from the pairwise agreements alone
(rank-one fit of log a_ij = log p_i + log p_j; for three networks p_i = sqrt(a_ij a_ik / a_jk)). Errors the
networks share (segmentation faults, a neuron pair no network separates) are invisible here too, so p_i is an
upper bound; how tight it is has to be read off recordings with ground truth, which is what --results is for.

Input: tracks saved by eval_accuracy --save_tracks (rescore_round.py), key pred_all: rows = volumes,
columns = predicted tracks, values = the object's id in that volume (NaN = none).

    python barlow_track/scripts/scratch/gt_free_proxy/cross_network_proxy.py \
        --tracks_dir <round>/tracks_v2 --results <round>/exp_results_v2.jsonl --out <round>/cross_network_proxy_v2.jsonl

Several --tracks_dir / --results (globs, quoted) pool networks of several rounds, e.g. the sensitivity rounds'
image-only, position-only and fused networks of one recording: --tracks_dir '<B>/round_sens_*/tracks_v2'.

Networks are grouped by recording (the record's lab, else the tag suffix after _trial<N>[r<k>]_). Two
populations: 'all' objects, and 'long' = objects that either network puts on a track covering at least
--min_cover of the volumes (the GT-free stand-in for "a real neuron": the accuracy is scored on GT neurons only).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def object_labels(pred_all, min_cover=0.5):
    """(keys, labels, is_long) of a pred_all table: one row per tracked object. keys = t * K + object id."""
    v = np.asarray(pred_all, dtype=float)
    n_t = v.shape[0]
    t_idx, col = np.nonzero(np.isfinite(v))
    ids = v[t_idx, col].astype(np.int64)
    keys = t_idx.astype(np.int64) * (1 << 32) + ids
    keys, first = np.unique(keys, return_index=True)  # an object on two tracks keeps its first
    col = col[first]
    cover = np.bincount(col, minlength=v.shape[1]) / max(n_t, 1)
    return keys, col, cover[col] >= min_cover


def pair_agreement(a, b):
    """Agreement of two networks' object labels (object_labels outputs) for the 'all' and 'long' populations."""
    ka, la, longa = a
    kb, lb, longb = b
    common, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    from scipy.optimize import linear_sum_assignment
    na, nb = int(la.max()) + 1 if len(la) else 1, int(lb.max()) + 1 if len(lb) else 1
    cont = np.zeros((na, nb))
    np.add.at(cont, (la[ia], lb[ib]), 1)
    r, c = linear_sum_assignment(cont, maximize=True)
    match = np.full(na, -1)
    match[r] = c
    same = match[la[ia]] == lb[ib]
    long_common = longa[ia] | longb[ib]
    n_union = len(ka) + len(kb) - len(common)
    # long objects seen by one network only: disagreements of the long population
    only_a = np.ones(len(ka), bool)
    only_a[ia] = False
    only_b = np.ones(len(kb), bool)
    only_b[ib] = False
    n_long = int(long_common.sum() + longa[only_a].sum() + longb[only_b].sum())
    return dict(all=float(same.sum() / n_union) if n_union else np.nan,
                long=float((same & long_common).sum() / n_long) if n_long else np.nan)


def rank_one_accuracy(agree):
    """p (n,) with agree[i, j] ~ p_i p_j off the diagonal (least squares in the log), clipped to [0, 1]."""
    a = np.asarray(agree, float)
    n = a.shape[0]
    if n < 3:
        return np.full(n, np.nan)
    rows, rhs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(a[i, j]) and a[i, j] > 0:
                e = np.zeros(n)
                e[i] = e[j] = 1
                rows.append(e)
                rhs.append(np.log(a[i, j]))
    if len(rows) < n:
        return np.full(n, np.nan)
    u = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return np.clip(np.exp(u), 0, 1)


def recording_of(tag, rec=None):
    if rec and rec.get('lab'):
        return rec['lab']
    m = re.search(r'_trial\d+(?:r\d+)?_(.+)$', tag)
    return m.group(1) if m else 'unknown'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracks_dir', required=True, nargs='+', help='one or more directories (globs allowed)')
    ap.add_argument('--results', default=None, nargs='+', help='jsonl(s) with the accuracy of each tag (optional)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--min_cover', type=float, default=0.5)
    ap.add_argument('--labs', nargs='+', default=None)
    args = ap.parse_args()
    import pandas as pd

    recs = {}
    for pattern in args.results or []:
        for path in sorted(glob.glob(pattern)):
            for line in open(path):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('tag'):
                    recs[r['tag']] = r
    groups = defaultdict(list)
    for f in sorted(f for pattern in args.tracks_dir for f in glob.glob(os.path.join(pattern, '*.h5'))):
        tag = os.path.splitext(os.path.basename(f))[0]
        groups[recording_of(tag, recs.get(tag))].append((tag, f))
    rows = []
    for lab, items in sorted(groups.items()):
        if args.labs and lab not in args.labs:
            continue
        if len(items) < 2:
            print(f"{lab}: {len(items)} network, skipped")
            continue
        objs = [object_labels(pd.read_hdf(f, 'pred_all').values, args.min_cover) for _, f in items]
        n = len(items)
        agree = {k: np.full((n, n), np.nan) for k in ('all', 'long')}
        for i in range(n):
            for j in range(i + 1, n):
                p = pair_agreement(objs[i], objs[j])
                for k in agree:
                    agree[k][i, j] = agree[k][j, i] = p[k]
        est = {k: rank_one_accuracy(agree[k]) for k in agree}
        print(f"\n{lab}: {n} networks")
        print(f"{'tag':60s} accuracy  est_long  est_all  mean_agree_long")
        for i, (tag, _) in enumerate(items):
            acc = (recs.get(tag) or {}).get('accuracy')
            row = dict(lab=lab, tag=tag, accuracy=acc, n_networks=n, min_cover=args.min_cover,
                       est_long=float(est['long'][i]), est_all=float(est['all'][i]),
                       mean_agree_long=float(np.nanmean(agree['long'][i])),
                       mean_agree_all=float(np.nanmean(agree['all'][i])))
            rows.append(row)
            print(f"{tag[-60:]:60s} {acc if acc is None else round(acc, 4)!s:8s}  {row['est_long']:.4f}    "
                  f"{row['est_all']:.4f}   {row['mean_agree_long']:.4f}")
    if args.out:
        with open(args.out, 'w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
    have = [r for r in rows if r['accuracy'] is not None and np.isfinite(r['est_long'])]
    if have:
        d = np.array([r['est_long'] - r['accuracy'] for r in have])
        print(f"\nest_long - accuracy over {len(have)} networks: mean {100 * d.mean():+.2f} pt, "
              f"s.d. {100 * d.std():.2f}, range {100 * d.min():+.2f} to {100 * d.max():+.2f}")
        for lab in sorted({r['lab'] for r in have}):
            dl = np.array([r['est_long'] - r['accuracy'] for r in have if r['lab'] == lab])
            print(f"  {lab:26s} n={len(dl):3d}  mean {100 * dl.mean():+.2f} pt  s.d. {100 * dl.std():.2f}")


if __name__ == '__main__':
    main()
