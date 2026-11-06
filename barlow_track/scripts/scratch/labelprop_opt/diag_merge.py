"""Fragment merging diagnostic: can final labels that are pieces of one track be told apart from distinct
neurons, and how much accuracy does lp_core.merge_fragments recover?

Runs one variant (default the tracker, 'base') on one case and RNG seed, caches its final labels, confidences
and consensus in --work, then
  1. for every candidate label pair (nearest centroids), the merge features (centroid ratio, volume overlap,
     consensus affinity) and whether the pair is a true fragment pair (same majority GT neuron), with AUCs;
  2. a grid over merge_ratio x merge_overlap x merge_aff: number of merges, official accuracy, acc_fast and
     the error decomposition, against the unmerged labels.

    python diag_merge.py --case round_bayes_wk3_zimmer_1210:36:zimmer_1210 --seed 0 --work $WORK --device cpu

Output: $WORK/diag_merge_<tag>_<variant>_s<seed>.json (and the cache merge_cache_<tag>_<variant>_s<seed>.npz).
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core  # noqa: E402
import sweep_labelprop as sw  # noqa: E402
from variants import BATCHES  # noqa: E402


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), by ranks."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', default='base', help='variant name from variants.py (any batch)')
    ap.add_argument('--work', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ratios', nargs='+', type=float, default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument('--overlaps', nargs='+', type=float, default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument('--affs', nargs='+', type=float, default=[0.0, 0.02, 0.05, 0.1])
    args = ap.parse_args()
    import torch
    from barlow_track.utils.utils_tracking import WormClusterTracker
    pool = {}
    for b in BATCHES.values():
        pool.update(b)
    vbase = lp_core.full_variant(pool[args.variant])
    assert vbase['merge_ratio'] == 0, 'diagnose an unmerged variant'
    rnd, trial, label = args.case.split(':')[:3]
    tag, weights, emb, cfg, ekw = sw.case_settings(rnd, int(trial), label)
    X, t2l, l2t, n_frames = sw.load_embedding(emb, args.work, tag)
    scorer = sw.Scorer(ekw, cfg, X.shape, t2l, l2t, n_frames, args.work, tag)
    seed = args.seed
    Xs = lp_core.embed(X, vbase, seed)
    tracker = WormClusterTracker(Xs, t2l, linear_ind_to_t_and_seg_id=l2t, seed=seed, verbose=0)
    cache = os.path.join(args.work, f'merge_cache_{tag}_{args.variant}_s{seed}.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        labels0, conf0, cons = d['labels'], d['conf'], d['cons']
        print(f"[{tag}] cached labels from {cache}", flush=True)
    else:
        t0 = time.time()
        device = None if args.device == 'cpu' else torch.device(args.device)
        graph = lp_core.build_graph(Xs, t2l, vbase, seed, device=device, gt_obj=scorer.gt_obj)
        seed_times = lp_core.select_seeds(t2l, vbase, seed, tracker.num_frames)
        with torch.no_grad():
            labs, probs = lp_core.propagate(graph, t2l, Xs.shape[0], seed_times, vbase, device=device)
        syn = lp_core.sync(labs, probs, t2l, vbase)
        labels0, conf0 = lp_core.finalize(syn, t2l, vbase, Xs=Xs, graph=graph)
        cons = np.asarray(syn['cons'].toarray() if hasattr(syn['cons'], 'toarray') else syn['cons'],
                          dtype=np.float32)
        np.savez(cache, labels=labels0, conf=conf0, cons=cons)
        print(f"[{tag}] {args.variant} s{seed} tracked in {time.time() - t0:.0f}s", flush=True)

    # 1) pair features vs truth
    g = scorer.gt_obj
    st = lp_core.label_stats(labels0, Xs, t2l)
    v_all = dict(vbase, merge_knn=10)
    pairs = lp_core.merge_candidates(st, v_all, cons=cons, labels=labels0)
    li, on = st['li'], st['on']
    go = g[on]
    L = len(st['ul'])
    n_gt = np.bincount(li[go >= 0], minlength=L)
    maj = np.full(L, -1)
    for a in range(L):
        sel = go[li == a]
        sel = sel[sel >= 0]
        if len(sel):
            maj[a] = np.bincount(sel).argmax()
    gt_share = n_gt / st['cnt']
    truth = {k: bool(maj[k[0]] >= 0 and maj[k[0]] == maj[k[1]]) for k in pairs}
    feats = {}
    for f in ('ratio', 'overlap', 'aff'):
        pos = [p[f] for k, p in pairs.items() if truth[k]]
        neg = [p[f] for k, p in pairs.items() if not truth[k]]
        sgn = -1 if f in ('ratio', 'overlap') else 1  # higher = more merge-like
        feats[f] = dict(auc=auc(sgn * np.array(pos), sgn * np.array(neg)),
                        pos_q=[float(x) for x in np.quantile(pos, [0.1, 0.5, 0.9])] if pos else None,
                        neg_q=[float(x) for x in np.quantile(neg, [0.1, 0.5, 0.9])] if neg else None)
    n_true = int(sum(truth.values()))
    print(f"[{tag}] {L} labels, {len(pairs)} candidate pairs, {n_true} true fragment pairs", flush=True)
    for f, r in feats.items():
        print(f"  {f:8s} AUC {r['auc']:.3f}  true q10/50/90 {r['pos_q']}  other {r['neg_q']}", flush=True)
    true_pairs = sorted(((p['ratio'], p['overlap'], p.get('aff', 0.0), int(st['cnt'][a]), int(st['cnt'][b]),
                          float(gt_share[a]), float(gt_share[b])) for (a, b), p in pairs.items()
                         if truth[(a, b)]))

    # 2) grid
    rows = []
    base_rec = scorer.score(tracker, labels0, conf0)
    base_rec.update(scorer.fast(labels0))
    print(f"  unmerged: acc {base_rec['accuracy']:.4f} fast {base_rec['acc_fast']:.4f} "
          f"bound {base_rec['acc_merge_bound']:.4f}", flush=True)
    for r, o, a in itertools.product(args.ratios, args.overlaps, args.affs):
        v = dict(vbase, merge_ratio=r, merge_overlap=o, merge_aff=a)
        lab, cf, merges = lp_core.merge_fragments(labels0, conf0, Xs, t2l, v, cons=cons)
        good = 0
        for kept, merged, _ in merges:
            ia, ib = np.searchsorted(st['ul'], kept), np.searchsorted(st['ul'], merged)
            good += int(maj[ia] >= 0 and maj[ia] == maj[ib])
        rec = scorer.score(tracker, lab, cf) if merges else dict(base_rec)
        rec.update(scorer.fast(lab))
        row = dict(merge_ratio=r, merge_overlap=o, merge_aff=a, n_merges=len(merges), n_good=good,
                   d_acc=rec['accuracy'] - base_rec['accuracy'], d_fast=rec['acc_fast'] - base_rec['acc_fast'],
                   **{k: rec[k] for k in ('accuracy', 'acc_fast', 'err_unlabelled', 'err_swapped', 'err_fragment',
                                          'err_frag_split', 'err_frag_distractor')})
        rows.append(row)
        print(f"  ratio {r:4.2f} ov {o:4.2f} aff {a:4.2f}: {len(merges):4d} merges ({good} same-neuron) "
              f"acc {rec['accuracy']:.4f} ({row['d_acc'] * 100:+.2f} pt) fast {row['d_fast'] * 100:+.2f} pt",
              flush=True)
    out = os.path.join(args.work, f'diag_merge_{tag}_{args.variant}_s{seed}.json')
    with open(out, 'w') as f:
        json.dump(dict(case=tag, variant=args.variant, seed=seed, n_labels=L, n_pairs=len(pairs),
                       n_true_pairs=n_true, features=feats, true_pairs=true_pairs[:200], unmerged=base_rec,
                       grid=rows), f)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
