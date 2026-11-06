"""Temporal chain voting diagnostic, on the cached consensus of diag_merge.py (no propagation rerun).

Chains are connected components of adjacent-volume mutual-NN links in the embedding (lp_core.temporal_chains).
Reports chain lengths and GT purity (how often a chain stays on one GT neuron), then scores
lp_core.finalize with chain_w over a grid, with and without temporal_smooth, against the plain finalize.

    python diag_chain.py --case round_bayes_wk3_zimmer_1210:36:zimmer_1210 --seed 0 --work $WORK

Output: $WORK/diag_chain_<tag>_<variant>_s<seed>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core  # noqa: E402
import sweep_labelprop as sw  # noqa: E402
from variants import BATCHES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', default='base')
    ap.add_argument('--work', required=True)
    ap.add_argument('--chain_ws', nargs='+', type=float, default=[0.25, 0.5, 1, 2, 5, 20])
    ap.add_argument('--tsmooths', nargs='+', type=float, default=[0.0, 0.5])
    args = ap.parse_args()
    from barlow_track.utils.utils_tracking import WormClusterTracker
    pool = {}
    for b in BATCHES.values():
        pool.update(b)
    vbase = lp_core.full_variant(pool[args.variant])
    rnd, trial, label = args.case.split(':')[:3]
    tag, weights, emb, cfg, ekw = sw.case_settings(rnd, int(trial), label)
    X, t2l, l2t, n_frames = sw.load_embedding(emb, args.work, tag)
    scorer = sw.Scorer(ekw, cfg, X.shape, t2l, l2t, n_frames, args.work, tag)
    Xs = lp_core.embed(X, vbase, args.seed)
    tracker = WormClusterTracker(Xs, t2l, linear_ind_to_t_and_seg_id=l2t, seed=args.seed, verbose=0)
    d = np.load(os.path.join(args.work, f'merge_cache_{tag}_{args.variant}_s{args.seed}.npz'))
    cons = d['cons'].astype(np.float64)
    R = float(vbase['num_seeds'])
    syn = dict(cons=cons, R=R)

    t0 = time.time()
    pairs = lp_core.adjacent_mnn_pairs(Xs, t2l)
    chains = lp_core.temporal_chains(Xs, t2l, pairs=pairs)
    g = scorer.gt_obj
    I, J = pairs
    both = (g[I] >= 0) & (g[J] >= 0)
    link_acc = float(np.mean(g[I][both] == g[J][both]))
    size = np.bincount(chains)
    obj_len = size[chains]
    # chain purity over GT objects: share of a chain's GT objects on its majority neuron
    gi = np.flatnonzero(g >= 0)
    pur_num = pur_den = 0
    order = np.argsort(chains[gi], kind='stable')
    cs, gs = chains[gi][order], g[gi][order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    for seg in np.split(np.arange(len(cs)), bounds):
        if len(seg) == 0:
            continue
        pur_num += np.bincount(gs[seg]).max()
        pur_den += len(seg)
    stats = dict(n_links=int(len(I)), link_acc_gt=link_acc, n_chains=int(len(size)),
                 obj_chain_len_q=[float(x) for x in np.quantile(obj_len, [0.1, 0.5, 0.9])],
                 frac_obj_in_chain_ge100=float(np.mean(obj_len >= 100)), gt_purity=float(pur_num / pur_den),
                 chain_s=round(time.time() - t0, 1))
    print(f"[{tag}] chains: {stats}", flush=True)

    rows = []
    ref = None
    for ts in args.tsmooths:
        for cw in [0.0] + args.chain_ws:
            v = dict(vbase, temporal_smooth=ts, chain_w=0.0)
            if cw > 0:
                # same as finalize with chain_w, reusing the chains
                c = cons
                if ts > 0:
                    sm = c.copy()
                    np.add.at(sm, I, ts * c[J])
                    np.add.at(sm, J, ts * c[I])
                    c = sm
                c = lp_core.chain_vote(c, Xs, t2l, cw, chains=chains)
                from barlow_track.utils import utils_spectral_relabeling as ss
                lab, pr = ss.enforce_temporal_uniqueness_hungarian(c, t2l, R=R, min_prob_threshold=v['min_prob_threshold'])
                lab, pr = lab[:, 0], pr[:, 0]
            else:
                lab, pr = lp_core.finalize(syn, t2l, v, Xs=Xs)
            rec = scorer.score(tracker, lab, pr)
            rec.update(scorer.fast(lab))
            if ref is None:
                ref = rec
            row = dict(temporal_smooth=ts, chain_w=cw, d_acc=rec['accuracy'] - ref['accuracy'],
                       **{k: rec[k] for k in ('accuracy', 'acc_fast', 'acc_merge_bound', 'err_unlabelled',
                                              'err_swapped', 'err_fragment', 'err_frag_distractor')})
            rows.append(row)
            print(f"  tsmooth {ts:.2f} chain_w {cw:5.2f}: acc {rec['accuracy']:.4f} ({row['d_acc'] * 100:+.2f} pt) "
                  f"unl {rec['err_unlabelled'] * 100:.2f} swap {rec['err_swapped'] * 100:.2f} "
                  f"frag {rec['err_fragment'] * 100:.2f}", flush=True)
    out = os.path.join(args.work, f'diag_chain_{tag}_{args.variant}_s{args.seed}.json')
    with open(out, 'w') as f:
        json.dump(dict(case=tag, variant=args.variant, seed=args.seed, chains=stats, grid=rows), f)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
