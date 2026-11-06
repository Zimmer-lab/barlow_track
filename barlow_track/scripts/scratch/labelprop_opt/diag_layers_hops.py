"""Per-run label accuracy against propagation depth, hop distance and time distance from the seed volume.

For a few seed volumes of one case (base graph and seeds, as the tracker), propagates one seed volume's
labels and checks every GT object's top-1 label at several layer counts: a label is correct if the
object is the same GT neuron as the seed object that owns the label (a distractor seed's label is wrong
on every GT object). This is the per-run picture BEFORE the sync, and it answers:

* does per-run accuracy fall as layers increase past some depth (long walks bleeding into other
  neurons' clusters), or does fewer layers only help by leaving far objects unlabelled (abstaining)?
* how accurate are labels on objects far from the seeds, in hops (the first layer at which an object
  receives any mass is its hop distance) and in volumes |t - t_seed|?

Modes: hard clamping (the tracker) and label spreading at --alphas. Labelled = the tracker's rule
(row-normalised top-1 probability >= 1.1 / C, C = objects in the seed volume).

    python diag_layers_hops.py --work $WORK --case round_bayes_wk2_samuel_137:50:samuel_137 --runs 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_labelprop as sw  # noqa: E402  (puts lp_core and the repo on sys.path)
import lp_core  # noqa: E402

LAYERS = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300)
HOP_BINS = (0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 10 ** 9)
DT_BINS = (0, 1, 10, 50, 200, 500, 10 ** 9)


def binned(values, keys, edges):
    """Mean of values per [edges[i], edges[i+1]) bin of keys, and counts."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (keys >= a) & (keys < b)
        out.append(dict(lo=int(a), hi=int(b) if b < 10 ** 9 else None, n=int(m.sum()),
                        mean=float(values[m].mean()) if m.any() else None))
    return out


def run_one(adj, seed_objs, gt_obj, lin_t, t_seed, alpha, torch):
    n = adj.shape[0]
    C = len(seed_objs)
    si = torch.from_numpy(seed_objs)
    Y = torch.zeros(n, C)
    Y[si, torch.arange(C)] = 1.0
    H = Y.clone()
    owner = gt_obj[seed_objs]                    # GT neuron of each label (-1: distractor seed)
    isgt = gt_obj >= 0
    notseed = np.ones(n, bool)
    notseed[seed_objs] = False
    ev = isgt & notseed                           # GT objects outside the seed volume
    present = np.isin(gt_obj, owner[owner >= 0])  # their neuron is in the seed volume (can be right)
    evp, eva = ev & present, ev & ~present
    first = np.full(n, -1, np.int64)              # hop distance: first layer with any mass
    first[seed_objs] = 0
    res, at = {}, {}
    for L in range(1, max(LAYERS) + 1):
        H = adj @ H
        if alpha is None:
            H[si] = Y[si]
        else:
            H = alpha * H + (1.0 - alpha) * Y
        rs = H.sum(1)
        newly = (first < 0) & (rs > 0).numpy()
        first[newly] = L
        if L in LAYERS:
            P = H / rs.clamp_min(1e-30)[:, None]
            p, lab = P.max(1)
            lab, p = lab.numpy(), p.numpy()
            labelled = (rs.numpy() > 0) & (p >= 1.1 / C)
            correct = labelled & isgt & (owner[lab] == gt_obj)
            res[L] = dict(coverage=float(labelled[ev].mean()), acc_labelled=float(correct[ev].sum() / max(labelled[ev].sum(), 1)),
                          acc_all=float(correct[ev].mean()),
                          frac_present=float(present[ev].mean()),
                          acc_present=float(correct[evp].mean()) if evp.any() else None,
                          cov_present=float(labelled[evp].mean()) if evp.any() else None,
                          cov_absent=float(labelled[eva].mean()) if eva.any() else None,
                          distractor_labelled=float(labelled[~isgt & notseed].mean()) if (~isgt & notseed).any() else None)
            at[L] = (labelled, correct)
    hops = np.where(first < 0, 10 ** 8, first)
    dt = np.abs(lin_t - t_seed)
    by = {}
    for L in (30, 100, 300):
        labelled, correct = at[L]
        by[L] = dict(acc_by_hop=binned(correct[ev].astype(float), hops[ev], HOP_BINS),
                     present_by_hop=binned(present[ev].astype(float), hops[ev], HOP_BINS),
                     accp_by_hop=binned(correct[evp].astype(float), hops[evp], HOP_BINS),
                     cov_by_hop=binned(labelled[ev].astype(float), hops[ev], HOP_BINS),
                     acc_by_dt=binned(correct[ev].astype(float), dt[ev], DT_BINS))
    return dict(by_layers=res, by_bins=by, hop_quantiles={q: float(np.quantile(hops[ev], q)) for q in (0.5, 0.9, 0.99, 1.0)},
                n_unreached=int((first[ev] < 0).sum()), C=C, n_distractor_seeds=int((owner < 0).sum()))


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--runs', type=int, default=5, help='seed volumes (the first of the tracker\'s seed list)')
    ap.add_argument('--alphas', type=float, nargs='*', default=[0.9, 0.97])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rnd, trial, label = args.case.split(':')[:3]
    tag, weights, emb, cfg, ekw = sw.case_settings(rnd, int(trial), label)
    X, t2l, l2t, n_frames = sw.load_embedding(emb, args.work, tag, None)
    scorer = sw.Scorer(ekw, cfg, X.shape, t2l, l2t, n_frames, args.work, tag)
    gt_obj = scorer.gt_obj
    lin_t = np.array([l2t[i][0] for i in range(X.shape[0])])
    v = lp_core.full_variant({})
    Xs = lp_core.embed(X, v, args.seed)
    t0 = time.time()
    graph = lp_core.build_graph(Xs, t2l, v, args.seed, device=None, gt_obj=gt_obj)
    print(f"[{tag}] graph {time.time() - t0:.0f}s", flush=True)
    from barlow_track.utils.utils_tracking import WormClusterTracker
    tracker = WormClusterTracker(Xs, t2l, linear_ind_to_t_and_seg_id=l2t, seed=args.seed, verbose=0)
    seed_times = lp_core.select_seeds(t2l, v, args.seed, tracker.num_frames)[:args.runs]
    out = dict(case=tag, seed=args.seed, seed_times=[int(t) for t in seed_times], runs=[])
    with torch.no_grad():
        for t in seed_times:
            objs = np.asarray(list(t2l[t]), dtype=np.int64)
            run = dict(t_seed=int(t))
            for a in [None] + list(args.alphas):
                tt = time.time()
                run['clamp' if a is None else f'spread{a:g}'] = run_one(graph['adj'], objs, gt_obj, lin_t, t, a, torch)
                print(f"[{tag}] t_seed {t} {'clamp' if a is None else a}: {time.time() - tt:.0f}s", flush=True)
            out['runs'].append(run)
    path = args.out or os.path.join(args.work, f'diag_layers_hops_{tag}_s{args.seed}.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    # summary: mean over runs
    print(f"\n[{tag}] per-run accuracy of GT objects outside the seed volume (mean over {len(seed_times)} runs)")
    for mode in ['clamp'] + [f'spread{a:g}' for a in args.alphas]:
        print(f"  {mode}")
        for L in LAYERS:
            r = [x[mode]['by_layers'][L] for x in out['runs']]
            mn = lambda k: np.mean([y[k] for y in r if y[k] is not None])  # noqa: E731
            print(f"    L={L:4d} cov {mn('coverage'):.3f}  acc_all {mn('acc_all'):.3f}  | neuron in seed vol "
                  f"({mn('frac_present'):.2f}): acc {mn('acc_present'):.3f} cov {mn('cov_present'):.3f}"
                  f"  | absent: cov {mn('cov_absent'):.3f}")
    print("  clamp, hop quantiles (50/90/99/max):",
          [np.mean([x['clamp']['hop_quantiles'][q] for x in out['runs']]) for q in (0.5, 0.9, 0.99, 1.0)])
    for L in (30, 100):
        print(f"  clamp L={L}, by hop distance: acc (all GT objects), share whose neuron is in the seed volume, "
              f"acc on those:")
        bb = lambda key, i: [x['clamp']['by_bins'][L][key][i]['mean'] for x in out['runs']  # noqa: E731
                             if x['clamp']['by_bins'][L][key][i]['mean'] is not None]
        for i, b in enumerate(out['runs'][0]['clamp']['by_bins'][L]['acc_by_hop']):
            ns = [x['clamp']['by_bins'][L]['acc_by_hop'][i]['n'] for x in out['runs']]
            if bb('acc_by_hop', i):
                ap = bb('accp_by_hop', i)
                print(f"    hops [{b['lo']}, {b['hi']}): n {int(np.mean(ns)):7d}  acc {np.mean(bb('acc_by_hop', i)):.3f}"
                      f"  present {np.mean(bb('present_by_hop', i)):.2f}  acc|present {np.mean(ap) if ap else float('nan'):.3f}")
        print(f"  clamp L={L}, accuracy by |t - t_seed|:")
        for i, b in enumerate(out['runs'][0]['clamp']['by_bins'][L]['acc_by_dt']):
            ms = [x['clamp']['by_bins'][L]['acc_by_dt'][i]['mean'] for x in out['runs']]
            ms = [m for m in ms if m is not None]
            if ms:
                print(f"    dt [{b['lo']}, {b['hi']}): acc {np.mean(ms):.3f}")
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
