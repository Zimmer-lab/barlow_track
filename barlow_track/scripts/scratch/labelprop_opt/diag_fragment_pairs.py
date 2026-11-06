"""What are the "distractor" partners of fragmented neurons? For project-source cases (segmentation metadata
with centroids and mask volumes), from the cached base labels of diag_merge.py:

for every GT neuron whose objects are split over several final labels, take its main label (the one holding
most of its objects) and each partner label (holding at least --min_objs of its objects). In the volumes
where both labels have an object, measure the physical distance between the two objects and their mask
volumes, and whether the partner object is a GT object (of which neuron). Reference: the distance from
each GT neuron to the nearest other GT neuron in the same volume.

Short partner distances (below a nucleus diameter) mean over-segmentation: one nucleus in two masks, with
the GT click in one or the other. Distances like the nearest-neuron distance mean a neighbouring nucleus
(an unannotated neuron when the partner object is not GT).

    python diag_fragment_pairs.py --case round_bayes_wk3_zimmer_1210:36:zimmer_1210 --seed 0 --work $WORK \
        --um_zxy 1.5 0.325 0.325

Output: $WORK/diag_fragment_pairs_<tag>_s<seed>.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_labelprop as sw  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', default='base')
    ap.add_argument('--work', required=True)
    ap.add_argument('--um_zxy', nargs=3, type=float, required=True, help='voxel size z x y in um')
    ap.add_argument('--min_objs', type=int, default=20, help="partner label: at least this many of the neuron's objects")
    args = ap.parse_args()
    rnd, trial, label = args.case.split(':')[:3]
    tag, weights, emb, cfg, ekw = sw.case_settings(rnd, int(trial), label)
    X, t2l, l2t, n_frames = sw.load_embedding(emb, args.work, tag)
    scorer = sw.Scorer(ekw, cfg, X.shape, t2l, l2t, n_frames, args.work, tag)
    assert scorer.seg_meta is not None, 'project-source cases only (needs segmentation metadata)'
    d = np.load(os.path.join(args.work, f'merge_cache_{tag}_{args.variant}_s{args.seed}.npz'))
    labels = d['labels']
    g = scorer.gt_obj
    n = len(labels)
    t_of = np.array([l2t[i][0] for i in range(n)])
    seg_of = np.array([l2t[i][2] for i in range(n)])
    um = np.array(args.um_zxy)

    meta = scorer.seg_meta.segmentation_metadata
    cent, vol = {}, {}

    def lookup(t, s):
        if t not in cent:
            df = meta[t]
            cent[t] = {int(k): np.array(c, float) * um for k, c in zip(df.index, df['centroids'])}
            vol[t] = {int(k): float(v) for k, v in zip(df.index, df['neuron_volume'])}
        return cent[t].get(int(s)), vol[t].get(int(s))

    # label per (t) for fast pairing
    by_lab_t = {}
    for i in np.flatnonzero(labels >= 1):
        by_lab_t[(int(labels[i]), int(t_of[i]))] = i
    rows = []
    n_gt = g.max() + 1
    for k in range(n_gt):
        objs = np.flatnonzero((g == k) & (labels >= 1))
        if len(objs) == 0:
            continue
        labs, cnt = np.unique(labels[objs], return_counts=True)
        main = labs[cnt.argmax()]
        for pl, pc in zip(labs, cnt):
            if pl == main or pc < args.min_objs:
                continue
            dist, v_main, v_part, partner_gt = [], [], [], []
            # volumes where the neuron sits in the partner label and the main label holds another object
            for i in objs[labels[objs] == pl]:
                t = int(t_of[i])
                j = by_lab_t.get((int(main), t))
                if j is None:
                    continue
                ci, vi = lookup(t, seg_of[i])
                cj, vj = lookup(t, seg_of[j])
                if ci is None or cj is None:
                    continue
                dist.append(float(np.linalg.norm(ci - cj)))
                v_main.append(vj)
                v_part.append(vi)
                partner_gt.append(int(g[j]))
            if not dist:
                continue
            pg = np.array(partner_gt)
            rows.append(dict(neuron=int(k), main=int(main), partner=int(pl), n_neuron_in_partner=int(pc),
                             n_both=len(dist), dist_med=float(np.median(dist)),
                             dist_q=[float(x) for x in np.quantile(dist, [0.1, 0.9])],
                             vol_neuron_med=float(np.median(v_part)), vol_other_med=float(np.median(v_main)),
                             other_is_gt_frac=float(np.mean(pg >= 0)),
                             other_gt_neurons=[int(x) for x in np.unique(pg[pg >= 0])]))
    # reference: nearest other GT neuron in the same volume, and median GT mask volume
    nn, gv = [], []
    for t in range(0, n_frames, max(1, n_frames // 100)):
        idx = [i for i in t2l.get(t, []) if g[i] >= 0]
        P = []
        for i in idx:
            c, v = lookup(t, seg_of[i])
            if c is not None:
                P.append(c)
                gv.append(v)
        if len(P) > 1:
            P = np.array(P)
            D = np.linalg.norm(P[:, None] - P[None], axis=-1)
            np.fill_diagonal(D, np.inf)
            nn.extend(D.min(1).tolist())
    ref = dict(nn_gt_dist_q=[float(x) for x in np.quantile(nn, [0.1, 0.5, 0.9])], gt_vol_med=float(np.median(gv)))
    w = np.array([r['n_both'] for r in rows])
    summ = dict(n_pairs=len(rows), objects=int(w.sum()) if len(w) else 0,
                dist_med_weighted=float(np.average([r['dist_med'] for r in rows], weights=w)) if len(w) else None,
                frac_pairs_closer_than_nn_q10=float(np.mean([r['dist_med'] < ref['nn_gt_dist_q'][0] for r in rows]))
                if rows else None,
                other_is_gt_frac_weighted=float(np.average([r['other_is_gt_frac'] for r in rows], weights=w))
                if len(w) else None)
    print(f"[{tag}] reference: nearest other GT neuron q10/50/90 {ref['nn_gt_dist_q']} um, GT mask volume "
          f"median {ref['gt_vol_med']:.0f} vox", flush=True)
    print(f"[{tag}] {summ}", flush=True)
    for r in sorted(rows, key=lambda r: -r['n_both'])[:25]:
        print(f"  neuron {r['neuron']:3d} main {r['main']:4d} partner {r['partner']:4d}: {r['n_both']:4d} vols, dist "
              f"{r['dist_med']:.2f} um ({r['dist_q'][0]:.2f}-{r['dist_q'][1]:.2f}), vol neuron/other "
              f"{r['vol_neuron_med']:.0f}/{r['vol_other_med']:.0f}, other is GT {r['other_is_gt_frac']:.2f}", flush=True)
    out = os.path.join(args.work, f'diag_fragment_pairs_{tag}_s{args.seed}.json')
    with open(out, 'w') as f:
        json.dump(dict(case=tag, seed=args.seed, reference=ref, summary=summ, pairs=rows), f)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
