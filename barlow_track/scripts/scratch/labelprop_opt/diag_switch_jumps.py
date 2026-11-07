"""At the moments a GT neuron's objects change final label, which one jumped: the GT click or the track?

For project-source cases (segmentation centroids), from the cached labels of diag_merge.py. For every GT
neuron, take its main label (the one holding most of its objects). A switch is a pair of adjacent volumes
(t-1, t) where the neuron's GT object is in the main label at one and in another label at the other, and the
main label has an object at both. At each switch compare
  gt_jump    |GT object centroid(t) - GT object centroid(t-1)|
  track_jump |main-label object centroid(t) - main-label object centroid(t-1)|
against the neuron's typical step (median GT step over volumes without a switch).

With --local K (default 8) positions are taken in a local frame: minus the mean centroid of the K GT
neurons nearest to this one (median distance), over those clicked in both volumes. That removes whole-worm
motion, which on Leifer is about one neighbour distance per volume.

With --click_nwb (an NWB whose final_tracks x/y/z are the manual clicks, as gt_tracks_from_clicks.py) the
click point itself is followed too: click_jump, its local-frame step at the switch. A GT object that jumps
while its click does not means the click stayed put and fell into the other of two touching masks
(segmentation / assignment); a click that jumps means the annotation moved to another nucleus.

gt_jump >> typical while track_jump ~ typical: the click moved to another (usually touching) mask and the
track stayed on its object -- the error is in the click or the segmentation, not the tracker.
track_jump >> typical while gt_jump ~ typical: the tracker moved the label to another object.

    python diag_switch_jumps.py --case round_bayes_wk2_leifer_reexport:57:leifer_stardist_strict --seed 0 \
        --work $WORK --um_zxy 1.5 0.3226 0.3226

Output: $WORK/diag_switch_jumps_<tag>_<variant>_s<seed>_local<K>.json
"""
import argparse
import collections
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
    ap.add_argument('--ratio', type=float, default=2.0, help='a jump is a step above ratio x the typical step')
    ap.add_argument('--click_nwb', default=None, help='NWB with the manual clicks (final_tracks x/y/z)')
    ap.add_argument('--local', type=int, default=8, help='reference GT neurons for the local frame (0: global)')
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
    cent = {}

    def c_of(i):
        t = int(t_of[i])
        if t not in cent:
            df = meta[t]
            cent[t] = {int(k): np.array(c, float) * um for k, c in zip(df.index, df['centroids'])}
        return cent[t].get(int(seg_of[i]))

    by_lab_t = {(int(labels[i]), int(t_of[i])): i for i in np.flatnonzero(labels >= 1)}
    clicks = None
    if args.click_nwb:
        import pandas as pd
        from wbfm.utils.projects.finished_project_data import ProjectData
        from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
        a = _load_nwb_arrays(ProjectData.load_final_project_data(args.click_nwb, allow_hybrid_loading=True, verbose=0))
        dummy = pd.DataFrame(index=pd.RangeIndex(n_frames))
        from barlow_track.utils.utils_ground_truth import align_gt_pred_time_index
        gal, _ = align_gt_pred_time_index(scorer.df_gt, dummy)
        names = list(gal.loc[:, (slice(None), scorer.match_col)].droplevel(1, axis=1).columns)
        col = [a['neurons'].index(nm) for nm in names]
        # (T, N_gt, 3) z x y in um, in the scorer's GT neuron order
        clicks = np.stack([a['gz'][:, col], a['gx'][:, col], a['gy'][:, col]], -1) * um
        print(f"[{tag}] clicks from {args.click_nwb}: {clicks.shape}", flush=True)
    n_gt = g.max() + 1
    gt_obj_at = {}  # (neuron, t) -> object
    for i in np.flatnonzero(g >= 0):
        gt_obj_at[(int(g[i]), int(t_of[i]))] = i
    refs = {}
    if args.local > 0:
        D = [[] for _ in range(n_gt)]
        for t in range(0, n_frames, max(1, n_frames // 200)):
            P = {k: c_of(gt_obj_at[(k, t)]) for k in range(n_gt) if (k, t) in gt_obj_at}
            P = {k: c for k, c in P.items() if c is not None}
            ks = list(P)
            if len(ks) < 2:
                continue
            A = np.array([P[k] for k in ks])
            M = np.linalg.norm(A[:, None] - A[None], axis=-1)
            for a, ka in enumerate(ks):
                D[ka].append({kb: M[a, b] for b, kb in enumerate(ks) if kb != ka})
        for k in range(n_gt):
            acc = {}
            for dd in D[k]:
                for kb, v in dd.items():
                    acc.setdefault(kb, []).append(v)
            med = sorted((np.median(v), kb) for kb, v in acc.items() if len(v) >= 10)
            refs[k] = [kb for _, kb in med[:args.local]]

    def frame(k, t0, t1):
        """local origins at t0 and t1 from the reference neurons clicked at both (None: too few)."""
        if args.local <= 0:
            return np.zeros(3), np.zeros(3)
        r0, r1 = [], []
        for kb in refs.get(k, []):
            a, b = gt_obj_at.get((kb, t0)), gt_obj_at.get((kb, t1))
            if a is None or b is None:
                continue
            ca, cb = c_of(a), c_of(b)
            if ca is not None and cb is not None:
                r0.append(ca)
                r1.append(cb)
        if len(r0) < 3:
            return None
        return np.mean(r0, 0), np.mean(r1, 0)
    rows, all_sw = [], []
    click_d = collections.defaultdict(list)
    for k in range(g.max() + 1):
        objs = np.flatnonzero((g == k) & (labels >= 1))
        if len(objs) < 20:
            continue
        labs, cnt = np.unique(labels[objs], return_counts=True)
        main = int(labs[cnt.argmax()])
        gt_at = {int(t_of[i]): i for i in np.flatnonzero(g == k)}
        ts = sorted(gt_at)
        steps, csteps, sws = [], [], []
        for t0, t1 in zip(ts[:-1], ts[1:]):
            if t1 != t0 + 1:
                continue
            i0, i1 = gt_at[t0], gt_at[t1]
            a, b = c_of(i0), c_of(i1)
            fr = frame(k, t0, t1)
            if a is None or b is None or fr is None:
                continue
            o0, o1 = fr
            gj = float(np.linalg.norm((b - o1) - (a - o0)))
            cj = float('nan')
            if clicks is not None and t1 < len(clicks):
                cj = float(np.linalg.norm((clicks[t1, k] - o1) - (clicks[t0, k] - o0)))
            in0, in1 = labels[i0] == main, labels[i1] == main
            if in0 == in1:
                steps.append(gj)
                csteps.append(cj)
                continue
            m0, m1 = by_lab_t.get((main, t0)), by_lab_t.get((main, t1))
            if m0 is None or m1 is None:
                continue
            ma, mb = c_of(m0), c_of(m1)
            if ma is None or mb is None:
                continue
            sws.append(dict(t=t1, into_main=bool(in1), gt_jump=gj, click_jump=cj,
                            track_jump=float(np.linalg.norm((mb - o1) - (ma - o0)))))
        if clicks is not None:
            # click to object distances: in error volumes (GT object off the main label, main label present)
            # to the GT object and to the main label's object; in correct volumes to the GT object
            for t, i in gt_at.items():
                if t >= len(clicks) or not np.all(np.isfinite(clicks[t, k])):
                    continue
                ci = c_of(i)
                if ci is None:
                    continue
                if labels[i] == main:
                    click_d['correct_to_gt'].append(float(np.linalg.norm(clicks[t, k] - ci)))
                    continue
                m = by_lab_t.get((main, t))
                cm = c_of(m) if m is not None else None
                if cm is None:
                    continue
                click_d['error_to_gt'].append(float(np.linalg.norm(clicks[t, k] - ci)))
                click_d['error_to_track'].append(float(np.linalg.norm(clicks[t, k] - cm)))
                click_d['error_gt_to_track'].append(float(np.linalg.norm(ci - cm)))
        if not sws or len(steps) < 10:
            continue
        typ = float(np.median(steps))
        ctyp = float(np.nanmedian(csteps)) if clicks is not None else float('nan')
        for s in sws:
            s['click_typical'] = ctyp
            s['click_jumped'] = bool(s['click_jump'] > args.ratio * max(ctyp, 0.3)) if clicks is not None else None
            s['neuron'] = int(k)
            s['typical'] = typ
            s['gt_jumped'] = s['gt_jump'] > args.ratio * max(typ, 0.3)
            s['track_jumped'] = s['track_jump'] > args.ratio * max(typ, 0.3)
        all_sw.extend(sws)
        rows.append(dict(neuron=int(k), main=main, n_switch=len(sws), typical_step=typ,
                         gt_jumped_frac=float(np.mean([s['gt_jumped'] for s in sws])),
                         track_jumped_frac=float(np.mean([s['track_jumped'] for s in sws])),
                         click_typical_step=ctyp,
                         click_jumped_frac=float(np.mean([bool(s['click_jumped']) for s in sws])) if clicks is not None
                         else None))
    gj = np.array([s['gt_jumped'] for s in all_sw])
    tj = np.array([s['track_jumped'] for s in all_sw])
    summ = dict(n_switches=len(all_sw), n_neurons=len(rows),
                only_gt_jumped=float(np.mean(gj & ~tj)), only_track_jumped=float(np.mean(tj & ~gj)),
                both=float(np.mean(gj & tj)), neither=float(np.mean(~gj & ~tj)),
                typical_step_med=float(np.median([r['typical_step'] for r in rows])),
                gt_jump_med=float(np.median([s['gt_jump'] for s in all_sw])),
                track_jump_med=float(np.median([s['track_jump'] for s in all_sw])))
    if clicks is not None:
        cjb = np.array([bool(s['click_jumped']) for s in all_sw])
        summ.update(click_jumped=float(np.mean(cjb)), gt_jumped_click_still=float(np.mean(gj & ~cjb)),
                    gt_jumped_click_jumped=float(np.mean(gj & cjb)),
                    click_jump_med=float(np.nanmedian([s['click_jump'] for s in all_sw])),
                    click_typical_med=float(np.nanmedian([r['click_typical_step'] for r in rows])))
        e_gt, e_tr = np.array(click_d['error_to_gt']), np.array(click_d['error_to_track'])
        summ['click_dist'] = {k: [float(x) for x in np.quantile(v, [0.25, 0.5, 0.75])] for k, v in click_d.items() if v}
        # error volumes where the click is nearer the track's object than the GT object it was assigned to
        summ['error_click_nearer_track'] = float(np.mean(e_tr < e_gt)) if len(e_gt) else None
        summ['n_error_volumes'] = int(len(e_gt))
    print(f"[{tag}] {summ}", flush=True)
    for r in sorted(rows, key=lambda r: -r['n_switch'])[:20]:
        print(f"  neuron {r['neuron']:3d}: {r['n_switch']:3d} switches, typical step {r['typical_step']:.2f} um, "
              f"GT jumped {r['gt_jumped_frac']:.2f}, track jumped {r['track_jumped_frac']:.2f}", flush=True)
    out = os.path.join(args.work, f'diag_switch_jumps_{tag}_{args.variant}_s{args.seed}_local{args.local}.json')
    with open(out, 'w') as f:
        json.dump(dict(case=tag, variant=args.variant, seed=args.seed, ratio=args.ratio, summary=summ,
                       neurons=rows, switches=all_sw), f)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
