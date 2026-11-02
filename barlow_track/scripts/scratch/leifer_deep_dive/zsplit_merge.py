"""Would merging z-stacked NeRVE masks fix the Leifer re-export's accuracy? (GT-free rule)

The deep dive (HANDOFF.md, finding 6) found that ~9 points go to non-GT masks sitting right
above or below the true GT mask. This tests whether those are z-splits of one cell, and what
merging them would buy, without re-embedding:

  pairs   per frame, every pair of masks in vertical voxel contact (label a at plane z, label b
          at plane z+1, same xy pixel); contact_frac = contacts / min(largest single-plane area
          of a, of b). No GT used. Writes <out>/pairs.npz.
  track   merge pairs with contact_frac >= --thresh (union-find per frame), collapse each group
          to one embedding row (--collapse largest: the largest mask's row; mean: the mean of the
          members' rows), map GT labels through the merge, then re-track and score exactly as
          decompose_errors.py does. Also reports how many of the baseline errors' stealing masks
          get merged into their GT mask, and how many GT-GT pairs get merged (false merges).
          --drop_frames removes whole volumes (e.g. the 4 corrupted ones) from tracking and
          scoring.

    python .../zsplit_merge.py pairs --out <dir> --workers 8
    python .../zsplit_merge.py track --out <dir> --emb <trial57 npz> --thresh 0.3 --collapse largest

Run with WBFM_NWB_LAZY_SEGMENTATION=1 (wbfm lazy NWB segmentation), or memory per worker is ~35 GB.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from barlow_track.scripts.scratch.leifer_deep_dive.decompose_errors import (NWB, PSEUDO, gt_labels, track,
                                                                            score, report)

CORRUPTED = (78, 405, 945, 1516)  # HANDOFF.md finding 2

_P = None


def _project():
    global _P
    if _P is None:
        os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
        from wbfm.utils.projects.finished_project_data import ProjectData
        _P = ProjectData.load_final_project_data(NWB, allow_hybrid_loading=True, verbose=0)
    return _P


def frame_pairs(t):
    seg = np.asarray(_project().raw_segmentation[t])  # (Z, X, Y)
    n = int(seg.max()) + 1
    plane_area = np.stack([np.bincount(seg[z].ravel(), minlength=n) for z in range(seg.shape[0])])
    max_area = plane_area.max(0)
    codes = []
    for z in range(seg.shape[0] - 1):
        a, b = seg[z], seg[z + 1]
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            lo, hi = np.minimum(a[m], b[m]).astype(np.int64), np.maximum(a[m], b[m]).astype(np.int64)
            codes.append(lo * 65536 + hi)
    if not codes:
        return np.zeros((0, 5))
    u, c = np.unique(np.concatenate(codes), return_counts=True)
    la, lb = u // 65536, u % 65536
    frac = c / np.minimum(max_area[la], max_area[lb])
    return np.stack([np.full(len(u), t), la, lb, c, frac], 1)


def cmd_pairs(args):
    T = _project().num_frames
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for r in ex.map(frame_pairs, range(T), chunksize=8) if len(r)]
    a = np.concatenate(rows)
    np.savez(os.path.join(args.out, 'pairs.npz'), t=a[:, 0].astype(int), a=a[:, 1].astype(int),
             b=a[:, 2].astype(int), contacts=a[:, 3].astype(int), frac=a[:, 4])
    print(f"{len(a)} touching pairs in {T} frames; contact_frac pct 10/50/90: "
          f"{np.round(np.percentile(a[:, 4], [10, 50, 90]), 3)}")


def _groups(pairs, thresh):
    """{t: {label: representative}} from union-find over pairs with frac >= thresh."""
    keep = pairs['frac'] >= thresh
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    for t, a, b in zip(pairs['t'][keep], pairs['a'][keep], pairs['b'][keep]):
        ra, rb = find((t, a)), find((t, b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    rep = {}
    for k in list(parent):
        r = find(k)
        rep.setdefault(k[0], {})[k[1]] = r[1]
    return rep


def cmd_track(args):
    from barlow_track.scripts.eval_accuracy import _load_cached_embeddings, _load_nwb_arrays
    pairs = dict(np.load(os.path.join(args.out, 'pairs.npz')))
    X, _, l2t, n_frames = _load_cached_embeddings(args.emb)
    tt = np.array([l2t[i][0] for i in range(len(X))])
    lab = np.array([l2t[i][1] for i in range(len(X))])
    src = _project()
    arrays = _load_nwb_arrays(src)
    names = arrays['neurons']
    labels_at = {}
    for t, l in zip(tt, lab):
        labels_at.setdefault(int(t), set()).add(int(l))
    gt_base = gt_labels(arrays['gs'], labels_at)

    rep = _groups(pairs, args.thresh) if args.thresh <= 1 else {}
    rep_of = lambda t, l: rep.get(int(t), {}).get(int(l), int(l))
    n_merged = sum(len(d) for d in rep.values())
    n_groups = sum(len(set(d.values())) for d in rep.values())

    # Diagnostics against the baseline errors and GT (GT used only to report, never to merge)
    gt_set = [set(gt_base[t][np.isfinite(gt_base[t])].astype(int)) for t in range(gt_base.shape[0])]
    gtgt = sum(1 for t, d in rep.items() for r in set(d.values())
               if sum(1 for l, rr in d.items() if rr == r and l in gt_set[t]) > 1)
    print(f"thresh {args.thresh}: {n_merged} masks in {n_groups} merged groups; groups holding >1 GT neuron: {gtgt}")
    if args.baseline_cells and os.path.exists(args.baseline_cells):
        c = np.load(args.baseline_cells)
        G, P = c['G'], c['P']
        mism = np.isfinite(G) & np.isfinite(P) & (G != P)
        ts, js = np.nonzero(mism)
        onto_non_gt = np.array([P[t, j] not in gt_set[t] for t, j in zip(ts, js)])
        same = np.array([rep_of(t, G[t, j]) == rep_of(t, P[t, j]) for t, j in zip(ts, js)])
        print(f"  baseline mismatches {mism.sum()}: onto non-GT {onto_non_gt.sum()}, of which merged into the "
              f"GT mask's group {(same & onto_non_gt).sum()} ({(same & onto_non_gt).sum() / max(onto_non_gt.sum(), 1):.2f})")

    # Collapse rows
    drop = set(args.drop_frames)
    key = np.array([rep_of(t, l) for t, l in zip(tt, lab)])
    order = np.lexsort((key, tt))
    newX, t2l, new_l2t = [], {}, {}
    voxels = None
    if args.collapse == 'largest':
        g = np.load(args.masks)
        voxels = {(int(t), int(l)): v for t, l, v in zip(g['t'], g['label'], g['voxels'])}
    i = 0
    while i < len(order):
        j = i
        t, k = tt[order[i]], key[order[i]]
        while j < len(order) and tt[order[j]] == t and key[order[j]] == k:
            j += 1
        members = order[i:j]
        i = j
        if int(t) in drop:
            continue
        if len(members) == 1 or args.collapse == 'mean':
            v = X[members].mean(0)
        else:
            v = X[max(members, key=lambda m: voxels.get((int(t), int(lab[m])), 0))]
        new = len(newX)
        newX.append(v)
        t2l.setdefault(int(t), []).append(new)
        new_l2t[new] = (int(t), int(k), int(k))
    Xk = np.asarray(newX, dtype=X.dtype)

    gs = arrays['gs'].copy()
    for t in range(gs.shape[0]):
        f = np.isfinite(gs[t])
        gs[t, f] = [rep_of(t, l) for l in gs[t, f]]
        if t in drop:
            gs[t] = np.nan
    new_labels_at = {}
    for n_, (t, l, _) in new_l2t.items():
        new_labels_at.setdefault(t, set()).add(l)
    gt_label = gt_labels(gs, new_labels_at)
    print(f"  {len(Xk)} of {len(X)} rows after merge/drop, {np.isfinite(gt_label).sum()} GT cells mapped "
          f"(baseline {np.isfinite(gt_base).sum()})", flush=True)

    t0 = time.time()
    df_pred = track(Xk, t2l, new_l2t, args.num_seeds, args.seed)
    print(f"tracked in {time.time() - t0:.0f} s", flush=True)
    from barlow_track.scripts.eval_accuracy import _ensure_ind_level
    df_gt = arrays['df_gt']
    try:
        fin, _ = src.get_final_tracks_only_finished_neurons()
        if fin is not None and not fin.empty:
            df_gt = _ensure_ind_level(fin)
    except Exception:
        pass
    stats, col_gt, col_pr = score(df_gt, gt_label, names, df_pred)
    print(f"accuracy {stats['accuracy']:.4f} misses {stats['total_misses']} mismatches "
          f"{stats['total_mismatches']} total {stats['total_ground_truth']}", flush=True)
    tag = f"merge_{args.thresh}_{args.collapse}" + ("_drop" if drop else "")
    np.savez(os.path.join(args.out, f'{tag}_cells.npz'), G=col_gt.values.astype(float),
             P=col_pr.values.astype(float), names=np.array(list(col_gt.columns)))
    report(col_gt.values.astype(float), col_pr.values.astype(float), list(col_gt.columns), new_labels_at)


def cmd_compare(args):
    """Merged runs score fewer GT cells (GT neurons merged together drop out), so compare on the
    same cells: the baseline restricted to the cells a merged run still scores."""
    b = np.load(args.baseline_cells)
    Gb, Pb = b['G'], b['P']
    ok_b = np.isfinite(Gb) & (Gb == Pb)
    print(f"baseline: {np.isfinite(Gb).sum()} cells, accuracy {ok_b.sum() / np.isfinite(Gb).sum():.4f}")
    for f in sorted(os.listdir(args.out)):
        if not (f.startswith('merge_') and f.endswith('_cells.npz')):
            continue
        m = np.load(os.path.join(args.out, f))
        assert list(m['names']) == list(b['names'])
        Gm, Pm = m['G'], m['P']
        T = min(len(Gm), len(Gb))
        vm = np.isfinite(Gm[:T])
        ok_m = vm & (Gm[:T] == Pm[:T])
        print(f"{f[:-10]}: {vm.sum()} cells, accuracy {ok_m.sum() / vm.sum():.4f}; baseline on the same cells "
              f"{ok_b[:T][vm].sum() / vm.sum():.4f}; cells dropped {np.isfinite(Gb[:T]).sum() - vm.sum()}, "
              f"baseline accuracy on them {ok_b[:T][np.isfinite(Gb[:T]) & ~vm].sum() / max((np.isfinite(Gb[:T]) & ~vm).sum(), 1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pairs', 'track', 'compare'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--emb')
    ap.add_argument('--masks', help='mask_geometry.py output (voxels per mask), for --collapse largest')
    ap.add_argument('--baseline_cells', help='decompose_errors.py all_cells.npz, for the stealing-mask diagnostic')
    ap.add_argument('--thresh', type=float, default=0.3, help='contact_frac to merge; > 1 merges nothing')
    ap.add_argument('--collapse', default='largest', choices=['largest', 'mean'])
    ap.add_argument('--drop_frames', type=int, nargs='*', default=[])
    ap.add_argument('--num_seeds', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {'pairs': cmd_pairs, 'track': cmd_track, 'compare': cmd_compare}[args.cmd](args)


if __name__ == '__main__':
    main()
