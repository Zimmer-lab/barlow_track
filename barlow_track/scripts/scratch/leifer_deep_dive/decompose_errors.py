"""Where a trial's Leifer re-export accuracy goes: re-track its cached embeddings and break
the errors down by neuron, frame, mask type and what the wrong label was.

Re-runs exactly the objective's tracking (SVD 50 scaled, label propagation, nandist
column matching, segmentation_id GT mapping) from the trial's emb_cache npz, on the CPU.
--subset drops embedding rows before tracking, to price each part of the detections:
  all     every detection (the objective)
  gt      only rows that are a GT neuron's mask at that frame (no distractors)
  nerve   only NeRVE masks (labels < 1000), dropping the GT-seeded pseudo masks

    python barlow_track/scripts/scratch/leifer_deep_dive/decompose_errors.py \
        --emb <round>/emb_cache/emb_..._trial57_leifer_reexport.npz --subset all --out <dir>

Writes <out>/<subset>_cells.npz (GT label and matched prediction per frame x neuron) and
prints the breakdown.
"""
import argparse
import os
import time

import numpy as np

NWB = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/leifer_reexport/'
       'Leifer_NeRVE_Worm1_reexport_A2.nwb')
PSEUDO = 1000  # pseudo mask labels are 1000 + GT neuron index (leifer_export/README.md)


def gt_labels(gs, labels_at):
    """eval_accuracy._nwb_all_objects with gt_mapping='segmentation_id', from the cached labels."""
    gt_label = np.full(gs.shape, np.nan)
    for t in range(gs.shape[0]):
        present = labels_at.get(t, set())
        js = np.flatnonzero(np.isfinite(gs[t]))
        ok = np.array([gs[t, j] in present for j in js], dtype=bool)
        vals, counts = np.unique(gs[t, js[ok]], return_counts=True)
        shared = np.isin(gs[t, js], vals[counts > 1])
        ok &= ~shared
        gt_label[t, js[ok]] = gs[t, js[ok]]
    return gt_label


def subset_rows(X, l2t, keep):
    idx = np.flatnonzero(keep)
    t2l, new_l2t = {}, {}
    for new, old in enumerate(idx):
        t, a, b = l2t[old]
        t2l.setdefault(t, []).append(new)
        new_l2t[new] = (t, a, b)
    return X[idx], t2l, new_l2t


def track(X, t2l, l2t, num_seeds, seed):
    from sklearn.decomposition import TruncatedSVD
    from barlow_track.utils.utils_seeding import seed_all
    from barlow_track.utils.utils_tracking import WormClusterTracker
    seed_all(seed)
    Xs = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=seed).fit_transform(X)
    tracker = WormClusterTracker(Xs, t2l, linear_ind_to_t_and_seg_id=l2t, seed=seed)
    return tracker.track_using_label_propagation_clusterer(num_seeds=num_seeds, device='cpu', recipe='legacy')


def score(df_gt_src, gt_label, neurons, df_pred):
    from barlow_track.utils.utils_ground_truth import (pad_with_nan_rows, align_gt_pred_time_index,
                                                       match_columns_to_gt, calculate_accuracy)
    df_gt = df_gt_src.copy()
    for n in df_gt.columns.get_level_values(0).unique():
        df_gt[(n, 'raw_neuron_ind_in_list')] = gt_label[:, neurons.index(n)]
    df_gt, df_pred = align_gt_pred_time_index(df_gt, df_pred)
    m = max(len(df_gt), len(df_pred))
    df_gt, df_pred = pad_with_nan_rows(df_gt, m), pad_with_nan_rows(df_pred, m)
    df_pr = match_columns_to_gt(df_gt, df_pred, 'raw_neuron_ind_in_list', gt_matching='nandist')
    if 'unmatched_neuron' in df_pr.columns.get_level_values(0):
        df_pr = df_pr.drop(columns='unmatched_neuron')
    col_gt = df_gt.loc[:, (slice(None), 'raw_neuron_ind_in_list')].droplevel(1, axis=1)
    col_pr = df_pr.loc[:, (slice(None), 'raw_neuron_ind_in_list')].droplevel(1, axis=1)
    stats = calculate_accuracy(col_gt, col_pr)
    col_pr = col_pr.reindex(index=col_gt.index, columns=col_gt.columns)
    return stats, col_gt, col_pr


def report(G, P, names, labels_at):
    valid = np.isfinite(G)
    ok = valid & (G == P)
    miss = valid & np.isnan(P)
    mism = valid & np.isfinite(P) & (G != P)
    n = valid.sum()
    print(f"cells {n}: correct {ok.sum() / n:.4f}, misses {miss.sum() / n:.4f}, mismatches {mism.sum() / n:.4f}")
    pseudo = valid & (G >= PSEUDO)
    for name, m in [('NeRVE-mask GT', valid & ~pseudo), ('pseudo-mask GT', pseudo)]:
        print(f"  {name}: {m.sum()} cells, accuracy {(ok & m).sum() / max(m.sum(), 1):.4f}")
    # What the wrong label was
    T = G.shape[0]
    gt_at = [set(G[t][np.isfinite(G[t])]) for t in range(T)]
    other_gt = np.zeros_like(mism)
    for t, j in zip(*np.nonzero(mism)):
        other_gt[t, j] = P[t, j] in gt_at[t]
    print(f"  mismatches onto another GT neuron's mask: {other_gt.sum()}, onto a non-GT detection: "
          f"{(mism & ~other_gt).sum()}")
    # Per neuron
    acc_n = ok.sum(0) / np.maximum(valid.sum(0), 1)
    order = np.argsort(acc_n)
    print("  per-neuron accuracy pct 0/10/25/50/75/100:",
          np.round(np.percentile(acc_n[valid.sum(0) > 0], [0, 10, 25, 50, 75, 100]), 3))
    print("  errors held by the worst 10 neurons:",
          f"{(valid & ~ok)[:, order[:10]].sum() / max((valid & ~ok).sum(), 1):.2f}")
    for j in order[:12]:
        nv = valid[:, j].sum()
        print(f"    {names[j]:>10s} acc {acc_n[j]:.3f} n {nv:4d} pseudo {pseudo[:, j].sum() / max(nv, 1):.2f} "
              f"miss {miss[:, j].sum():4d} mism {mism[:, j].sum():4d} onto-GT {other_gt[:, j].sum():4d}")
    # Per frame
    acc_t = ok.sum(1) / np.maximum(valid.sum(1), 1)
    has = valid.sum(1) > 0
    print("  per-frame accuracy pct 0/1/5/10/50:", np.round(np.percentile(acc_t[has], [0, 1, 5, 10, 50]), 3))
    n_det = np.array([len(labels_at.get(t, ())) for t in range(T)])
    n_nerve = np.array([sum(l < PSEUDO for l in labels_at.get(t, ())) for t in range(T)])
    for name, m in [('no NeRVE masks', has & (n_nerve == 0)), ('>= 200 detections', has & (n_det >= 200)),
                    ('frames 1524+', has & (np.arange(T) >= 1523))]:
        if m.any():
            print(f"  frames with {name}: {m.sum()}, accuracy {(ok[m]).sum() / valid[m].sum():.4f}")
    # Time structure: accuracy in 16 blocks
    blocks = np.array_split(np.arange(T), 16)
    print("  accuracy by time block:", ' '.join(f"{ok[b].sum() / max(valid[b].sum(), 1):.2f}" for b in blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', required=True)
    ap.add_argument('--nwb', default=NWB)
    ap.add_argument('--subset', default='all', choices=['all', 'gt', 'nerve'])
    ap.add_argument('--num_seeds', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from barlow_track.scripts.eval_accuracy import _load_cached_embeddings, _load_nwb_arrays
    from barlow_track.scripts.scratch.leifer_deep_dive.lazy_nwb import load_nwb_project
    os.makedirs(args.out, exist_ok=True)

    X, _, l2t, n_frames = _load_cached_embeddings(args.emb)
    lab = np.array([l2t[i][1] for i in range(len(X))])
    tt = np.array([l2t[i][0] for i in range(len(X))])
    src = load_nwb_project(args.nwb)
    arrays = _load_nwb_arrays(src)
    names = arrays['neurons']
    all_labels_at = {}
    for t, l in zip(tt, lab):
        all_labels_at.setdefault(int(t), set()).add(int(l))
    gt_label = gt_labels(arrays['gs'], all_labels_at)

    if args.subset == 'gt':
        keep = np.array([l in set(gt_label[t][np.isfinite(gt_label[t])]) for t, l in zip(tt, lab)])
    elif args.subset == 'nerve':
        keep = lab < PSEUDO
        gt_label[gt_label >= PSEUDO] = np.nan
    else:
        keep = np.ones(len(X), dtype=bool)
    Xk, t2l, l2tk = subset_rows(X, l2t, keep)
    labels_at = {}
    for i in range(len(Xk)):
        labels_at.setdefault(l2tk[i][0], set()).add(l2tk[i][1])
    print(f"[{args.subset}] {len(Xk)} of {len(X)} rows, {n_frames} frames, "
          f"{np.isfinite(gt_label).sum()} GT cells mapped", flush=True)

    t0 = time.time()
    df_pred = track(Xk, t2l, l2tk, args.num_seeds, args.seed)
    print(f"tracked in {time.time() - t0:.0f} s, {df_pred.shape}", flush=True)
    df_gt = arrays['df_gt']
    try:
        fin, _ = src.get_final_tracks_only_finished_neurons()
        if fin is not None and not fin.empty:
            from barlow_track.scripts.eval_accuracy import _ensure_ind_level
            df_gt = _ensure_ind_level(fin)
    except Exception:
        pass
    stats, col_gt, col_pr = score(df_gt, gt_label, names, df_pred)
    print(f"accuracy {stats['accuracy']:.4f} misses {stats['total_misses']} mismatches "
          f"{stats['total_mismatches']} total {stats['total_ground_truth']}", flush=True)
    cols = list(col_gt.columns)
    G, P = col_gt.values.astype(float), col_pr.values.astype(float)
    np.savez(os.path.join(args.out, f'{args.subset}_cells.npz'), G=G, P=P, names=np.array(cols))
    report(G, P, cols, labels_at)


if __name__ == '__main__':
    main()
