"""TEST_PLAN T1: deformation per dataset, and trial-59 accuracy against per-frame deformation (R3.W1 / W3 / Q1).

GT positions only (no images, no GPU). Distances in units of the recording's median within-frame nearest-neighbour
(NN) distance between ALL segmented detections (gt_positions.detection_nn), after z is scaled by the recording's ViSNet
z_scale. Not the GT-only NN: Zimmer and Leifer annotate a subset of the neurons, which inflates their GT NN distance
(~13-14 vs ~7 voxels) and understated their deformation in the first run (2026-11-24). The table keeps both NN
distances and the GT's share of detections.

1. Dataset table (t1_dataset_table.tsv): per lag, over up to --pairs sampled frame pairs: raw displacement / NN
   (median, p90, fraction > 1 NN), non-rigid residual after a similarity fit (rotation + translation + uniform
   scale) / NN (median, p90), rotation of that fit (median deg), and at the 60 s lag the std of log
   neighbour-distance ratio for pairs initially 2.5-10 NN apart (local_geometry_change.py's mid-range bin).
   Lags: 1 volume (what the tracker sees between consecutive volumes) and 1 s, 10 s, 60 s at each lab's volume rate
   (gt_positions.VPS). Fixed-frame lags confound posture variety with frame rate (100 frames = 59 s Flavell, 10 s
   Samuel), so cross-lab comparisons use the seconds lags.
2. Per-frame scores (t1_frames_<lab>.tsv): d_lag1(t) = median residual / NN between t and t+1 (speed);
   d_novel(t) = min over ~50 evenly spaced reference frames (|t - r| >= 5) of the median residual / NN
   (how unlike any reference posture frame t is; label propagation seeds from such frames); rot_lag1(t);
   acc(t) = per-frame accuracy of the trial-59 tracks (round_shared_v1/tracks_v2), mean over replicates.
3. Accuracy by decile of d_novel and d_lag1 (t1_deciles.tsv), the top-decile drop and the Spearman correlation.
4. The 5 most deformed non-overlapping 20-frame segments per recording (t1_segments.tsv): candidates for the
   multi-frame figure and the supplementary videos.
5. Per-neuron non-rigid residual / NN at lag 10 (neuron_deformation_<lab>.npz), used by modality_contribution.py.

Usage (wbfm env, from the repo root):
  PYTHONPATH=.:barlow_track/scripts/scratch/revision_t1_t2 python barlow_track/scripts/scratch/revision_t1_t2/deformation_table.py \
      --out /lisc/.../barlow_track_revisions/revision_analysis [--labs zimmer_1128 samuel_137]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gt_positions import GT_PATHS, ROUNDS, VPS, detection_nn, load_positions, nn_distance, similarity_fit, track_accuracy

LAGS_S = (1, 10, 60)  # seconds; plus the 1-volume lag
MIN_POINTS = 10


def pair_stats(a, b, nn):
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if ok.sum() < MIN_POINTS:
        return None
    a, b = a[ok], b[ok]
    res, angle, _ = similarity_fit(a, b)
    return ok, np.linalg.norm(b - a, axis=1) / nn, np.linalg.norm(res, axis=1) / nn, angle, a / nn, b / nn


def dataset_row(lab, pos, nn, rng, n_pairs, nn_gt, det):
    T = len(pos)
    gt_per_frame = float(np.isfinite(pos).all(-1).sum(1).mean())
    row = dict(lab=lab, n_frames=T, n_neurons=pos.shape[1], gt_per_frame=gt_per_frame,
               detections_per_frame=det['detections_per_frame'],
               gt_share_of_detections=gt_per_frame / det['detections_per_frame'],
               nn_det_voxels=nn, nn_gt_voxels=nn_gt, vps=VPS[lab], duration_s=T / VPS[lab])
    lags = [('1vol', 1)] + [(f'{s}s', int(round(s * VPS[lab]))) for s in LAGS_S]
    for name, lag in lags:
        ts = np.arange(T - lag)
        ts = rng.choice(ts, size=min(n_pairs, len(ts)), replace=False)
        raw, res, ang, lr = [], [], [], []
        for t in ts:
            s = pair_stats(pos[t], pos[t + lag], nn)
            if s is None:
                continue
            _, r, e, a, pa, pb = s
            raw.append(r), res.append(e), ang.append(a)
            if name == '60s':
                da = np.linalg.norm(pa[:, None] - pa[None], axis=-1)
                db = np.linalg.norm(pb[:, None] - pb[None], axis=-1)
                iu = np.triu_indices(len(pa), 1)
                m = (da[iu] >= 2.5) & (da[iu] < 10)
                lr.append(np.log(db[iu][m] / da[iu][m]))
        raw, res = np.concatenate(raw), np.concatenate(res)
        row.update({f'raw_med_{name}': np.median(raw), f'raw_p90_{name}': np.percentile(raw, 90),
                    f'raw_frac_gt1NN_{name}': (raw > 1).mean(),
                    f'nonrigid_med_{name}': np.median(res), f'nonrigid_p90_{name}': np.percentile(res, 90),
                    f'rot_deg_med_{name}': np.median(ang)})
        if name == '60s':
            row['logdist_std_mid_60s'] = np.std(np.concatenate(lr))
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}


def frame_scores(pos, nn, n_refs=50):
    T = len(pos)
    d_lag1, rot = np.full(T, np.nan), np.full(T, np.nan)
    for t in range(T - 1):
        s = pair_stats(pos[t], pos[t + 1], nn)
        if s is not None:
            d_lag1[t], rot[t] = np.median(s[2]), s[3]
    refs = np.unique(np.linspace(0, T - 1, n_refs).astype(int))
    d_novel = np.full(T, np.inf)
    for r in refs:
        for t in range(T):
            if abs(t - r) < 5:
                continue
            s = pair_stats(pos[r], pos[t], nn)
            if s is not None:
                d_novel[t] = min(d_novel[t], np.median(s[2]))
    d_novel[~np.isfinite(d_novel)] = np.nan
    return d_lag1, d_novel, rot


def neuron_deformation(pos, nn, rng, n_pairs, lag=10):
    """Per neuron: median non-rigid residual / NN over sampled lag-10 pairs."""
    T, N = pos.shape[:2]
    vals = [[] for _ in range(N)]
    for t in rng.choice(np.arange(T - lag), size=min(n_pairs, T - lag), replace=False):
        s = pair_stats(pos[t], pos[t + lag], nn)
        if s is None:
            continue
        for j, e in zip(np.flatnonzero(s[0]), s[2]):
            vals[j].append(e)
    return np.array([np.median(v) if v else np.nan for v in vals])


def trial59_frame_accuracy(lab, index):
    """Per-frame accuracy of every trial-59 replicate's v2 tracks on this recording, mean over replicates."""
    files = sorted(glob.glob(f'{ROUNDS}/round_shared_v1/tracks_v2/round_shared_v1_trial59*_{lab}.h5'))
    files = [f for f in files if re.search(rf'trial59(r\d+)?_{re.escape(lab)}\.h5$', f)]
    accs = []
    for f in files:
        correct, valid = track_accuracy(f)
        a = (correct.sum(1) / valid.sum(1).replace(0, np.nan)).reindex(index)
        accs.append(a.to_numpy())
    return np.nanmean(np.stack(accs), 0) if accs else np.full(len(index), np.nan), len(files)


def decile_table(lab, score_name, score, acc):
    ok = np.isfinite(score) & np.isfinite(acc)
    s, a = score[ok], acc[ok]
    edges = np.percentile(s, np.linspace(0, 100, 11))
    q = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, 9)
    rows = [dict(lab=lab, score=score_name, decile=d + 1, score_lo=round(edges[d], 3), score_hi=round(edges[d + 1], 3),
                 n_frames=int((q == d).sum()), acc_mean=round(float(a[q == d].mean()), 4)) for d in range(10)]
    rho = spearmanr(s, a).correlation
    summary = dict(lab=lab, score=score_name, spearman=round(float(rho), 3),
                   acc_top_decile=round(float(a[q == 9].mean()), 4), acc_rest=round(float(a[q < 9].mean()), 4),
                   drop_top_decile_pt=round(100 * float(a[q < 9].mean() - a[q == 9].mean()), 2))
    return rows, summary


def top_segments(lab, d_novel, acc, index, k=5, width=20):
    T = len(d_novel)
    m = pd.Series(d_novel).rolling(width, min_periods=width // 2).mean().to_numpy()
    out, taken = [], np.zeros(T, bool)
    for end in np.argsort(-np.nan_to_num(m, nan=-np.inf)):
        start = end - width + 1
        if start < 0 or taken[start:end + 1].any() or not np.isfinite(m[end]):
            continue
        taken[start:end + 1] = True
        out.append(dict(lab=lab, frame_start=int(index[start]), frame_end=int(index[end]),
                        d_novel_mean=round(float(m[end]), 3),
                        acc_mean=round(float(np.nanmean(acc[start:end + 1])), 4)))
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--labs', nargs='+', default=list(GT_PATHS))
    ap.add_argument('--pairs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cache = os.path.join(args.out, 'cache')

    table, dec_rows, dec_sum, segs = [], [], [], []
    for lab in args.labs:
        rng = np.random.default_rng([args.seed, sum(map(ord, lab))])
        pos, names, index = load_positions(lab, cache)
        det = detection_nn(lab, cache)
        nn, nn_gt = det['nn_det'], nn_distance(pos)
        table.append(dataset_row(lab, pos, nn, rng, args.pairs, nn_gt, det))
        print(table[-1], flush=True)

        np.savez(os.path.join(args.out, f'neuron_deformation_{lab}.npz'), names=np.array(names), nn=nn,
                 nonrigid_lag10=neuron_deformation(pos, nn, rng, args.pairs))

        d_lag1, d_novel, rot = frame_scores(pos, nn)
        acc, n_rep = trial59_frame_accuracy(lab, index)
        pd.DataFrame(dict(frame=index, d_lag1=d_lag1, d_novel=d_novel, rot_lag1=rot, acc_t59=acc)).to_csv(
            os.path.join(args.out, f't1_frames_{lab}.tsv'), sep='\t', index=False)
        for name, sc in (('d_novel', d_novel), ('d_lag1', d_lag1)):
            r, s = decile_table(lab, name, sc, acc)
            s['n_replicates'] = n_rep
            dec_rows += r
            dec_sum.append(s)
            print(s, flush=True)
        segs += top_segments(lab, d_novel, acc, index)

    pd.DataFrame(table).to_csv(os.path.join(args.out, 't1_dataset_table.tsv'), sep='\t', index=False)
    pd.DataFrame(dec_rows).to_csv(os.path.join(args.out, 't1_deciles.tsv'), sep='\t', index=False)
    pd.DataFrame(dec_sum).to_csv(os.path.join(args.out, 't1_decile_summary.tsv'), sep='\t', index=False)
    pd.DataFrame(segs).to_csv(os.path.join(args.out, 't1_segments.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
