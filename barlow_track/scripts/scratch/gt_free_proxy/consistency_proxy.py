"""GT-free tracking self-consistency of sweep trials, from their cached embeddings.

For every trial of a round with an embedding in <round>/emb_cache, re-run label propagation exactly as
eval_accuracy --mode trained does (50-component scaled SVD, WormClusterTracker, 25 seed volumes, seed 0)
and record, next to the trial's recorded accuracy (exp_results.jsonl), measures that need no GT:

  agree_mean / agree_p10  fraction of objects whose label in one propagation run, mapped to the consensus
                          by spectral synchronization, equals the consensus label (mean / 10th pct over runs)
  conf_mean / conf_med    consensus confidence per object (spectral_sync probs_out, top label)
  assigned_frac           fraction of objects with a consensus label
  dup_frac                fraction of (volume, label) cells claimed by 2+ objects (track conflicts)
  complete_tracks_50/90   labels present in >= 50% / 90% of volumes, over the median objects per volume
  ari_seed1               (with --second_seed) adjusted Rand index of the consensus labels against a
                          second full run with seed 1 (different seed volumes and kNN graph)

Writes one JSON line per trial to --out; skips tags already there (resumable).

    python barlow_track/scripts/scratch/gt_free_proxy/consistency_proxy.py \
        --round .../config_files/round_bayes_wk3_zimmer_1210 --out proxy.jsonl --device cuda
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))


def consensus_run(Xs, time_to_lin, lin_to_t_seg, seed, device, num_seeds=25):
    from barlow_track.utils.utils_tracking import WormClusterTracker, select_seed_times
    from barlow_track.utils.utils_label_propagation import multi_seed_propagation
    from barlow_track.utils.utils_spectral_relabeling import spectral_sync_from_topk
    t0 = time.time()
    tracker = WormClusterTracker(Xs, dict(time_to_lin), linear_ind_to_t_and_seg_id=lin_to_t_seg, seed=seed)
    # Same defaults as WormClusterTracker.track_using_label_propagation_clusterer
    seed_times = select_seed_times(tracker.time_index_to_linear_feature_indices, num_seeds=num_seeds, seed=seed)
    labelings, probabilities = multi_seed_propagation(
        tracker.X_svd, seed_times, tracker.time_index_to_linear_feature_indices, k=20, device=device,
        random_state=seed, return_top_k=2, num_layers=100, softmax=False, tau=0.02)
    print(f'  propagation {time.time() - t0:.0f}s', flush=True); t0 = time.time()
    perms, final_labels, confidence, _ = spectral_sync_from_topk(
        labelings, probabilities, time_index_to_linear_feature_indices=tracker.time_index_to_linear_feature_indices,
        input_probability_threshold=0.1, verbose=False)
    print(f'  spectral sync {time.time() - t0:.0f}s', flush=True)
    return labelings, perms, final_labels[:, 0], confidence[:, 0]


def metrics(labelings, perms, final, conf, lin_to_t_seg, n_frames):
    has = final >= 0
    agree = []
    for lab, perm in zip(labelings, perms):
        # Run labels are 1-based; perm is indexed by column (label - 1) and gives the consensus
        # column (-1 = none); final labels are consensus column + 1 (enforce_temporal_uniqueness_hungarian).
        top = np.asarray(lab)[:, 0]
        perm = np.asarray(perm)
        ok = (top >= 1) & (top - 1 < len(perm))
        mapped = np.full(len(top), -2)
        col = perm[top[ok] - 1]
        mapped[ok] = np.where(col >= 0, col + 1, -2)
        agree.append(float(np.mean(mapped[has] == final[has])) if has.any() else np.nan)
    agree = np.array(agree)
    t = np.array([lin_to_t_seg[i][0] for i in range(len(final))])
    cells = {}
    for ti, lab in zip(t[has], final[has]):
        cells[(ti, lab)] = cells.get((ti, lab), 0) + 1
    counts = np.array(list(cells.values()))
    per_label_frames = {}
    for (ti, lab) in cells:
        per_label_frames[lab] = per_label_frames.get(lab, 0) + 1
    cover = np.array(list(per_label_frames.values())) / n_frames
    med_obj = np.median(np.bincount(t, minlength=n_frames))
    return dict(agree_mean=float(np.nanmean(agree)), agree_p10=float(np.nanpercentile(agree, 10)),
                conf_mean=float(np.mean(conf[has])) if has.any() else np.nan,
                conf_med=float(np.median(conf[has])) if has.any() else np.nan,
                assigned_frac=float(has.mean()), dup_frac=float(np.mean(counts > 1)),
                complete_tracks_50=float((cover >= 0.5).sum() / med_obj),
                complete_tracks_90=float((cover >= 0.9).sum() / med_obj),
                n_objects=int(len(final)), median_objects=float(med_obj))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--second_seed', action='store_true')
    ap.add_argument('--trials', nargs='*', type=int, default=None)
    args = ap.parse_args()

    import torch
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import adjusted_rand_score

    acc = {}
    with open(os.path.join(args.round, 'exp_results.jsonl')) as f:
        for line in f:
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get('accuracy') is not None:
                acc[x['tag']] = x
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)['tag'] for l in f if l.strip()}
    for emb in sorted(glob.glob(os.path.join(args.round, 'emb_cache', 'emb_*_trained_*.npz'))):
        tag = re.sub(r'^emb_.*?_trained_', '', os.path.basename(emb))[:-4]
        trial = int(re.search(r'_trial(\d+)_', tag).group(1))
        if tag in done or tag not in acc or (args.trials and trial not in args.trials):
            continue
        rec = acc[tag]
        t0 = time.time()
        with np.load(emb, allow_pickle=True) as d:
            X = d['X']; time_to_lin = d['time_to_lin'].item(); lin_to_t_seg = d['lin_to_t_seg'].item()
            n_frames = int(d['n_frames'])
        if not np.isfinite(X).all():
            continue
        svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=0)
        Xs = svd.fit_transform(X)
        print(f'{tag}: {X.shape}, svd {time.time() - t0:.0f}s', flush=True)
        try:
            labelings, perms, final, conf = consensus_run(Xs, time_to_lin, lin_to_t_seg, 0, args.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            labelings, perms, final, conf = consensus_run(Xs, time_to_lin, lin_to_t_seg, 0, 'cpu')
        out = dict(tag=tag, round=os.path.basename(args.round), trial=trial, accuracy=rec['accuracy'],
                   train_config={k: rec.get('train_config', {}).get(k) for k in
                                 ('fusion', 'num_frames', 'epochs', 'lr', 'lambd_obj', 'self_layers')})
        out.update(metrics(labelings, perms, final, conf, lin_to_t_seg, n_frames))
        if args.second_seed:
            _, _, final1, _ = consensus_run(Xs, time_to_lin, lin_to_t_seg, 1, args.device)
            out['ari_seed1'] = float(adjusted_rand_score(final, final1))
        out['seconds'] = round(time.time() - t0, 1)
        with open(args.out, 'a') as f:
            f.write(json.dumps(out) + '\n')
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()
                    if k not in ('tag', 'round', 'train_config')}, flush=True)
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
