"""GT-free self-consistency of a label-propagation tracking run.

Label propagation tracks from several seed volumes and spectral synchronization merges the per-seed
labelings into one consensus (WormClusterTracker.track_using_label_propagation_clusterer). How well the
individual runs agree with that consensus needs no ground truth, and on the Samuel recordings it ranks
networks by tracking accuracy (Spearman 0.93-0.94 over 76-109 GT-tuned sweep trials, 0.87-0.89 after
controlling for the swept settings; validation loss: at most ~0.5), see
barlow_track/scripts/scratch/gt_free_proxy/. On near-saturated recordings (Zimmer 1210) it separates
broken from working networks but plateaus among the good ones.

Metrics (labelprop_consistency):
  agree_mean / agree_p10  fraction of objects whose top label in one run, mapped to the consensus by
                          that run's spectral-sync permutation, equals the consensus label; mean and
                          10th percentile over runs (objects without a consensus label left out)
  conf_mean / conf_med    consensus confidence (spectral sync) of the objects with a consensus label
  assigned_frac           fraction of objects with a consensus label
  dup_frac                fraction of (volume, label) cells claimed by 2+ objects
  complete_tracks_50/90   labels present in >= 50% / 90% of volumes, over the median objects per volume
"""
import numpy as np

METRICS = ('agree_mean', 'agree_p10', 'conf_mean', 'conf_med', 'assigned_frac', 'dup_frac',
           'complete_tracks_50', 'complete_tracks_90')


def labelprop_consistency(labelings, perms, final_labels, confidence, lin_to_t, n_frames):
    """GT-free consistency metrics of one label-propagation tracking run.

    labelings      per-seed (N, k) label arrays from multi_seed_propagation; labels are 1-based,
                   0 or negative = none
    perms          per-seed permutations from spectral_sync_from_topk, indexed by run column
                   (label - 1), giving the consensus column (-1 = none)
    final_labels   (N,) consensus labels = consensus column + 1 (negative = none)
    confidence     (N,) consensus confidence
    lin_to_t       (N,) volume index of each object, or a mapping linear index -> (t, seg id)
    n_frames       number of volumes
    """
    final = np.asarray(final_labels)
    conf = np.asarray(confidence, dtype=float)
    if not isinstance(lin_to_t, np.ndarray):
        lin_to_t = np.array([lin_to_t[i][0] for i in range(len(final))])
    t = np.asarray(lin_to_t, dtype=np.int64)
    has = final >= 0
    agree = []
    for lab, perm in zip(labelings, perms):
        top = np.asarray(lab)
        top = top[:, 0] if top.ndim == 2 else top
        perm = np.asarray(perm)
        ok = (top >= 1) & (top - 1 < len(perm))
        mapped = np.full(len(top), -2)
        col = perm[top[ok] - 1]
        mapped[ok] = np.where(col >= 0, col + 1, -2)
        agree.append(float(np.mean(mapped[has] == final[has])) if has.any() else np.nan)
    agree = np.array(agree, dtype=float)
    cells = {}
    for ti, lab in zip(t[has], final[has]):
        cells[(ti, lab)] = cells.get((ti, lab), 0) + 1
    counts = np.array(list(cells.values()))
    per_label = {}
    for (_, lab) in cells:
        per_label[lab] = per_label.get(lab, 0) + 1
    cover = np.array(list(per_label.values()), dtype=float) / max(int(n_frames), 1)
    med_obj = float(np.median(np.bincount(t, minlength=int(n_frames)))) if len(t) else 0.0
    nan = float('nan')
    return dict(
        agree_mean=float(np.nanmean(agree)) if np.isfinite(agree).any() else nan,
        agree_p10=float(np.nanpercentile(agree, 10)) if np.isfinite(agree).any() else nan,
        conf_mean=float(np.mean(conf[has])) if has.any() else nan,
        conf_med=float(np.median(conf[has])) if has.any() else nan,
        assigned_frac=float(has.mean()) if len(has) else nan,
        dup_frac=float(np.mean(counts > 1)) if len(counts) else nan,
        complete_tracks_50=float((cover >= 0.5).sum() / med_obj) if med_obj else nan,
        complete_tracks_90=float((cover >= 0.9).sum() / med_obj) if med_obj else nan,
        n_runs=int(len(agree)), n_objects=int(len(final)), median_objects=med_obj)


def tracker_consistency(tracker, lin_to_t, n_frames):
    """labelprop_consistency of a WormClusterTracker's last spectral-sync label-propagation run, or None."""
    runs = getattr(tracker, 'labelprop_runs', None)
    if not runs:
        return None
    return labelprop_consistency(runs['labelings'], runs['perms'], runs['final_labels'], runs['confidence'],
                                 lin_to_t, n_frames)
