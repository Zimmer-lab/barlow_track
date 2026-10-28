"""Frame-pair matching accuracy: a cheap tracking proxy to log during training.

The Barlow validation loss scores augmented view pairs of held-out frames under the
training augmentation, so it measures invariance to synthetic transforms and is not
comparable across augmentation settings. This instead embeds unaugmented crops of
frames the training never sampled, exactly as eval_accuracy --mode trained does
(same crops, normalization and descriptor stage), matches every object of one frame
to one of another by optimal assignment on cosine distance (untracked objects
included, as distractors), and scores the GT-named neurons present in both frames.
Frames are far apart in time, so this is closer to what label propagation needs
than consecutive-frame matching. Draws no random numbers.
"""
import itertools
import logging

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def held_out_frames(num_frames, used_frames, n):
    """n frames evenly spaced over the frames not in used_frames (sorted)."""
    used = set(int(t) for t in used_frames)
    free = [t for t in range(int(num_frames)) if t not in used]
    if n <= 0 or not free:
        return []
    idx = np.unique(np.round(np.linspace(0, len(free) - 1, min(n, len(free)))).astype(int))
    return [free[i] for i in idx]


def _seg_to_name(project_data, t):
    tracks = project_data.final_tracks
    if tracks is None:
        return {}
    try:
        row = tracks.loc[t, (slice(None), 'raw_segmentation_id')].droplevel(1)
    except KeyError:
        return {}
    return {int(v): k for k, v in row.items() if v == v}


def _iter_keypoint_frames(project_data, frames):
    """_iter_project_frames for position-only models: metadata centroids, no volumes.

    Keypoints are every detection's centroid normalized by the volume shape, as the
    training views see them (keypoints_only); crops are (N, 1, 1, 1, 1) placeholders.
    """
    from barlow_track.utils.volume_data import VolumeCoordsDataset, get_centroids_for_volume
    shape = tuple(int(s) for s in project_data.red_data.shape[1:])
    for t in frames:
        zxy, seg = get_centroids_for_volume(project_data, int(t))
        if len(zxy) < 2:
            continue
        kpts = VolumeCoordsDataset._normalize(torch.from_numpy(zxy.astype(np.float32)), shape)
        yield dict(t=t, crops=torch.zeros((len(zxy), 1, 1, 1, 1)), kpts=kpts,
                   meta=[(None, int(s)) for s in seg])


def frame_pair_matching_accuracy(model, project_data, frames, target_sz, device, stages=('auto',),
                                 crop_window='pad', round_centers=False, keypoints_only=False):
    """{stage: mean accuracy over frame pairs} and the number of pairs scored.

    A pair's accuracy is the fraction of GT-named neurons present in both frames
    whose assigned partner carries the same name. Restores the model's train/eval
    mode afterwards. crop_window: GPU edge crops, as eval_accuracy --crop_window;
    round_centers: as eval_accuracy --round_crop_centers. keypoints_only (position-only
    models, off by default): embed metadata keypoints without loading volumes.
    """
    import torchio as tio
    from barlow_track.scripts.eval_accuracy import _iter_project_frames, _trained_stage_emb
    if keypoints_only and 'backbone' in stages:
        # Placeholder crops have no image to embed; every stage would be lost with it
        raise TypeError("keypoints_only frames carry no crops: drop the 'backbone' stage")
    normalizer = tio.RescaleIntensity(percentiles=(5, 99.5))  # in sync with training and eval
    was_training = model.training
    model.eval()
    per_frame = []  # (names (N,), {stage: unit-norm embedding (N, D)})
    frame_iter = _iter_keypoint_frames(project_data, frames) if keypoints_only else _iter_project_frames(
        project_data, target_sz, normalizer, frames, device=device,
        crop_window=crop_window, round_centers=round_centers)
    try:
        with torch.no_grad():
            for f in frame_iter:
                seg2name = _seg_to_name(project_data, f['t'])
                names = np.array([seg2name.get(int(seg)) for _, seg in f['meta']], dtype=object)
                embs = {}
                for stage in stages:
                    e = np.asarray(_trained_stage_emb(model, f['crops'], f['kpts'], stage, device), dtype=np.float64)
                    embs[stage] = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
                per_frame.append((names, embs))
    finally:
        model.train(was_training)
    out = {stage: [] for stage in stages}
    for (na, ea), (nb, eb) in itertools.combinations(per_frame, 2):
        shared = set(n for n in na if n is not None) & set(n for n in nb if n is not None)
        if not shared:
            continue
        for stage in stages:
            rows, cols = linear_sum_assignment(1.0 - ea[stage] @ eb[stage].T)
            correct = sum(1 for r, c in zip(rows, cols) if na[r] in shared and na[r] == nb[c])
            out[stage].append(correct / len(shared))
    n_pairs = len(out[stages[0]])
    if n_pairs == 0:
        logging.warning("frame_pair_matching_accuracy: no frame pair shares a GT-named neuron")
    return {s: (float(np.mean(v)) if v else float('nan')) for s, v in out.items()}, n_pairs
