"""Does a trained position branch identify neurons between augmented views far better than between real frames?

local_geometry_change.py showed that the training views change neighbour distances less
than real motion does; this asks whether the network's position features exploit that.
For each fused checkpoint, embed a frame's detections with the position branch alone
(norm_pos(encode_position(kpts))), keypoints normalized as in training, and match two
point sets by Hungarian assignment on cosine distance:
  - augmented: the two training views of one frame, drawn with the checkpoint's own
    global_augment / position_augment (the keypoint half of
    VolumeCoordsDataset._augmented_view; no crops needed);
  - real: frames t and t + lag, as embedded at eval (no augmentation).
Score: fraction of GT neurons (identified by raw_segmentation_id) present in both sets
that are assigned to themselves. Every checkpoint is also scored with its position
branch re-initialized (same architecture, seed 0), as a reference for how much the
geometry gives an untrained encoder.

Usage (wbfm env):
  python position_branch_matching.py --weights A/resnet50.pth B/resnet50.pth --out out.jsonl
"""
import argparse
import copy
import json
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

PROJECT = os.path.expanduser('~/.cache/barlow_scratch/local_data/flavell/2025_07_01/project_config.yaml')


def augmented_keypoints(points, shape, rng, global_args, position_args):
    """(kept points (K, 3), kept original indices (K,)) for one training view, keypoints only."""
    from barlow_track.utils.volume_data import (
        apply_position_jitter, in_bounds_mask, sample_dropout_keep, sample_elastic_field,
        sample_global_affine, split_global_args, transform_points)
    affine_args, elastic_args = split_global_args(global_args)
    to_volume = bool((global_args or {}).get('apply_to_volume', True))
    R, t_vec = sample_global_affine(rng, **affine_args)
    field = sample_elastic_field(rng, shape, **elastic_args)
    pts = transform_points(points, shape, R, t_vec, field=field)
    pts = apply_position_jitter(pts, rng, position_args.get('jitter_std', 0.0))
    valid = in_bounds_mask(pts, shape) if to_volume else np.ones(len(pts), dtype=bool)
    valid_idx = np.where(valid)[0]
    _, keep_rel = sample_dropout_keep(len(valid_idx), rng, dropout_p=position_args.get('dropout_p', 0.0),
                                      min_keep=position_args.get('min_keep', 2))
    return pts[valid_idx][keep_rel], valid_idx[keep_rel]


def embed(model, points, shape, device):
    """(N, 3) voxel keypoints -> (N, D) unit-norm position descriptors, standardized per frame.

    The raw descriptors share a large per-frame component (cosine ~0.99 between
    neurons); the Barlow loss standardizes each feature over a frame's objects, so
    do the same before comparing.
    """
    from barlow_track.utils.volume_data import VolumeCoordsDataset
    kpts = VolumeCoordsDataset._normalize(torch.as_tensor(points, dtype=torch.float32), shape).to(device)
    with torch.no_grad():
        d = model.norm_pos(model.encode_position(kpts))
    d = (d - d.mean(0, keepdim=True)) / d.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return F.normalize(d, dim=1).cpu().numpy()


def match_accuracy(da, db, ids_a, ids_b, gt_ids):
    """Fraction of gt_ids present in both sets that the Hungarian assignment maps to themselves."""
    rows, cols = linear_sum_assignment(1 - da @ db.T)
    assigned = {ids_a[r]: ids_b[c] for r, c in zip(rows, cols)}
    shared = [i for i in gt_ids if i in set(ids_a) and i in set(ids_b)]
    if not shared:
        return np.nan, 0
    return float(np.mean([assigned.get(i) == i for i in shared])), len(shared)


def reinit_position_branch(model, seed=0):
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    for mod in list(m.visnet_enc.modules()) + list(m.norm_pos.modules()):
        if hasattr(mod, 'reset_parameters') and mod is not m.visnet_enc:
            mod.reset_parameters()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--project', default=PROJECT)
    ap.add_argument('--lags', type=int, nargs='+', default=[1, 10, 100, 500, 1000])
    ap.add_argument('--pairs', type=int, default=100, help='frame pairs per lag, and frames for augmented views')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='jsonl of per-checkpoint results')
    args = ap.parse_args()
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_project_gt
    from barlow_track.utils.barlow import load_barlow_model
    from barlow_track.utils.volume_data import get_centroids_for_volume

    project_data = ProjectData.load_final_project_data(args.project, allow_hybrid_loading=True, verbose=0)
    shape = tuple(project_data.red_data.shape[1:])
    df = _load_project_gt(args.project)
    names = list(df.columns.get_level_values(0).unique())
    gt_seg = df.loc[:, [(n, 'raw_segmentation_id') for n in names]]
    gt_seg_at = {int(t): set(int(s) for s in row if np.isfinite(s)) for t, row in zip(df.index, gt_seg.values)}
    frames = sorted(gt_seg_at)

    rng = np.random.RandomState(args.seed)
    aug_frames = list(rng.choice(frames, size=args.pairs, replace=False))
    real_pairs = {lag: [(int(t), int(t) + lag) for t in rng.choice([t for t in frames if t + lag in gt_seg_at],
                                                                  size=args.pairs, replace=False)]
                  for lag in args.lags}
    needed = set(aug_frames) | {t for pairs in real_pairs.values() for p in pairs for t in p}
    dets = {t: get_centroids_for_volume(project_data, t) for t in sorted(needed)}
    print(f'{len(names)} GT neurons, {len(frames)} frames, shape {shape}; '
          f'median detections {np.median([len(dets[t][1]) for t in dets]):.0f}, '
          f'median GT per frame {np.median([len(gt_seg_at[t]) for t in frames]):.0f}')

    for weights in args.weights:
        device, model, margs = load_barlow_model(weights)
        model.eval()
        global_args = getattr(margs, 'global_augment', None) or {}
        position_args = getattr(margs, 'position_augment', None) or {}
        for label, m in [('trained', model), ('reinit', reinit_position_branch(model).eval())]:
            aug_rng = np.random.RandomState(args.seed + 1)
            aug = []
            for t in aug_frames:
                zxy, seg = dets[t]
                pa, ia = augmented_keypoints(zxy, shape, aug_rng, global_args, position_args)
                pb, ib = augmented_keypoints(zxy, shape, aug_rng, global_args, position_args)
                acc, n = match_accuracy(embed(m, pa, shape, device), embed(m, pb, shape, device),
                                        list(seg[ia]), list(seg[ib]), gt_seg_at[t])
                aug.append(acc)
            real = {}
            for lag, pairs in real_pairs.items():
                accs = []
                for t1, t2 in pairs:
                    (z1, s1), (z2, s2) = dets[t1], dets[t2]
                    # Segmentation ids are per frame; carry GT identity across frames by neuron name.
                    row1, row2 = df.loc[t1, (slice(None), 'raw_segmentation_id')], df.loc[t2, (slice(None), 'raw_segmentation_id')]
                    to_name1 = {int(v): n for (n, _), v in row1.items() if np.isfinite(v)}
                    to_name2 = {int(v): n for (n, _), v in row2.items() if np.isfinite(v)}
                    ids1 = [to_name1.get(int(s), ('det1', int(s))) for s in s1]
                    ids2 = [to_name2.get(int(s), ('det2', int(s))) for s in s2]
                    acc, n = match_accuracy(embed(m, z1, shape, device), embed(m, z2, shape, device),
                                            ids1, ids2, set(to_name1.values()) & set(to_name2.values()))
                    accs.append(acc)
                real[lag] = float(np.nanmean(accs))
            # Participation ratio of the descriptor spectrum: how many dimensions carry neuron identity.
            sv = np.linalg.svd(embed(m, dets[aug_frames[0]][0], shape, device), compute_uv=False)
            rec = dict(weights=weights, branch=label, aug_views=float(np.nanmean(aug)),
                       eff_rank=float((sv ** 2).sum() ** 2 / (sv ** 4).sum()),
                       real_by_lag={str(k): v for k, v in real.items()}, pairs=args.pairs, seed=args.seed,
                       global_augment=global_args, position_augment=position_args)
            print(json.dumps({k: rec[k] for k in ('weights', 'branch', 'aug_views', 'eff_rank', 'real_by_lag')}), flush=True)
            if args.out:
                with open(args.out, 'a') as f:
                    f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
