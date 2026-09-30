"""Leifer A/B from the NWB itself: positions (GT xyz) + red_data, no segmentation needed.

Crops around each named neuron's GT xyz per frame, embeds with untrained_leifer
weights (image / position / attention modes, same 64-d descriptor space),
clustering (global, or labelprop with --cluster labelprop), accuracy vs GT on
raw_neuron_ind_in_list level.
Usage: python exp_leifer.py --mode attention [--cluster labelprop --num_seeds 10] [--skip_embed]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

RESULTS = '/tmp/claude/exp_results.jsonl'
NWB = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb'
WEIGHTS = '/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/untrained_leifer/trial_0/resnet50.pth'


def log_result(rec):
    with open(RESULTS, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['image', 'position', 'posonly', 'attention', 'trained'], required=True)
    ap.add_argument('--weights', default=None)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cluster', choices=['global', 'labelprop'], default='labelprop')
    ap.add_argument('--num_seeds', type=int, default=25)
    ap.add_argument('--skip_embed', action='store_true')
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.utils.barlow import load_barlow_model
    from barlow_track.utils.barlow_superglue import BarlowVolumeAttention, BarlowWithPosition
    from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid
    from barlow_track.utils.volume_data import VolumeCoordsDataset
    from barlow_track.utils.utils_tracking import WormClusterTracker
    from barlow_track.utils.utils_ground_truth import calculate_accuracy, pad_with_nan_rows
    from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
    from barlow_track.utils.siamese import ResidualEncoder3D
    from types import SimpleNamespace
    import torchio as tio

    t0 = time.time()
    p = ProjectData.load_final_project_data(NWB, verbose=0)
    device = torch.device('cpu')
    _, paper_model, margs = load_barlow_model(WEIGHTS)
    paper_model = paper_model.to(device).eval()
    target_sz = np.array([8, 64, 64])
    tmodel = None
    if args.weights:
        _, tmodel, targs = load_barlow_model(args.weights)
        tmodel = tmodel.to(device).eval()
        target_sz = np.array(getattr(targs, 'target_sz', [targs.target_sz_z, targs.target_sz_xy, targs.target_sz_xy]))
    if args.mode == 'trained':
        assert args.weights, '--mode trained requires --weights'
    D = margs.embedding_dim

    def _mk(cls, **kw):
        d = dict(embedding_dim=D, projector=margs.projector,
                 projector_final=margs.projector_final, lambd=0.0051, lambd_obj=0.67,
                 fusion='add', fusion_norm='none', keypoint_encoder_layers=[32, 64],
                 self_layers=2)
        d.update(kw)
        a = SimpleNamespace(**d)
        m = cls(a, backbone=ResidualEncoder3D, in_channels=1, num_levels=2, f_maps=4,
                crop_sz=target_sz).to(device).eval()
        m.backbone.load_state_dict(paper_model.backbone.state_dict())
        return m

    fmodel = amodel = pomodel = None
    if args.mode != 'trained':
        fmodel = _mk(BarlowWithPosition)
        amodel = _mk(BarlowVolumeAttention, fusion_norm='layernorm')
        pomodel = _mk(BarlowWithPosition, fusion='position_only', fusion_norm='layernorm')

    df_gt = p.final_tracks
    neurons = list(df_gt.columns.get_level_values(0).unique())
    # Vectorize GT access once (per-cell iloc on a 1536x395 MultiIndex is ~50ms)
    _gx = df_gt.loc[:, (slice(None), 'x')].values.astype(float)
    _gy = df_gt.loc[:, (slice(None), 'y')].values.astype(float)
    _gz = df_gt.loc[:, (slice(None), 'z')].values.astype(float)
    _gr = df_gt.loc[:, (slice(None), 'raw_neuron_ind_in_list')].values.astype(float)
    _gs = df_gt.loc[:, (slice(None), 'raw_segmentation_id')].values.astype(float)
    normalizer = tio.RescaleIntensity(percentiles=(5, 100))
    n_frames = p.num_frames if args.max_frames is None else min(args.max_frames, p.num_frames)

    t1 = time.time()
    emb_path = f'/tmp/claude/emb_leifer_{args.mode}.npz'
    if args.skip_embed and os.path.exists(emb_path):
        d = np.load(emb_path, allow_pickle=True)
        X, time_to_lin, lin_to_t_seg = d['X'], d['time_to_lin'].item(), d['lin_to_t_seg'].item()
        print(f"[leifer/{args.mode}] loaded saved embeddings {X.shape}", flush=True)
    else:
        X_parts, time_to_lin, lin_to_t_seg = [], defaultdict(list), {}
        i_lin = 0
        for t in range(n_frames):
            vol = np.asarray(p.red_data[t, ...], dtype=np.float32)
            sz = np.array([1, *vol.shape])
            names, crops_l, zxy_l, meta_l = [], [], [], []
            fin = np.isfinite(_gx[t]) & np.isfinite(_gy[t]) & np.isfinite(_gz[t])
            for j in np.flatnonzero(fin):
                z, x, y = float(_gz[t, j]), float(_gx[t, j]), float(_gy[t, j])
                dat, _ = get_3d_crop_using_bbox_or_centroid([z, x, y], sz, target_sz, vol)
                names.append(neurons[j])
                crops_l.append(dat)
                zxy_l.append([z, x, y])
                meta_l.append((int(_gr[t, j]), int(_gs[t, j])))
            if len(names) < 2:
                continue
            crops = normalizer(torch.from_numpy(np.stack(crops_l))).unsqueeze(1).float()
            kpts = VolumeCoordsDataset._normalize(
                torch.from_numpy(np.array(zxy_l, dtype=np.float32)), vol.shape)
            with torch.no_grad():
                if args.mode == 'image':
                    emb = torch.cat([paper_model.backbone(crops[j:j + 32].to(device)).cpu()
                                     for j in range(0, len(crops), 32)], 0).numpy()
                elif args.mode == 'trained':
                    if hasattr(tmodel, 'contextual_descriptors'):
                        emb = torch.cat([tmodel.contextual_descriptors(
                            crops[j:j + 32].to(device),
                            kpts[j:j + 32].to(device)).cpu()
                            for j in range(0, len(crops), 32)], 0).numpy()
                    elif hasattr(tmodel, 'fused_descriptors'):
                        emb = torch.cat([tmodel.fused_descriptors(
                            crops[j:j + 32].to(device),
                            kpts[j:j + 32].to(device)).cpu()
                            for j in range(0, len(crops), 32)], 0).numpy()
                    else:
                        emb = torch.cat([tmodel.backbone(crops[j:j + 32].to(device)).cpu()
                                         for j in range(0, len(crops), 32)], 0).numpy()
                elif args.mode == 'position':
                    emb = torch.cat([fmodel.fused_descriptors(crops[j:j + 32].to(device),
                                                              kpts[j:j + 32].to(device)).cpu()
                                     for j in range(0, len(crops), 32)], 0).numpy()
                elif args.mode == 'posonly':
                    emb = torch.cat([pomodel.fused_descriptors(crops[j:j + 32].to(device),
                                                               kpts[j:j + 32].to(device)).cpu()
                                     for j in range(0, len(crops), 32)], 0).numpy()
                else:
                    emb = torch.cat([amodel.contextual_descriptors(crops[j:j + 32].to(device),
                                                                   kpts[j:j + 32].to(device)).cpu()
                                     for j in range(0, len(crops), 32)], 0).numpy()
            X_parts.append(emb)
            for (raw_ind, seg) in meta_l:
                time_to_lin[t].append(i_lin)
                lin_to_t_seg[i_lin] = (t, raw_ind, seg)
                i_lin += 1
        X = np.vstack(X_parts)
        print(f"[leifer/{args.mode}] embedded {n_frames} frames, {X.shape} in {time.time()-t1:.0f}s", flush=True)
        np.savez(emb_path, X=X, time_to_lin=dict(time_to_lin), lin_to_t_seg=lin_to_t_seg)

    from sklearn.decomposition import TruncatedSVD
    Xs = TruncatedSVD(n_components=min(50, X.shape[1] - 1)).fit_transform(X)
    tracker = WormClusterTracker(Xs, dict(time_to_lin), linear_ind_to_t_and_seg_id=lin_to_t_seg)
    t2 = time.time()
    if args.cluster == 'labelprop':
        df_pred = tracker.track_using_label_propagation_clusterer(num_seeds=args.num_seeds)
    else:
        df_pred = tracker.track_using_global_clusterer()
    print(f"[leifer/{args.mode}/{args.cluster}] tracked in {time.time()-t2:.0f}s; df {df_pred.shape}", flush=True)

    col = 'raw_neuron_ind_in_list'
    max_len = max(len(df_gt), len(df_pred))
    df_pred = pad_with_nan_rows(df_pred, max_len)
    df_gt_p = pad_with_nan_rows(df_gt, max_len)
    df_r, _, _, _ = rename_columns_using_matching(df_gt_p, df_pred, column=col)
    cgt = df_gt_p.loc[:, (slice(None), col)].droplevel(1, axis=1)
    cpr = df_r.loc[:, (slice(None), col)].droplevel(1, axis=1)
    stats = calculate_accuracy(cgt, cpr)
    log_result(dict(lab='leifer', mode=args.mode, cluster=args.cluster, seed=args.seed,
                    n_frames=n_frames, accuracy=float(stats['accuracy']),
                    misses=int(stats['misses'].sum().sum()),
                    mismatches=int(stats['mismatches'].sum().sum()),
                    total=int(stats['total_ground_truth']),
                    minutes=(time.time() - t0) / 60))


if __name__ == '__main__':
    main()
