"""Experiment: untrained image-only vs position-fusion tracking accuracy.

Compares, with IDENTICAL backbone weights (paper untrained checkpoints):
  A) image-only backbone embeddings (paper baseline)
  B) backbone + random KeypointEncoder position fusion (new)

Pipeline per dataset: batched embed all frames -> SVD50 -> WormClusterTracker
-> label_propagation (paper mode) -> df -> rename_columns_using_matching vs GT
-> calculate_accuracy.

Usage (wbfm env):
  python exp_accuracy.py --lab zimmer --max_frames 50   # quick signal
  python exp_accuracy.py --lab zimmer                   # full run
Results appended as JSON lines to /tmp/claude/exp_results.jsonl
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

RESULTS = '/tmp/claude/exp_results.jsonl'

LABS = {
    'zimmer': dict(
        project='/tmp/claude/eval_projects/zimmer/project_config.yaml',
        gt='/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format/project_config.yaml',
        weights='/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/untrained_zimmer/trial_0/resnet50.pth',
    ),
    'flavell': dict(
        project='/tmp/claude/eval_projects/flavell/project_config.yaml',
        gt='/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/images_for_charlie/flavell_data.nwb',
        weights='/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/untrained_flavell/trial_0/resnet50.pth',
    ),
    'leifer': dict(
        project='/tmp/claude/eval_projects/leifer/project_config.yaml',
        gt='/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb',
        weights='/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/untrained_leifer/trial_0/resnet50.pth',
    ),
    'samuel': dict(
        project='/tmp/claude/eval_projects/samuel/project_config.yaml',
        gt='/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data/153.nwb',
        weights='/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/untrained_samuel/trial_0/resnet50.pth',
    ),
}


def log_result(rec):
    with open(RESULTS, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lab', required=True, choices=list(LABS))
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--mode', choices=['image', 'position', 'posonly', 'attention', 'trained', 'both'], default='both')
    ap.add_argument('--weights', default=None, help='trained checkpoint for mode=trained (load_barlow_model)')
    ap.add_argument('--cluster', choices=['global', 'labelprop'], default='labelprop')
    ap.add_argument('--num_seeds', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip_embed', action='store_true',
                    help='reuse saved embeddings from a previous run instead of re-embedding')
    ap.add_argument('--fuse_norm', action='store_true',
                    help='L2-normalize visual and position descriptors before adding (scale-matched fusion)')
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.utils.barlow import load_barlow_model
    from barlow_track.utils.barlow_superglue import BarlowWithPosition
    from barlow_track.utils.data_loading import get_bbox_data_for_volume_with_label
    from barlow_track.utils.volume_data import VolumeCoordsDataset, extract_crops, load_volume
    from barlow_track.utils.utils_tracking import WormClusterTracker
    from barlow_track.utils.utils_ground_truth import calculate_accuracy
    from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
    import torchio as tio

    spec = LABS[args.lab]
    t0 = time.time()
    project_data = ProjectData.load_final_project_data(spec['project'], allow_hybrid_loading=True, verbose=0)
    gpu, paper_model, margs = load_barlow_model(spec['weights'])
    # Force CPU: no GPU on this machine; keep device explicit
    device = torch.device('cpu')
    paper_model = paper_model.to(device).eval()
    target_sz = np.array(getattr(margs, 'target_sz', [margs.target_sz_z, margs.target_sz_xy, margs.target_sz_xy]))
    tmodel, targs, t_target_sz = None, None, None
    if args.weights:
        _, tmodel, targs = load_barlow_model(args.weights)
        tmodel = tmodel.to(device).eval()
        t_target_sz = np.array(getattr(targs, 'target_sz', [targs.target_sz_z, targs.target_sz_xy, targs.target_sz_xy]))
    if args.mode == 'trained':
        assert args.weights, '--mode trained requires --weights'
        target_sz = t_target_sz
    print(f"[{args.lab}] model target {list(target_sz)}, emb {margs.embedding_dim}", flush=True)

    # Fusion/attention models sharing the SAME backbone weights (isolates new components)
    from types import SimpleNamespace
    from barlow_track.utils.siamese import ResidualEncoder3D
    fargs = SimpleNamespace(embedding_dim=margs.embedding_dim, projector=margs.projector,
                            projector_final=margs.projector_final, lambd=0.0051, lambd_obj=0.67,
                            fusion='add', fusion_norm='none', keypoint_encoder_layers=[32, 64])
    fmodel = amodel = pmodel = None
    if args.mode != 'trained':
        fmodel = BarlowWithPosition(fargs, backbone=ResidualEncoder3D,
                                    in_channels=1, num_levels=2, f_maps=4, crop_sz=target_sz).to(device).eval()
        fmodel.backbone.load_state_dict(paper_model.backbone.state_dict())
    from barlow_track.utils.barlow_superglue import BarlowVolumeAttention
    aargs = SimpleNamespace(embedding_dim=margs.embedding_dim, projector=margs.projector,
                            projector_final=margs.projector_final, lambd=0.0051, lambd_obj=0.67,
                            fusion='concat', fusion_norm='layernorm', keypoint_encoder_layers=[32, 64],
                            self_layers=2)
    if args.mode != 'trained':
        amodel = BarlowVolumeAttention(aargs, backbone=ResidualEncoder3D,
                                       in_channels=1, num_levels=2, f_maps=4, crop_sz=target_sz).to(device).eval()
        amodel.backbone.load_state_dict(paper_model.backbone.state_dict())
        # NOTE: 'attention' now means the concat default; 'position' keeps legacy add-fusion.
        pmodel = BarlowWithPosition(
        SimpleNamespace(embedding_dim=margs.embedding_dim, projector=margs.projector,
                        projector_final=margs.projector_final, lambd=0.0051, lambd_obj=0.67,
                        fusion='position_only', fusion_norm='layernorm',
                        keypoint_encoder_layers=[32, 64]),
        backbone=ResidualEncoder3D, in_channels=1, num_levels=2, f_maps=4,
        crop_sz=target_sz).to(device).eval()

    normalizer = tio.RescaleIntensity(percentiles=(5, 100))
    n_frames = project_data.num_frames if args.max_frames is None else min(args.max_frames, project_data.num_frames)

    modes = ['image', 'position'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        emb_path = f'/tmp/claude/emb_{args.lab}_{mode}{"_norm" if args.fuse_norm and mode == "position" else ""}.npz'
        if args.skip_embed and os.path.exists(emb_path):
            d = np.load(emb_path, allow_pickle=True)
            X, time_to_lin, lin_to_t_seg, n_frames = (
                d['X'], d['time_to_lin'].item(), d['lin_to_t_seg'].item(), int(d['n_frames']))
            print(f"[{args.lab}/{mode}] loaded saved embeddings {X.shape}", flush=True)
            did_embed = False
        else:
            did_embed = True
            t1 = time.time()
            X_parts, time_to_lin, lin_to_t_seg = [], defaultdict(list), {}
            i_lin = 0
            frame_list = list(range(n_frames))
            for t in frame_list:
                vol = load_volume(project_data, t)
                crops_d, seg2name, _ = get_bbox_data_for_volume_with_label(
                    project_data, t, target_sz=target_sz, include_untracked=True)
                names = sorted(crops_d)
                if len(names) < 2:
                    continue  # no relative geometry; matches legacy len>1 filter; same frames skipped in both modes
                name_to_seg = {}
                for n in names:
                    if n in seg2name.values():
                        name_to_seg[n] = int([k for k, v in seg2name.items() if v == n][0])
                    else:
                        name_to_seg[n] = int(n.split('_')[-1])  # untracked_time_{t}_{ind}_{seg}
                crops = torch.from_numpy(np.stack([crops_d[n] for n in names]).astype(np.float32))
                crops = normalizer(crops).unsqueeze(1).float()
                # Keypoints for every non-image mode (trained models use them iff
                # the checkpoint actually contains a position branch).
                kpts = None
                if mode != 'image':
                    zxy = np.array([[r['z'], r['x'], r['y']] for r in
                                    _rows_for_names(project_data, t, names, name_to_seg)], dtype=np.float32)
                    kpts = VolumeCoordsDataset._normalize(torch.from_numpy(zxy), vol.shape)
                with torch.no_grad():
                    if mode == 'image':
                        emb = torch.cat([paper_model.backbone(crops[j:j + 32].to(device)).cpu()
                                         for j in range(0, len(crops), 32)], 0).numpy()
                    elif mode == 'trained':
                        # Best descriptor the checkpoint offers (position model if
                        # it was trained as one, else its backbone).
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
                    else:
                        # Position-family descriptors (pre-projector, 64-d): same dim as
                        # backbone, so the A/B isolates the new components.
                        # 'position': plain add-fusion (fuse_norm diagnostic optional).
                        # 'attention': layernorm fusion + intra-volume self-attention.
                        chunks = []
                        for j in range(0, len(crops), 32):
                            cj, kj = crops[j:j + 32].to(device), kpts[j:j + 32].to(device)
                            if mode == 'attention':
                                d = amodel.contextual_descriptors(cj, kj)
                            elif mode == 'posonly':
                                d = pmodel.fused_descriptors(cj, kj)
                            else:
                                d = fmodel.fused_descriptors(cj, kj)
                            if args.fuse_norm:
                                # scale-matched diagnostic: unit-norm each branch, then add
                                with torch.no_grad():
                                    v = fmodel.backbone(cj)
                                    p = fmodel.encode_position(kj)
                                    d = (v / v.norm(dim=1, keepdim=True).clamp_min(1e-6)
                                         + p / p.norm(dim=1, keepdim=True).clamp_min(1e-6))
                            chunks.append(d.cpu())
                        emb = torch.cat(chunks, 0).numpy()
                X_parts.append(emb)
                for n in names:
                    seg = name_to_seg[n]
                    try:
                        raw_ind = int(project_data.segmentation_metadata.mask_index_to_i_in_array(t, seg))
                    except (FileNotFoundError, IndexError, KeyError):
                        raw_ind = seg
                    time_to_lin[t].append(i_lin)
                    lin_to_t_seg[i_lin] = (t, raw_ind, seg)
                    i_lin += 1
        if did_embed:
            X = np.vstack(X_parts)
            print(f"[{args.lab}/{mode}] embedded {n_frames} frames, {X.shape} in {time.time()-t1:.0f}s", flush=True)
            np.savez(emb_path, X=X, time_to_lin=dict(time_to_lin), lin_to_t_seg=lin_to_t_seg, n_frames=n_frames)

        from sklearn.decomposition import TruncatedSVD
        Xs = TruncatedSVD(n_components=min(50, X.shape[1] - 1)).fit_transform(X)
        tracker = WormClusterTracker(Xs, dict(time_to_lin), linear_ind_to_t_and_seg_id=lin_to_t_seg)
        t2 = time.time()
        # Label propagation is the paper's final clustering step (global mode
        # only for quick debugging).
        if args.cluster == 'labelprop':
            df_pred = tracker.track_using_label_propagation_clusterer(num_seeds=args.num_seeds)
        else:
            df_pred = tracker.track_using_global_clusterer()
        print(f"[{args.lab}/{mode}] tracked in {time.time()-t2:.0f}s; df {df_pred.shape}", flush=True)
        from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
        df_pred = add_metadata_to_df_raw_ind(df_pred, project_data.segmentation_metadata)

        # Accuracy vs GT on raw_segmentation_id level (paper recipe)
        gt_data = ProjectData.load_final_project_data(spec['gt'], allow_hybrid_loading=True, verbose=0)
        df_gt = gt_data.get_final_tracks_only_finished_neurons()[0]
        if df_gt is None or df_gt.empty:
            df_gt = gt_data.final_tracks
        from barlow_track.utils.utils_ground_truth import pad_with_nan_rows
        max_len = max(len(df_gt), len(df_pred))
        df_pred = pad_with_nan_rows(df_pred, max_len)
        df_gt = pad_with_nan_rows(df_gt, max_len)
        df_pred_r, _, _, _ = rename_columns_using_matching(df_gt, df_pred, column='raw_segmentation_id')
        col_gt = df_gt.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
        col_pr = df_pred_r.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
        stats = calculate_accuracy(col_gt, col_pr)
        log_result(dict(lab=args.lab, mode=mode, seed=args.seed, n_frames=n_frames,
                        cluster=args.cluster, num_seeds=args.num_seeds,
                        fuse_norm=bool(args.fuse_norm and mode == 'position'),
                        accuracy=float(stats['accuracy']),
                        misses=int(stats['misses'].sum().sum()),
                        mismatches=int(stats['mismatches'].sum().sum()),
                        total=int(stats['total_ground_truth']),
                        minutes=(time.time() - t0) / 60))


def _rows_for_names(project_data, t, names, name_to_seg):
    row_data, col_names = project_data.segmentation_metadata.get_all_neuron_metadata_for_single_time(t, as_dataframe=False)
    import pandas as pd
    mdata = pd.DataFrame(dict(zip(col_names, row_data)))
    out = []
    for n in names:
        seg = name_to_seg[n]
        row = mdata[mdata['raw_segmentation_id'] == seg].iloc[0]
        out.append(row)
    return out


if __name__ == '__main__':
    main()
