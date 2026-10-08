"""Experiment: untrained image-only vs position-fusion tracking accuracy.

Compares, with IDENTICAL backbone weights (paper untrained checkpoints):
  A) image-only backbone embeddings (paper baseline)
  B) backbone + random KeypointEncoder position fusion (new)

Pipeline per dataset: batched embed all frames -> SVD50 -> WormClusterTracker
-> label_propagation (paper mode) -> df -> rename_columns_using_matching vs GT
-> calculate_accuracy.

Two frame sources (--source):
  project: crops via segmentation metadata (standard path; needs
      1-segmentation/metadata.pickle in the working copy).
  nwb: crops centered on GT xyz read straight from a GT NWB's red_data
      (for datasets whose segmentation is gone, e.g. leifer: its analyzed
      project's metadata.pickle no longer exists on disk, but the NWB
      carries both the volume and GT xyz). Works for any lab whose GT is
      an NWB with red_data + final_tracks (flavell, leifer, samuel).

Usage (wbfm env):
  python eval_accuracy.py --lab zimmer --max_frames 50   # quick signal
  python eval_accuracy.py --lab zimmer                   # full run
  python eval_accuracy.py --lab leifer --source nwb --mode attention
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


def log_result(rec, results_path=None):
    with open(results_path or RESULTS, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec, indent=1))


def _normalize_frame(emb, center, l2):
    """Per-frame descriptor normalization (eval-time collapse diagnostic).

    center: subtract the per-frame mean (removes a shared volume-mean
    component c_i ~= m + r_i). l2: row-wise unit-norm. Applied in that
    order when both are set; each is a no-op on empty input.
    """
    emb = np.asarray(emb, dtype=np.float64)
    if emb.size == 0:
        return emb
    if center:
        emb = emb - emb.mean(axis=0, keepdims=True)
    if l2:
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(n, 1e-12)
    return emb


def _trained_stage_emb(tmodel, crops, kpts, stage, device, chunk=32):
    """One frame of trained-model descriptors at the requested stage.

    backbone: tmodel.backbone(crops); fused: fused_descriptors (fallback
    backbone); contextual: contextual_descriptors (fallback fused, then
    backbone); projected: post-projector space via embed_with_position /
    embed; auto: best available (legacy behavior).

    NOTE: fused/contextual/projected-with-position run as a single
    full-volume forward (no chunking). KENC/GNN use InstanceNorm over N and
    attention softmax runs over N keys, so per-32 chunking is inexact (same
    blast radius as track_using_barlow.embed_volumes_with_position). The
    backbone-only path is per-sample (GroupNorm) and stays chunked to bound
    3D-conv memory. chunk is kept for back-compat and only applies there.
    """
    import torch as _torch

    def _cat_nokpts(fn):
        return _torch.cat([fn(crops[j:j + chunk].to(device)).cpu()
                           for j in range(0, len(crops), chunk)], 0)

    has_ctx = hasattr(tmodel, 'contextual_descriptors')
    has_fused = hasattr(tmodel, 'fused_descriptors')
    has_pos_embed = hasattr(tmodel, 'embed_with_position')
    if stage == 'auto':
        if has_ctx:
            stage = 'contextual'
        elif has_fused:
            stage = 'fused'
        else:
            stage = 'backbone'
    if stage == 'backbone':
        return _cat_nokpts(tmodel.backbone).numpy()
    _crops_d, _kpts_d = crops.to(device), kpts.to(device) if kpts is not None else None
    with _torch.no_grad():
        if stage == 'fused':
            if has_fused:
                return tmodel.fused_descriptors(_crops_d, _kpts_d).cpu().numpy()
            return _cat_nokpts(tmodel.backbone).numpy()
        if stage == 'contextual':
            if has_ctx:
                return tmodel.contextual_descriptors(_crops_d, _kpts_d).cpu().numpy()
            if has_fused:
                return tmodel.fused_descriptors(_crops_d, _kpts_d).cpu().numpy()
            return _cat_nokpts(tmodel.backbone).numpy()
        if stage == 'projected':
            if has_pos_embed:
                return tmodel.embed_with_position(_crops_d, _kpts_d).cpu().numpy()
            if hasattr(tmodel, 'embed'):
                return tmodel.embed(_crops_d).cpu().numpy()
            proj = tmodel.projector(tmodel.backbone(_crops_d)).cpu()
            return proj.numpy()
    raise ValueError(f"Unknown descriptor_stage '{stage}'")


def _centroid_boxes(zxy, target_sz, vol_shape, round_centroids=True):
    """Integer (z0, z1, x0, x1, y0, y1) boxes replicating
    data_loading.get_3d_crop_using_bbox_or_centroid (centroid input).

    round_centroids mirrors the 3-value branch (int(np.round)); False mirrors
    the 6-value branch with duplicated centroids (int((c + c) / 2), i.e.
    truncation). The project path uses duplicated centroids, the NWB path
    plain ones.
    """
    tz, tx, ty = int(target_sz[0]), int(target_sz[1]), int(target_sz[2])
    Z, X, Y = (int(s) for s in vol_shape)
    boxes = []
    for z, x, y in np.asarray(zxy, dtype=float):
        if round_centroids:
            zm, xm, ym = int(round(z)), int(round(x)), int(round(y))
        else:
            zm, xm, ym = int(z), int(x), int(y)
        z0 = min(max(zm - tz // 2, 0), Z); z1 = min(max(zm + tz // 2, 0), Z)
        if z1 - z0 > tz:
            z1 = z0 + tz
        x0 = min(max(xm - tx // 2, 0), X); x1 = min(max(xm + tx // 2, 0), X)
        if x1 - x0 > tx:
            x1 = x0 + tx
        y0 = min(max(ym - ty // 2, 0), Y); y1 = min(max(ym + ty // 2, 0), Y)
        if y1 - y0 > ty:
            y1 = y0 + ty
        boxes.append((z0, z1, x0, x1, y0, y1))
    return boxes


def _extract_crops_gpu(vol_t, zxy, target_sz, round_centroids=True):
    """GPU replica of volume_data.extract_crops: centroid boxes, pad-before zeros.

    vol_t: (Z, X, Y) tensor already on the target device. Returns
    (N, tz, tx, ty) on the same device. Bit-identical values to the numpy path
    (same box math, same zero padding); only the slicing device differs.
    """
    tz, tx, ty = int(target_sz[0]), int(target_sz[1]), int(target_sz[2])
    out = torch.zeros((len(zxy), tz, tx, ty), dtype=vol_t.dtype, device=vol_t.device)
    for i, (z0, z1, x0, x1, y0, y1) in enumerate(
            _centroid_boxes(zxy, target_sz, vol_t.shape, round_centroids)):
        crop = vol_t[z0:z1, x0:x1, y0:y1]
        if crop.numel():
            dz, dx, dy = tz - crop.shape[0], tx - crop.shape[1], ty - crop.shape[2]
            out[i, dz:, dx:, dy:] = crop
    return out


def _rescale_gpu(x, lo_pct=5.0, hi_pct=99.5):
    """GPU replica of tio RescaleIntensity(percentiles=(lo, hi)).

    Global cutoffs over the whole stack (as tio does), linear rescale,
    clipped to [0, 1]. Quantiles computed in float64 to match np.percentile.
    """
    with torch.no_grad():
        q = torch.quantile(x.detach().double().reshape(-1),
                           torch.tensor([lo_pct / 100.0, hi_pct / 100.0],
                                        dtype=torch.float64, device=x.device))
        lo, hi = q[0].to(x.dtype), q[1].to(x.dtype)
        denom = (hi - lo).clamp_min(1e-12)
        return ((x - lo) / denom).clamp_(0, 1)


def _embed_frame(mode, crops, kpts, models, tmodel, args, device, chunk=32):
    """Embed one frame's crops under the requested mode (no grad)."""
    paper_model, fmodel, amodel, pmodel = (models['paper'], models['fmodel'],
                                          models['amodel'], models['pmodel'])
    if mode == 'image':
        # Backbone-only: per-sample ops, chunking is exact.
        return torch.cat([paper_model.backbone(crops[j:j + chunk].to(device)).cpu()
                          for j in range(0, len(crops), chunk)], 0).numpy()
    if mode == 'trained':
        return _trained_stage_emb(tmodel, crops, kpts, args.descriptor_stage, device, chunk)
    # Position-family descriptors (pre-projector): same dim as backbone.
    # 'position': plain add-fusion (fuse_norm diagnostic optional).
    # 'attention': layernorm fusion + intra-volume self-attention.
    # 'posonly': pure geometry benchmark.
    # Full-volume forward (no chunking): InstanceNorm stats + attention
    # softmax both run over N; see _trained_stage_emb.
    cj, kj = crops.to(device), kpts.to(device)
    with torch.no_grad():
        if mode == 'attention':
            d = amodel.contextual_descriptors(cj, kj)
        elif mode == 'posonly':
            d = pmodel.fused_descriptors(cj, kj)
        else:
            d = fmodel.fused_descriptors(cj, kj)
        if args.fuse_norm:
            # scale-matched diagnostic: unit-norm each branch, then add
            v = fmodel.backbone(cj)
            p = fmodel.encode_position(kj)
            d = (v / v.norm(dim=1, keepdim=True).clamp_min(1e-6)
                 + p / p.norm(dim=1, keepdim=True).clamp_min(1e-6))
    return d.cpu().numpy()


def _iter_project_frames(project_data, target_sz, normalizer, frame_list, device=None):
    """Yield per-frame dicts (crops, kpts, meta) from segmentation crops.

    meta: list of (raw_ind, seg) aligned with crops rows. Frames with < 2
    detections are skipped (no relative geometry; matches legacy filter).
    With a CUDA device, the volume is transferred once per frame and crops +
    intensity normalization happen on the GPU (numerically identical to CPU).
    """
    from barlow_track.utils.data_loading import get_bbox_data_for_volume_with_label
    from barlow_track.utils.volume_data import VolumeCoordsDataset, load_volume
    use_gpu = device is not None and device.type == 'cuda'
    for t in frame_list:
        vol = load_volume(project_data, t)
        dat_d, seg2name, _ = get_bbox_data_for_volume_with_label(
            project_data, t, target_sz=target_sz, include_untracked=True,
            skip_crops=use_gpu)
        names = sorted(dat_d)
        if len(names) < 2:
            continue
        name_to_seg = {}
        for n in names:
            if n in seg2name.values():
                name_to_seg[n] = int([k for k, v in seg2name.items() if v == n][0])
            else:
                name_to_seg[n] = int(n.split('_')[-1])  # untracked_time_{t}_{ind}_{seg}
        if use_gpu:
            zxy = np.array([dat_d[n] for n in names], dtype=np.float32)
            vol_t = torch.from_numpy(np.ascontiguousarray(vol)).to(device, dtype=torch.float32)
            crops = _rescale_gpu(
                _extract_crops_gpu(vol_t, zxy, target_sz, round_centroids=False)).unsqueeze(1)
        else:
            crops = torch.from_numpy(np.stack([dat_d[n] for n in names]).astype(np.float32))
            crops = normalizer(crops).unsqueeze(1).float()
            zxy = np.array([[r['z'], r['x'], r['y']] for r in
                            _rows_for_names(project_data, t, names, name_to_seg)], dtype=np.float32)
        kpts = VolumeCoordsDataset._normalize(torch.from_numpy(zxy), vol.shape)
        meta = []
        for n in names:
            seg = name_to_seg[n]
            try:
                raw_ind = int(project_data.segmentation_metadata.mask_index_to_i_in_array(t, seg))
            except (FileNotFoundError, IndexError, KeyError):
                # Do NOT substitute the mask/segmentation id: it lives in a
                # different ID space than the array index and would misjoin
                # (or IndexError) in add_metadata_to_df_raw_ind, which
                # skips NaN raw_ind gracefully (counts as a miss).
                raw_ind = np.nan
            meta.append((raw_ind, seg))
        yield dict(t=t, crops=crops, kpts=kpts, meta=meta)


def _ensure_ind_level(df_gt):
    """Expose segmentation ids as 'raw_neuron_ind_in_list' when missing.

    Some NWBs (e.g. flavell) only carry raw_segmentation_id. The tracker
    output always uses a 'raw_neuron_ind_in_list' level, so expose the
    segmentation ids under that standard name to keep matching working.
    """
    import pandas as pd
    level1 = set(df_gt.columns.get_level_values(1).unique())
    if 'raw_neuron_ind_in_list' not in level1:
        if 'raw_segmentation_id' not in level1:
            raise ValueError(f"GT final_tracks has no usable neuron-id column; level1={sorted(level1)}")
        ids = df_gt.loc[:, (slice(None), 'raw_segmentation_id')]
        ids.columns = pd.MultiIndex.from_arrays(
            [ids.columns.get_level_values(0),
             ['raw_neuron_ind_in_list'] * len(ids.columns)])
        df_gt = pd.concat([df_gt, ids], axis=1)
    return df_gt


def _load_nwb_arrays(nwb_project):
    """Vectorized GT access for an NWB project (per-cell iloc is ~50ms)."""
    df_gt = _ensure_ind_level(nwb_project.final_tracks)
    neurons = list(df_gt.columns.get_level_values(0).unique())
    return dict(
        df_gt=df_gt,
        neurons=neurons,
        gx=df_gt.loc[:, (slice(None), 'x')].values.astype(float),
        gy=df_gt.loc[:, (slice(None), 'y')].values.astype(float),
        gz=df_gt.loc[:, (slice(None), 'z')].values.astype(float),
        gr=df_gt.loc[:, (slice(None), 'raw_neuron_ind_in_list')].values.astype(float),
        gs=df_gt.loc[:, (slice(None), 'raw_segmentation_id')].values.astype(float),
    )


def _iter_nwb_frames(nwb_project, arrays, target_sz, normalizer, frame_list, device=None):
    """Yield per-frame dicts (crops, kpts, meta) cropped around GT xyz.

    Same dict schema as _iter_project_frames; meta entries are
    (raw_neuron_ind_in_list, raw_segmentation_id) from the NWB directly.
    With a CUDA device, crops + intensity normalization happen on the GPU.
    """
    from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid
    from barlow_track.utils.volume_data import VolumeCoordsDataset
    use_gpu = device is not None and device.type == 'cuda'
    gx, gy, gz, gr, gs = arrays['gx'], arrays['gy'], arrays['gz'], arrays['gr'], arrays['gs']
    for t in frame_list:
        vol = np.asarray(nwb_project.red_data[t, ...], dtype=np.float32)
        sz = np.array([1, *vol.shape])
        zxy_l, meta_l = [], []
        fin = np.isfinite(gx[t]) & np.isfinite(gy[t]) & np.isfinite(gz[t]) \
            & np.isfinite(gr[t]) & np.isfinite(gs[t])
        for j in np.flatnonzero(fin):
            z, x, y = float(gz[t, j]), float(gx[t, j]), float(gy[t, j])
            zxy_l.append([z, x, y])
            meta_l.append((int(gr[t, j]), int(gs[t, j])))
        if len(zxy_l) < 2:
            continue
        zxy = np.array(zxy_l, dtype=np.float32)
        if use_gpu:
            vol_t = torch.from_numpy(np.ascontiguousarray(vol)).to(device)
            crops = _rescale_gpu(_extract_crops_gpu(vol_t, zxy, target_sz)).unsqueeze(1)
        else:
            crops_l = [get_3d_crop_using_bbox_or_centroid([z, x, y], sz, target_sz, vol)[0]
                       for z, x, y in zxy_l]
            crops = normalizer(torch.from_numpy(np.stack(crops_l))).unsqueeze(1).float()
        kpts = VolumeCoordsDataset._normalize(
            torch.from_numpy(zxy), vol.shape)
        yield dict(t=t, crops=crops, kpts=kpts, meta=meta_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lab', required=True, choices=list(LABS))
    ap.add_argument('--source', choices=['project', 'nwb'], default='project',
                    help="frame data source: 'project' crops via segmentation metadata; "
                         "'nwb' crops around GT xyz from the GT NWB itself (use for leifer, "
                         "whose segmentation metadata is gone)")
    ap.add_argument('--nwb', default=None,
                    help='override NWB path for --source nwb (default: the lab GT entry)')
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--mode', choices=['image', 'position', 'posonly', 'attention', 'trained', 'both'], default='both')
    ap.add_argument('--weights', default=None, help='trained checkpoint for mode=trained (load_barlow_model)')
    ap.add_argument('--cluster', choices=['global', 'labelprop'], default='labelprop')
    ap.add_argument('--num_seeds', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0,
                    help='single seed for every random step: head init, SVD, seed times, kNN graph')
    ap.add_argument('--skip_embed', action='store_true',
                    help='reuse saved embeddings from a previous run instead of re-embedding')
    ap.add_argument('--fuse_norm', action='store_true',
                    help='L2-normalize visual and position descriptors before adding (scale-matched fusion)')
    ap.add_argument('--descriptor_stage', choices=['auto', 'backbone', 'fused', 'contextual', 'projected'],
                    default='auto',
                    help='trained-model descriptor to evaluate (auto=best available: contextual>fused>backbone, legacy behavior)')
    ap.add_argument('--center_per_volume', action='store_true',
                    help='subtract the per-frame descriptor mean before saving embeddings')
    ap.add_argument('--l2_per_volume', action='store_true',
                    help='row-wise L2-normalize descriptors before saving embeddings')
    ap.add_argument('--device', default='cpu',
                    help="torch device for embedding (use 'cuda' on a GPU node)")
    ap.add_argument('--project', default=None,
                    help="override working-copy project for --source project (default: the lab entry)")
    ap.add_argument('--results_jsonl', default=None,
                    help='where to append result records (default: /tmp/claude/exp_results.jsonl)')
    ap.add_argument('--emb_dir', default=None,
                    help='where to cache embeddings (default: /tmp/claude)')
    ap.add_argument('--tag', default=None,
                    help='label recorded with each result and added to embedding cache filenames '
                         '(use e.g. the trial name so caches from different checkpoints do not collide)')
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    # One seed for everything. seed_all covers the parts that still read global
    # streams (random head init below, sklearn's SVD); the tracker and the kNN
    # graph get the seed explicitly (see barlow_track/utils/utils_seeding.py).
    from barlow_track.utils.utils_seeding import seed_all
    seed_all(args.seed)

    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.utils.barlow import load_barlow_model
    from barlow_track.utils.barlow_superglue import BarlowWithPosition
    from barlow_track.utils.utils_tracking import WormClusterTracker
    from barlow_track.utils.utils_ground_truth import calculate_accuracy
    from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
    import torchio as tio

    spec = LABS[args.lab]
    t0 = time.time()
    nwb_path = args.nwb or spec['gt']
    if args.source == 'nwb' and not str(nwb_path).endswith('.nwb'):
        raise ValueError(f"--source nwb needs an NWB file, got {nwb_path!r} (pass --nwb)")
    # Embedding source: working-copy project, or the GT NWB itself.
    project_path = args.project or spec['project']
    src_data = ProjectData.load_final_project_data(
        nwb_path if args.source == 'nwb' else project_path,
        allow_hybrid_loading=True, verbose=0)
    device = torch.device(args.device)
    tmodel, targs, t_target_sz = None, None, None
    if args.weights:
        _, tmodel, targs = load_barlow_model(args.weights)
        tmodel = tmodel.to(device).eval()
        t_target_sz = np.array(getattr(targs, 'target_sz', [targs.target_sz_z, targs.target_sz_xy, targs.target_sz_xy]))
    if args.mode == 'trained':
        assert args.weights, '--mode trained requires --weights'
        # The checkpoint carries its own architecture; the lab reference
        # weights are unused here, so don't load them (also avoids stale-arch
        # migration warnings for legacy reference checkpoints).
        paper_model, margs = None, targs
        target_sz = t_target_sz
    else:
        gpu, paper_model, margs = load_barlow_model(spec['weights'])
        paper_model = paper_model.to(device).eval()
        if args.source == 'nwb':
            # Old leifer-script convention: fixed crop size unless a trained
            # checkpoint dictates its own target.
            target_sz = np.array([8, 64, 64])
        else:
            target_sz = np.array(getattr(margs, 'target_sz', [margs.target_sz_z, margs.target_sz_xy, margs.target_sz_xy]))
    print(f"[{args.lab}/{args.source}] model target {list(target_sz)}, emb {margs.embedding_dim}", flush=True)

    # Fusion/attention models sharing the SAME backbone weights (isolates new components).
    # NOTE (measurement confound): only the backbone weights are loaded here;
    # KENC/GNN/projector heads are RANDOMLY INITIALIZED, and fmodel (add/none)
    # differs from amodel (concat/layernorm) in BOTH fusion type and norm. So
    # cross-mode position-family numbers compare untrained heads with a double
    # confound -- do not draw architecture conclusions from them. The
    # trustworthy trained-model comparison is --mode trained (with
    # --descriptor_stage backbone/fused/contextual/projected), which embeds
    # with the checkpoint's own heads and args.
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
    models = dict(paper=paper_model, fmodel=fmodel, amodel=amodel, pmodel=pmodel)

    normalizer = tio.RescaleIntensity(percentiles=(5, 99.5))  # in sync with training
    n_frames = src_data.num_frames if args.max_frames is None else min(args.max_frames, src_data.num_frames)
    frame_list = list(range(n_frames))
    if args.source == 'nwb':
        arrays = _load_nwb_arrays(src_data)
        frame_iter_fn = lambda: _iter_nwb_frames(src_data, arrays, target_sz, normalizer, frame_list,
                                                device=device)
    else:
        frame_iter_fn = lambda: _iter_project_frames(src_data, target_sz, normalizer, frame_list,
                                                     device=device)

    modes = ['image', 'position'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        suffix = f"{mode}_stage{args.descriptor_stage}" if mode == 'trained' and args.descriptor_stage != 'auto' else mode
        if args.fuse_norm and mode == 'position':
            suffix += "_norm"
        if args.center_per_volume:
            suffix += "_centered"
        if args.l2_per_volume:
            suffix += "_l2"
        emb_dir = args.emb_dir or '/tmp/claude'
        os.makedirs(emb_dir, exist_ok=True)
        tag_suffix = f"_{args.tag}" if args.tag else ""
        emb_path = os.path.join(emb_dir, f'emb_{args.lab}_{suffix}{tag_suffix}.npz')
        if args.skip_embed and os.path.exists(emb_path):
            d = np.load(emb_path, allow_pickle=True)
            X, time_to_lin, lin_to_t_seg = d['X'], d['time_to_lin'].item(), d['lin_to_t_seg'].item()
            # Older leifer-script caches lack n_frames; fall back to frame count.
            n_frames = int(d['n_frames']) if 'n_frames' in d else len(time_to_lin)
            print(f"[{args.lab}/{mode}] loaded saved embeddings {X.shape}", flush=True)
            did_embed = False
        else:
            did_embed = True
            t1 = time.time()
            X_parts, time_to_lin, lin_to_t_seg = [], defaultdict(list), {}
            i_lin = 0
            for fr in frame_iter_fn():
                t, crops, meta = fr['t'], fr['crops'], fr['meta']
                # Keypoints for every non-image mode (trained models use them iff
                # the checkpoint actually contains a position branch).
                kpts = fr['kpts'] if mode != 'image' else None
                with torch.no_grad():
                    emb = _embed_frame(mode, crops, kpts, models, tmodel, args, device)
                if args.center_per_volume or args.l2_per_volume:
                    emb = _normalize_frame(emb, args.center_per_volume, args.l2_per_volume)
                X_parts.append(emb)
                for (raw_ind, seg) in meta:
                    time_to_lin[t].append(i_lin)
                    lin_to_t_seg[i_lin] = (t, raw_ind, seg)
                    i_lin += 1
        if did_embed:
            X = np.vstack(X_parts)
            print(f"[{args.lab}/{mode}] embedded {n_frames} frames, {X.shape} in {time.time()-t1:.0f}s", flush=True)
            np.savez(emb_path, X=X, time_to_lin=dict(time_to_lin), lin_to_t_seg=lin_to_t_seg, n_frames=n_frames)
            # Fail here with the trial tag, not pages later inside sklearn:
            # non-finite embeddings mean a poisoned/NaN checkpoint.
            n_bad = int(np.isnan(X).sum())
            if n_bad:
                bad_rows = np.where(np.isnan(X).any(axis=1))[0]
                raise ValueError(
                    f"[{args.tag}] {n_bad} NaN entries in {len(bad_rows)} embedding rows "
                    f"(rows {bad_rows[0]}-{bad_rows[-1]}); refusing to track a NaN "
                    f"checkpoint (weights: {args.weights})")

        from sklearn.decomposition import TruncatedSVD
        Xs = TruncatedSVD(n_components=min(50, X.shape[1] - 1),
                          random_state=args.seed).fit_transform(X)
        tracker = WormClusterTracker(Xs, dict(time_to_lin), linear_ind_to_t_and_seg_id=lin_to_t_seg,
                                     seed=args.seed)
        t2 = time.time()
        # Label propagation is the paper's final clustering step (global mode
        # only for quick debugging).
        if args.cluster == 'labelprop':
            df_pred = tracker.track_using_label_propagation_clusterer(
                num_seeds=args.num_seeds, device=device if device.type == 'cuda' else None)
        else:
            df_pred = tracker.track_using_global_clusterer()
        print(f"[{args.lab}/{mode}] tracked in {time.time()-t2:.0f}s; df {df_pred.shape}", flush=True)

        if args.source == 'nwb':
            # GT ids come straight from the NWB; match on neuron index, no
            # segmentation-metadata join (GT lacks seg ids at match time).
            # Mirror the project path's finished-neuron semantics when the
            # source exposes them; otherwise fall back to full final_tracks.
            df_gt = arrays['df_gt']
            gt_filter = 'full final_tracks'
            try:
                _df_fin, _ = src_data.get_final_tracks_only_finished_neurons()
            except Exception:
                _df_fin = None
            if _df_fin is not None and not _df_fin.empty:
                df_gt = _ensure_ind_level(_df_fin)
                gt_filter = 'finished neurons'
            print(f"[{args.lab}/nwb] GT: {len(df_gt.columns.get_level_values(0).unique())} neurons "
                  f"({gt_filter}), {len(df_gt)} frames", flush=True)
            match_col = 'raw_neuron_ind_in_list'
        else:
            from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
            df_pred = add_metadata_to_df_raw_ind(df_pred, src_data.segmentation_metadata)
            # Accuracy vs GT on raw_segmentation_id level (paper recipe)
            gt_data = ProjectData.load_final_project_data(spec['gt'], allow_hybrid_loading=True, verbose=0)
            df_gt = gt_data.get_final_tracks_only_finished_neurons()[0]
            if df_gt is None or df_gt.empty:
                df_gt = gt_data.final_tracks
            print(f"[{args.lab}/project] GT: {len(df_gt.columns.get_level_values(0).unique())} neurons, "
                  f"{len(df_gt)} frames; pred: {len(df_pred)} frames", flush=True)
            match_col = 'raw_segmentation_id'
        from barlow_track.utils.utils_ground_truth import pad_with_nan_rows, align_gt_pred_time_index
        # Restrict GT to the actually-evaluated time range (e.g. --max_frames):
        # without this, every unevaluated GT frame counts as a miss and
        # accuracy scales as ~(n_frames / total_frames) * true_accuracy.
        df_gt, df_pred = align_gt_pred_time_index(df_gt, df_pred)
        max_len = max(len(df_gt), len(df_pred))
        df_pred = pad_with_nan_rows(df_pred, max_len)
        df_gt = pad_with_nan_rows(df_gt, max_len)
        df_pred_r, _, _, _ = rename_columns_using_matching(
            df_gt, df_pred, column=match_col, try_to_fix_inf=True)
        if 'unmatched_neuron' in df_pred_r.columns.get_level_values(0):
            # Defensive: wbfm drops these when pred has more columns than GT,
            # but a residual duplicate label would crash reindex below.
            df_pred_r = df_pred_r.drop(columns='unmatched_neuron')
        col_gt = df_gt.loc[:, (slice(None), match_col)].droplevel(1, axis=1)
        col_pr = df_pred_r.loc[:, (slice(None), match_col)].droplevel(1, axis=1)
        stats = calculate_accuracy(col_gt, col_pr)
        _fusion = {'image': None, 'position': 'add', 'attention': 'concat',
                   'posonly': 'position_only', 'trained': getattr(targs, 'fusion', None) if targs else None}.get(mode)
        train_config = None
        if mode == 'trained' and args.weights:
            # Record the exact training setup alongside the result: the
            # train_config.yaml saved next to the checkpoint (plain
            # yaml.safe_load types, so the record stays JSON-serializable).
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.weights)),
                                    'train_config.yaml')
            if os.path.isfile(cfg_path):
                import yaml
                with open(cfg_path) as f:
                    train_config = yaml.safe_load(f)
        log_result(dict(lab=args.lab, mode=mode, seed=args.seed, n_frames=n_frames,
                        source=args.source,
                        tag=args.tag, weights=os.path.abspath(args.weights) if args.weights else None,
                        train_config=train_config,
                        cluster=args.cluster, num_seeds=args.num_seeds,
                        fuse_norm=bool(args.fuse_norm and mode == 'position'),
                        fusion=_fusion,
                        head_init='trained' if mode == 'trained' else 'random-backbone-only',
                        descriptor_stage=args.descriptor_stage if mode == 'trained' else None,
                        center_per_volume=bool(args.center_per_volume),
                        l2_per_volume=bool(args.l2_per_volume),
                        accuracy=float(stats['accuracy']),
                        misses=int(stats['total_misses']),
                        mismatches=int(stats['total_mismatches']),
                        total=int(stats['total_ground_truth']),
                        minutes=(time.time() - t0) / 60),
                     args.results_jsonl)


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
