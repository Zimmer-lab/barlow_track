"""Competitor baseline: track a GT recording with the published fDNC model and score it.

fDNC (Yu et al. 2021) matches the neuron point cloud of one volume to a template
point cloud (positions only). Tracking = match every frame to a template frame of
the same recording; each template neuron is one track. This script:

1. reads per-frame neuron positions from the same objects our eval embeds
   (project source: every segmented object, index = raw_neuron_ind_in_list;
   nwb source: the GT objects of the NWB, as eval_accuracy --source nwb does),
2. converts zxy voxels -> fDNC units (xyz, 1 unit = 84 um, mean-centred; the
   wbfm wrapper's PhysicalUnitConversion.zimmer2leifer),
3. runs the pretrained fDNC model (fDNC/model/model.bin, nh128 nl6) for each
   frame vs the template, with fDNC's own Hungarian assignment,
4. builds a track dataframe (columns = template neurons, values =
   raw_neuron_ind_in_list), and scores it with the exact steps of
   eval_accuracy.run_eval (align/pad, match_columns_to_gt 'nandist',
   calculate_accuracy).

With several --templates, each template is tracked and scored separately, and
the per-template matches are also combined by wbfm's multi-template vote
(combine_multiple_template_matches) after mapping every template's neurons to
the first template's neurons via the first template's own fDNC match of the
other template frame (so no GT is used for the combination).

--nwb_objects all tracks every detection of the NWB's raw_segmentation (ids =
mask labels) and maps the GT to mask labels exactly as eval_accuracy
--nwb_objects all does (the revision's Flavell setting). --reuse_dir rescores
cached <tag>_<variant>_df.pickle track tables with the current scoring code
after checking each against the loaded detections (no tracking).

Outputs (never inside the project): <out_dir>/<tag>_*.pickle and a record per
variant appended to <out_dir>/fdnc_results.jsonl; per recording also
<dataset>_detections.npz (per-frame detection ids / positions) and
<dataset>_scored_gt.h5 (key gt: the GT matrix the score is computed against,
same layout as eval_accuracy --save_tracks), for the pairing check.

Run with:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/charles/Current_work/repos/fDNC_Neuron_ID:<barlow_track repo> \
  python fdnc_track_and_score.py --dataset zimmer_1128 --templates 10
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FDNC_ROOT = '/home/charles/Current_work/repos/fDNC_Neuron_ID'
DEFAULT_MODEL = os.path.join(FDNC_ROOT, 'fDNC', 'model', 'model.bin')
OUT_DIR = '/home/charles/.cache/barlow_scratch/fdnc'


# ---------------------------------------------------------------- fDNC core
def load_fdnc_model(model_path=DEFAULT_MODEL):
    """Exactly the model fDNC.src.DNC_predict._encode_using_pytorch builds, loaded once."""
    from fDNC.src.model import NIT_Registration
    model = NIT_Registration(input_dim=3, n_hidden=128, n_layer=6, p_rotate=0, feat_trans=0, cuda=False)
    params = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params['state_dict'])
    return model.to(torch.device('cpu')).eval()


def fdnc_match(model, template_pos, test_pos):
    """fDNC's predict_matches without reloading the model per call.

    Mirrors _encode_using_pytorch (no colour) + _calculate_matches_from_encoding +
    matches2indices. Returns a list of (template_idx, test_idx, prob), one per
    matched test neuron (test neurons left out by the Hungarian assignment when
    there are more test than template neurons are dropped).
    """
    from fDNC.src.DNC_predict import _calculate_matches_from_encoding
    pt_batch = [template_pos[:, :3], test_pos[:, :3]]
    with torch.no_grad():
        _, output_pairs = model(pt_batch, match_dict=None, ref_idx=0, mode='eval')
    num_neui = len(pt_batch[1])
    p_m = output_pairs['p_m'][1].detach().cpu().numpy()[:num_neui, :]
    col, p_m, prob_m, row = _calculate_matches_from_encoding(0, p_m)
    return [(int(c), int(r), float(prob_m[r, c])) for r, c in zip(row, col)]


ORIENTS = ('default', 'swap', 'mirror', 'swap_mirror')


def fdnc_reference_rms():
    """Median RMS radius (fDNC units) of fDNC's 11 real NeuroPAL reference worms.

    Data/test_neuropal_all_features/{*.data, mov/*.data}, loaded with fDNC's own
    pre_matt (which converts to the model's units). Used to rescale another
    microscope's point clouds to the size the model was trained/validated on,
    without looking at any GT.
    """
    import glob
    from fDNC.src.DNC_predict import pre_matt
    d = os.path.join(FDNC_ROOT, 'Data', 'test_neuropal_all_features')
    fs = sorted(glob.glob(os.path.join(d, '*.data')) + glob.glob(os.path.join(d, 'mov', '*.data')))
    r = []
    for f in fs:
        p = pre_matt(f)['pts'][:, :3]
        p = p - p.mean(0)
        r.append(np.sqrt((p ** 2).sum(1).mean()))
    return float(np.median(r)), len(fs)


def rms_radius(pts):
    p = pts - pts.mean(0)
    return float(np.sqrt((p ** 2).sum(1).mean()))


def to_fdnc_units(zxy, um_xy, um_z, orient='default', scale=1.0):
    """zxy voxels -> fDNC xyz (1 unit = 84 um, mean-centred).

    orient 'default' is wbfm's PhysicalUnitConversion.zimmer2leifer: output
    columns (input y, input x, z). 'swap' = (x, y, z); 'mirror' flips the sign of
    z; 'swap_mirror' both. These are the four axis conventions that are not
    related by an in-plane rotation: the two chiralities times the two sides a
    worm can lie on (fDNC's README: a left-side worm is rotated 180 deg by
    negating x and z).
    """
    zxy = np.asarray(zxy, dtype=float)
    if orient not in ORIENTS:
        raise ValueError(orient)
    if orient.startswith('swap'):
        out = np.stack([zxy[:, 1] * um_xy, zxy[:, 2] * um_xy, zxy[:, 0] * um_z], 1) / 84.0
    else:
        out = np.stack([zxy[:, 2] * um_xy, zxy[:, 1] * um_xy, zxy[:, 0] * um_z], 1) / 84.0
    out -= out.mean(0)
    if orient.endswith('mirror'):
        out[:, 2] *= -1
    return out * scale


# ---------------------------------------------------------------- data access
def load_frames(args):
    """Per-frame object positions and ids.

    Returns (frames, df_gt, match_col, src_data, info) where frames[t] =
    (zxy (M,3) voxels, ids (M,)) and ids live in the match_col ID space of the GT
    (project: raw_neuron_ind_in_list of the segmentation metadata, converted to
    raw_segmentation_id later via add_metadata_to_df_raw_ind; nwb: the NWB's
    raw_neuron_ind_in_list of the GT objects, as eval_accuracy --source nwb).
    """
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_project_gt, _load_nwb_arrays, _ensure_ind_level
    path = args.nwb if args.source == 'nwb' else args.project
    src = ProjectData.load_final_project_data(path, allow_hybrid_loading=True, verbose=0)
    n_frames = src.num_frames if args.max_frames is None else min(args.max_frames, src.num_frames)
    frames = {}
    info = {}
    if args.source == 'project':
        sm = src.segmentation_metadata
        n_bad = 0
        for t in range(n_frames):
            zxy = np.asarray(src.get_centroids_as_numpy(t), dtype=float).reshape(-1, 3)
            frames[t] = (zxy, np.arange(len(zxy)))
            if t % 200 == 0:
                # check: row i of get_centroids_as_numpy is raw_neuron_ind_in_list i
                row, cols = sm.get_all_neuron_metadata_for_single_time(t, as_dataframe=False)
                m = pd.DataFrame(dict(zip(cols, row)))
                m = m.sort_values('raw_neuron_ind_in_list')
                ok = (len(m) == len(zxy)
                      and np.allclose(m[['z', 'x', 'y']].values.astype(float), zxy)
                      and np.array_equal(m['raw_neuron_ind_in_list'].values.astype(int), np.arange(len(zxy))))
                n_bad += int(not ok)
        if n_bad:
            raise RuntimeError(f"{n_bad} checked frames: centroid rows != raw_neuron_ind_in_list order")
        df_gt = None if args.no_score else _load_project_gt(args.gt or args.project)
        match_col = 'raw_segmentation_id'
        puc = src.physical_unit_conversion
        um_xy, um_z = puc.zimmer_fluroscence_um_per_pixel_xy, puc.zimmer_um_per_pixel_z
    else:
        arrays = _load_nwb_arrays(src)
        gx, gy, gz, gr, gs = arrays['gx'], arrays['gy'], arrays['gz'], arrays['gr'], arrays['gs']
        nwb_all = args.nwb_objects == 'all'
        if nwb_all:
            # Every detection of the NWB's raw_segmentation (eval_accuracy --nwb_objects all), ids = mask labels
            from barlow_track.scripts.eval_accuracy import _nwb_all_objects
            objects, gt_label = _nwb_all_objects(src, arrays, list(range(n_frames)), gt_mapping=args.nwb_gt_mapping)
            for t in range(n_frames):
                labels, cen = objects[t]
                frames[t] = (np.asarray(cen, dtype=float).reshape(-1, 3), np.asarray(labels, dtype=int))
        else:
            for t in range(n_frames):
                fin = np.isfinite(gx[t]) & np.isfinite(gy[t]) & np.isfinite(gz[t]) \
                    & np.isfinite(gr[t]) & np.isfinite(gs[t])
                js = np.flatnonzero(fin)
                frames[t] = (np.stack([gz[t, js], gx[t, js], gy[t, js]], 1), gr[t, js].astype(int))
        df_gt = arrays['df_gt']
        info['gt_filter'] = 'full final_tracks'
        try:
            _df_fin, _ = src.get_final_tracks_only_finished_neurons()
        except Exception:
            _df_fin = None
        if _df_fin is not None and not _df_fin.empty:
            df_gt = _ensure_ind_level(_df_fin)
            info['gt_filter'] = 'finished neurons'
        if nwb_all:
            # The GT in the predictions' ID space (mask labels), exactly as eval_accuracy.run_eval does
            df_gt = df_gt.copy()
            if len(df_gt) != len(gt_label):
                raise ValueError(f"GT has {len(df_gt)} frames, the NWB arrays {len(gt_label)}")
            for n in df_gt.columns.get_level_values(0).unique():
                df_gt[(n, 'raw_neuron_ind_in_list')] = gt_label[:, arrays['neurons'].index(n)]
            info['gt_filter'] += ', mapped to mask labels'
        info.update(nwb_objects=args.nwb_objects, nwb_gt_mapping=args.nwb_gt_mapping if nwb_all else None)
        match_col = 'raw_neuron_ind_in_list'
        um_xy, um_z = args.um_xy, args.um_z
        if um_xy is None or um_z is None:
            # NWB imaging-volume grid_spacing, ordered (x, y, z) like the image axes
            import h5py
            with h5py.File(args.nwb, 'r') as h:
                opt = h['general/optophysiology']
                gs = [np.asarray(opt[k]['grid_spacing'][()], dtype=float) for k in opt
                      if 'grid_spacing' in opt[k]]
            if not gs:
                raise ValueError("No grid_spacing in the NWB; pass --um_xy and --um_z")
            gs = gs[0]
            if not np.isclose(gs[0], gs[1]):
                raise ValueError(f"anisotropic xy grid_spacing {gs}")
            um_xy = um_xy if um_xy is not None else float(gs[0])
            um_z = um_z if um_z is not None else float(gs[2])
            info['grid_spacing_from_nwb'] = gs.tolist()
    info.update(n_frames=n_frames, um_xy=um_xy, um_z=um_z,
                objects_per_frame=float(np.mean([len(v[0]) for v in frames.values()])))
    return frames, df_gt, match_col, src, info


# ---------------------------------------------------------------- tracking
def track_with_template(model, frames, i_template, um_xy, um_z, orient='default', threshold=0.0, scale=1.0):
    template_zxy = frames[i_template][0]
    template = to_fdnc_units(template_zxy, um_xy, um_z, orient, scale)
    all_matches = {}
    for t, (zxy, _) in frames.items():
        if len(zxy) < 2:
            all_matches[t] = []
            continue
        m = fdnc_match(model, template, to_fdnc_units(zxy, um_xy, um_z, orient, scale))
        all_matches[t] = [x for x in m if x[2] > threshold]
    return all_matches, len(template)


def matches_to_df(all_matches, frames, n_template, n_frames, prefix='fdnc'):
    """Columns (fdnc_XXX, raw_neuron_ind_in_list|likelihood); NaN where unmatched."""
    ind = np.full((n_frames, n_template), np.nan)
    lik = np.full((n_frames, n_template), np.nan)
    for t, ms in all_matches.items():
        ids = frames[t][1]
        for (i_tmp, i_test, p) in ms:
            ind[t, i_tmp] = ids[i_test]
            lik[t, i_tmp] = p
    names = [f"{prefix}_{i:03d}" for i in range(n_template)]
    data = {}
    for j, n in enumerate(names):
        data[(n, 'likelihood')] = lik[:, j]
        data[(n, 'raw_neuron_ind_in_list')] = ind[:, j]
    df = pd.DataFrame(data, index=range(n_frames))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def combine_templates(per_template, frames, template_ids, n_frames):
    """Multi-template vote (wbfm combine_multiple_template_matches), GT-free.

    Every template k's neuron indices are mapped to template 0's neuron indices
    with fDNC's own match of template-k's frame to template 0. Then each frame's
    candidate (template0_neuron, test_neuron, conf) list from all templates is
    resolved by wbfm's bipartite vote.
    """
    from wbfm.utils.general.utils_networkx import calc_bipartite_from_candidates
    from wbfm.utils.external.custom_errors import NoMatchesError
    t0 = template_ids[0]
    n0 = len(frames[t0][0])
    maps = [dict((i, i) for i in range(n0))]
    for tk in template_ids[1:]:
        # per_template[0] already holds template-0 -> frame tk matches: (i0, i_k, p)
        m0k = per_template[0][tk]
        maps.append({i_k: i0 for (i0, i_k, p) in m0k})
    K = len(template_ids)
    combined = {}
    for t in range(n_frames):
        cand = []
        for k in range(K):
            for (i_tmp, i_test, p) in per_template[k][t]:
                if i_tmp in maps[k]:
                    cand.append((maps[k][i_tmp], i_test, p / K))
        try:
            matches, conf, _ = calc_bipartite_from_candidates(cand, min_confidence_after_sum=0.1)
            combined[t] = [(int(m[0]), int(m[1]), float(c)) for m, c in zip(matches, conf)]
        except NoMatchesError:
            combined[t] = []
    return combined, n0


# ---------------------------------------------------------------- scoring
def score_like_run_eval(df_pred, df_gt, match_col, src, gt_matching='nandist', return_gt=False):
    """The scoring block of eval_accuracy.run_eval, verbatim in effect."""
    from barlow_track.utils.utils_ground_truth import (pad_with_nan_rows, align_gt_pred_time_index,
                                                       match_columns_to_gt, calculate_accuracy)
    if match_col == 'raw_segmentation_id':
        from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
        df_pred = add_metadata_to_df_raw_ind(df_pred, src.segmentation_metadata)
    df_gt, df_pred = align_gt_pred_time_index(df_gt, df_pred)
    max_len = max(len(df_gt), len(df_pred))
    df_pred = pad_with_nan_rows(df_pred, max_len)
    df_gt = pad_with_nan_rows(df_gt, max_len)
    df_pred_r = match_columns_to_gt(df_gt, df_pred, match_col, gt_matching=gt_matching)
    if 'unmatched_neuron' in df_pred_r.columns.get_level_values(0):
        df_pred_r = df_pred_r.drop(columns='unmatched_neuron')
    col_gt = df_gt.loc[:, (slice(None), match_col)].droplevel(1, axis=1)
    col_pr = df_pred_r.loc[:, (slice(None), match_col)].droplevel(1, axis=1)
    stats = calculate_accuracy(col_gt, col_pr)
    # How many GT neurons got a predicted column at all
    n_gt = col_gt.shape[1]
    n_assigned = int(len(set(col_pr.columns) & set(col_gt.columns)))
    if return_gt:
        return stats, n_gt, n_assigned, col_gt
    return stats, n_gt, n_assigned


def code_revision():
    """git HEAD of this repo, whether the tree is dirty, and the sha1 of this script (for the records)."""
    import hashlib
    import subprocess
    try:
        head = subprocess.check_output(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', REPO, 'status', '--porcelain', '--',
                                              os.path.abspath(__file__)], text=True).strip())
    except Exception:
        head, dirty = None, None
    with open(os.path.abspath(__file__), 'rb') as f:
        sha = hashlib.sha1(f.read()).hexdigest()[:12]
    return dict(git_head=head, script_modified_vs_head=dirty, script_sha1=sha)


def save_detections(path, frames):
    """Per-frame detection ids and zxy positions, for the pairing check against the BarlowTrack embedding cache."""
    ts = sorted(frames)
    np.savez_compressed(path, t=np.concatenate([np.full(len(frames[t][1]), t) for t in ts]),
                        ids=np.concatenate([np.asarray(frames[t][1], dtype=float) for t in ts]),
                        zxy=np.concatenate([np.asarray(frames[t][0], dtype=float).reshape(-1, 3) for t in ts]))


def check_reused_df(df_pred, frames, n_template, label):
    """A cached track dataframe belongs to these detections: same template size, every id a detection of its frame."""
    names = df_pred.columns.get_level_values(0).unique()
    if len(names) != n_template:
        raise RuntimeError(f"{label}: cached df has {len(names)} template neurons, the template frame now has {n_template}")
    ind = df_pred.loc[:, (slice(None), 'raw_neuron_ind_in_list')].values
    if len(ind) != len(frames):
        raise RuntimeError(f"{label}: cached df has {len(ind)} frames, the loader {len(frames)}")
    n_bad = 0
    for t in range(len(ind)):
        v = ind[t][np.isfinite(ind[t])]
        n_bad += int((~np.isin(v, frames[t][1])).sum())
    if n_bad:
        raise RuntimeError(f"{label}: {n_bad} cached matches are not detections of their frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='record label, e.g. zimmer_1128')
    ap.add_argument('--source', choices=['project', 'nwb'], default='project')
    ap.add_argument('--project', default=None)
    ap.add_argument('--gt', default=None)
    ap.add_argument('--nwb', default=None)
    ap.add_argument('--nwb_objects', choices=['gt', 'all'], default='gt',
                    help="nwb source: 'gt' tracks the GT objects (oracle detections); 'all' every detection of "
                         "the NWB's raw_segmentation with the GT mapped to mask labels (eval_accuracy --nwb_objects all)")
    ap.add_argument('--nwb_gt_mapping', choices=['centroid', 'segmentation_id'], default='centroid',
                    help='with --nwb_objects all: as eval_accuracy --nwb_gt_mapping')
    ap.add_argument('--um_xy', type=float, default=None)
    ap.add_argument('--um_z', type=float, default=None)
    ap.add_argument('--templates', type=int, nargs='+', default=[10])
    ap.add_argument('--n_spread', type=int, default=0,
                    help='add this many extra templates evenly spread over the video (after --templates)')
    ap.add_argument('--orient', choices=ORIENTS, nargs='+', default=['default'],
                    help='axis convention(s) (default = wbfm zimmer2leifer); several run one after another '
                         'on the frames loaded once')
    ap.add_argument('--rescale', choices=['none', 'ref'], default='none',
                    help="ref: one factor per recording, rms(fDNC reference worms) / rms(first template "
                         "cloud), applied to every frame (GT-free size normalization)")
    ap.add_argument('--threshold', type=float, default=0.0,
                    help='drop fDNC matches with prob <= threshold (0 keeps all Hungarian matches)')
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--model_path', default=DEFAULT_MODEL)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--tag', default=None, help='tag prefix; the orientation is appended')
    ap.add_argument('--reuse_dir', default=None,
                    help='rescore the <reuse_tag>_<variant>_df.pickle track tables in this directory instead of '
                         'tracking (each is checked against the loaded detections first)')
    ap.add_argument('--reuse_dataset', default=None,
                    help='dataset label in the cached tags (default --dataset), e.g. leifer_stardist')
    ap.add_argument('--save_tracks', action='store_true',
                    help='project source: also write <tag>_<variant>_tracks.h5, key pred_all (one column per '
                         'template neuron, values = mask labels), for score_gt_variants.py')
    ap.add_argument('--no_score', action='store_true',
                    help='skip the built-in scoring (e.g. a project whose GT is scored offline)')
    args = ap.parse_args()
    if args.save_tracks and args.source != 'project':
        raise ValueError('--save_tracks needs --source project (mask labels from the segmentation metadata)')
    if args.source == 'project' and args.project is None:
        from barlow_track.scripts.multiproject_scripts.run_trials_on_all_ground_truth import DATASETS
        spec = DATASETS[args.dataset]
        args.project = os.path.join(spec['project'], 'project_config.yaml')
        args.gt = args.gt or os.path.join(spec['gt'], 'project_config.yaml')
    if args.source == 'nwb' and args.nwb is None:
        from barlow_track.scripts.multiproject_scripts.run_trials_on_all_ground_truth import DATASETS
        args.nwb = DATASETS[args.dataset]['nwb']
    for p in (args.project, args.gt, args.nwb):
        if p and os.path.abspath(args.out_dir).startswith(os.path.dirname(os.path.abspath(p))):
            raise ValueError("out_dir must not be inside the data project")
    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    templates_given = list(args.templates)

    def make_tag(dataset, orient):
        return (f"fdnc_{dataset}_t{'-'.join(map(str, templates_given))}"
                f"{f'+{args.n_spread}spread' if args.n_spread else ''}_{orient}"
                f"{'_rescaled' if args.rescale == 'ref' else ''}")

    rev = code_revision()
    t_start = time.time()
    frames, df_gt, match_col, src, info = load_frames(args)
    n_frames = info['n_frames']
    print(f"[{args.dataset}] loaded {n_frames} frames, {info['objects_per_frame']:.1f} objects/frame, "
          f"GT {df_gt.columns.get_level_values(0).nunique() if df_gt is not None else '-'} neurons "
          f"({time.time()-t_start:.0f}s), code {rev}", flush=True)
    save_detections(os.path.join(args.out_dir, f"{args.dataset}_detections.npz"), frames)
    if args.n_spread:
        extra = np.linspace(0, n_frames - 1, args.n_spread + 2)[1:-1].round().astype(int).tolist()
        # A spread frame without detections (e.g. a gap in an NWB's GT objects) moves to the nearest frame
        # with at least the median number of detections (earlier frame on a tie). Detection counts only, no GT
        # accuracy; frames that already have >= 2 detections are never moved, so earlier runs are unchanged.
        counts = np.array([len(frames[t][0]) for t in range(n_frames)])
        med = np.median(counts)
        subs = {}
        for k, t in enumerate(extra):
            if counts[t] < 2:
                ok = np.flatnonzero(counts >= med)
                new = int(ok[np.argmin(np.abs(ok - t) + 0.1 * (ok > t))])
                subs[t] = new
                extra[k] = new
        if subs:
            print(f"[{args.dataset}] spread template frames without detections moved: {subs}", flush=True)
        info['template_substitutions'] = {str(k): v for k, v in subs.items()}
        args.templates = list(args.templates) + [t for t in extra if t not in args.templates]
    bad = [t for t in args.templates if t not in frames or len(frames[t][0]) < 2]
    if bad:
        raise ValueError(f"template frames {bad} missing or with < 2 objects")
    scale = 1.0
    if args.rescale == 'ref':
        ref_rms, n_ref = fdnc_reference_rms()
        scale = ref_rms / rms_radius(to_fdnc_units(frames[args.templates[0]][0], info['um_xy'], info['um_z']))
        info.update(ref_rms=ref_rms, n_ref_worms=n_ref)
    info['scale_factor'] = scale
    print(f"[{args.dataset}] templates {args.templates}, scale factor {scale:.3f}", flush=True)
    model = load_fdnc_model(args.model_path) if args.reuse_dir is None else None
    results_jsonl = os.path.join(args.out_dir, 'fdnc_results.jsonl')
    gt_saved = [False]

    for orient in args.orient:
        tag = (f"{args.tag}_{orient}" if args.tag else make_tag(args.dataset, orient))
        reuse_tag = make_tag(args.reuse_dataset or args.dataset, orient) if args.reuse_dir else None
        per_template = []

        def record(variant, all_matches, n_template, t_track, templates, df_pred=None, reused_from=None):
            if df_pred is None:
                df_pred = matches_to_df(all_matches, frames, n_template, n_frames)
            t1 = time.time()
            if args.save_tracks:
                from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
                lab = add_metadata_to_df_raw_ind(df_pred.copy(), src.segmentation_metadata)
                pa = lab.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1).astype(float)
                pa.to_hdf(os.path.join(args.out_dir, f"{tag}_{variant}_tracks.h5"), key='pred_all', mode='w')
            if args.no_score:
                stats = stats_v = dict(accuracy=np.nan, total_misses=-1, total_mismatches=-1, total_ground_truth=-1)
                n_gt = n_assigned = -1
            else:
                stats, n_gt, n_assigned, col_gt = score_like_run_eval(df_pred, df_gt, match_col, src,
                                                                      return_gt=True)
                stats_v, _, _ = score_like_run_eval(df_pred, df_gt, match_col, src, gt_matching='gt_valid')
                if not gt_saved[0]:
                    # the GT matrix the score is computed against, to compare with the BarlowTrack tracks' 'gt'
                    col_gt.astype(float).to_hdf(os.path.join(args.out_dir, f"{args.dataset}_scored_gt.h5"),
                                                key='gt', mode='w')
                    gt_saved[0] = True
            lik = df_pred.loc[:, (slice(None), 'likelihood')].values
            rec = dict(tag=f"{tag}_{variant}", dataset=args.dataset, method='fDNC', variant=variant,
                       model=os.path.abspath(args.model_path), source=args.source,
                       project=args.project, gt=args.gt or args.nwb, templates=templates,
                       orient=orient, threshold=args.threshold,
                       n_template_neurons=int(n_template), n_gt_neurons=int(n_gt),
                       n_gt_neurons_assigned_a_track=n_assigned, match_col=match_col,
                       gt_matching='nandist', accuracy=float(stats['accuracy']),
                       misses=int(stats['total_misses']), mismatches=int(stats['total_mismatches']),
                       total=int(stats['total_ground_truth']),
                       accuracy_gt_valid_matching=float(stats_v['accuracy']),
                       rescale=args.rescale,
                       mean_match_prob=float(np.nanmean(lik)) if np.isfinite(lik).any() else float('nan'),
                       n_matches=int(np.isfinite(lik).sum()),
                       track_seconds=t_track, score_seconds=time.time() - t1,
                       reused_from=reused_from, code=rev, **info)
            with open(results_jsonl, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            if reused_from is None:
                with open(os.path.join(args.out_dir, f"{tag}_{variant}_df.pickle"), 'wb') as f:
                    pickle.dump(df_pred, f)
            print(json.dumps({k: rec[k] for k in ('tag', 'accuracy', 'accuracy_gt_valid_matching', 'misses',
                                                  'mismatches', 'total', 'n_template_neurons',
                                                  'n_gt_neurons_assigned_a_track', 'mean_match_prob',
                                                  'track_seconds')}), flush=True)

        variants = [(f"single{i}", [i]) for i in args.templates]
        if len(args.templates) > 1:
            variants.append(("multi", list(args.templates)))
        if args.reuse_dir is not None:
            for variant, templates in variants:
                path = os.path.join(args.reuse_dir, f"{reuse_tag}_{variant}_df.pickle")
                with open(path, 'rb') as f:
                    df_pred = pickle.load(f)
                n_template = len(frames[templates[0]][0])
                check_reused_df(df_pred, frames, n_template, path)
                record(variant, None, n_template, 0.0, templates, df_pred=df_pred, reused_from=path)
            continue

        for i_tmp in args.templates:
            t1 = time.time()
            all_matches, n_template = track_with_template(model, frames, i_tmp, info['um_xy'], info['um_z'],
                                                          orient=orient, threshold=args.threshold, scale=scale)
            t_track = time.time() - t1
            print(f"[{tag}] template {i_tmp}: {n_template} neurons, tracked {n_frames} frames in {t_track:.0f}s "
                  f"({t_track / n_frames * 1000:.0f} ms/frame)", flush=True)
            with open(os.path.join(args.out_dir, f"{tag}_template{i_tmp}_matches.pickle"), 'wb') as f:
                pickle.dump(all_matches, f)
            per_template.append(all_matches)
            record(f"single{i_tmp}", all_matches, n_template, t_track, [i_tmp])

        if len(args.templates) > 1:
            t1 = time.time()
            combined, n0 = combine_templates(per_template, frames, args.templates, n_frames)
            record("multi", combined, n0, time.time() - t1, list(args.templates))
    print(f"[{args.dataset}] done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
