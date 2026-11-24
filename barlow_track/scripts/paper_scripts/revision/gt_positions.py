"""GT neuron positions per benchmark recording, cached, and the saved tracks' per-neuron / per-frame accuracy.

Shared by t1_deformation_table.py (TEST_PLAN T1) and t2_modality_contribution.py (T2).

Positions are (T, N, 3) in (z, x, y), z multiplied by the recording's ViSNet z_scale (the value its trial-59 network
used: xy voxels per z voxel), so distances are in roughly isotropic xy-voxel units. Every analysis divides them by
the recording's median nearest-neighbour (NN) distance, so the unit cancels.
"""
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np
import pandas as pd

_F = '/lisc/data/scratch/neurobiology/zimmer/fieseler'
_PAPER = f'{_F}/barlow_track_paper'
_ZIM = f'{_F}/wbfm_projects/manually_annotated/paper_data'
ROUNDS = f'{_F}/barlow_track_revisions/config_files'

# GT source of each benchmark recording, as eval_accuracy recorded it in round_shared_v1/exp_results_v2.jsonl
GT_PATHS = {
    'zimmer_1123': f'{_ZIM}/2022-11-23_worm11_updated_format/project_config.yaml',
    'zimmer_1128': f'{_ZIM}/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format/project_config.yaml',
    'zimmer_1210': f'{_ZIM}/ZIM2165_Gcamp7b_worm1-2022-12-10_updated_format/project_config.yaml',
    'flavell': f'{_PAPER}/flavell_data/images_for_charlie/flavell_data.nwb',
    'flavell_2022031603': f'{_PAPER}/flavell_data/forCharlie_2/2022-03-16-03/flavell_data.nwb',
    'flavell_2022032402': f'{_PAPER}/flavell_data/forCharlie_2/2022-03-24-02/flavell_data.nwb',
    'flavell_2022033001': f'{_PAPER}/flavell_data/forCharlie_2/2022-03-30-01/flavell_data.nwb',
    'flavell_2022033002': f'{_PAPER}/flavell_data/forCharlie_2/2022-03-30-02/flavell_data.nwb',
    'samuel': f'{_PAPER}/samuel_data/153.nwb',
    'samuel_137': f'{_PAPER}/samuel_data/137.nwb',
    'samuel_184': f'{_PAPER}/samuel_data/thresholded_segmentation/184-004.nwb',
    'leifer_stardist_strict': f'{_F}/barlow_track_revisions/wbfm_tracker_baseline/leifer_resegment/'
                              'Leifer_NeRVE_Worm1_reexport_A2_stardist_strictgt',
}
# visnet_z_scale of each recording's trial-59 network (round_shared_v1 train_config)
Z_SCALE = {'zimmer_1123': 4.615, 'zimmer_1128': 4.615, 'zimmer_1210': 4.615, 'leifer_stardist_strict': 4.65,
           'samuel': 3.889, 'samuel_137': 3.889, 'samuel_184': 3.889, 'flavell': 1.0, 'flavell_2022031603': 1.0,
           'flavell_2022032402': 1.0, 'flavell_2022033001': 1.0, 'flavell_2022033002': 1.0}
# Volumes per second (paper Table 1; Samuel from Susoy et al. 2021, the same rate assumed for 153 / 137 / 184)
VPS = {'zimmer_1123': 3.5, 'zimmer_1128': 3.5, 'zimmer_1210': 3.5, 'leifer_stardist_strict': 6.0,
       'samuel': 10.0, 'samuel_137': 10.0, 'samuel_184': 10.0, 'flavell': 1.7, 'flavell_2022031603': 1.7,
       'flavell_2022032402': 1.7, 'flavell_2022033001': 1.7, 'flavell_2022033002': 1.7}
TUNING = ['zimmer_1123', 'zimmer_1210', 'flavell_2022032402', 'flavell_2022033002', 'samuel_137', 'samuel_184']


# Segmentation that holds every detection of the recording (the objects the tracker sees). Default: the GT source
# itself (Zimmer projects; Flavell NWBs, evaluated with nwb_objects all; Samuel NWBs, whose segmentation holds only GT
# neurons, so their detection NN equals the GT NN). Leifer's GT project is the strict-GT copy; detections are in the
# StarDist project it was made from.
DET_PATHS = dict(GT_PATHS, leifer_stardist_strict=f'{_F}/barlow_track_revisions/wbfm_tracker_baseline/leifer_resegment/'
                                                  'Leifer_NeRVE_Worm1_reexport_A2_stardist/project_config.yaml')


def detection_nn(lab, cache_dir, every=10):
    """Median within-frame NN distance over ALL segmented detections (z-scaled), and mean detections per frame.

    The unit for deformation: the GT annotates a subset of neurons on Zimmer (61-131) and Leifer (79), so the
    GT-only NN distance is inflated there and would understate their deformation. Cached as json.
    """
    import json
    fn = os.path.join(cache_dir, f'detection_nn_{lab}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from scipy.spatial import cKDTree
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.utils.volume_data import get_centroids_for_volume
    p = ProjectData.load_final_project_data(DET_PATHS[lab], allow_hybrid_loading=True, verbose=0)
    if p.segmentation_metadata is not None:
        centroids = lambda t: get_centroids_for_volume(p, t)[0]  # noqa: E731
    else:
        # NWB source: regionprops centroids of the raw segmentation, (z, x, y) like the GT positions, as
        # eval_accuracy._nwb_all_objects reads every detection
        from skimage.measure import regionprops
        centroids = lambda t: [r.centroid for r in regionprops(np.asarray(p.raw_segmentation[t]))]  # noqa: E731
    d, counts = [], []
    for t in range(0, p.num_frames, every):
        zxy = np.asarray(centroids(t), dtype=float).reshape(-1, 3)
        zxy = zxy[np.isfinite(zxy).all(1)]
        counts.append(len(zxy))
        if len(zxy) < 3:
            continue
        zxy[:, 0] *= Z_SCALE[lab]
        d.append(cKDTree(zxy).query(zxy, 2)[0][:, 1])
    out = dict(nn_det=float(np.median(np.concatenate(d))), detections_per_frame=float(np.mean(counts)),
               source=DET_PATHS[lab], every=every)
    json.dump(out, open(fn, 'w'))
    return out


def load_positions(lab, cache_dir):
    """(pos (T, N, 3) z-scaled zxy, names, frame index (T,) int), cached as npz in cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f'gt_positions_{lab}.npz')
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=False)
        return d['pos'], list(d['names']), d['index']
    from barlow_track.scripts.eval_accuracy import _load_project_gt
    df = _load_project_gt(GT_PATHS[lab])
    names = [str(n) for n in df.columns.get_level_values(0).unique()]
    pos = np.stack([df.loc[:, [(n, c) for n in names]].to_numpy(dtype=float) for c in ('z', 'x', 'y')], -1)
    pos[..., 0] *= Z_SCALE[lab]
    index = np.asarray(df.index, dtype=float).astype(int)
    np.savez(fn, pos=pos, names=np.array(names), index=index)
    return pos, names, index


def nn_distance(pos, every=10):
    """Median within-frame nearest-neighbour distance over every `every`-th frame."""
    out = []
    for t in range(0, len(pos), every):
        p = pos[t][np.isfinite(pos[t]).all(1)]
        if len(p) < 3:
            continue
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        out.append(d.min(1))
    return float(np.median(np.concatenate(out)))


def similarity_fit(a, b):
    """Residuals of b against the best similarity transform of a onto b (Umeyama), rotation angle (deg), scale."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / A.var(0).sum()
    res = b - (scale * A @ R.T + mb)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return res, angle, scale


def track_accuracy(h5_path):
    """Per-neuron and per-frame accuracy of one saved track file, as calculate_accuracy computes them.

    Returns (correct, valid) boolean DataFrames (frames x GT neurons), indexed by int frame.
    """
    gt = pd.read_hdf(h5_path, 'gt')
    pr = pd.read_hdf(h5_path, 'pred_matched').reindex(index=gt.index, columns=gt.columns)
    valid = gt.notna()
    correct = valid & (gt == pr)
    idx = np.asarray(gt.index, dtype=float).astype(int)
    valid.index = idx
    correct.index = idx
    return correct, valid


def aligned(pos, names, index, valid):
    """Positions reindexed onto a track file's frames and neurons: (T', N', 3), plus the coverage check.

    coverage = fraction of the track file's GT-valid entries that have a finite position; should be ~1.
    """
    df_idx = {t: i for i, t in enumerate(index)}
    col = {n: j for j, n in enumerate(names)}
    rows = [df_idx.get(t, -1) for t in valid.index]
    cols = [col.get(str(n), -1) for n in valid.columns]
    out = np.full((len(rows), len(cols), 3), np.nan)
    for i, r in enumerate(rows):
        if r < 0:
            continue
        ok = [j for j, c in enumerate(cols) if c >= 0]
        out[i, ok] = pos[r, [cols[j] for j in ok]]
    fin = np.isfinite(out).all(-1)
    v = valid.to_numpy()
    coverage = float((fin & v).sum() / max(v.sum(), 1))
    return out, coverage
