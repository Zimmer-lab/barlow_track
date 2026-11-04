"""Why each GT click did or did not become GT in a re-segmented project: one category per click.

gt_tracks_from_clicks.py assigns a click to the nearest segmented-object centroid within max_dist_um
whose mask holds the click voxel, unless that object is nearest to another GT click too. Its counts
lump very different cases into "no object within max_dist" (a click inside a large object whose
centroid is farther than max_dist, a click on background next to a mask, a neuron the new
segmentation never detected). This script replays the same rule and puts every click (t, neuron) in
exactly one category, first match wins:

  empty_volume          the volume has no objects (the corrupted volumes)
  assigned              the rule assigned it (also flagged: its object holds another neuron's click)
  clicks_coincide       another neuron's click at t lies within --coincide_um (GT points on top of each other)
  merged_object         the click lies in a mask that also holds another neuron's click, or the rule
                        found its nearest object shared (one object, two neurons)
  in_mask_far_centroid  the click lies in a mask no other click is in, but the rule rejected it (centroid
                        farther than max_dist_um, or nearest centroid is another object): recoverable
  background_near_mask  no mask at the click voxel, nearest mask voxel within --edge_um
  missed_found_by_ref   no mask within --edge_um, but the reference segmentation (the GT NWB's, NeRVE)
                        has a mask at the click voxel: the new segmentation missed a detected object
  background_both       no mask near the click in either segmentation

    WBFM_NWB_LAZY_SEGMENTATION=1 python .../gt_click_exclusions.py --project .../project_config.yaml \
        --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb --max_dist_um 1.5 --out <dir>
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CATS = ['empty_volume', 'assigned', 'clicks_coincide', 'merged_object', 'in_mask_far_centroid',
        'background_near_mask', 'missed_found_by_ref', 'background_both']
_S = {}


def _init(project, gt_nwb, args):
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    _S['prj'] = ProjectData.load_final_project_data(project, allow_hybrid_loading=True, verbose=0)
    _S['ref'] = ProjectData.load_final_project_data(gt_nwb, allow_hybrid_loading=True, verbose=0)
    _S['args'] = args


def classify_frame(job):
    """Categories (N,) and extras for the clicks g (N, 3 voxels, z x y; NaN rows = no click) at t."""
    from scipy.ndimage import distance_transform_edt
    from wbfm.utils.neuron_matching.utils_matching import calc_nearest_neighbor_matches
    t, g = job
    a, prj, ref = _S['args'], _S['prj'], _S['ref']
    scale = np.asarray(a['voxel_um'], float)
    N = len(g)
    cat = np.full(N, -1)
    centroid_dist = np.full(N, np.nan)  # um, click to the centroid of the mask it lies in
    edge_dist = np.full(N, np.nan)  # um, click to the nearest mask voxel (background clicks)
    assigned_shares = np.zeros(N, bool)
    js = np.flatnonzero(np.isfinite(g).all(1))
    if not len(js):
        return t, cat, centroid_dist, edge_dist, assigned_shares
    c = np.asarray(prj.get_centroids_as_numpy(t), float)
    if not len(c):
        cat[js] = CATS.index('empty_volume')
        return t, cat, centroid_dist, edge_dist, assigned_shares
    seg = np.asarray(prj.raw_segmentation[t])
    vox = np.clip(np.round(g[js]).astype(int), 0, np.array(seg.shape) - 1)
    lab_at = seg[vox[:, 0], vox[:, 1], vox[:, 2]]

    # The rule, as gt_tracks_from_clicks.assign_clicks
    m, _ = calc_nearest_neighbor_matches(g[js] * scale, c * scale, max_dist=a['max_dist_um'])
    m = np.asarray(m, dtype=int).reshape(-1, 2)
    md = prj.segmentation_metadata
    matched_label = np.full(len(js), -1)
    matched_label[m[:, 0]] = [md.i_in_array_to_mask_index(t, i1) for i1 in m[:, 1]]
    nearest_obj = np.full(len(js), -1)
    nearest_obj[m[:, 0]] = m[:, 1]
    vals, cnt = np.unique(m[:, 1], return_counts=True)
    rule_shared = np.isin(nearest_obj, vals[cnt > 1]) & (nearest_obj >= 0)
    inside = (matched_label >= 0) & (lab_at == matched_label)
    rule_assigned = inside & ~rule_shared

    # Clicks of other neurons in the same mask, and nearby clicks
    in_mask = lab_at > 0
    labs, lcnt = np.unique(lab_at[in_mask], return_counts=True)
    mask_shared = np.isin(lab_at, labs[lcnt > 1]) & in_mask
    p = g[js] * scale
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    np.fill_diagonal(d, np.inf)
    coincide = d.min(1) < a['coincide_um'] if len(js) > 1 else np.zeros(len(js), bool)

    # Centroid distance for clicks inside a mask
    lab2row = {int(md.i_in_array_to_mask_index(t, i)): i for i in range(len(c))}
    for k in np.flatnonzero(in_mask):
        r = lab2row.get(int(lab_at[k]))
        if r is not None:
            centroid_dist[js[k]] = float(np.linalg.norm((g[js[k]] - c[r]) * scale))

    bg = ~in_mask & ~rule_assigned
    if bg.any():
        dt = distance_transform_edt(seg == 0, sampling=scale)
        edge_dist[js[bg]] = dt[vox[bg, 0], vox[bg, 1], vox[bg, 2]]
        rseg = np.asarray(ref.raw_segmentation[t])
        ref_has = rseg[vox[:, 0], vox[:, 1], vox[:, 2]] > 0
    for k, j in enumerate(js):
        if rule_assigned[k]:
            cat[j] = CATS.index('assigned')
            assigned_shares[j] = mask_shared[k]
        elif coincide[k]:
            cat[j] = CATS.index('clicks_coincide')
        elif mask_shared[k] or rule_shared[k]:
            cat[j] = CATS.index('merged_object')
        elif in_mask[k]:
            cat[j] = CATS.index('in_mask_far_centroid')
        elif edge_dist[j] <= a['edge_um']:
            cat[j] = CATS.index('background_near_mask')
        elif ref_has[k]:
            cat[j] = CATS.index('missed_found_by_ref')
        else:
            cat[j] = CATS.index('background_both')
    return t, cat, centroid_dist, edge_dist, assigned_shares


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True)
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--max_dist_um', type=float, default=1.5)
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.5, 0.3226, 0.3226])
    ap.add_argument('--coincide_um', type=float, default=1.5)
    ap.add_argument('--edge_um', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=None, help='first N volumes only (a quick check)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    gt = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    arr = _load_nwb_arrays(gt)
    G = np.stack([arr['gz'], arr['gx'], arr['gy']], 2)  # (T, N, 3)
    if args.frames:
        G = G[:args.frames]
    T, N, _ = G.shape
    cat = np.full((T, N), -1)
    cdist, edist = np.full((T, N), np.nan), np.full((T, N), np.nan)
    shares = np.zeros((T, N), bool)
    kw = dict(max_dist_um=args.max_dist_um, voxel_um=args.voxel_um, coincide_um=args.coincide_um,
              edge_um=args.edge_um)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.project, args.gt_nwb, kw)) as ex:
        for t, c, cd, ed, sh in ex.map(classify_frame, ((t, G[t]) for t in range(T)), chunksize=4):
            cat[t], cdist[t], edist[t], shares[t] = c, cd, ed, sh
    np.savez(os.path.join(args.out, 'click_categories.npz'), cat=cat, centroid_dist=cdist, edge_dist=edist,
             assigned_shares=shares, cats=np.array(CATS), neurons=np.array(arr['neurons']))
    clicks = cat >= 0
    summary = dict(args=vars(args), clicks=int(clicks.sum()),
                   counts={k: int((cat == i).sum()) for i, k in enumerate(CATS)},
                   assigned_but_object_holds_another_click=int(shares.sum()))
    summary['fractions'] = {k: v / summary['clicks'] for k, v in summary['counts'].items()}
    far = cat == CATS.index('in_mask_far_centroid')
    if far.any():
        summary['in_mask_far_centroid_dist_um_pct_10_50_90'] = np.percentile(cdist[far], [10, 50, 90]).tolist()
    per_neuron = {}
    for i, k in enumerate(CATS):
        per_neuron[k] = ((cat == i).sum(0) / np.maximum(clicks.sum(0), 1)).round(4).tolist()
    summary['per_neuron_fraction'] = per_neuron
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in summary if k not in ('per_neuron_fraction',)}, indent=1))


if __name__ == '__main__':
    main()
