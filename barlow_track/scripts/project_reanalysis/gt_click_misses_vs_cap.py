"""Are GT clicks without a StarDist mask dim objects the 180-object cap removed, or true misses?

wbfm's 3d segmentation (perform_post_processing_3d) keeps only the max_number_of_objects brightest
StarDist objects per volume (mean red intensity), so in a volume at the cap a dim neuron can be
segmented and then deleted. Reads gt_click_exclusions.py's click_categories.npz and reports:

  objects per volume (StarDist, and NeRVE's on a sample of volumes) and how many volumes sit at the cap
  distance from each click to the nearest StarDist mask, per category
  the rate of no-mask clicks in capped vs uncapped volumes
  red intensity at each click (mean of a 3x5x5 voxel box, z x y) relative to the volume's assigned clicks,
  per category, and the dimmest kept object's mean intensity in capped volumes

    WBFM_NWB_LAZY_SEGMENTATION=1 python .../gt_click_misses_vs_cap.py --project .../project_config.yaml \
        --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb --categories .../click_categories.npz --out <dir>
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_S = {}
BOX = (1, 2, 2)  # half-widths z, x, y


def _init(project, gt_nwb):
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    _S['prj'] = ProjectData.load_final_project_data(project, allow_hybrid_loading=True, verbose=0)
    _S['ref'] = ProjectData.load_final_project_data(gt_nwb, allow_hybrid_loading=True, verbose=0)


def frame_stats(job):
    """Objects in the volume, intensity at each click, and the dimmest object's mean intensity."""
    from skimage.measure import regionprops
    t, g, count_ref = job
    prj = _S['prj']
    red = np.asarray(prj.red_data[t], float)
    seg = np.asarray(prj.raw_segmentation[t])
    n_obj = len(np.unique(seg)) - 1
    inten = np.full(len(g), np.nan)
    for j in np.flatnonzero(np.isfinite(g).all(1)):
        c = np.round(g[j]).astype(int)
        sl = tuple(slice(max(ci - h, 0), ci + h + 1) for ci, h in zip(c, BOX))
        box = red[sl]
        if box.size:
            inten[j] = box.mean()
    obj_means = np.array([p.intensity_mean for p in regionprops(seg, intensity_image=red)]) if n_obj else np.zeros(0)
    n_ref = len(np.unique(np.asarray(_S['ref'].raw_segmentation[t]))) - 1 if count_ref else -1
    return t, n_obj, n_ref, inten, (obj_means.min() if len(obj_means) else np.nan), np.median(red)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True)
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--categories', required=True, help="gt_click_exclusions.py's click_categories.npz")
    ap.add_argument('--cap', type=int, default=180, help='postprocessing_params.max_number_of_objects')
    ap.add_argument('--ref_every', type=int, default=10, help='count NeRVE objects every Nth volume')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--frames', type=int, default=None, help='first N volumes only (a quick check)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    d = np.load(args.categories)
    cat, cats, edge = d['cat'][:args.frames], list(d['cats']), d['edge_dist'][:args.frames]
    gt = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    arr = _load_nwb_arrays(gt)
    G = np.stack([arr['gz'], arr['gx'], arr['gy']], 2)[:len(cat)]
    T, N = cat.shape
    n_obj, n_ref = np.zeros(T, int), np.full(T, -1)
    inten, dimmest, med = np.full((T, N), np.nan), np.full(T, np.nan), np.full(T, np.nan)
    jobs = ((t, G[t], t % args.ref_every == 0) for t in range(T))
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.project, args.gt_nwb)) as ex:
        for t, n, nr, it, dm, md in ex.map(frame_stats, jobs, chunksize=4):
            n_obj[t], n_ref[t], inten[t], dimmest[t], med[t] = n, nr, it, dm, md

    A = cats.index('assigned')
    # Intensity relative to the median assigned click of the same volume (bleaching, motion)
    ref_level = np.array([np.nanmedian(inten[t][cat[t] == A]) if (cat[t] == A).any() else np.nan for t in range(T)])
    rel = inten / ref_level[:, None]
    capped = n_obj >= args.cap
    live = n_obj > 0
    no_mask = np.isin(cat, [cats.index(c) for c in ('background_near_mask', 'missed_found_by_ref', 'background_both')])
    clicks = cat >= 0
    pct = lambda x: np.nanpercentile(x, [10, 50, 90]).round(3).tolist() if np.isfinite(x).any() else None
    out = dict(args=vars(args), n_volumes=int(T), n_volumes_with_objects=int(live.sum()),
               objects_per_volume_pct_0_10_50_90_100=np.percentile(n_obj[live], [0, 10, 50, 90, 100]).tolist(),
               volumes_at_cap=int(capped.sum()),
               nerve_objects_per_volume_pct_0_10_50_90_100=np.percentile(n_ref[n_ref >= 0], [0, 10, 50, 90, 100]).tolist(),
               no_mask_click_rate_capped=float(no_mask[capped].sum() / max(clicks[capped].sum(), 1)),
               no_mask_click_rate_uncapped=float(no_mask[live & ~capped].sum() / max(clicks[live & ~capped].sum(), 1)),
               clicks_in_capped_volumes=int(clicks[capped].sum()),
               dimmest_kept_object_rel_capped_pct=pct(dimmest[capped] / ref_level[capped]),
               dimmest_kept_object_rel_uncapped_pct=pct(dimmest[live & ~capped] / ref_level[live & ~capped]),
               volume_median_rel_pct=pct(med / ref_level))
    per_cat = {}
    for i, k in enumerate(cats):
        m = cat == i
        if not m.any():
            continue
        per_cat[k] = dict(n=int(m.sum()), in_capped_volume=int((m & capped[:, None]).sum()),
                          intensity_rel_to_assigned_pct_10_50_90=pct(rel[m]),
                          dist_to_mask_um_pct_10_50_90=pct(edge[m]) if np.isfinite(edge[m]).any() else None)
    out['per_category'] = per_cat
    # Edge distances of every click outside a mask (assigned-by-mask-rule ones are <= 1 um)
    bg = no_mask & np.isfinite(edge)
    out['no_mask_dist_um_hist'] = {f'{a}-{b}': int(((edge >= a) & (edge < b) & bg).sum())
                                   for a, b in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 1e9)]}
    np.savez(os.path.join(args.out, 'click_misses_vs_cap.npz'), n_obj=n_obj, n_ref=n_ref, inten=inten,
             dimmest=dimmest, ref_level=ref_level)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
