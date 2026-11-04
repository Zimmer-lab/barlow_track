"""Ground-truth tracks for a re-segmented wbfm project, from manual GT clicks in an NWB.

Used for the Leifer NeRVE worm1 recording re-segmented with stardist
(wbfm_tracker_baseline/leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist), whose GT is the
Leifer lab's manual clicks (manualSummary.manualPoints, 79 neurons) stored as NeuronCentroids in
Leifer_NeRVE_Worm1_reexport_A2.nwb. The project's red volumes must be the NWB's voxel for voxel
(checked: identical).

Assignment rule, per volume t and GT neuron j with a click at p (z, x, y voxels):
  1. Candidates are the project's segmented objects at t (centroids from
     project_data.get_centroids_as_numpy, the segmentation metadata).
  2. Match p to its nearest object centroid with wbfm's calc_nearest_neighbor_matches, distances in um
     (voxel size from --voxel_um, z x y), keeping only matches within --max_dist_um.
  3. The click voxel (rounded p) must lie inside that object's mask.
  4. An object claimed by two or more GT neurons at t is assigned to none of them (a likely merge of
     neighbouring neurons; the GT is not forced onto one object).
  Otherwise neuron j is NaN (no GT) at t. No GT is moved, interpolated or added.

--rule mask (the default above is --rule centroid) drops the centroid distance, which rejects clicks
inside large objects (a z voxel is 1.5 um and stardist objects span ~6 planes; gt_click_exclusions.py
found 9% of the Leifer clicks inside an object whose centroid is 1.5-2.7 um away):
  1. The click's object is the mask at the click voxel; if the voxel is background, the nearest mask
     voxel within --edge_um (um distances; ties keep the first in a fixed offset order).
  2. A click with another neuron's click within --coincide_um at t is assigned to none (GT points on
     top of each other).
  3. An object whose mask holds two or more neurons' click voxels at t (coincident clicks included) is
     assigned to none of them (merge). A click matched only through the edge search takes its object
     only if no other click claims it; it never removes a click inside the mask.
  Otherwise the click is assigned to its object.

--rule segmentation_id takes the object from the GT NWB's own raw_segmentation_id, for a project whose
masks are the NWB's (the NeRVE project Leifer_NeRVE_Worm1_reexport_A2: NeRVE's masks plus pseudo masks,
label 1000 + GT index, watershed around clicks NeRVE missed):
  1. A click without a raw_segmentation_id, or with one at or above --max_label (the pseudo masks: built
     from the click itself, not by the segmentation), is assigned to none.
  2. A label held by two or more neurons' clicks at t is assigned to none of them (merge).
  3. A label that is not in the project's segmentation metadata is assigned to none.
  Otherwise the click is assigned to its labelled object.

The output is a final_3d_tracks_df in the same format as the paper projects' tracks (columns per neuron:
raw_neuron_ind_in_list plus everything wbfm's add_metadata_to_df_raw_ind fills from the segmentation),
and a JSON next to it with the parameters, inputs and counts of every exclusion reason.

    python barlow_track/scripts/project_reanalysis/gt_tracks_from_clicks.py \
        --project .../project_config.yaml --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb \
        --max_dist_um 1.0 [--write] [--set_final_tracks]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def _git_rev():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(['git', '-C', here, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.call(['git', '-C', here, 'diff', '--quiet', 'HEAD', '--', os.path.abspath(__file__)])
        untracked = subprocess.check_output(['git', '-C', here, 'ls-files', os.path.abspath(__file__)], text=True)
        return rev + ('' if untracked.strip() and not dirty else ' (script uncommitted or modified)')
    except Exception:
        return None


def assign_clicks(project_data, gx, gy, gz, max_dist_um, voxel_um, masks=None):
    """Assign GT clicks (T, N arrays, voxels) to segmented objects; see module docstring.

    Returns raw_ind (T, N) of row indices into each volume's segmentation metadata (NaN when not
    assigned) and a dict of counts per outcome.
    """
    from wbfm.utils.neuron_matching.utils_matching import calc_nearest_neighbor_matches
    scale = np.asarray(voxel_um, dtype=float)  # z, x, y
    T, N = gx.shape
    raw_ind = np.full((T, N), np.nan)
    counts = dict(gt_clicks=0, no_objects_in_volume=0, no_object_within_max_dist=0,
                  click_outside_matched_mask=0, object_shared_with_another_gt_neuron=0, assigned=0)
    masks = project_data.raw_segmentation if masks is None else masks
    vol_shape = np.array(masks.shape[1:])
    for t in range(T):
        g = np.stack([gz[t], gx[t], gy[t]], 1)
        js = np.flatnonzero(np.isfinite(g).all(1))
        counts['gt_clicks'] += len(js)
        if not len(js):
            continue
        c = project_data.get_centroids_as_numpy(t)
        if not len(c):
            counts['no_objects_in_volume'] += len(js)
            continue
        m, _ = calc_nearest_neighbor_matches(g[js] * scale, np.asarray(c, dtype=float) * scale,
                                             max_dist=max_dist_um)
        m = np.asarray(m, dtype=int).reshape(-1, 2)
        counts['no_object_within_max_dist'] += len(js) - len(m)
        if not len(m):
            continue
        seg = np.asarray(masks[t])
        labels = np.array([project_data.segmentation_metadata.i_in_array_to_mask_index(t, i1) for i1 in m[:, 1]])
        vox = np.clip(np.round(g[js[m[:, 0]]]).astype(int), 0, vol_shape - 1)
        inside = seg[vox[:, 0], vox[:, 1], vox[:, 2]] == labels
        counts['click_outside_matched_mask'] += int((~inside).sum())
        # Shared objects: any object nearest to 2+ GT clicks (counted before the inside filter, so a
        # merged object is never kept for one neuron because the other's click fell just outside).
        vals, cnt = np.unique(m[:, 1], return_counts=True)
        shared = np.isin(m[:, 1], vals[cnt > 1])
        counts['object_shared_with_another_gt_neuron'] += int((shared & inside).sum())
        keep = inside & ~shared
        raw_ind[t, js[m[keep, 0]]] = m[keep, 1]
        counts['assigned'] += int(keep.sum())
    return raw_ind, counts


def assign_clicks_by_mask(project_data, gx, gy, gz, voxel_um, edge_um, coincide_um, masks=None):
    """--rule mask (module docstring): same outputs as assign_clicks."""
    scale = np.asarray(voxel_um, dtype=float)  # z, x, y
    T, N = gx.shape
    raw_ind = np.full((T, N), np.nan)
    counts = dict(gt_clicks=0, no_objects_in_volume=0, coincident_with_another_click=0,
                  no_mask_within_edge=0, object_shared_with_another_gt_neuron=0, label_not_in_metadata=0,
                  assigned=0, assigned_in_mask=0, assigned_on_mask_edge=0)
    masks = project_data.raw_segmentation if masks is None else masks
    vol_shape = np.array(masks.shape[1:])
    md = project_data.segmentation_metadata
    # Background clicks look for a mask voxel at these offsets, nearest first
    r = np.floor(edge_um / scale).astype(int)
    offs = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in r], indexing='ij'), -1).reshape(-1, 3)
    dist = np.linalg.norm(offs * scale, axis=1)
    keep = (dist > 0) & (dist <= edge_um)
    offs = offs[keep][np.argsort(dist[keep], kind='stable')]
    for t in range(T):
        g = np.stack([gz[t], gx[t], gy[t]], 1)
        js = np.flatnonzero(np.isfinite(g).all(1))
        counts['gt_clicks'] += len(js)
        if not len(js):
            continue
        n_obj = len(project_data.get_centroids_as_numpy(t))
        if not n_obj:
            counts['no_objects_in_volume'] += len(js)
            continue
        seg = np.asarray(masks[t])
        vox = np.clip(np.round(g[js]).astype(int), 0, vol_shape - 1)
        lab = seg[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.int64)
        on_edge = np.zeros(len(js), bool)
        for k in np.flatnonzero(lab == 0):
            v = vox[k] + offs
            ok = ((v >= 0) & (v < vol_shape)).all(1)
            hits = seg[v[ok, 0], v[ok, 1], v[ok, 2]]
            if (hits > 0).any():
                lab[k] = hits[np.argmax(hits > 0)]
                on_edge[k] = True
        p = g[js] * scale
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        np.fill_diagonal(d, np.inf)
        coincide = d.min(1) < coincide_um
        labeled = lab > 0
        inside = labeled & ~on_edge
        vals, cnt = np.unique(lab[inside], return_counts=True)
        shared = np.isin(lab, vals[cnt > 1]) & inside
        # Edge clicks: dropped if their object holds an inside click, or another edge click claims it
        all_vals, all_cnt = np.unique(lab[labeled], return_counts=True)
        shared |= on_edge & np.isin(lab, all_vals[all_cnt > 1])
        lab2row = {int(md.i_in_array_to_mask_index(t, i)): i for i in range(n_obj)}
        for k, j in enumerate(js):
            if coincide[k]:
                counts['coincident_with_another_click'] += 1
            elif not labeled[k]:
                counts['no_mask_within_edge'] += 1
            elif shared[k]:
                counts['object_shared_with_another_gt_neuron'] += 1
            elif int(lab[k]) not in lab2row:
                counts['label_not_in_metadata'] += 1
            else:
                raw_ind[t, j] = lab2row[int(lab[k])]
                counts['assigned'] += 1
                counts['assigned_on_mask_edge' if on_edge[k] else 'assigned_in_mask'] += 1
    return raw_ind, counts


def assign_clicks_by_segmentation_id(project_data, gx, gs, max_label=None):
    """--rule segmentation_id (module docstring): same outputs as assign_clicks. gx (T, N) marks the
    clicks (finite where neuron j was clicked at t); gs (T, N) is their raw_segmentation_id."""
    T, N = gs.shape
    raw_ind = np.full((T, N), np.nan)
    counts = dict(gt_clicks=0, no_objects_in_volume=0, no_segmentation_id=0, pseudo_mask_label=0,
                  object_shared_with_another_gt_neuron=0, label_not_in_metadata=0, assigned=0)
    md = project_data.segmentation_metadata
    for t in range(T):
        js = np.flatnonzero(np.isfinite(gx[t]))
        counts['gt_clicks'] += len(js)
        if not len(js):
            continue
        n_obj = len(project_data.get_centroids_as_numpy(t))
        if not n_obj:
            counts['no_objects_in_volume'] += len(js)
            continue
        lab = gs[t, js]
        has = np.isfinite(lab) & (lab > 0)
        pseudo = has & (lab >= max_label) if max_label is not None else np.zeros(len(js), bool)
        ok = has & ~pseudo
        vals, cnt = np.unique(lab[ok], return_counts=True)
        shared = ok & np.isin(lab, vals[cnt > 1])
        lab2row = {int(md.i_in_array_to_mask_index(t, i)): i for i in range(n_obj)}
        for k, j in enumerate(js):
            if not has[k]:
                counts['no_segmentation_id'] += 1
            elif pseudo[k]:
                counts['pseudo_mask_label'] += 1
            elif shared[k]:
                counts['object_shared_with_another_gt_neuron'] += 1
            elif int(lab[k]) not in lab2row:
                counts['label_not_in_metadata'] += 1
            else:
                raw_ind[t, j] = lab2row[int(lab[k])]
                counts['assigned'] += 1
    return raw_ind, counts



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True, help='project_config.yaml of the re-segmented project')
    ap.add_argument('--gt_nwb', required=True, help='NWB whose final_tracks x/y/z are the manual GT clicks')
    ap.add_argument('--rule', choices=['centroid', 'mask', 'segmentation_id'], default='centroid')
    ap.add_argument('--max_label', type=int, default=None,
                    help='--rule segmentation_id: labels at or above this are pseudo masks, assigned to none')
    ap.add_argument('--max_dist_um', type=float, default=1.0, help='--rule centroid')
    ap.add_argument('--edge_um', type=float, default=1.0, help='--rule mask: background clicks this close to a mask')
    ap.add_argument('--coincide_um', type=float, default=1.5, help='--rule mask: clicks of two neurons this close')
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.5, 0.3226, 0.3226],
                    help='z x y voxel size in um (default: the Leifer NWB grid_spacing)')
    ap.add_argument('--out_name', default='3-tracking/manual_annotation/gt_tracks_from_clicks.h5',
                    help='output path relative to the project directory')
    ap.add_argument('--write', action='store_true', help='write the h5 + provenance JSON (default: dry run)')
    ap.add_argument('--set_final_tracks', action='store_true',
                    help="point the project's tracking_config final_3d_tracks_df at the output")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    from wbfm.utils.projects.finished_project_data import ProjectData
    from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays

    gt = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    prj = ProjectData.load_final_project_data(args.project, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(gt)
    neurons = a['neurons']
    if a['gx'].shape[0] != prj.num_frames:
        raise ValueError(f"GT has {a['gx'].shape[0]} volumes, the project {prj.num_frames}")

    if args.rule == 'segmentation_id':
        raw_ind, counts = assign_clicks_by_segmentation_id(prj, a['gx'], a['gs'], args.max_label)
    elif args.rule == 'mask':
        raw_ind, counts = assign_clicks_by_mask(prj, a['gx'], a['gy'], a['gz'], args.voxel_um, args.edge_um,
                                                args.coincide_um)
    else:
        raw_ind, counts = assign_clicks(prj, a['gx'], a['gy'], a['gz'], args.max_dist_um, args.voxel_um)
    n_click = np.isfinite(a['gx']).sum(0)
    coverage = np.isfinite(raw_ind).sum(0) / np.maximum(n_click, 1)
    summary = dict(counts, fraction_assigned=counts['assigned'] / max(counts['gt_clicks'], 1),
                   neurons=len(neurons),
                   per_neuron_coverage_percentiles={p: float(np.percentile(coverage, p)) for p in (0, 10, 50, 90, 100)},
                   volumes_with_no_objects=[int(t) for t in range(prj.num_frames)
                                            if not len(prj.get_centroids_as_numpy(t))])
    print(json.dumps(summary, indent=1))
    if not args.write:
        return

    df = pd.DataFrame(raw_ind, columns=pd.MultiIndex.from_product([neurons, ['raw_neuron_ind_in_list']]))
    # Manual GT: likelihood 1 where assigned (wbfm's helper also fails without this column).
    lik = pd.DataFrame(np.where(np.isfinite(raw_ind), 1.0, np.nan),
                       columns=pd.MultiIndex.from_product([neurons, ['likelihood']]))
    df = pd.concat([df, lik], axis=1)
    df = add_metadata_to_df_raw_ind(df, prj.segmentation_metadata)
    df = df.sort_index(axis=1)
    project_dir = os.path.dirname(os.path.abspath(args.project))
    out = os.path.join(project_dir, args.out_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_hdf(out, key='df_with_missing', mode='w')
    prov = dict(
        created=datetime.datetime.now().isoformat(timespec='seconds'),
        script='barlow_track/scripts/project_reanalysis/gt_tracks_from_clicks.py',
        barlow_track_git=_git_rev(),
        project=os.path.abspath(args.project), gt_nwb=os.path.abspath(args.gt_nwb),
        gt_source="NWB final_tracks x/y/z = manual GT clicks (manualSummary.manualPoints)",
        rule=("nearest segmented-object centroid within max_dist_um (wbfm calc_nearest_neighbor_matches, "
              "um distances); click voxel inside that object's mask; objects nearest to 2+ GT neurons "
              "assigned to none; otherwise NaN") if args.rule == 'centroid' else
             ("segmentation_id: the GT NWB's raw_segmentation_id; labels >= max_label (pseudo masks) and "
              "labels held by 2+ neurons' clicks assigned to none; otherwise NaN") if args.rule == 'segmentation_id' else
             ("mask: the mask at the click voxel, else the nearest mask voxel within edge_um; clicks with "
              "another neuron's click within coincide_um assigned to none; masks holding 2+ neurons' clicks "
              "assigned to none; otherwise NaN"),
        rule_name=args.rule, max_dist_um=args.max_dist_um if args.rule == 'centroid' else None,
        edge_um=args.edge_um if args.rule == 'mask' else None,
        coincide_um=args.coincide_um if args.rule == 'mask' else None,
        max_label=args.max_label if args.rule == 'segmentation_id' else None, voxel_um_zxy=list(args.voxel_um),
        output=out, shape=list(df.shape), summary=summary)
    with open(os.path.splitext(out)[0] + '_provenance.json', 'w') as f:
        json.dump(prov, f, indent=1)
    print('wrote', out, df.shape)

    if args.set_final_tracks:
        cfg = prj.project_config.get_tracking_config()
        cfg.config['final_3d_tracks_df'] = args.out_name
        cfg.update_self_on_disk()
        print('set final_3d_tracks_df ->', args.out_name)


if __name__ == '__main__':
    main()
