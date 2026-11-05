"""Does StarDist segment a recording better after resampling it to the object size its model was trained on?

The wbfm StarDist model (students_and_lukas_3d_zarr) was trained on Zimmer volumes, whose nuclei have a
median per-object spread of ~3.0 voxels in xy and ~1.5 in z (voxel_size_from_masks.py). Samuel 153's are
~1.7 / 1.0 voxels (0.45 um pixels, 1.75 um planes), and StarDist on the raw volumes leaves 36% of the GT
clicks outside every mask and puts 34% in masks shared by 2+ GT neurons (r261105-3532825).

For a few volumes, this zooms the red volume by (fz, fxy, fxy) (linear interpolation), segments with the
model as wbfm does (percentile normalization 1-99.8, predict_instances), maps the labels back to the
original grid (nearest neighbour), and scores the GT clicks with the strict rule of gt_tracks_from_clicks
(--rule mask --edge_um 0): a click is assigned if the mask at its voxel holds no other neuron's click.
Coincident-click exclusion is left out (46 of 120,923 Samuel clicks).

    python barlow_track/scripts/project_reanalysis/stardist_rescale_test.py --project .../153_stardist/project_config.yaml \
        --gt_nwb .../samuel_data/153.nwb --zooms 1,1 1,1.4 1.17,1.75 --n_volumes 10
"""
import argparse
import json
import os
import sys

import numpy as np


def score_clicks(labels, clicks):
    """labels (Z, X, Y); clicks (N, 3) z, x, y voxels (NaN rows = no click). Returns counts."""
    ok = np.isfinite(clicks).all(1)
    p = np.round(clicks[ok]).astype(int)
    inside = np.all((p >= 0) & (p < np.array(labels.shape)), axis=1)
    lab = np.zeros(len(p), dtype=np.int64)
    lab[inside] = labels[p[inside, 0], p[inside, 1], p[inside, 2]]
    in_mask = lab > 0
    u, c = np.unique(lab[in_mask], return_counts=True)
    shared = np.isin(lab, u[c > 1]) & in_mask
    _, sizes = np.unique(labels[labels > 0], return_counts=True)
    return dict(clicks=int(ok.sum()), in_mask=int(in_mask.sum()), shared=int(shared.sum()),
                assigned=int((in_mask & ~shared).sum()), objects=int(len(sizes)),
                median_voxels=float(np.median(sizes)) if len(sizes) else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True)
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--zooms', nargs='+', default=['1,1', '1,1.4', '1.17,1.75'], help='fz,fxy pairs')
    ap.add_argument('--n_volumes', type=int, default=10)
    ap.add_argument('--model', default='students_and_lukas_3d_zarr')
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.75, 0.45, 0.45],
                    help='z x y voxel size in um, for the object volumes (default: Samuel 153 NWB)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    from scipy.ndimage import zoom
    from csbdeep.utils import normalize
    from wbfm.utils.projects.finished_project_data import ProjectData
    from wbfm.utils.segmentation.util.utils_model import get_stardist_model
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays

    gt = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    prj = ProjectData.load_final_project_data(args.project, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(gt)
    model = get_stardist_model(args.model)
    T = prj.num_frames
    vols = np.unique(np.linspace(0, T - 1, args.n_volumes).astype(int))
    totals = {z: dict(clicks=0, in_mask=0, shared=0, assigned=0, objects=[], vox=[]) for z in args.zooms}
    for t in vols:
        vol = np.asarray(prj.red_data[int(t)], dtype=np.float32)
        clicks = np.stack([a['gz'][t], a['gx'][t], a['gy'][t]], axis=1)
        for zs in args.zooms:
            fz, fxy = (float(v) for v in zs.split(','))
            big = zoom(vol, (fz, fxy, fxy), order=1) if (fz, fxy) != (1.0, 1.0) else vol
            lab, _ = model.predict_instances(normalize(big, 1, 99.8, axis=(0, 1, 2)), show_tile_progress=False)
            if lab.shape != vol.shape:
                # nearest-neighbour back to the original grid: sample at each original voxel's centre
                idx = [np.minimum(((np.arange(n) + 0.5) * f).astype(int), m - 1)
                       for n, f, m in zip(vol.shape, (fz, fxy, fxy), lab.shape)]
                lab = lab[np.ix_(*idx)]
            s = score_clicks(lab, clicks)
            for k in ('clicks', 'in_mask', 'shared', 'assigned'):
                totals[zs][k] += s[k]
            totals[zs]['objects'].append(s['objects'])
            totals[zs]['vox'].append(s['median_voxels'])
            print(f"t={t} zoom {zs}: {s}", flush=True)
    for zs, d in totals.items():
        rec = dict(zoom=zs, volumes=vols.tolist(), clicks=d['clicks'],
                   frac_in_mask=d['in_mask'] / d['clicks'], frac_shared=d['shared'] / d['clicks'],
                   frac_assigned=d['assigned'] / d['clicks'], median_objects=float(np.median(d['objects'])),
                   median_object_um3=float(np.median(d['vox']) * np.prod(args.voxel_um)))
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
