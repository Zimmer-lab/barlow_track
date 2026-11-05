"""GT-free check of a recording's voxel size from the shape of its segmented objects.

C. elegans head nuclei are roughly the same physical size in every lab, so the per-object spread of
mask voxels (standard deviation of voxel coordinates, a size measure robust to the mask boundary
rule) in voxel units gives the voxel size up to one common factor: sigma_vox = sigma_um / voxel_um.
Comparing recordings against Zimmer (voxels 1.5 x 0.325 x 0.325 um, known) gives
    voxel_um(rec) ~= voxel_um(zimmer) * sigma_vox(zimmer) / sigma_vox(rec)
separately for z and xy. Axial blur and segmenter differences bias z (and to a lesser degree xy), so
treat this as a consistency check of the stated voxel sizes (a 1.5x pixel-size error shows up
plainly), not a measurement.

    python barlow_track/scripts/scratch/voxel_size_from_masks.py --labels zimmer_1128 samuel_137 samuel_184
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


def object_spreads(seg, min_voxels=20):
    """Per-object std of voxel coordinates (z, x, y) and voxel count for one (Z, X, Y) label volume."""
    seg = np.asarray(seg)
    lab = seg.ravel()
    keep = lab > 0
    idx = np.nonzero(keep)[0]
    lab = lab[keep]
    coords = np.stack(np.unravel_index(idx, seg.shape), axis=1).astype(np.float64)
    uniq, inv, counts = np.unique(lab, return_inverse=True, return_counts=True)
    s1 = np.zeros((len(uniq), 3)); s2 = np.zeros((len(uniq), 3))
    for d in range(3):
        s1[:, d] = np.bincount(inv, coords[:, d], minlength=len(uniq))
        s2[:, d] = np.bincount(inv, coords[:, d] ** 2, minlength=len(uniq))
    mean = s1 / counts[:, None]
    var = np.maximum(s2 / counts[:, None] - mean ** 2, 0)
    ok = counts >= min_voxels
    return np.sqrt(var[ok]), counts[ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--projects', nargs='*', default=None,
                    help='label=project_config.yaml overrides / additions')
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--out', default=None, help='JSON lines output')
    args = ap.parse_args()

    from barlow_track.scripts.scratch.make_per_recording_rounds import TRAIN_PROJECTS
    from wbfm.utils.projects.finished_project_data import ProjectData
    projects = dict(TRAIN_PROJECTS)
    for kv in args.projects or []:
        k, v = kv.split('=', 1)
        projects[k] = v

    for label in args.labels:
        p = ProjectData.load_final_project_data(projects[label], verbose=0)
        seg = p.raw_segmentation
        T = seg.shape[0]
        frames = np.unique(np.linspace(0, T - 1, args.n_frames).astype(int))
        sig, cnt = [], []
        for t in frames:
            s, c = object_spreads(np.asarray(seg[int(t)]))
            sig.append(s); cnt.append(c)
        sig = np.concatenate(sig); cnt = np.concatenate(cnt)
        sxy = np.sqrt((sig[:, 1] ** 2 + sig[:, 2] ** 2) / 2)
        rec = dict(label=label, project=projects[label], volume_shape=list(seg.shape[1:]),
                   frames=frames.tolist(), n_objects=int(len(cnt)),
                   median_voxels=float(np.median(cnt)),
                   sigma_z_vox=float(np.median(sig[:, 0])), sigma_xy_vox=float(np.median(sxy)),
                   sigma_x_vox=float(np.median(sig[:, 1])), sigma_y_vox=float(np.median(sig[:, 2])),
                   ratio_xy_over_z=float(np.median(sxy / np.maximum(sig[:, 0], 1e-6))))
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
