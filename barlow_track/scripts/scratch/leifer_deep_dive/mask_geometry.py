"""Geometry of every detection in the Leifer re-export A2: centroid, size, and where the GT
clicks are relative to it, to tell distractors that are other cells from fragments or
duplicates of GT neurons.

Per mask (frame t, label): centroid (z, x, y voxels), voxels, the nearest GT click of that
frame and its distance in um (xy 0.3226, z 1.5; leifer_export/README.md), the number of GT
clicks inside the mask, and whether the mask is a GT neuron's (NeuronSegmentationID).

    python barlow_track/scripts/scratch/leifer_deep_dive/mask_geometry.py --out <dir>/masks.npz
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NWB = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/leifer_reexport/'
       'Leifer_NeRVE_Worm1_reexport_A2.nwb')
UM = np.array([1.5, 0.3226, 0.3226])  # z, x, y


def _load():
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    p = ProjectData.load_final_project_data(NWB, allow_hybrid_loading=True, verbose=0)
    return p, _load_nwb_arrays(p)


_P = _A = None


def frame_rows(t):
    from skimage.measure import regionprops
    global _P, _A
    if _P is None:
        _P, _A = _load()
    seg = np.asarray(_P.raw_segmentation[t])
    props = regionprops(seg)
    g = np.stack([_A['gz'][t], _A['gx'][t], _A['gy'][t]], 1)
    fin = np.isfinite(g).all(1)
    js = np.flatnonzero(fin)
    gi = np.round(g[js]).astype(int)
    shape = np.array(seg.shape)
    inb = (gi >= 0).all(1) & (gi < shape).all(1)
    click_label = np.zeros(len(js), dtype=int)
    click_label[inb] = seg[tuple(gi[inb].T)]
    gt_labels = set(_A['gs'][t, js][np.isfinite(_A['gs'][t, js])].astype(int))
    out = []
    for r in props:
        c = np.array(r.centroid)
        if len(js):
            d = np.sqrt((((g[js] - c) * UM) ** 2).sum(1))
            k = d.argmin()
            near_j, near_d = int(js[k]), float(d[k])
        else:
            near_j, near_d = -1, np.inf
        out.append((t, r.label, *c, r.area, near_j, near_d, int((click_label == r.label).sum()),
                    int(r.label in gt_labels)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    p, _ = _load()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for fr in ex.map(frame_rows, range(p.num_frames), chunksize=8) for r in fr]
    a = np.array(rows, dtype=float)
    np.savez(args.out, t=a[:, 0].astype(int), label=a[:, 1].astype(int), zxy=a[:, 2:5], voxels=a[:, 5],
             near_gt=a[:, 6].astype(int), near_um=a[:, 7], clicks_inside=a[:, 8].astype(int),
             is_gt=a[:, 9].astype(bool))
    print(f"{len(a)} masks in {p.num_frames} frames -> {args.out}")


if __name__ == '__main__':
    main()
