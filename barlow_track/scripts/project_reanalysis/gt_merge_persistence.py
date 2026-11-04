"""How persistent are the merges the GT conversion excludes (two GT neurons' clicks inside one mask)?

gt_tracks_from_clicks.py assigns a mask held by two or more neurons' clicks at t to none of them. A merge
that lasts one frame is segmentation noise; one that lasts hundreds of frames is a pair of neurons the
segmentation cannot separate, and both drop out of the GT (and the tracker never sees them apart). For each
project (its masks, the same NWB clicks) this reads the mask label at every click voxel and reports, per
pair of neurons sharing a label: frames merged, longest run, and the click distance while merged (clicks
under ~1.5 um apart are two annotations on one cell rather than a segmentation merge).

    python .../gt_merge_persistence.py --projects stardist=.../project_config.yaml nerve=... \
        --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb --out <dir> [--max_label 1000]
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def longest_run(frames):
    frames = sorted(frames)
    best = run = 1 if frames else 0
    for a, b in zip(frames[:-1], frames[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--projects', nargs='+', required=True, help='NAME=project_config.yaml')
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--max_label', type=int, default=None, help='ignore mask labels >= this (pseudo masks)')
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.5, 0.3226, 0.3226])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    scale = np.asarray(args.voxel_um)
    nwb = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(nwb)
    names = a['neurons']
    C = np.stack([a['gz'], a['gx'], a['gy']], 2)
    out = dict(args=vars(args))
    for name, cfg in (p.split('=', 1) for p in args.projects):
        prj = ProjectData.load_final_project_data(cfg, allow_hybrid_loading=True, verbose=0)
        seg = prj.raw_segmentation
        T = min(seg.shape[0], C.shape[0])
        shape = np.array(seg.shape[1:])

        def one(t):
            v = np.asarray(seg[t])
            ok = np.isfinite(C[t]).all(1)
            lab = np.zeros(len(names), int)
            idx = np.round(C[t][ok]).astype(int).clip(0, shape - 1)
            lab[ok] = v[idx[:, 0], idx[:, 1], idx[:, 2]]
            return t, lab

        labs = np.zeros((T, len(names)), int)
        with ThreadPoolExecutor(args.workers) as ex:
            for t, lab in ex.map(one, range(T)):
                labs[t] = lab
        if args.max_label is not None:
            labs[labs >= args.max_label] = 0
        pairs = defaultdict(list)
        dist = defaultdict(list)
        cells_in_merge = 0
        for t in range(T):
            l = labs[t]
            vals, counts = np.unique(l[l > 0], return_counts=True)
            for v in vals[counts > 1]:
                js = np.flatnonzero(l == v)
                cells_in_merge += len(js)
                for i1 in range(len(js)):
                    for i2 in range(i1 + 1, len(js)):
                        key = (names[js[i1]], names[js[i2]])
                        pairs[key].append(t)
                        dist[key].append(float(np.linalg.norm((C[t, js[i1]] - C[t, js[i2]]) * scale)))
        rows = sorted(({'pair': list(k), 'frames': len(f), 'longest_run': longest_run(f),
                        'click_dist_um_median': round(float(np.median(dist[k])), 2)} for k, f in pairs.items()),
                      key=lambda r: -r['frames'])
        n_clicks = int(np.isfinite(C[:T]).all(2).sum())
        fr = np.array([r['frames'] for r in rows]) if rows else np.zeros(0)
        lr = np.array([r['longest_run'] for r in rows]) if rows else np.zeros(0)
        cd = np.array([r['click_dist_um_median'] for r in rows]) if rows else np.zeros(0)
        out[name] = dict(
            clicks=n_clicks, clicks_in_a_shared_mask=int(cells_in_merge),
            frac_clicks_in_a_shared_mask=round(cells_in_merge / max(n_clicks, 1), 4),
            pairs=len(rows),
            pair_frames_in_pairs_merged_100_plus_frames=int(fr[fr >= 100].sum()),
            pairs_merged_100_plus_frames=int((fr >= 100).sum()),
            pairs_with_a_run_of_50_plus=int((lr >= 50).sum()),
            pair_frames_with_clicks_under_1_5um=int(fr[cd < 1.5].sum()),
            top_pairs=rows[:15])
        print(name, json.dumps({k: v for k, v in out[name].items() if k != 'top_pairs'}), flush=True)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
