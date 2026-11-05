"""NeRVE's own tracks of the Leifer worm1 recording as saved tracks for score_gt_variants.py.

NeRVE (Nguyen et al. 2017) writes outputFiles/pointStatsNew.mat: per volume (stackIdx = t + 1) its detections
(pointIdx 1..n, rawPoints, ...) and straightPoints / trackIdx, which are LONGER than the detections: rows
past n are NeRVE's 'error correction' points (its tutorial, wholeBrainTutorial_pointstats.m), tracked positions
with no detection and so no mask. trackIdx (1..119, NaN = untracked) is NeRVE's identity across volumes.

In the re-export (scratch/leifer_export/export_leifer_nwb.py) every NeRVE detection is a mask whose label is its
pointIdx, so a detection's track is a mask label directly. Extra points are placed through that volume's
CLstraight/pointStatsNNNNN.mat lookups (raw col / row / frame offset = transformx / y / z indexed at the rounded
straightPoints, the rule that reproduces rawPoints for the detections) and the same frame map as the export
(rule A), then given the label of the mask at that voxel in each segmentation passed with --seg (0 = none).

Writes to --out:
  nerve_dets_tracks.h5            pred_all, detections only (the headline: NeRVE scored on its own masks)
  nerve_ext_<seg>_tracks.h5       pred_all, detections + extra points that land in a mask of <seg>
  nerve_points.npz                every tracked point (T, tracks, 3) in re-export voxels (z, x632, y600),
                                  kind 1 = detection, 2 = extra point, 0 = none; for distance-based scores
  nerve_tracks_summary.json       counts and the checks below
Checks: detection labels exist in the project's metadata; mask centroids agree with rawPoints; the mapping of
detections' straightPoints reproduces rawPoints and hits their own mask label in each --seg.

    python .../nerve_tracks.py --project .../Leifer_NeRVE_Worm1_reexport_A2_nervegt/project_config.yaml \
        --seg nervemasks=.../nervegt/project_config.yaml --seg nwbseg=.../Leifer_NeRVE_Worm1_reexport_A2.nwb \
        --out <dir> [--workers 16]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.io as sio

RAW = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1'


def _git_rev():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(['git', '-C', here, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.call(['git', '-C', here, 'diff', '--quiet', 'HEAD', '--', os.path.abspath(__file__)])
        return rev + (' (script modified)' if dirty else '')
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True, help='re-export project with NeRVE masks (metadata check)')
    ap.add_argument('--seg', nargs='*', default=[], help='NAME=project_config.yaml or .nwb: segmentations '
                                                          'the extra points are looked up in')
    ap.add_argument('--raw', default=RAW)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--max_volumes', type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scratch', 'leifer_export'))
    import export_leifer_nwb as ex
    from wbfm.utils.projects.finished_project_data import ProjectData

    raw = ex.Path(args.raw)
    pn = sio.loadmat(raw / 'outputFiles' / 'pointStatsNew.mat', squeeze_me=True,
                     struct_as_record=False)['pointStatsNew']
    if not np.array_equal([p.stackIdx for p in pn], np.arange(1, len(pn) + 1)):
        raise SystemExit('pointStatsNew stackIdx is not 1..n')
    prj = ProjectData.load_final_project_data(args.project, allow_hybrid_loading=True, verbose=0)
    T = prj.num_frames
    n_vol = min(len(pn), T) if args.max_volumes is None else args.max_volumes
    n_tracks = int(max(np.nanmax(np.atleast_1d(p.trackIdx).astype(float)) for p in pn
                       if np.isfinite(np.atleast_1d(p.trackIdx).astype(float)).any()))
    vols = ex.load_volume_table(raw)
    segs, seg_projects = {}, []  # keep the ProjectData alive: dropping it closes the NWB file under the array
    for spec in args.seg:
        name, path = spec.split('=', 1)
        sp_ = prj if os.path.abspath(path) == os.path.abspath(args.project) else \
            ProjectData.load_final_project_data(path, allow_hybrid_loading=True, verbose=0)
        seg_projects.append(sp_)
        segs[name] = sp_.raw_segmentation
    md = prj.segmentation_metadata
    seg_lock = threading.Lock()  # h5py-backed segmentations are not safe to read from several threads

    def one(t):
        p = pn[t]
        n = int(np.size(p.pointIdx))
        tr = np.atleast_1d(p.trackIdx).astype(float)
        sp = np.atleast_2d(p.straightPoints).astype(float)
        fin = np.isfinite(tr) & np.isfinite(sp[:len(tr), 0])
        out = dict(t=t, n_det=n, det_tracked=int(fin[:n].sum()), ext_tracked=int(fin[n:].sum()))
        det = [(int(tr[k]) - 1, k + 1) for k in np.flatnonzero(fin[:n])]  # (track col, label = pointIdx)
        m = md.segmentation_metadata[t] if t in md.segmentation_metadata else None
        labs_md = set(int(x) for x in (m.index if m is not None else []))
        out['det_label_not_in_md'] = int(sum(lab not in labs_md for _, lab in det))
        if n and len(labs_md):
            rp = np.atleast_2d(p.rawPoints)
            ks = [k for k in range(n) if k + 1 in labs_md]
            c = np.array([np.asarray(m.loc[k + 1, 'centroids'], float) for k in ks])
            out['det_centroid_dxy_px'] = float(np.median(np.abs(c[:, 1:] - rp[ks][:, [1, 0]])))
        pos = np.full((n_tracks, 3), np.nan)
        kind = np.zeros(n_tracks, np.int8)
        ext_lab = {s: {} for s in segs}
        out['ext_unmapped'] = 0
        f = raw / 'outputFiles' / 'CLstraight' / f'pointStats{t + 1:05d}.mat'
        try:
            ps = sio.loadmat(f)['pointStats'][0, 0]
            tx, ty, tz = (ps[k].astype(np.int64) for k in ('transformx', 'transformy', 'transformz'))
        except Exception as e:
            out['clstraight_error'] = repr(e)[:200]
            out['ext_unmapped'] = out['ext_tracked']
            return out, det, pos, kind, ext_lab
        fmap = ex.frame_map(t, vols, 'A')
        inv = np.full(max(vols['start'][t] + 64, fmap.max() + 1), -1, int)
        inv[fmap[fmap >= 0]] = np.flatnonzero(fmap >= 0)
        rows_k = np.flatnonzero(fin)
        idx = np.round(sp[rows_k]).astype(int) - 1
        ok = (idx >= 0).all(1) & (idx < np.array(tx.shape)).all(1)
        xyz = np.full((len(rows_k), 3), -1, np.int64)
        xyz[ok] = np.stack([tx[tuple(idx[ok].T)], ty[tuple(idx[ok].T)], tz[tuple(idx[ok].T)]], 1)
        valid = ok & (xyz > 0).all(1)  # 0 = NaN in MATLAB
        # rawPoints convention: (col, row, frame offset); re-export voxel (z, x632 = row, y600 = col)
        frame = vols['start'][t] + xyz[:, 2]
        z = np.where(valid, inv[np.clip(frame, 0, len(inv) - 1)], -1)
        vox = np.stack([z, xyz[:, 1], xyz[:, 0]], 1)
        valid &= (z >= 0) & (vox[:, 1] < ex.NX) & (vox[:, 2] < ex.NY)
        is_det = rows_k < n
        if is_det.any():
            rp = np.atleast_2d(p.rawPoints)[rows_k[is_det]]
            g = valid[is_det]
            out['det_map_err_px'] = float(np.median(np.abs(xyz[is_det][g][:, :2] - rp[g][:, :2]))) if g.any() else None
        with seg_lock:
            vol = {s: np.asarray(seg[t]) for s, seg in segs.items()}
        for s, v in vol.items():
            if is_det.any():
                d = valid & is_det
                hit = v[tuple(vox[d].T)] == rows_k[d] + 1
                out[f'det_map_hits_own_label_{s}'] = [int(hit.sum()), int(d.sum())]
        for j, r in enumerate(rows_k):
            col = int(tr[r]) - 1
            if r < n:
                kind[col] = 1
                c = m.loc[r + 1, 'centroids'] if m is not None and (r + 1) in labs_md else None
                pos[col] = np.asarray(c, float) if c is not None else (vox[j] if valid[j] else np.nan)
                continue
            if not valid[j]:
                out['ext_unmapped'] += 1
                continue
            kind[col] = 2
            pos[col] = vox[j]
            for s, v in vol.items():
                lab = int(v[tuple(vox[j])])
                ext_lab[s][col] = lab
                if 0 < lab <= n:  # on a NeRVE detection: one another track holds, or an untracked one
                    key = f'ext_on_tracked_det_{s}' if fin[lab - 1] else f'ext_on_untracked_det_{s}'
                    out[key] = out.get(key, 0) + 1
        return out, det, pos, kind, ext_lab

    P = np.full((T, n_tracks), np.nan)
    E = {s: np.full((T, n_tracks), np.nan) for s in segs}
    POS = np.full((T, n_tracks, 3), np.nan)
    KIND = np.zeros((T, n_tracks), np.int8)
    stats = []
    with ThreadPoolExecutor(args.workers) as pool:
        for out, det, pos, kind, ext_lab in pool.map(one, range(n_vol)):
            t = out['t']
            for col, lab in det:
                P[t, col] = lab
            POS[t], KIND[t] = pos, kind
            for s in segs:
                for col, lab in ext_lab[s].items():
                    if lab > 0:
                        E[s][t, col] = lab
            stats.append(out)
            if t % 100 == 0:
                print(json.dumps(out), flush=True)
    names = [f'nerve_{k + 1:03d}' for k in range(n_tracks)]
    pd.DataFrame(P, columns=names).to_hdf(os.path.join(args.out, 'nerve_dets_tracks.h5'), key='pred_all', mode='w')
    for s in segs:
        both = np.where(np.isfinite(P), P, E[s])
        pd.DataFrame(both, columns=names).to_hdf(os.path.join(args.out, f'nerve_ext_{s}_tracks.h5'),
                                                 key='pred_all', mode='w')
    np.savez_compressed(os.path.join(args.out, 'nerve_points.npz'), pos=POS, kind=KIND, names=np.array(names))
    S = pd.DataFrame(stats)
    summary = dict(script='barlow_track/scripts/project_reanalysis/nerve_tracks.py', barlow_track_git=_git_rev(),
                   project=os.path.abspath(args.project), segs=dict(s.split('=', 1) for s in args.seg),
                   volumes=n_vol, tracks=n_tracks, detections=int(S.n_det.sum()),
                   tracked_detections=int(S.det_tracked.sum()), tracked_extra_points=int(S.ext_tracked.sum()),
                   extra_points_unmapped=int(S.ext_unmapped.sum()),
                   det_labels_not_in_metadata=int(S.det_label_not_in_md.sum()),
                   det_centroid_dxy_px_median=float(S.det_centroid_dxy_px.median()),
                   det_map_err_px_median=float(S.det_map_err_px.median()),
                   clstraight_errors={int(r.t): r.clstraight_error for r in S.itertuples()
                                      if isinstance(getattr(r, 'clstraight_error', None), str)})
    for s in segs:
        h = np.array([x for x in S[f'det_map_hits_own_label_{s}'] if isinstance(x, list)])
        summary[f'det_map_hits_own_label_{s}'] = round(float(h[:, 0].sum() / max(h[:, 1].sum(), 1)), 4)
        ext = (KIND == 2)
        summary[f'extra_points_in_a_mask_{s}'] = int((np.isfinite(E[s]) & ext).sum())
        summary[f'extra_points_in_background_{s}'] = int((~np.isfinite(E[s]) & ext).sum())
        lab = E[s][ext & np.isfinite(E[s])]
        summary[f'extra_points_in_label_ge_1000_{s}'] = int((lab >= 1000).sum())
        for k in ('tracked', 'untracked'):
            c = f'ext_on_{k}_det_{s}'
            summary[f'extra_points_on_{k}_nerve_detection_{s}'] = int(S[c].fillna(0).sum()) if c in S else 0
    with open(os.path.join(args.out, 'nerve_tracks_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
