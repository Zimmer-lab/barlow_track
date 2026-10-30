"""Measure the old NWB's (Leifer_NeRVE_Worm1.nwb) frame map and compare it to export rule A. Read-only.

For every volume and z plane, find the .dat frame (offset from the volume's stackList start) whose red
part equals the plane: -1 = all-zero plane, -2 = no exact match, -3 = several (duplicated .dat frames).
Then compare with export_leifer_nwb.frame_map(t, 'A'). Takes ~20 min with 8 processes (network bound).

Usage: python old_map_check.py --out old_map.npy            (measure, then compare)
       python old_map_check.py --cached old_map.npy         (compare only)
"""
import argparse
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import export_leifer_nwb as E  # noqa: E402

OLD = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb'
SLAB = 48  # the old file's time chunk


def measure_slab(args):
    i0, start, nfr = args
    dat = np.memmap(E.RAW / 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, *E.DAT_SHAPE)
    with h5py.File(OLD, 'r') as f:
        old = f['acquisition/CalciumImageSeries/data'][i0:i0 + SLAB, :, :, :, 0]
    res = np.full((len(old), old.shape[-1]), -2, int)
    for j in range(len(old)):
        st, n = start[i0 + j], nfr[i0 + j]
        lo, hi = max(st - 4, 0), min(st + n + 4, len(dat))
        cand = np.asarray(dat[lo:hi])[:, :, :E.NX].transpose(0, 2, 1)
        for k in range(old.shape[-1]):
            p = old[j, :, :, k]
            if not p.any():
                res[j, k] = -1
                continue
            hits = [lo + q - st for q in range(len(cand)) if np.array_equal(cand[q], p)]
            res[j, k] = hits[0] if len(hits) == 1 else (-3 if hits else -2)
    return i0, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--cached', default=None)
    ap.add_argument('--n_workers', type=int, default=8)
    args = ap.parse_args()
    vols = E.load_volume_table()
    T = len(vols['start'])
    if args.cached:
        M = np.load(args.cached)
    else:
        M = np.full((T, 32), -9, int)
        t0 = time.time()
        with Pool(args.n_workers) as pool:
            jobs = [(i0, vols['start'], vols['nfr']) for i0 in range(0, T, SLAB)]
            for i0, r in pool.imap_unordered(measure_slab, jobs):
                M[i0:i0 + len(r)] = r
                print(i0, f'{time.time() - t0:.0f}s', flush=True)
        if args.out:
            np.save(args.out, M)
    print('codes', dict(Counter(M.ravel()[M.ravel() < 0])))
    A = np.stack([E.frame_map(t, vols, 'A') - np.where(E.frame_map(t, vols, 'A') >= 0, vols['start'][t], 0)
                  for t in range(T)])
    measured = M >= 0
    zero = M == -1
    bad_frame = measured & (M != A)
    bad_zero = zero & (A != -1)
    unmatched = M <= -2
    print(f'planes: {measured.sum()} matched to one frame, {zero.sum()} zero, {unmatched.sum()} unmatched/duplicate')
    print(f'rule A disagrees on {bad_frame.sum()} matched planes and {bad_zero.sum()} zero planes; volumes',
          sorted(set(np.flatnonzero(bad_frame.any(1) | bad_zero.any(1)))))
    print('unmatched (-2) / duplicate (-3) planes (t, z):', np.argwhere(unmatched).tolist())


if __name__ == '__main__':
    main()
