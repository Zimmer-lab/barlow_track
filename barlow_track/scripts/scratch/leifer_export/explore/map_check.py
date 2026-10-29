"""Check (read-only) the frame->volume mapping of the existing Leifer NWB and
how NeRVE pointStats rawPoints relate to NWB centroid coordinates.

Run: python -I map_check.py
"""
import numpy as np
import scipy.io as sio
import h5py
from scipy.spatial import cKDTree

L = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/'
W = L + 'Nguyen2017NeuronTracking/worm1/'

m = sio.loadmat(W + 'manualSummary.mat')
sl = m['stackList'].ravel().astype(int)
h = sio.loadmat(W + 'hiResData.mat')['dataAll'][0, 0]
s = h['stackIdx'].ravel().astype(int)
Z = h['Z'].ravel()
off = 12
print('stackList[i] == stackIdx[i+12]?', np.array_equal(sl[:-off], s[off:]))

f = h5py.File(L + 'Leifer_NeRVE_Worm1.nwb', 'r')
g = f['processing/CalciumActivity/NeuronCentroids/CentroidSeries']
c = g['control'][:]; d = g['data'][:]; ts = g['timestamps'][:]
mp = m['manualPoints']


def manual(t):
    a = mp[t, 0]
    return np.array([[np.nan if np.size(a[i, j]) == 0 else float(np.ravel(a[i, j])[0])
                      for j in range(3)] for i in range(a.shape[0])])


def nwb_z_from_frame(t):
    """Predicted mapping frame(dat, 0-based) -> nwb z for volume t (stack t+1)."""
    fr = np.where(sl == t + 1)[0]
    zz = Z[fr + off]  # hiRes index = dat index + 12
    up = zz[-1] > zz[0]
    bottom = fr[0] if up else fr[-1]
    return fr, up, bottom


# 1) verify z mapping hypothesis on all volumes with GT
bad = 0; nfr = []
for t in range(1536):
    sel = ts == t
    if not sel.any():
        continue
    a = manual(t)[c[sel]]
    fr, up, bottom = nwb_z_from_frame(t)
    nfr.append(len(fr))
    pred = 31 - np.abs(a[:, 2] - bottom)
    if np.nanmax(np.abs(pred - d[sel, 2])) > 1e-3:
        bad += 1
        if bad < 4:
            print('mismatch t', t, pred[:3], d[sel, 2][:3], 'up', up, 'nfr', len(fr))
print('z-mapping mismatches:', bad, 'frames/vol', np.unique(nfr, return_counts=True))
print('GT-less volumes:', sorted(set(range(1536)) - set(np.unique(ts).astype(int))))

# 2) rawPoints vs NWB centroids for a few volumes
for sidx in [1, 2, 100, 101, 1000]:
    ps = sio.loadmat(W + f'outputFiles/CLstraight/pointStats{sidx:05d}.mat')['pointStats'][0, 0]
    rp = ps['rawPoints'].astype(float)
    t = sidx - 1
    sel = ts == t
    gt = d[sel]  # (row632, col600, z)
    fr, up, bottom = nwb_z_from_frame(t)
    best = None
    for name, cand in {
        'rp(0,1)': rp[:, [0, 1]], 'rp(1,0)': rp[:, [1, 0]],
    }.items():
        for zname, zc in {
            'z=31-(z-1)': 31 - (rp[:, 2] - 1), 'z=31-z': 31 - rp[:, 2],
            'z=31-(n-z)': 31 - (len(fr) - rp[:, 2]), 'z=31-(n-1-z)': 31 - (len(fr) - 1 - rp[:, 2]),
        }.items():
            P = np.c_[cand, zc]
            dist, _ = cKDTree(P * [1, 1, 4]).query(gt * [1, 1, 4])  # z weighted (1.5um vs .33um)
            score = np.median(dist)
            if best is None or score < best[0]:
                best = (score, name, zname, np.percentile(dist, [50, 90]).round(2))
    print(f'stack {sidx} up={up} n_raw={len(rp)} n_gt={sel.sum()} best={best}')
