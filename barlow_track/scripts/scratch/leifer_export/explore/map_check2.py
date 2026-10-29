"""Second read-only check: exact NWB z rule, rawPoints offsets, baseImg components."""
import numpy as np, scipy.io as sio, h5py
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
L = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/'
W = L + 'Nguyen2017NeuronTracking/worm1/'
m = sio.loadmat(W + 'manualSummary.mat'); sl = m['stackList'].ravel().astype(int)
h = sio.loadmat(W + 'hiResData.mat')['dataAll'][0, 0]; Z = h['Z'].ravel()
f = h5py.File(L + 'Leifer_NeRVE_Worm1.nwb', 'r')
g = f['processing/CalciumActivity/NeuronCentroids/CentroidSeries']
c = g['control'][:]; d = g['data'][:]; ts = g['timestamps'][:]
mp = m['manualPoints']
def manual(t):
    a = mp[t, 0]
    return np.array([[np.nan if np.size(a[i, j]) == 0 else float(np.ravel(a[i, j])[0]) for j in range(3)] for i in range(a.shape[0])])
bad = 0
for t in range(1536):
    sel = ts == t
    if not sel.any(): continue
    fr = np.where(sl == t + 1)[0]; zz = Z[fr + 12]; up = zz[-1] > zz[0]; st = fr[0]
    a = manual(t)[c[sel]]
    pred = (st + 31 - a[:, 2]) if up else (a[:, 2] - st - 2)
    if np.nanmax(np.abs(pred - d[sel, 2])) > 1e-3: bad += 1
print('rule up: z=start+31-f ; down: z=f-start-2 ; mismatches', bad)
for sidx in [1, 2, 500, 1000, 1500]:
    ps = sio.loadmat(W + f'outputFiles/CLstraight/pointStats{sidx:05d}.mat')['pointStats'][0, 0]
    rp = ps['rawPoints'].astype(float); t = sidx - 1; sel = ts == t
    fr = np.where(sl == sidx)[0]; zz = Z[fr + 12]; up = zz[-1] > zz[0]
    res = {}
    for zoff in [-1, 0, 1]:
        zr = rp[:, 2] + zoff  # candidate 0-based frame offset from start
        zn = (31 - zr) if up else (zr - 2)
        P = np.c_[rp[:, 1], rp[:, 0], zn]
        dist, j = cKDTree(P * [1, 1, 4]).query(d[sel] * [1, 1, 4])
        ok = dist < 6
        res[zoff] = (np.median(dist).round(2), ok.sum(), (d[sel][ok] - P[j[ok]]).mean(0).round(2))
    bi = ps['baseImg']; lab, n = ndi.label(bi)
    print(sidx, 'up', up, 'nraw', len(rp), 'ngt', sel.sum(), res, '| baseImg comps', n, 'vox', bi.sum(), 'sum Volume', ps['Volume'].sum())
