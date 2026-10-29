"""Read-only: for up/down stacks, is NeRVE rawPoints z in acquisition order or ascending-Z order?
Compare in dat-frame space: GT frame (manualPoints z) vs rawPoints mapped by each hypothesis."""
import numpy as np, scipy.io as sio
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
W = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1/'
m = sio.loadmat(W + 'manualSummary.mat'); sl = m['stackList'].ravel().astype(int); mp = m['manualPoints']
h = sio.loadmat(W + 'hiResData.mat')['dataAll'][0, 0]; Z = h['Z'].ravel()
p = sio.loadmat(W + 'outputFiles/pointStatsNew.mat')['pointStatsNew'][0]
zw = 4.0
res = {True: {'acq': [], 'rev': []}, False: {'acq': [], 'rev': []}}
for t in range(0, 1523, 7):
    a = mp[t, 0]
    gt = np.array([[np.nan if np.size(a[i, j]) == 0 else float(np.ravel(a[i, j])[0]) for j in range(3)] for i in range(a.shape[0])])
    gt = gt[~np.isnan(gt[:, 0])]
    rp = p[t]['rawPoints'].astype(float)
    if len(rp) == 0 or len(gt) == 0: continue
    fr = np.where(sl == t + 1)[0]; st, n = fr[0], len(fr)
    up = Z[fr[-1] + 12] > Z[fr[0] + 12]
    G = np.c_[gt[:, 1], gt[:, 0], gt[:, 2] * zw]
    for hyp, fz in {'acq': st + rp[:, 2], 'rev': st + 1 + (n - rp[:, 2])}.items():
        R = np.c_[rp[:, 1], rp[:, 0], fz * zw]
        D = cdist(G, R); r, c = linear_sum_assignment(D)
        res[up][hyp].append(np.median(D[r, c]))
for up in res:
    print('up' if up else 'down', {k: (len(v), np.round(np.median(v), 2)) for k, v in res[up].items()})
