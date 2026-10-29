"""Read-only: how many GT (manualPoints) neurons have a NeRVE detection (pointStatsNew.rawPoints) nearby, per volume.
Matching in raw (row, col, dat-frame) space; rawPoints = (col, row, frame offset from stack start)."""
import numpy as np, scipy.io as sio
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
W = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1/'
m = sio.loadmat(W + 'manualSummary.mat'); sl = m['stackList'].ravel().astype(int); mp = m['manualPoints']
p = sio.loadmat(W + 'outputFiles/pointStatsNew.mat')['pointStatsNew'][0]
zw = 1.5 / 0.3226  # weight frames vs pixels (assumes 1.5 um/frame, uncertain)
rec, nzero, prec, dists = [], 0, [], []
for t in range(1523):
    a = mp[t, 0]
    gt = np.array([[np.nan if np.size(a[i, j]) == 0 else float(np.ravel(a[i, j])[0]) for j in range(3)] for i in range(a.shape[0])])
    gt = gt[~np.isnan(gt[:, 0])]  # (col?,row?,frame) as in manual: col0=600-axis? -> manual[:,0]=NWB dim2? see report
    rp = p[t]['rawPoints'].astype(float)
    if len(rp) == 0:
        nzero += 1; continue
    st = np.where(sl == t + 1)[0][0]
    # manual (c0,c1,frame); NWB = (c1, c0, z). rawPoints (x, y, zoff) with NWB dim1 = y = rp[:,1]
    G = np.c_[gt[:, 1], gt[:, 0], gt[:, 2] * zw]
    R = np.c_[rp[:, 1], rp[:, 0], (rp[:, 2] + st) * zw]
    D = cdist(G, R)
    r, c = linear_sum_assignment(D)
    ok = D[r, c] < 8  # px-equivalent (~2.6 um)
    rec.append(ok.mean()); dists += list(D[r, c][ok])
print('volumes with 0 detections:', nzero)
print('GT recall by NeRVE detections (Hungarian, <8px-eq): mean %.3f, pct' % np.mean(rec), np.percentile(rec, [5, 25, 50, 75]).round(3))
print('matched distance median %.2f p90 %.2f' % (np.median(dists), np.percentile(dists, 90)))
