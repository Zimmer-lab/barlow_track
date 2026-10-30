"""Per-axis +-1 shift check of warped NeRVE masks vs raw red intensity."""
import sys
import numpy as np, scipy.io as sio
from scipy import ndimage as ndi
W = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1/'
m = sio.loadmat(W + 'manualSummary.mat'); sl = m['stackList'].ravel().astype(int)
h = sio.loadmat(W + 'hiResData.mat')['dataAll'][0, 0]; Z = h['Z'].ravel()
dat = np.memmap(W + 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, 600, 1200)
for sidx in [int(s) for s in sys.argv[1:]]:
    fr = np.where(sl == sidx)[0]; st, n = fr[0], len(fr); up = Z[fr[-1] + 12] > Z[fr[0] + 12]
    ps = sio.loadmat(W + f'outputFiles/CLstraight/pointStats{sidx:05d}.mat')['pointStats'][0, 0]
    bi = ps['baseImg']
    lab = np.transpose(ndi.label(np.transpose(bi, (2, 1, 0)))[0], (2, 1, 0))
    tx, ty, tz = (ps[k].astype(int) for k in ('transformx', 'transformy', 'transformz'))
    v = (lab > 0) & (tz > 0)
    rows, cols, frs = ty[v], tx[v], tz[v]
    red = np.asarray(dat[st - 3: st + n + 4])[:, :, :632].transpose(2, 1, 0).astype(float)  # (632,600,f) from st-3
    out = []
    for ax in range(3):
        r = []
        for d in (-1, 0, 1):
            sh = [0, 0, 0]; sh[ax] = d
            rr = np.clip(rows + sh[0], 0, 631); cc = np.clip(cols + sh[1], 0, 599); ff = frs + 3 + sh[2]
            r.append(red[rr, cc, ff].mean())
        out.append(np.round(r, 0))
    # also: does the per-label maximum frame appear at 0? intensity-weighted offset per axis
    print(sidx, 'up' if up else 'down', 'n', n, 'tz range', frs.min(), frs.max(), '| row(ty)', out[0], 'col(tx)', out[1], 'frame(st+tz)', out[2])
