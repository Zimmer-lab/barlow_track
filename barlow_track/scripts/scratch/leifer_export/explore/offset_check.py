"""Read-only: check 0/1-based offsets of GT (manualPoints / NWB centroids) against raw .dat intensities."""
import numpy as np, scipy.io as sio, h5py
from scipy import ndimage as ndi
L = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/'
W = L + 'Nguyen2017NeuronTracking/worm1/'
dat = np.memmap(W + 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, 600, 1200)
f = h5py.File(L + 'Leifer_NeRVE_Worm1.nwb', 'r')
g = f['processing/CalciumActivity/NeuronCentroids/CentroidSeries']
c = g['control'][:]; d = g['data'][:]; ts = g['timestamps'][:]
m = sio.loadmat(W + 'manualSummary.mat'); sl = m['stackList'].ravel().astype(int)
h = sio.loadmat(W + 'hiResData.mat')['dataAll'][0, 0]; Z = h['Z'].ravel()
allz = []; allxy = []
for t in [0, 1, 200, 201, 1200]:
    fr = np.where(sl == t + 1)[0]; up = Z[fr[-1] + 12] > Z[fr[0] + 12]; st = fr[0]
    frames = np.arange(st - 2, st + 36)
    red = np.asarray(dat[frames])[:, :, :632].transpose(0, 2, 1).astype(float)  # (f, 632, 600)
    red = ndi.gaussian_filter(red, (0.7, 1.5, 1.5))
    gt = d[ts == t]
    # NWB z -> dat frame
    fz = (st + 31 - gt[:, 2]) if up else (gt[:, 2] + st + 2)
    dz = []; dxy = []
    for (r, cc, _), fz_ in zip(gt, fz):
        r0, c0, f0 = int(round(r)), int(round(cc)), int(round(fz_)) - frames[0]
        if not (6 <= r0 < 626 and 6 <= c0 < 594 and 3 <= f0 < len(frames) - 3): continue
        win = red[f0 - 3:f0 + 4, r0 - 5:r0 + 6, c0 - 5:c0 + 6]
        k = np.unravel_index(np.argmax(win), win.shape)
        dz.append(k[0] - 3); dxy.append((r0 - 5 + k[1] - r, c0 - 5 + k[2] - cc))
    dz = np.array(dz); dxy = np.array(dxy)
    allz += list(dz); allxy += list(dxy)
    print(t, 'up', up, 'n', len(dz), 'frame offset of intensity peak vs GT: mean %.2f' % dz.mean(), np.bincount(dz + 3), 'xy peak - GT median', np.median(dxy, 0).round(2))
print('overall z', np.mean(allz).round(2), 'xy', np.median(np.array(allxy), 0).round(2))
