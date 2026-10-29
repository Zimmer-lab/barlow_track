"""Read-only prototype: rasterize NeRVE CLstraight segmentation (baseImg + transformx/y/z) into raw
(row632, col600, frame) space for one volume; check holes and alignment to red intensity.
Usage: python -I warp_proto.py [stackIdx]"""
import sys
import numpy as np, scipy.io as sio
from scipy import ndimage as ndi
W = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1/'
sidx = int(sys.argv[1]) if len(sys.argv) > 1 else 100
m = sio.loadmat(W + 'manualSummary.mat'); sl = m['stackList'].ravel().astype(int)
fr = np.where(sl == sidx)[0]; st, n = fr[0], len(fr)
ps = sio.loadmat(W + f'outputFiles/CLstraight/pointStats{sidx:05d}.mat')['pointStats'][0, 0]
bi = ps['baseImg']
lab = np.transpose(ndi.label(np.transpose(bi, (2, 1, 0)))[0], (2, 1, 0))  # MATLAB column-major order == pointIdx
tx, ty, tz = (ps[k].astype(int) for k in ('transformx', 'transformy', 'transformz'))
v = (lab > 0) & (tz > 0)
seg = np.zeros((632, 600, n + 2), np.uint16)  # frames st .. st+n+1
rows, cols, frs = ty[v], tx[v], tz[v]
ok = (rows < 632) & (cols < 600) & (frs < n + 2)
seg[rows[ok], cols[ok], frs[ok]] = lab[v][ok]
nl = lab.max()
filled = sum(ndi.binary_fill_holes(ndi.binary_closing(seg == k, iterations=1)).sum() for k in range(1, nl + 1))
print(f'stack {sidx}: n frames {n}, labels {nl}, raw voxels labelled {np.count_nonzero(seg)}, after per-label closing+fill {filled}')
sizes = np.bincount(seg.ravel())[1:]
print('raw voxels per label pct', np.percentile(sizes[sizes > 0], [5, 50, 95]))
dat = np.memmap(W + 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, 600, 1200)
red = np.asarray(dat[st - 2: st + n + 4])[:, :, :632].transpose(2, 1, 0).astype(float)  # (632,600,frames) from st-2
bg = np.median(red)
for dz in [-1, 0, 1]:
    for dr in [-1, 0, 1]:
        mask = seg > 0
        idx = np.nonzero(mask)
        vals = red[idx[0] + dr, idx[1], idx[2] + 2 + dz]
        if dr == 0 or dz == 0:
            print(f'shift frame {dz:+d} row {dr:+d}: mean red in mask {vals.mean():.1f} (bg median {bg:.0f})')
