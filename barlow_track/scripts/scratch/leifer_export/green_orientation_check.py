"""Which green -> red mapping is right? Compare candidate orientations / flips / frame offsets / control-point
conventions by how well warped green overlaps the red nuclei (NLS-GCaMP6s and NLS-tagRFP are both nuclear).
Read-only.

Scores per candidate, over the z planes of a few volumes (frame map A, NeRVE masks):
  corr   Pearson correlation of red and warped green within 3 px of the NeRVE masks
  ratio  mean green inside NeRVE masks / mean green outside (within the warped green field)

Usage: python green_orientation_check.py [--volumes 100 800 1500]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import ndimage as ndi
from skimage.transform import AffineTransform, ProjectiveTransform

sys.path.insert(0, str(Path(__file__).parent))
import export_leifer_nwb as E  # noqa: E402


def coords_from(tform, red_offset=0.):
    """Green (dim0, dim1) sampled for each red pixel, control points read as MATLAB (x = dim 1, y = dim 0)."""
    rr, cc = np.mgrid[0:E.NX, 0:E.NY].astype(float)
    uv = tform.inverse(np.c_[cc.ravel(), rr.ravel()] + red_offset)
    return np.stack([uv[:, 1], uv[:, 0]]).reshape(2, E.NX, E.NY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--volumes', type=int, nargs='+', default=[100, 800, 1500])
    args = ap.parse_args()
    a = sio.loadmat(E.RAW / 'alignments.mat')['alignments'][0, 0]['S2AHiRes'][0, 0]
    A, S = a['Aall'], a['Sall']

    def fit(src, dst, cls=ProjectiveTransform):
        t = cls()
        assert t.estimate(src, dst)
        return t

    H = fit(A, S)
    Hswap = fit(A[:, ::-1], S[:, ::-1])
    rr, cc = np.mgrid[0:E.NX, 0:E.NY].astype(float)
    ident = np.stack([rr, cc])  # crops overlaid, no warp
    # name: (coords, green transform before warp, frame offset)
    cands = {
        'NeRVE projective (export)': (coords_from(H), None, 0),
        'frame offset -1': (coords_from(H), None, -1),
        'frame offset +1': (coords_from(H), None, 1),
        'frame offset -2': (coords_from(H), None, -2),
        'frame offset +2': (coords_from(H), None, 2),
        # transform fit with the columns of Aall / Sall read as (row, col) but applied as (x, y): a transpose
        'control points as (row, col)': (coords_from(Hswap), None, 0),
        'green flipped along 632/566 axis': (coords_from(H), 'flip0', 0),
        'green flipped along 600 axis': (coords_from(H), 'flip1', 0),
        'green rotated 180': (coords_from(H), 'rot180', 0),
        'no warp (crops overlaid)': (ident, None, 0),
        '1-based points (red +1 px)': (coords_from(H, red_offset=1.), None, 0),
        'affine fit': (coords_from(fit(A, S, AffineTransform)), None, 0),
    }
    vols = E.load_volume_table()
    dat = np.memmap(E.RAW / 'sCMOS_Frames_U16_1200x600.dat', dtype='<u2', mode='r').reshape(-1, *E.DAT_SHAPE)
    score = {k: dict(corr=[], gin=0., nin=0, gout=0., nout=0) for k in cands}
    for t in args.volumes:
        fm = E.frame_map(t, vols, 'A')
        m = E.nerve_masks((t, fm, vols['start'][t], 32))
        seg = np.zeros(E.NX * E.NY * 32, bool)
        seg[m['idx']] = True
        seg = seg.reshape(E.NX, E.NY, 32)
        for k in np.flatnonzero(fm >= 0):
            mk = seg[:, :, k]
            if mk.sum() < 200:
                continue
            region = ndi.binary_dilation(mk, iterations=3)
            red = np.asarray(dat[fm[k]])[:, :E.NX].T.astype(float)
            for name, (coords, op, dt) in cands.items():
                gp = np.asarray(dat[fm[k] + dt])[:, 634:1200].T
                gp = {None: gp, 'flip0': gp[::-1], 'flip1': gp[:, ::-1], 'rot180': gp[::-1, ::-1]}[op]
                g = ndi.map_coordinates(gp.astype(np.float32), coords, order=1, mode='constant', cval=0.)
                valid = g > 0
                s = score[name]
                s['corr'].append(np.corrcoef(g[region & valid], red[region & valid])[0, 1])
                s['gin'] += g[mk & valid].sum()
                s['nin'] += int((mk & valid).sum())
                s['gout'] += g[~mk & valid].sum()
                s['nout'] += int((~mk & valid).sum())
    print(f'volumes {args.volumes}, {len(score["NeRVE projective (export)"]["corr"])} planes')
    rows = []
    for name, s in score.items():
        rows.append((np.mean(s['corr']), name, (s['gin'] / s['nin']) / (s['gout'] / s['nout'])))
    for c, name, ratio in sorted(rows, reverse=True):
        print(f'  {name:36s} corr {c:.4f}   in/out green {ratio:.3f}')


if __name__ == '__main__':
    main()
