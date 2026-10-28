"""Is a tracks table the Flavell GT, or a tracker's output? Compare by position, not ids.

Segmentation ids and neuron names differ between the flavell sources (project tracks file,
the project's NWB copy, images_for_charlie/flavell_data.nwb), so this matches neurons by
centroid. Each neuron of A is paired with the neuron of B whose track is closest over
the shared frames (median distance, Hungarian on neurons); per entry, A agrees with B when
B's paired neuron is present within --tol voxels in that frame. A GT copy agrees on nearly
every entry; a tracker run on the same detections agrees on its accuracy, ~0.9.

Usage (wbfm env):
  python flavell_gt_identity.py --a <tracks.h5 | project_config.yaml | .nwb> --b <same> [--offset 0]
"""
import argparse
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PAPER = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data'


def load(path):
    if path.endswith('.h5'):
        return pd.read_hdf(path)
    from barlow_track.scripts.eval_accuracy import _load_project_gt
    return _load_project_gt(path)


def xyz(df):
    """(frames, neurons, 3) array in z, x, y order, index reset to 0..T-1 by position."""
    names = list(dict.fromkeys(df.columns.get_level_values(0)))
    arr = np.stack([df.loc[:, (names, f)].droplevel(1, axis=1)[names].to_numpy(dtype=float)
                    for f in ('z', 'x', 'y')], axis=-1)
    return names, np.asarray(df.index, dtype=float), arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default=f'{PAPER}/projects/2025_07_01/3-tracking/barlow_tracker/df_barlow_tracks.h5')
    ap.add_argument('--b', default=f'{PAPER}/images_for_charlie/flavell_data.nwb')
    ap.add_argument('--offset', type=int, default=0, help='frame t of A is compared to frame t + offset of B')
    ap.add_argument('--tol', type=float, default=2.0, help='voxels')
    args = ap.parse_args()

    na, ta, A = xyz(load(args.a))
    nb, tb, B = xyz(load(args.b))
    tb_pos = {t: i for i, t in enumerate(tb)}
    rows = [(i, tb_pos[t + args.offset]) for i, t in enumerate(ta) if t + args.offset in tb_pos]
    ia, ib = map(np.array, zip(*rows))
    A, B = A[ia], B[ib]
    print(f'A {len(na)} neurons, B {len(nb)} neurons, {len(rows)} shared frames (offset {args.offset})')

    # Median distance between every A track and every B track over frames where both exist
    d = np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=-1)  # (T, na, nb)
    med = np.nanmedian(d, axis=0)
    cost = np.where(np.isfinite(med), med, 1e6)
    ra, rb = linear_sum_assignment(cost)
    paired = cost[ra, rb] < args.tol
    print(f'neurons paired within {args.tol} voxels (median): {paired.sum()} of {len(ra)}; '
          f'median pair distance {np.median(cost[ra, rb][paired]):.3f}')

    da = d[:, ra[paired], rb[paired]]
    present_a = np.isfinite(A[:, ra[paired], 0])
    agree = np.isfinite(da) & (da < args.tol)
    n_a = int(np.isfinite(A[..., 0]).sum())
    print(f'A entries {n_a}; on paired neurons {int(present_a.sum())}; '
          f'agree with B {agree.sum() / max(present_a.sum(), 1):.4f} of paired, '
          f'{agree.sum() / max(n_a, 1):.4f} of all A entries')
    exact = np.isfinite(da) & (da < 1e-3)
    print(f'identical position (<1e-3 voxel): {exact.sum() / max(present_a.sum(), 1):.4f} of paired entries')

    # Where A disagrees: is its partner absent in that frame (filtering only removes entries,
    # so this would be A having data where B has none), or present elsewhere (A sits on a
    # different neuron)? For the latter, does A's position coincide with another B neuron?
    bad = present_a & ~agree
    partner_missing = bad & ~np.isfinite(da)
    partner_far = bad & np.isfinite(da)
    print(f'disagreeing entries {int(bad.sum())}: partner absent in B {int(partner_missing.sum())}, '
          f'partner present but >= {args.tol} voxels away {int(partner_far.sum())}')
    if partner_far.any():
        q = np.percentile(da[partner_far], [10, 50, 90])
        print(f'  distance to partner when far: p10 {q[0]:.1f}, median {q[1]:.1f}, p90 {q[2]:.1f} voxels')
    a_pos = A[:, ra[paired]]
    d_any = np.linalg.norm(a_pos[:, :, None, :] - B[:, None, :, :], axis=-1)  # (T, paired, nb)
    nearest = np.nanmin(np.where(np.isfinite(d_any), d_any, np.inf), axis=-1)
    for name, m in [('partner absent', partner_missing), ('partner far', partner_far)]:
        if m.any():
            print(f'  {name}: A sits within {args.tol} voxels of some other B neuron in '
                  f'{np.mean(nearest[m] < args.tol):.3f} of these')
    # Unpaired A neurons: tracks with no B counterpart
    print(f'unpaired A neurons {int((~paired).sum())} with {int(np.isfinite(A[:, ra[~paired], 0]).sum())} entries')


if __name__ == '__main__':
    main()
