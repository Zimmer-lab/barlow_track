"""Compare the GT eval_accuracy loads from a project config vs from an NWB.

Benchmark cells with --source project score against _load_project_gt(project_config),
others against _load_project_gt(nwb). For flavell these differ in size (216,238 vs
231,268 entries). This reports neurons, entries and per-cell agreement of
raw_segmentation_id on the shared neurons and frames.

Usage (wbfm env):
  python compare_gt_sources.py --a project_config.yaml --b data.nwb
"""
import argparse
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np

PAPER = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data'


def seg_table(df):
    s = df.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
    return s.apply(lambda c: c.astype(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default=f'{PAPER}/projects/2025_07_01/project_config.yaml')
    ap.add_argument('--b', default=f'{PAPER}/images_for_charlie/flavell_data.nwb')
    args = ap.parse_args()
    from barlow_track.scripts.eval_accuracy import _load_project_gt

    sa, sb = seg_table(_load_project_gt(args.a)), seg_table(_load_project_gt(args.b))
    for name, s in [('a', sa), ('b', sb)]:
        print(f'{name}: {s.shape[1]} neurons x {s.shape[0]} frames (index {s.index.min()}..{s.index.max()}), '
              f'entries {int(np.isfinite(s.values).sum())}')
    names = sorted(set(sa.columns) & set(sb.columns))
    frames = sorted(set(sa.index) & set(sb.index))
    print(f'shared neurons {len(names)}; only a {len(set(sa.columns) - set(sb.columns))}; '
          f'only b {len(set(sb.columns) - set(sa.columns))}; shared frames {len(frames)}')
    A, B = sa.loc[frames, names].values, sb.loc[frames, names].values
    fa, fb = np.isfinite(A), np.isfinite(B)
    both = fa & fb
    print(f'shared cells: both {both.sum()}, only a {(fa & ~fb).sum()}, only b {(~fa & fb).sum()}; '
          f'seg id equal where both {np.mean(A[both] == B[both]):.4f}')
    if len(set(sb.columns) - set(sa.columns)) or (~fa & fb).sum():
        miss = sorted(set(sb.columns) - set(sa.columns))[:10]
        print('  first neurons only in b:', miss)


if __name__ == '__main__':
    main()
