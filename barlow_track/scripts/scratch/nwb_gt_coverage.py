"""Are eval_accuracy._load_nwb_arrays' per-field GT arrays aligned by neuron?

_load_nwb_arrays builds gx/gy/gz/gr/gs with df.loc[:, (slice(None), field)], which
returns each field's columns in the frame's own column order. If that order differs
between fields, column j of gx and column j of gs belong to different neurons:
_iter_nwb_frames then crops at one neuron's xyz and labels the crop with another's ids,
and drops entries where the two neurons' presence differs (guaranteed misses).

Prints, per NWB: whether each field's neuron order matches x, how many entries the
eval's (positional) mask keeps vs a name-aligned mask, and the accuracy cap the
positional mask implies.

Usage (wbfm env):
  python nwb_gt_coverage.py --nwb a.nwb b.nwb ...
"""
import argparse
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np

PAPER = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper'
DEFAULT_NWBS = [f'{PAPER}/flavell_data/images_for_charlie/flavell_data.nwb',
                f'{PAPER}/leifer_data/Leifer_NeRVE_Worm1.nwb']
FIELDS = ['x', 'y', 'z', 'raw_neuron_ind_in_list', 'raw_segmentation_id']


def check(nwb_path):
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays, _ensure_ind_level

    src = ProjectData.load_final_project_data(nwb_path, allow_hybrid_loading=True, verbose=0)
    df = _ensure_ind_level(src.final_tracks)
    orders = {c: list(df.loc[:, (slice(None), c)].columns.get_level_values(0)) for c in FIELDS}
    print(f'== {nwb_path}')
    print('  same neuron order as x:', {c: orders[c] == orders['x'] for c in FIELDS})

    a = _load_nwb_arrays(src)
    positional = np.ones_like(a['gx'], dtype=bool)
    for k in ['gx', 'gy', 'gz', 'gr', 'gs']:
        positional &= np.isfinite(a[k])
    names = a['neurons']
    aligned = np.ones((len(df), len(names)), dtype=bool)
    for c in FIELDS:
        aligned &= np.isfinite(df.loc[:, [(n, c) for n in names]].values.astype(float))
    has_id = np.isfinite(df.loc[:, [(n, 'raw_neuron_ind_in_list') for n in names]].values.astype(float))
    print(f'  GT entries with an id: {has_id.sum()}; cropped by eval (positional): {positional.sum()}; '
          f'name-aligned: {aligned.sum()}')
    print(f'  accuracy cap from never-cropped entries (all final_tracks): {positional.sum() / has_id.sum():.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nwb', nargs='+', default=DEFAULT_NWBS)
    args = ap.parse_args()
    for p in args.nwb:
        try:
            check(p)
        except Exception as e:
            print(f'== {p}: failed ({type(e).__name__}: {e})')


if __name__ == '__main__':
    main()
