"""Check volume_data's metadata-table centroid path against wbfm's per-frame accessor.

For each project: frames that wbfm lists as empty must come back empty, and on a random
sample of frames the (z, x, y) and segmentation ids must be identical to
get_all_neuron_metadata_for_single_time's (the slow path, ~2 s a frame).

Usage (wbfm env): python check_fast_centroids.py PROJECT_CONFIG [...] [--n 30]
"""
import argparse
import os
import time

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np
import pandas as pd


def slow(project_data, t):
    row_data, column_names = project_data.segmentation_metadata.get_all_neuron_metadata_for_single_time(
        t, as_dataframe=False)
    if len(column_names) == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    m = pd.DataFrame(dict(zip(column_names, row_data))).dropna(subset=['z', 'x', 'y'])
    return m[['z', 'x', 'y']].to_numpy(dtype=float), m['raw_segmentation_id'].to_numpy(dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('projects', nargs='+')
    ap.add_argument('--n', type=int, default=30)
    args = ap.parse_args()
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.utils.volume_data import _centroids_from_metadata_table
    for path in args.projects:
        p = ProjectData.load_final_project_data(path, allow_hybrid_loading=True, verbose=0)
        empty = set(int(t) for t in p.segmentation_metadata.volumes_with_no_neurons)
        frames = np.random.RandomState(0).choice(p.num_frames, size=min(args.n, p.num_frames), replace=False)
        bad, t_fast, t_slow = [], 0.0, 0.0
        for t in sorted(set(int(f) for f in frames) | empty):
            t0 = time.time()
            fast = _centroids_from_metadata_table(p, t)
            t_fast += time.time() - t0
            if fast is None:
                bad.append((t, 'fast path returned None'))
                continue
            if t in empty:
                if len(fast[0]):
                    bad.append((t, f'wbfm lists it empty, fast path has {len(fast[0])}'))
                continue
            t0 = time.time()
            ref = slow(p, t)
            t_slow += time.time() - t0
            if not (np.array_equal(fast[0], ref[0]) and np.array_equal(fast[1], ref[1])):
                bad.append((t, f'differs: {fast[0].shape} vs {ref[0].shape}'))
        print(f'{path}: {len(frames)} sampled + {len(empty)} empty frames, {len(bad)} mismatches; '
              f'fast {t_fast:.2f}s, slow {t_slow:.1f}s total')
        for b in bad[:10]:
            print('   ', b)


if __name__ == '__main__':
    main()
