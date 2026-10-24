"""Which flavell GT agrees with the segmentation masks?

For each GT entry (neuron, t) with finite x, y, z and raw_segmentation_id, read the
project's mask volume at the GT position and check whether it equals the GT's
raw_segmentation_id. A GT whose ids were assigned against a different segmentation
(or stale metadata) shows a low hit rate. Run on a subset of frames.

Usage (wbfm env):
  python gt_mask_consistency.py --frames 0 200 400 800 1200 1500
"""
import argparse
import os

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np

PAPER = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data'


def hit_rate(df_gt, masks, frames):
    names = list(df_gt.columns.get_level_values(0).unique())
    stats = dict(n=0, hit=0, background=0, other=0)
    for t in frames:
        if t not in df_gt.index:
            continue
        vol = np.asarray(masks[t])
        row = df_gt.loc[t]
        for n in names:
            x, y, z, s = (float(row[(n, k)]) for k in ['x', 'y', 'z', 'raw_segmentation_id'])
            if not np.all(np.isfinite([x, y, z, s])):
                continue
            zi, xi, yi = (int(np.clip(round(v), 0, vol.shape[i] - 1)) for i, v in enumerate([z, x, y]))
            m = int(vol[zi, xi, yi])
            stats['n'] += 1
            stats['hit' if m == int(s) else ('background' if m == 0 else 'other')] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default=f'{PAPER}/projects/2025_07_01/project_config.yaml')
    ap.add_argument('--nwb', default=f'{PAPER}/images_for_charlie/flavell_data.nwb')
    ap.add_argument('--frames', type=int, nargs='+', default=[0, 200, 400, 800, 1200, 1500])
    args = ap.parse_args()
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_project_gt

    project = ProjectData.load_final_project_data(args.project, verbose=0)
    masks = project.raw_segmentation
    print('masks', masks.shape)
    for label, path in [('project', args.project), ('nwb', args.nwb)]:
        df = _load_project_gt(path)
        df.index = df.index.astype(int)
        s = hit_rate(df, masks, args.frames)
        print(f"{label}: {s['n']} entries; mask at GT xyz == GT seg id {s['hit'] / max(s['n'], 1):.3f}, "
              f"background {s['background'] / max(s['n'], 1):.3f}, other mask {s['other'] / max(s['n'], 1):.3f}")


if __name__ == '__main__':
    main()
