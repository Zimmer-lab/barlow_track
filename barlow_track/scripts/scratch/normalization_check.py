"""Does crop normalization explain the gap between eval_accuracy's embedding and the paper's?

Embeds the first frames of a project with a checkpoint's backbone, through
eval_accuracy._iter_project_frames on CPU, once per normalization percentile pair,
and compares each to the paper's saved embedding.zarr row by row (matched on (t, seg)).
The paper (v0.1.0) normalized with RescaleIntensity(percentiles=(5, 100));
eval_accuracy and training since d1ff47a use (5, 99.5).

Usage (wbfm env):
  python normalization_check.py --frames 30
"""
import argparse
import os
import pickle

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.expanduser('~/.cache/barlow_scratch/numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.expanduser('~/.cache/barlow_scratch/mpl'))

import numpy as np
import torch

PAPER_DIR = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/'
             'flavell/baseline/2025_07_01trial_0/3-tracking/barlow_tracker')
PROJECT = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/'
           '2025_07_01/project_config.yaml')
WEIGHTS = '/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/baseline_flavell/trial_0/resnet50.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper_dir', default=PAPER_DIR)
    ap.add_argument('--project', default=PROJECT)
    ap.add_argument('--weights', default=WEIGHTS)
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--percentiles', nargs='+', default=['5,100', '5,99.5'])
    args = ap.parse_args()

    import zarr
    import torchio as tio
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.utils.barlow import load_barlow_model
    from barlow_track.scripts.eval_accuracy import _iter_project_frames

    Xp = np.asarray(zarr.open(os.path.join(args.paper_dir, 'embedding.zarr'), 'r'))
    with open(os.path.join(args.paper_dir, 'linear_ind_to_t_and_seg_id.pickle'), 'rb') as f:
        key_p = {(int(t), int(s)): i for i, (t, r, s) in pickle.load(f).items()}

    _, model, margs = load_barlow_model(args.weights)
    model = model.cpu().eval()
    target_sz = np.array(getattr(margs, 'target_sz', [margs.target_sz_z, margs.target_sz_xy, margs.target_sz_xy]))
    project_data = ProjectData.load_final_project_data(args.project)

    for pct in args.percentiles:
        lo, hi = map(float, pct.split(','))
        normalizer = tio.RescaleIntensity(percentiles=(lo, hi))
        A, B = [], []
        for fr in _iter_project_frames(project_data, target_sz, normalizer, list(range(args.frames)), device=None):
            with torch.no_grad():
                emb = model.backbone(fr['crops']).numpy()
            for row, (raw, seg) in zip(emb, fr['meta']):
                j = key_p.get((fr['t'], int(seg)))
                if j is not None:
                    A.append(row)
                    B.append(Xp[j])
        A, B = np.array(A), np.array(B)
        cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
        print(f'percentiles {pct}: {len(A)} rows, max abs diff {np.abs(A - B).max():.3g}, '
              f'cosine median {np.median(cos):.5f} min {cos.min():.5f}', flush=True)


if __name__ == '__main__':
    main()
