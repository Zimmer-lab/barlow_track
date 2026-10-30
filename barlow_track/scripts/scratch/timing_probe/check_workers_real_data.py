"""Real-data check that DataLoader workers do not change the training views.

Builds the position data module of a round_wk1 config twice (num_workers 0 and N), runs
the trainer's loop shape (one loader, set_epoch per epoch) over a few frames, and asserts
every batch tensor is identical. Prints the data-loading wall time of each.

    python barlow_track/scripts/scratch/timing_probe/check_workers_real_data.py --label samuel --workers 4
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from barlow_track.utils.volume_data import VolumeCoordsDataModule
from wbfm.utils.projects.finished_project_data import ProjectData

ROUNDS = Path(__file__).resolve().parents[1]


def batches(cfg, project, num_workers, epochs):
    dm = VolumeCoordsDataModule(
        project_data=project, num_frames=cfg['num_frames'], batch_size=1,
        train_fraction=cfg['train_fraction'], val_fraction=cfg['val_fraction'],
        target_sz=np.array([cfg['target_sz_z'], cfg['target_sz_xy'], cfg['target_sz_xy']]),
        global_args=cfg.get('global_augment'), photometric_args=cfg.get('crop_photometric'),
        position_args=cfg.get('position_augment'), seed=int(cfg.get('seed', 43)),
        num_workers=num_workers)
    dm.setup()
    loader = dm.train_dataloader()
    out, t0 = [], time.perf_counter()
    for epoch in range(epochs):
        dm.set_epoch(epoch)
        out.append(list(loader))
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='samuel')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=2)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROUNDS / f'round_wk1_{args.label}' / 'train_config.yaml').read_text())
    cfg['num_frames'] = args.frames
    project = ProjectData.load_final_project_data(cfg['project_path'], allow_hybrid_loading=True)
    ref, t_ref = batches(cfg, project, 0, args.epochs)
    par, t_par = batches(cfg, project, args.workers, args.epochs)
    n = 0
    for e in range(args.epochs):
        assert len(ref[e]) == len(par[e]), (e, len(ref[e]), len(par[e]))
        for a, b in zip(ref[e], par[e]):
            assert all(torch.equal(x, y) for x, y in zip(a, b)), f"epoch {e}: batches differ"
            n += 1
    assert not torch.equal(ref[0][0][0], ref[1][0][0]), "epochs did not rotate the augmentation"
    print(f"{args.label}: {n} batches identical with num_workers 0 and {args.workers}; "
          f"load time {t_ref:.1f} s vs {t_par:.1f} s", flush=True)


if __name__ == '__main__':
    main()
