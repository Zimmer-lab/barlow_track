"""Score trained checkpoints with the frame-pair matching proxy (utils/val_matching.py).

Checks whether the proxy ranks checkpoints as full-video tracking accuracy does, before
it is trusted as a per-epoch training metric. Frames are evenly spaced over the whole
video (the checkpoints' training frames are not excluded, ~19% overlap at 300 of 1600).

Usage (wbfm env):
  python val_matching_on_checkpoints.py --project P/project_config.yaml --n_frames 20 \
      --weights a/resnet50.pth b/resnet50.pth ... [--jsonl out.jsonl]
"""
import argparse
import json
import time

import numpy as np
import torch

from barlow_track.utils.barlow import load_barlow_model
from barlow_track.utils.val_matching import frame_pair_matching_accuracy, held_out_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--n_frames', type=int, default=20)
    ap.add_argument('--jsonl')
    ap.add_argument('--crop_window', choices=['pad', 'slide'], default='pad')
    args = ap.parse_args()
    from wbfm.utils.projects.finished_project_data import ProjectData
    project_data = ProjectData.load_final_project_data(args.project, allow_hybrid_loading=True)
    frames = held_out_frames(project_data.num_frames, [], args.n_frames)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for w in args.weights:
        t0 = time.time()
        _, model, targs = load_barlow_model(w)
        model = model.to(device).eval()
        target_sz = np.array([targs.target_sz_z, targs.target_sz_xy, targs.target_sz_xy])
        stages = ('auto', 'backbone') if hasattr(model, 'fused_descriptors') else ('backbone',)
        acc, n_pairs = frame_pair_matching_accuracy(model, project_data, frames, target_sz, device, stages=stages,
                                                    crop_window=args.crop_window)
        rec = dict(weights=w, frames=frames, n_pairs=n_pairs, minutes=(time.time() - t0) / 60,
                   crop_window=args.crop_window if device.type == 'cuda' else 'cpu',
                   **{f'match_{k}': v for k, v in acc.items()})
        print(json.dumps(rec), flush=True)
        if args.jsonl:
            with open(args.jsonl, 'a') as f:
                print(json.dumps(rec), file=f)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
