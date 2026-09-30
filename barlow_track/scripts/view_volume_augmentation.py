"""Visualize Step 1 (global-before-crop) augmentation in napari.

Example:
    python barlow_track/scripts/view_volume_augmentation.py \
        --project_path /path/to/project_config.yaml --frame 0
"""
import argparse

from barlow_track.utils.volume_augment_viewer import (
    DEFAULT_TARGET_SZ,
    ViewerParams,
    load_project_for_viewer,
    show_volume_augmentation,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize volume augmentation in napari')
    parser.add_argument('--project_path', type=str, required=True,
                        help='Path to project folder or project_config.yaml')
    parser.add_argument('--frame', type=int, default=0, help='Timepoint to display')
    parser.add_argument('--target_sz', type=int, nargs=3, default=list(DEFAULT_TARGET_SZ),
                        help='Crop size as "z x y", e.g. --target_sz 8 64 64')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for the augmentation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Loading project from {args.project_path} ...", flush=True)
    project_data = load_project_for_viewer(args.project_path)
    print(f"Project loaded: {project_data.num_frames} frames.", flush=True)
    print(f"Augmenting frame {args.frame} with crop size {tuple(args.target_sz)} ...",
          flush=True)
    params = ViewerParams(frame=args.frame, seed=args.seed)
    main, crop, state = show_volume_augmentation(
        project_data, target_sz=tuple(args.target_sz), params=params)
    print(f"Viewers ready: frame {state.params.frame} with "
          f"{state.num_points} neurons ({state.num_kept} kept).", flush=True)

    import napari
    print("Starting napari event loop ...", flush=True)
    napari.run()
    print("napari event loop exited.", flush=True)
