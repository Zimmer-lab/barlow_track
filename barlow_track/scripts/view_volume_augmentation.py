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
    project_data = load_project_for_viewer(args.project_path)
    params = ViewerParams(frame=args.frame, seed=args.seed)
    show_volume_augmentation(project_data, target_sz=tuple(args.target_sz), params=params)

    import napari
    napari.run()
