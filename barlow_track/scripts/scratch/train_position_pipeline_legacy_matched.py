"""Train through the position data path made as close to the legacy crop pipeline as it goes.

Combines the two earlier wrappers: fused_descriptors returns the backbone output only
(train_position_pipeline_visual_only.py) and each view gets the preprint's per-crop
torchio transform, view 1 then view 2 (train_position_with_legacy_crop_aug.py). The
config sets p_global_affine 0 and no position jitter or dropout, so the whole volume is
not transformed before cropping, and reaugment_each_epoch true, so views are re-drawn
each epoch as the legacy dataset does. What is left differing from the legacy pipeline:
crops at metadata centroids of the raw volume (legacy: regionprops bbox midpoints),
the frame and object selection of VolumeCoordsDataModule, and its training loop path.
Near 0.95 puts the gap in the global volume augmentation; near 0.80 in what is left.
Evaluate with --descriptor_stage backbone only (see the visual-only wrapper).

Usage (wbfm env): same as train_barlow_clusterer.py
  python train_position_pipeline_legacy_matched.py -p train_config.yaml
"""
import runpy
import sys
from pathlib import Path

from ruamel.yaml import YAML

import barlow_track.utils.volume_data as vd
from barlow_track.utils.barlow_superglue import BarlowWithPosition
from train_position_pipeline_visual_only import _visual_only
from train_position_with_legacy_crop_aug import AlternatingLegacyTransform

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / 'train_barlow_clusterer.py'


def main():
    config_fname = sys.argv[sys.argv.index('-p') + 1]
    with open(config_fname, 'r') as f:
        cfg = dict(YAML(typ='safe').load(f))
    if float(cfg.get('global_augment', {}).get('p_global_affine', 1.0)) != 0.0:
        raise ValueError("legacy-matched run expects global_augment.p_global_affine 0")
    # Transform reads global_augment too; the legacy crop path must not apply it.
    cfg.pop('global_augment', None)
    BarlowWithPosition.fused_descriptors = _visual_only
    vd.build_photometric_transform = lambda photometric_args=None: AlternatingLegacyTransform(cfg)
    runpy.run_path(str(TRAIN_SCRIPT), run_name='__main__')


if __name__ == '__main__':
    main()
