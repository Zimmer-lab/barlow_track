"""Train the position pipeline with the preprint's per-crop augmentation, without editing repo code.

With use_position true, train_barlow_clusterer uses VolumeCoordsDataModule, whose
per-crop transform (volume_data.build_photometric_transform) is blur + noise +
normalization only. The preprint (legacy crop pipeline, barlow.Transform) applied,
per view and on top of normalization:
  view 1: RandomAffine(degrees=(180, 0, 0)), RandomBlur
  view 2: RandomBlur, RandomAffine(degrees=(180, 0, 0)), RandomAffine(degrees=(180, 180, 0, 0, 0, 0)),
          RandomElasticDeformation(zxy_RandomElasticDeformation), RandomNoise
with the probabilities from the config (p_RandomAffine_base, p_RandomBlur_base, p_RandomBlur,
p_RandomAffine, p_RandomAffine_flip, p_RandomElasticDeformation, p_RandomNoise, std_RandomNoise).
This wrapper replaces build_photometric_transform with a callable that alternates the two
legacy view transforms; VolumeCoordsDataset.__getitem__ builds view 1 then view 2, so with
num_workers 0 each pair gets one of each. Global augmentation and position branch unchanged.

Usage (wbfm env): same as train_barlow_clusterer.py
  python train_position_with_legacy_crop_aug.py -p train_config.yaml
"""
import runpy
import sys
from pathlib import Path

from ruamel.yaml import YAML

import barlow_track.utils.volume_data as vd
from barlow_track.utils.barlow import Transform

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / 'train_barlow_clusterer.py'


class AlternatingLegacyTransform:
    def __init__(self, transform_args):
        legacy = Transform(dict(transform_args))
        self.views = [legacy.transform, legacy.transform_prime]
        self.calls = 0

    def __call__(self, x):
        t = self.views[self.calls % 2]
        self.calls += 1
        return t(x)


def main():
    config_fname = sys.argv[sys.argv.index('-p') + 1]
    with open(config_fname, 'r') as f:
        cfg = dict(YAML(typ='safe').load(f))
    # Transform reads global_augment too; global geometry is already applied to the volume here.
    cfg.pop('global_augment', None)
    vd.build_photometric_transform = lambda photometric_args=None: AlternatingLegacyTransform(cfg)
    runpy.run_path(str(TRAIN_SCRIPT), run_name='__main__')


if __name__ == '__main__':
    main()
