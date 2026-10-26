"""Train through the position pipeline with the position branch cut out, without editing repo code.

With use_position true, train_barlow_clusterer uses VolumeCoordsDataModule (global
augmentation of the whole volume before cropping) and BarlowWithPosition, whose
fused_descriptors adds a ViSNet position descriptor to the image descriptor. Within a
training pair both views are the same volume under a rigid transform, so position alone
can nearly solve the Barlow loss and the image branch need not learn identity.
This wrapper makes fused_descriptors return the backbone output only, so the model
trains on the global-augmentation pipeline with image features alone (the position
branch gets no gradient). Evaluate it with --descriptor_stage backbone only: a fresh
eval process loads the unpatched class, whose fused stage would add the untrained
position branch.

Usage (wbfm env): same as train_barlow_clusterer.py
  python train_position_pipeline_visual_only.py -p train_config.yaml
"""
import runpy
from pathlib import Path

from barlow_track.utils.barlow_superglue import BarlowWithPosition

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / 'train_barlow_clusterer.py'


def _visual_only(self, y, kpts, scores=None):
    return self.backbone(y)


def main():
    BarlowWithPosition.fused_descriptors = _visual_only
    runpy.run_path(str(TRAIN_SCRIPT), run_name='__main__')


if __name__ == '__main__':
    main()
