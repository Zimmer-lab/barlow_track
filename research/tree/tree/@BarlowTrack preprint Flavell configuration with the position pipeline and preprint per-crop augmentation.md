---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a17931-4f48-7ed7-9bfa-ab4552d3e3f6
status: done
priority: normal
dateCreated: 2026-10-25T15:17
dateModified: 2026-10-25T15:17
completedDate: 2026-10-25
run: barlow_track@d8cbddd:barlow_track/scripts/scratch/paper_flavell_use_position_legacyaug/run.sh
data:
  - /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_legacyaug
projects:
  - "[[barlowtrack]]"
---

## Description

The use_position cell of the preprint Flavell config ([[@BarlowTrack preprint Flavell configuration with the position pipeline]]: emb 64, batchnorm projector, 11,900 steps, seed 0) with the position pipeline's photometric-only per-crop transform replaced by the preprint's alternating per-crop view transforms (barlow.Transform.transform / transform_prime: RandomAffine 180 degrees in plane, a fixed 180 degree rotation, RandomElasticDeformation (1, 3, 3) at p = 1, noise), via scratch/train_position_with_legacy_crop_aug.py. The global affine before cropping and ViSNet add fusion are unchanged. Scored with eval_accuracy.py --source project against the NWB GT at descriptor_stage auto (fused) and backbone, 25 label-propagation seeds. Run r261025-2666148, about 23 h of training.
