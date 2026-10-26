---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a17a19-87bd-7c95-9f45-a1b5dfde26c1
status: done
priority: normal
dateCreated: 2026-10-26T16:21
dateModified: 2026-10-26T16:21
completedDate: 2026-10-26
run: barlow_track@8d2480c:barlow_track/scripts/scratch/paper_flavell_position_pipeline_visual_only/run.sh
data:
  - /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_visual_only
projects:
  - "[[barlowtrack]]"
---

## Description

The use_position cell of the preprint Flavell config ([[@BarlowTrack preprint Flavell configuration with the position pipeline]]: VolumeCoordsDataModule with global affine before cropping and photometric-only per-crop transform, emb 64, batchnorm projector, 50 epochs x 300 frames, seed 0) with BarlowWithPosition.fused_descriptors replaced by the backbone output (scratch/train_position_pipeline_visual_only.py), so the position branch takes no part in training. Reads a local mirror of the Flavell project. Scored with eval_accuracy.py --source project against the NWB GT at descriptor_stage backbone, 25 label-propagation seeds. Run r261026-1561729 (attempts r261026-1482855 and r261026-1549224 died on EIO reading over /lisc).
