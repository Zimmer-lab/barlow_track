---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17932-6621-7398-8121-321e354ed94c
status: in-progress
priority: high
dateCreated: 2026-10-26T14:11
dateModified: 2026-10-26T14:11
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Separate the position pipeline's data path from position fusion: train the use_position cell of the preprint Flavell config with the position branch cut out of the descriptor, so the image backbone alone carries the Barlow loss under the same global augmentation.

## Method

Same as [[@BarlowTrack preprint Flavell configuration with the position pipeline]] (use_position true, emb 64, batchnorm projector, 50 epochs x 300 frames, 11,900 steps, seed 0, VolumeCoordsDataModule with global affine and photometric-only per-crop transform), with BarlowWithPosition.fused_descriptors replaced by the backbone output (scratch/train_position_pipeline_visual_only.py). Scored with eval_accuracy.py --source project against the NWB GT at descriptor_stage backbone, 25 label-propagation seeds. barlow_track@8d2480c:barlow_track/scripts/scratch/paper_flavell_position_pipeline_visual_only/run.sh, reading a local mirror of the Flavell project and NWB. Run r261026-1561729; attempts r261026-1482855 and r261026-1549224 died on EIO reading the zarr over /lisc.

What would answer it, against the legacy pipeline's 0.952-0.957 and the position pipeline's roughly 0.79-0.92 across training seeds (NWB GT): 0.93 or above means the data path is fine for image features and fusion costs the accuracy (supports [[CLM - Position fusion lets the position pipeline skip learning appearance on Flavell]]); 0.88 or below means the data path itself costs it (supports [[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]]); in between needs a second training seed.

## Results

## Conclusion

## Caveats
