---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17932-66a4-7b50-a536-a58ca1587314
status: in-progress
priority: high
dateCreated: 2026-10-26T18:11:48+01:00
dateModified: 2026-10-26T22:25:15+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Test whether non-rigid augmentation between the two training views, matched to real Flavell motion, recovers the position pipeline's Flavell accuracy. After [[EVD - Image features alone in the position pipeline]] the data path, not fusion, holds the accuracy, and non-rigid deformation between views is the largest thing it lacks compared to real frames.

## Method

The use_position cell of the preprint Flavell config (emb 64, batchnorm projector, 11,900 steps, seed 0) with global_augment elastic_std (1.1, 4.1, 4.1), elastic_length 41 and position jitter_std 0.3 instead of 1.0 (914141c; the match in [[EVD - The position augmentation deforms mid-range neighbour distances less than real motion does]]), num_workers 6, local Flavell mirror; barlow_track@803401c:barlow_track/scripts/scratch/paper_flavell_use_position_elastic/run.sh. Scored with eval_accuracy.py --source project against the NWB GT at descriptor_stage auto and backbone. Run r261026-1722467. The fused arm comes first because it has three training seeds to compare against (0.769, 0.795, 0.848); the image-only arm and second seeds follow depending on the result.

What would answer it: fused 0.90 or above (beyond the fused seed range) means non-rigid deformation is most of what the data path lacks; inside 0.77-0.85 means it is not, and the next suspects are global resampling and the other data-module differences; the backbone score says whether a gain comes through the image features.

## Results

## Conclusion

## Caveats
