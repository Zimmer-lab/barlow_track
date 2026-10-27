---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17932-66a4-7b50-a536-a58ca1587314
status: in-progress
priority: high
dateCreated: 2026-10-26T18:11:48+01:00
dateModified: 2026-10-27T04:01:27+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Test whether non-rigid augmentation between the two training views, matched to real Flavell motion, recovers the position pipeline's Flavell accuracy. After [[EVD - Image features alone in the position pipeline]] the data path, not fusion, holds the accuracy, and non-rigid deformation between views is the largest thing it lacks compared to real frames.

## Method

The use_position cell of the preprint Flavell config (emb 64, batchnorm projector, 11,900 steps, seed 0) with global_augment elastic_std (1.1, 4.1, 4.1), elastic_length 41 and position jitter_std 0.3 instead of 1.0 (914141c; the match in [[EVD - The position augmentation deforms mid-range neighbour distances less than real motion does]]), local Flavell mirror; barlow_track@01e6be3:barlow_track/scripts/scratch/paper_flavell_use_position_elastic/run.sh. The first attempt (r261026-1722467, num_workers 6) crashed on BadZipFile because forked workers share the zarr ZipStore handle; rerun serially (num_workers 0) as r261026-1730274. Scored with eval_accuracy.py --source project against the NWB GT at descriptor_stage auto and backbone.

What would answer it: the position pipeline's training seed alone spans 0.77-0.93 on Flavell ([[EVD - Position-pipeline Flavell accuracy spans 12 points across training seeds]]; this exact cell 0.769 / 0.795 / 0.848 over three seeds, 2 of 20 position networks above 0.90). Fused 0.93 or above, beyond every position-pipeline network seen, means non-rigid deformation is most of what the data path lacks. 0.86-0.93 is inside the upper seed range and needs a second seed before it is read. 0.85 or below means it is not, and the next suspects are global resampling and the other data-module differences. The backbone score says whether a gain comes through the image features.

## Results

## Conclusion

## Caveats
