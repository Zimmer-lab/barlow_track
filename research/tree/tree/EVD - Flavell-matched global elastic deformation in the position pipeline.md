---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17932-66a4-7b50-a536-a58ca1587314
status: open
priority: high
dateCreated: 2026-10-26T18:11:48+01:00
dateModified: 2026-10-26T18:11:48+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Test whether non-rigid augmentation between the two training views, matched to real Flavell motion, recovers the position pipeline's Flavell accuracy. Both candidate answers predict a gain: it is the per-crop deformation the data path lacks, and it makes position a harder cue in training.

## Method

Planned, after the image-only result and the rescored training-seed spread: the use_position cell of the preprint Flavell config with global_augment elastic_std (1.1, 4.1, 4.1), elastic_length 41 and position jitter_std about 0.3 (914141c; matched in [[EVD - The position augmentation deforms mid-range neighbour distances less than real motion does]]), fused and image-only, two training seeds each, scored against the NWB GT. Which arm comes first depends on the image-only result.

What would answer it: a gain beyond the seed spread (about 4 points) in the fused arm; the image-only arm says whether the gain comes through the image features.

## Results

## Conclusion

## Caveats
