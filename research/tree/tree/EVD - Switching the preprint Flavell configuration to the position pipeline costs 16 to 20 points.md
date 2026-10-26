---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1776f-7f1c-7ed3-aaf6-d06884ab63a6
status: done
priority: normal
dateCreated: 2026-10-25T15:10
dateModified: 2026-10-26T09:59:37+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-26
---

## Purpose

Test whether the switch from the legacy crop pipeline to the position pipeline (global augmentation before cropping + ViSNet add fusion), with everything else at the preprint's size, explains the revision networks' gap.

## Method

The preprint Flavell config with only use_position switched on (emb 64, batchnorm, 11,900 steps), seed 0, scored --source project against the NWB GT at descriptor_stage auto and backbone (r261025-2444559).

## Results

0.7953 fused, 0.7559 backbone-only, vs 0.957 for the same config with use_position off. Validation loss rose over training (0.045 -> 0.054). Zero-placeholder crops 4,206 (about 0.2%). Training about 6x faster than the legacy pipeline (3.5 h vs 23 h).

## Conclusion

At the preprint's size, the position pipeline loses 16-20 points on Flavell; emb 512, layernorm and 75 epochs (trial27) recover most of it (0.924). A candidate cause: the position pipeline's per-crop transform is blur and noise only, while the preprint applied per-crop RandomAffine (180 degrees in plane), a fixed 180 degree rotation and RandomElasticDeformation with p = 1.

## Caveats

One seed, one dataset. use_position changes the augmentation and adds position fusion together, so this cannot say which costs the points; the backbone-only score suggests the image features suffer. The per-crop augmentation hypothesis is under test.
