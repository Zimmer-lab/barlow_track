---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17770-1eee-7463-b6c5-4242dbbda28e
status: done
priority: high
dateCreated: 2026-10-26T09:59:59+01:00
dateModified: 2026-10-26T18:10:50+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
outcome: failed
completedDate: 2026-10-26
---

## Purpose

Test whether the missing per-crop rotation and elastic augmentation explain the position pipeline's 16-20 point Flavell loss ([[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]]).

## Method

Identical to the use_position cell (preprint Flavell config, use_position true, emb 64, batchnorm, 11,900 steps, seed 0) but with the preprint's alternating per-crop view transforms replacing build_photometric_transform: barlow_track@d8cbddd:barlow_track/scripts/scratch/paper_flavell_use_position_legacyaug/run.sh (with scratch/train_position_with_legacy_crop_aug.py). Scored --source project against the NWB GT at descriptor_stage auto and backbone, 25 label-propagation seeds. Run r261025-2666148, log /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_legacyaug/run.log.
What would answer it: near 0.95 supports the claim (the augmentation is the gap); still near 0.80 opposes it and points at position fusion or global augmentation itself.

## Results

0.6571 fused (misses 30,258, mismatches 49,052 of 231,268), 0.4870 backbone-only (49,395 / 69,240), vs 0.7953 / 0.7559 for the same cell without per-crop augmentation and 0.957 for the legacy pipeline. The backbone falls below an untrained encoder (~0.70). No collapse (fused effective rank about 30, as without it); train loss reaches about 1.2e-4 in both and validation loss rises over training in both (0.050 -> 0.069 here). Zero-placeholder crops 4,206 (about 0.2%), as without it.

## Conclusion

Adding the preprint's per-crop rotation and elastic deformation to the position pipeline lowers Flavell accuracy by 14 points fused and 27 points backbone-only (one seed), so the missing per-crop augmentation is not what holds the position pipeline below the preprint. The backbone is hit hardest, which fits a position shortcut: the harder the crops are to match by appearance, the more the network leans on position, which is a weak cue across time.

## Caveats

One training seed, and the position pipeline's training-seed spread on Flavell is about +-4 points, small next to a 14-27 point drop. The per-crop transforms stack on the global affine (two rotations), so this is heavier augmentation than the preprint's, not the preprint's augmentation in a new pipeline. Does not separate augmentation from fusion; the image-only run does.
