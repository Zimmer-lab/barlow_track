---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-e980-708e-bd8a-ac55204e429c
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: medium
completedDate: 2025-11-12
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find which augmentations carry the training signal.

## Method

Fig. S1f: networks trained with only one augmentation (Blur, Affine, ElasticDeformation, Flip, Noise) vs baseline and untrained.

## Results

Affine-only and Flip-only networks reach nearly baseline accuracy: Leifer 97.4% (no loss), Flavell 94.2%. Blur-only, Noise-only and Elastic-only networks are close to untrained networks (Zimmer ≈0.6-0.7, Flavell ≈0.7-0.8).

## Conclusion

Geometric augmentations (affine, 180° flip) alone train near-baseline networks, while intensity and elastic augmentations alone barely improve on an untrained network.

## Caveats

As reported in the paper; not rerun. Flavell volumes are motion corrected with rare 180° flips, which explains why flip matters there (Methods 4.1).
