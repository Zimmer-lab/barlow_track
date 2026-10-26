---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a17931-af25-7caf-81e1-2a27bf1eebb9
status: open
priority: normal
dateCreated: 2026-10-26T09:00
dateModified: 2026-10-26T09:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Claim

In the position pipeline, the two views of a training pair are one volume under a near-rigid transform, so the ViSNet position features (invariant to rotation and translation) nearly solve the Barlow loss by themselves and the image branch need not learn neuron appearance. Across time, where neurons move non-rigidly, position is a weak cue, so tracking accuracy falls.

## Rationale

The fused network's own backbone scores below its fused output (0.756 vs 0.795) and adding heavy per-crop augmentation pushes the backbone to 0.487, below an untrained encoder, while the fused score falls less (0.657). Validation loss rises over training in both runs without collapse. Real Flavell frames change neighbour distances 1.3-2x more than the current augmentation in the 2.5-10 nearest-neighbour range (scratch/local_geometry_change.py), so the training views are easier to match by position than real frames are.

## Scope

A working hypothesis, low confidence: one training seed per run, Flavell only. Predicts that the image-only run (position branch removed, same data path) beats the fused 0.795, and that non-rigid augmentation between views (global elastic deformation, stronger point dropout) or a separate image-only loss recovers accuracy. Competes with [[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]].
