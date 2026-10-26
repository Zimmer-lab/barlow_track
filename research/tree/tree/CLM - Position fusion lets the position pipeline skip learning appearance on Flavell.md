---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a17931-af25-7caf-81e1-2a27bf1eebb9
status: done
priority: normal
dateCreated: 2026-10-26T09:00
dateModified: 2026-10-26T22:24:56+01:00
confidence: refuted
projects:
  - "[[barlowtrack]]"
completedDate: 2026-10-26
---

## Claim

In the position pipeline, the two views of a training pair are one volume under a near-rigid transform, so the ViSNet position features (invariant to rotation and translation) nearly solve the Barlow loss by themselves and the image branch need not learn neuron appearance. Across time, where neurons move non-rigidly, position is a weak cue, so tracking accuracy falls.

## Rationale

The fused network's own backbone scores below its fused output (0.756 vs 0.795) and adding heavy per-crop augmentation pushes the backbone to 0.487, below an untrained encoder, while the fused score falls less (0.657). Validation loss rises over training in both runs without collapse. Real Flavell frames change neighbour distances 1.3-2x more than the current augmentation in the 2.5-10 nearest-neighbour range ([[EVD - The position augmentation deforms mid-range neighbour distances less than real motion does]]).

## Scope

Refuted as the main cause of the position pipeline's Flavell deficit (2026-10-26): it predicted that removing position from training recovers most of the 14 points, and the image-only arm scored 0.819, inside the fused arm's training-seed range ([[EVD - Image features alone in the position pipeline]]); its validation loss rises too. A smaller effect remains possible: the fused network's own backbone (0.756) is below the image-only backbone (0.819), one seed each. Competed with [[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]].
