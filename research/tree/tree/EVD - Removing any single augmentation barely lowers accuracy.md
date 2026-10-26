---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-e91c-7d0d-aebc-8e1ef5b722b5
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

Test which training augmentations are necessary.

## Method

Fig. S1e: baseline network vs networks with one augmentation removed (Blur, Affine, ElasticDeformation, Flip, Noise) on the three labs.

## Results

All remove-one networks stay at ≈0.9-0.97; the largest drop is a modest one on Zimmer without ElasticDeformation and Noise.

## Conclusion

No single augmentation is necessary: removing any one keeps accuracy within a few points of baseline on all three labs.

## Caveats

As reported in the paper; not rerun. Few networks per condition; the co-author found the surrounding sentence hard to follow.
