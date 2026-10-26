---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-e9e7-74a0-b1ce-4c5ad5d5fe5e
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

Separate the contribution of training from that of the clustering.

## Method

Fig. S1e "Untrained": randomly initialised encoder (untrained_<lab> networks, emb 64, crop 8x64x64), label propagation.

## Results

Leifer 86%, Zimmer 62%, Flavell ≈70%.

## Conclusion

Label propagation on random-CNN features already tracks 62-86% of neurons, comparable to some published methods on Leifer data; training adds 11-35 points.

## Caveats

As reported in the paper; not rerun. One untrained network per lab as far as shown.
