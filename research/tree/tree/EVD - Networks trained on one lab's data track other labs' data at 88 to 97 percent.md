---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-7247-7570-92cf-1f34bbddd00c
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

Test whether one trained network can be reused on other recordings, of the same lab or another lab, instead of training per recording.

## Method

Fig. 3h: mean accuracy of each lab's baseline network (rows) on each lab's ground-truth recordings (columns), label propagation, excluding the network's own training recording. Leifer has one recording, so no within-lab cell.

## Results

Train Flavell: Flavell 0.97, Leifer 0.97, Zimmer 0.96. Train Zimmer: Flavell 0.94, Leifer 0.97, Zimmer 0.96. Train Leifer: Flavell 0.88, Zimmer 0.96.

## Conclusion

Within-lab reuse matches per-recording training (0.96-0.97), and cross-lab reuse loses at most about 3 points except Leifer-trained networks on Flavell data (0.88).

## Caveats

As reported in the paper; not rerun. Means over few recordings (3 Zimmer, 5 Flavell, 1 Leifer); one network per lab. The z-crop differs for Flavell (16 vs 8); how cross-lab crops were handled is not stated. The co-author markup flags the Fig. 3h reference text.
