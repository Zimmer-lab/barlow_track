---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17932-2f5c-7d96-a9c6-6e09ce4c4296
status: done
priority: normal
dateCreated: 2026-10-26T15:18
dateModified: 2026-10-26T15:18
confidence: medium
completedDate: 2026-10-26
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

ViSNet position features see distances and angles to neighbours within its cutoff, and are invariant to rotation and translation. Measure whether the two training views of the position pipeline change that local geometry as much as real motion between frames does; if less, position is an easier cue in training than in tracking.

## Method

[[@BarlowTrack neighbour-distance change in real frames and training augmentation]]: std of log neighbour-distance ratio by initial distance in NN units, real frames at lag 100 vs the current augmentation (global affine, scale jitter 0.1, jitter 1 voxel, no elastic), and a grid of global elastic settings, on Zimmer (zimmer_1128), Flavell, Leifer and Samuel GT positions. No runs logged; inline from the committed script.

## Results

Std log distance ratio, bins 0-1.5/1.5-2.5/2.5-4/4-6/6-10/10-cutoff NN, real lag 100 vs current augmentation:
- Zimmer 0.26/0.22/0.18/0.14/0.11/0.10 vs 0.21/0.14/0.11/0.10/0.09/0.09
- Flavell 0.18/0.18/0.16/0.13/0.13/0.12 vs 0.27/0.17/0.12/0.10/0.09/0.08
- Leifer 0.36/0.24/0.18/0.13/0.11/0.10 vs 0.22/0.13/0.11/0.10/0.10/0.10
- Samuel 0.33/0.29/0.25/0.20/0.14/0.10 vs 0.35/0.22/0.15/0.11/0.09/0.09

RMS log mismatch of the current augmentation 0.35-0.59; best elastic matches (std z,x,y voxels / length / jitter): Zimmer (0.2,2.9,2.9)/19/0.5 at 0.068, Flavell (1.1,4.1,4.1)/41/0.14-0.34 at 0.075, Leifer (0.3,3.1,3.1)/42/2.1 at 0.040, Samuel (0.2,2.3,2.3)/10/0.26 at 0.031. Real scale change p5-p95 about 0.82-1.10, matching scale_jitter 0.1.

## Conclusion

In all four datasets the position augmentation changes neighbour distances in the 2.5-10 NN range 1.3-2x less than 100 frames of real motion, gets the far range about right through scale jitter, and over-perturbs the nearest pairs on Flavell (jitter is in voxels; Flavell's NN distance is 6.8). A global elastic field with xy std 2-4 voxels matches real motion to within 0.03-0.08 RMS log ratio everywhere; Flavell needs the largest z deformation.

## Caveats

Geometry only: says the training views are easier to match by position than real frames, not that the network uses that. 100 frame pairs per dataset; Leifer estimates vary with the frame sample (current mismatch 0.39 vs 0.59 in two samples). Lag 100 is the target; tracking spans all lags. GT positions only, not segmentation centroids.
