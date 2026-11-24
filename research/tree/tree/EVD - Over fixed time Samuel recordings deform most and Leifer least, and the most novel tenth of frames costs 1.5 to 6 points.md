---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a20cea-ed97-754f-a1f4-e2372a10a755
status: done
priority: normal
dateCreated: 2026-11-24T10:40
dateModified: 2026-11-24T10:40
confidence: medium
completedDate: 2026-11-24
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Reviewer 3 (W1, W3, Q1) asks whether the benchmarks contain large deformation, says Leifer is easy, and asks for
high-deformation data. Questions: how much does each benchmark deform, in units comparable across labs? Is Samuel
(Susoy et al. 2021) the high-deformation case? How much does trial-59 accuracy drop on the most deformed frames
within each recording?

## Method

[[@BarlowTrack deformation per dataset and accuracy against deformation]]. GT positions only.

- **Unit:** the median within-volume nearest-neighbour (NN) distance between all segmented detections, after z
  scaling.
- **Motion measures:** raw displacement, and the non-rigid residual after the best similarity transform (rotation,
  translation, uniform scale). Both are taken at a lag of 1 volume (what the tracker sees) and at 1 s, 10 s and
  60 s (comparable across volume rates).
- **Rotation:** the rotation of the similarity fit.
- **Neighbour-distance change:** the std of the log neighbour-distance ratio over 60 s, for pairs initially
  2.5-10 NN apart.
- **Per-frame novelty (d_novel):** the smallest median non-rigid residual between frame t and 50 reference
  frames spread over the recording. Per-frame accuracy of trial 59's v2 tracks is binned by deciles of it.

## Results

Median non-rigid residual in NN units, per volume / over 10 s / over 60 s, with the 60-s neighbour-distance change:

| Recording (vol/s) | per volume | 10 s | 60 s | log-dist std 60 s | rotation 60 s (deg) |
|---|---|---|---|---|---|
| Samuel 153 (10) | 0.19 | 0.82 | **1.21** | 0.155 | 18 |
| Samuel 137 (10) | **0.60** | **1.05** | 1.11 | **0.194** | 26 |
| Samuel 184 (10) | 0.35 | 0.98 | 1.13 | 0.168 | 20 |
| Zimmer 1123 / 1128 / 1210 (3.5) | 0.23 / 0.32 / 0.23 | 0.80 / 1.08 / 0.87 | 0.69 / 0.96 / 1.01 | 0.10-0.15 | 61-85 |
| Flavell ×5 (1.7) | 0.39-0.52 | 0.78-0.89 | 0.78-0.94 | 0.12-0.14 | 7-15 |
| Leifer StarDist (6) | 0.42 | **0.68** | **0.69** | **0.109** | 43 |

- **Raw displacement per volume** (median, NN units): Flavell 1.5-2.7 (74-90% of neurons move more than 1 NN),
  Leifer 1.24, Zimmer 0.71-1.13, Samuel 0.23-0.93.
- **Unit check:** NN distance between detections vs GT only, in voxels: Zimmer 11.5-12.2 vs 12.9-13.2,
  Leifer 11.7 vs 14.3, Flavell 6.3-6.8 vs 6.6-7.0, Samuel 6.8-7.4 vs 6.7-7.5.
- **Accuracy on the most novel tenth of frames vs the rest** (trial 59, v2):
  - drop of 1.5-5.7 pt on 10 of 12 recordings (Zimmer 1128: .934 vs .990; Leifer: .902 vs .940; Flavell:
    1.5-3.4 pt);
  - Samuel 137: **11.9 pt** (.811 vs .930);
  - Spearman of per-frame accuracy with novelty: -0.18 to -0.53 on 10 recordings; Zimmer 1210 and Samuel 153
    are slightly positive.
- **Per-volume speed** (d_lag1) costs less: at most 3 pt in the top decile.
- **Worst segment:** Zimmer 1128 frames 1422-1441, at .59 accuracy.

## Conclusion

- **Leifer is the least deformed benchmark.** It is lowest on non-rigid motion over 10 s and 60 s and on
  neighbour-distance change, tied only with Zimmer 1123. This agrees with R3 and holds after both corrections.
- **All three Samuel recordings deform most over fixed time.** They are highest on non-rigid residual and
  neighbour-distance change over 60 s. This changes the first run, where fixed-frame lags made only Samuel 137
  stand out. Samuel 153 moves little between volumes because it is recorded at 10 Hz, so it is easy for a
  tracker despite its posture variety.
- **Per volume, Flavell and Samuel 137 are hardest.** At 1.7 Hz, Flavell neurons travel 1.5-2.7 NN between
  volumes. Zimmer's difficulty is different: whole-body rotation (61-85 degrees in 60 s, against 7-15 for the
  motion-corrected Flavell data).
- **Within a recording, deformation costs a few points.** Accuracy on the most novel tenth of frames is 1.5-6 pt
  lower on most recordings, and 12 pt lower on Samuel 137. This answers R3.W1 better than comparing datasets: the
  tracker keeps working on the deformed frames, with a measurable cost.
- **Use in the response:** call Samuel the high-deformation lab "over seconds", and name Samuel 137 as the
  recording where deformation costs most.

## Caveats

- Physical units are not checked: the voxel sizes in paper Table 1 put Zimmer's detection NN at about 4.2 µm and
  Flavell's at about 2.1 µm, which is implausible for the same head. Everything here is in NN units, so it does
  not depend on them; the visnet_z_scale values do.
- Samuel's volume rate (10 Hz) is assumed for 137 and 184.
- Samuel 184's detections come from the thresholded segmentation, which has about 10% fewer masks than GT neurons.
  Its NN distance is unaffected (6.77 vs 6.70 voxels).
- Per-frame accuracy of tuning recordings comes from one network. d_novel's reference frames are evenly spaced,
  not the tracker's actual seed volumes.
- The similarity fit removes rotation, so non-rigid numbers understate Zimmer's difficulty.
- GT positions only: segmentation errors are not part of the deformation measure.
