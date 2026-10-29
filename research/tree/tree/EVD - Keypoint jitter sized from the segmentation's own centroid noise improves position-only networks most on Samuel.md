---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a188f1-a7cf-70fb-916b-3adfa8a0c45b
status: done
priority: normal
dateCreated: 2026-10-29T19:35:00+01:00
dateModified: 2026-10-29T19:35:00+01:00
confidence: medium
completedDate: 2026-10-29
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Test whether position jitter sized per axis from the recording's own detection noise (no GT) beats the isotropic 1-voxel jitter used so far in position-only networks.

## Method

`position_augment.jitter_std: auto` (volume_data.py, cb4e869 / fc206be) sets per-axis jitter from consecutive-frame centroid residuals after an affine ICP (utils/dataset_stats.py), using only the segmentation: about (0.30, 0.91, 0.94) voxels (z, x, y) on Zimmer, (0.80, 0.87, 0.85) on Flavell, (0.10, 0.30, 0.30) on Samuel. Local fast loop, epoch-14 frame-pair proxy, seeds 43 and 44 (batch 15 r261029-3707539 on Zimmer; batch 18 r261029-3914302 on Samuel 153 and Flavell). Cluster: round_posonly2_samuel full-video accuracy (label propagation, NWB GT objects, seed 43), jitter 1.0 vs auto at each cutoff_nn x dropout cell.

## Results

Proxy (s43 / s44): Zimmer 0.719 / 0.645 -> 0.768 / 0.693; Samuel 0.504 / 0.568 -> 0.549 / 0.609; Flavell 0.781 / 0.722 -> 0.783 / 0.728. Jitter 0 on Zimmer cost 0.09; a GT-matched (0.2, 1, 1) jitter gave +0.07 (diagnostic only, never used). Full video, round_posonly2_samuel: jitter 1.0 gives 0.694-0.727 (trials 0-3), auto 0.768 / 0.904 / 0.936 (trials 5 / 4 / 6); the best is auto with cutoff_nn 6 and no dropout.

## Conclusion

Matching the keypoint jitter to the segmentation's own centroid noise, estimated without GT, helps wherever the default 1-voxel jitter is far from that noise: +0.05 proxy on Zimmer (z noise 0.3), up to +0.2 full-video accuracy on Samuel (noise 0.1-0.3 voxels, so 1.0 drowned the geometry), and nothing on Flavell, whose noise is already about 0.8-0.9 voxels per axis.

## Caveats

Proxy results are epoch 14 with two seeds; the cluster numbers are one seed per cell (second seeds still running) and track GT objects. The estimate assumes most frame-to-frame motion is affine; Flavell's larger residual may contain real non-rigid motion rather than detection noise.
