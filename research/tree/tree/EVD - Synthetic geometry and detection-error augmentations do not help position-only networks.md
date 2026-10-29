---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a188f1-a853-782c-8537-83c72aca4c73
status: done
priority: normal
dateCreated: 2026-10-29T19:35:01+01:00
dateModified: 2026-10-29T19:35:01+01:00
confidence: medium
completedDate: 2026-10-29
outcome: failed
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find augmentations of the keypoint views that make a position-only network more robust: synthetic segmentation errors, non-rigid geometry, real frame pairs.

## Method

Local fast loop on Zimmer 2022-11-28, position-only ViSNet (RBF fix on), epoch-14 frame-pair proxy against a base of 0.719 (seed 43; batches 15-16 also seed 44, base 0.645). Batch 14 (r261029-3636490): spurious detections p 0.2, merges 0.05, splits 0.05. Batch 15-16 (r261029-3707539, r261029-3720770, queue r261029-3738930 / r261029-3751540): keypoint bending (bend_max), per-axis scale jitter, plane cuts (cut_p), global scale jitter 0 / 0.2, real frame pairs as views (pair_p 0.5, GT-free pseudo-matches from an affine ICP plus local refinement, 42 matches per frame at 0.964 precision on Zimmer, 0.82 on Flavell). Code 62f6943, 2676174, a524744 (volume_data.py, all off by default).

## Results

Seed 43, against 0.719: spurious 0.666, merge 0.700, split 0.713 (with anisotropic jitter all three together 0.691 vs that jitter alone 0.785); bend 0.707; per-axis scale 0.695; plane cut 0.664; global scale jitter 0 / 0.2 0.686 / 0.659; real pairs 0.580. Bend and axis scale were within seed noise on both seeds; cut, global scale and every real-pair setting were lower on both.

## Conclusion

None of these augmentations helps the position-only network; detection errors, plane cuts and real frame pairs cost 0.01-0.14 on the proxy, bending and per-axis scale are neutral. What did help was matching the jitter to the detection noise ([[EVD - Keypoint jitter sized from the segmentation's own centroid noise improves position-only networks most on Samuel]]).

## Caveats

Proxy only (no full-video check), Zimmer only, mostly one seed; magnitudes were not tuned beyond one or two values each. Real pairs need matched objects, which the GT-free matcher gives only for the subset that moves near-affinely, so the views lose many objects. All stay in the code, off by default.
