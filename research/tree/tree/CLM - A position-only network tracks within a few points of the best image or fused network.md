---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a188f2-2fa9-7f4b-bae9-c43b7ae20523
status: open
priority: normal
dateCreated: 2026-10-29T19:35:35+01:00
dateModified: 2026-10-29T19:35:35+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Claim

With the ViSNet RBF fix, a network trained on neuron centroids alone (no image features) tracks a recording it was trained on almost as well as the best image-only or fused network: Zimmer 2022-11-28 0.961 vs 0.970 fused, Flavell 2022-03-31-01 0.98 vs 0.963 image-only (GT objects), Samuel 153 0.936 vs 0.951 image-only (GT objects). The recipe that got there: RBF fix, width 512, 600 training frames, no or little object dropout, jitter sized from the segmentation's centroid noise.

## Why low confidence

Best cells are single seeds; Flavell and Samuel track GT objects (oracle detections), Zimmer the full segmentation; and position-only has not yet been compared to image networks under the same budget (600 frames vs trial 27's 100). Whether positions add to image features in a fused network is open (round_fused_fix_zimmer, running).
