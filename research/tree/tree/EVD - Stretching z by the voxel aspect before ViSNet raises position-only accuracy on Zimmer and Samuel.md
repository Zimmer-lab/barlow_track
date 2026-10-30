---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a18aa3-1d89-7ce0-813f-61888b04058b
status: open
priority: normal
dateCreated: 2026-10-30T03:28:28+01:00
dateModified: 2026-10-30T03:28:28+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
Keypoints reach ViSNet in voxel units, which compresses z by the voxel aspect (Zimmer 1.5 / 0.325 um = 4.6).
Multiplying z by about that factor before the graph (`visnet_z_scale`) raised full-video position-only
accuracy from 0.922 to 0.967 on Zimmer 1128 (z_scale 4.6) and from 0.741 to 0.983 on Samuel 153 (z_scale 5),
from 300-frame, 15-epoch fast-loop networks. Both are above the best position-only networks so far (0.961,
0.936). On Flavell it did not help: z_scale 2 was flat and 4 cost 0.10-0.12 on the proxy.

## Method
Fast local loop (`posonly_fast.py`), position-only ViSNet with the RBF fix, batch-18 base with GT-free jitter
(`jitter_std: auto`), 300 frames, 15 epochs; `visnet_z_scale` from barlow_track@d9cfd1a.
- Batch 22 (barlow_track@d152ab3, run r261029-4126504), epoch-14 proxy (`val_val_match_accuracy_auto`),
  seeds 43 / 44: Zimmer z_scale 1 / 1.55 / 3 / 4.6 = .762 / .778 / .823 / .865 and .693 / .735 / .784 / .802;
  Samuel 1 / 2.9 / 5 = .548 / .848 / .865 and .611 / .917 / .918; Flavell 1 / 2 / 4 = .785 / .775 / .680 and
  .727 / .739 / .610.
- Batch 23 (barlow_track@beda007, runs r261029-4158341, r261030-102486), seed 43: proxy Zimmer 6 / 8 = .874 /
  .889, Samuel 8 / 12 = .794 / .551. Full-video accuracy (`eval_accuracy.py`, label propagation) at z_scale 1
  vs 4.6 (Zimmer) and 1 vs 5 (Samuel).

## Caveats
Full-video numbers are one seed each. z_scale values were chosen from voxel sizes and GT-free
nearest-neighbour spacing ratios, but the best value per dataset was picked on GT-based proxies, so a
per-recording rule (voxel aspect, or the xy/z nearest-neighbour ratio) still needs to be fixed in advance
and tested on the other recordings. Flavell's and Samuel's physical voxel sizes are placeholders in their
projects. Fused and image networks are untested with it. It probably explains part of the dataset-dependent
cutoff_nn effect ([[EVD - A scale-normalised ViSNet cutoff helps position-only tracking on Samuel and hurts it on Zimmer]]).
