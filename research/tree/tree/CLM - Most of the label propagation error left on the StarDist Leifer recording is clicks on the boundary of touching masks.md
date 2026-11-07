---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1b39b-b14a-7c92-a97c-e89830a62ad1
status: open
priority: normal
dateCreated: 2026-11-07T02:24:47+01:00
dateModified: 2026-11-07T02:24:47+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Claim

On the paper's Leifer recording (StarDist masks, strict click GT), about 5 of the remaining 6 pt of tracker error are not tracker swaps. The tracker keeps one touching mask while the GT object flickers between it and its neighbour. That happens where the manual click sits off-centre, near the boundary between the two masks. Tracker changes that win elsewhere (c_all_gh: +0.4 pt here, +1.1 pt on average) leave these errors in place, so the strict-click accuracy on this recording is close to its ceiling for this segmentation and GT conversion.

## Scope

One network (wk2 t57), base tracker, seed 0 for the switch analysis; batch 2 (3 seeds) for the error decomposition. Which of the two masks is right cannot be told from positions alone. Zimmer 1210 fragments are real tracker swaps by the same test ([[CLM - Label propagation fragment errors on recordings with distractors are long swaps with an unannotated neighbouring nucleus]]).
