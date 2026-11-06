---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b26c-b087-7ac0-af36-66610b44930f
status: done
priority: high
dateCreated: 2026-11-06T20:53:49+01:00
dateModified: 2026-11-06T23:05:34+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: failed
completedDate: 2026-11-06
---

## Purpose

Batch 1 showed that no tracker setting reduces fragmentation ([[EVD - Fragment errors of label propagation split into split tracks and distractor tracks]]), while perfect merging of final labels would recover 2.4 pt on Z1210 and 1.1 pt on F0330 (8.9 pt on the earlier 100-volume Leifer smoke). Test whether a GT-free merge rule recovers part of that bound.

## Method

`lp_core.merge_fragments` (barlow_track@3fab4ee, labelprop_opt): candidate pairs of final labels by nearest embedding centroids (10 per label); merge greedily when the centroid distance is within `merge_ratio` RMS spreads, the two groups share at most `merge_overlap` of the smaller one's volumes, and optionally the consensus affinity is at least `merge_aff`. The smaller group takes the larger's label; per volume the more confident object keeps it. `diag_merge.py` caches the base labels at seed 0 (CPU), reports how well each pair feature separates true fragment pairs (same majority GT neuron) from other candidate pairs (AUC), and scores a grid ratio {0.5, 0.75, 1, 1.5, 2, 3} × overlap {0.02, 0.05, 0.1, 0.2} × affinity {0, 0.02, 0.05, 0.1}. Run r261106-602653 on Z1210 t36 and F0330 t61; r261106-606521 (Leifer StarDist t57) queued after it.

## Results

| case | labels | candidate pairs | true pairs | overlap of true pairs, q10 / median | AUC ratio / overlap / affinity | best grid point |
|---|---|---|---|---|---|---|
| Z1210 | 176 | 1067 | 85 | 0.76 / 0.98 | .78 / .68 / .84 | 2 merges, +0.25 pt (ratio 3, overlap 0.2) |
| F0330 | 173 | 1100 | 36 | 0.73 / 0.99 | .83 / .53 / .95 | no merges anywhere |
| Leifer StarDist t57 | 148 | 917 | 98 | 0.80 / 0.985 | .83 / .66 / .85 | no merges anywhere |

Unmerged accuracy Z1210 .9312 (merge bound .9601), F0330 .9716 (bound .9811), Leifer StarDist .9371 (bound .9878).

## Conclusion

Merging final labels cannot recover the fragment errors on Z1210, F0330 or Leifer StarDist: the label pairs that share a GT neuron run side by side through 98-99% of the same volumes (median), instead of covering different stretches of time, so no merge rule that keeps one object per label per volume finds them (at most 2 merges, +0.25 pt, over 96 settings). Consensus affinity does identify the partner pairs (AUC .84 Zimmer, .95 Flavell), but what those pairs need is a per-volume choice between two objects, not a merge. [[EVD - Fragmented neurons on Zimmer swap with a neighbouring nucleus for long stretches]] shows what the partner objects are.

## Caveats

One network per recording, seed 0 only. The merge bound counts each label's GT objects for its majority neuron, which only per-volume reassignment can reach, so it overstates what merging could do. The 8.9 pt Leifer bound came from a 100-volume smoke of the NWB re-export; at full length on StarDist masks the bound is 5.1 pt and none of it is reachable by merging ([[EVD - On Leifer StarDist the fragment errors follow an unannotated touching neighbour mask]]).
