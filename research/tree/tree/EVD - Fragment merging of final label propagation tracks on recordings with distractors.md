---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b26c-b087-7ac0-af36-66610b44930f
status: in-progress
priority: high
dateCreated: 2026-11-06T20:53:49+01:00
dateModified: 2026-11-06T20:53:49+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Batch 1 showed that no tracker setting reduces fragmentation ([[EVD - Fragment errors of label propagation split into split tracks and distractor tracks]]), while perfect merging of final labels would recover 2.4 pt on Z1210 and 1.1 pt on F0330 (8.9 pt on the earlier 100-volume Leifer smoke). Test whether a GT-free merge rule recovers part of that bound.

## Method

`lp_core.merge_fragments` (barlow_track@3fab4ee, labelprop_opt): candidate pairs of final labels by nearest embedding centroids (10 per label); merge greedily when the centroid distance is within `merge_ratio` RMS spreads, the two groups share at most `merge_overlap` of the smaller one's volumes, and optionally the consensus affinity is at least `merge_aff`. The smaller group takes the larger's label; per volume the more confident object keeps it. `diag_merge.py` caches the base labels of Z1210 t36 and F0330 t61 at seed 0 (CPU), reports how well each pair feature separates true fragment pairs (same majority GT neuron) from others (AUC), and scores a grid ratio {0.5..3} x overlap {0.02..0.2} x affinity {0..0.1}. A setting chosen here must then pass on the confirmation cases, including Leifer.

## Results

Running.

## Conclusion

Pending.

## Caveats

Thresholds chosen on two dev recordings at one seed; the merge never splits tracks, so it cannot fix a distractor track that holds a neuron in volumes where the neuron's own track also has an object.
