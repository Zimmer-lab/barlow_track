---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-7c22-7888-a923-151c0151861c
status: open
priority: high
dateCreated: 2026-10-23T00:05:06+02:00
dateModified: 2026-10-23T00:05:06+02:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Make sure the evaluation and tracking code is correct before re-scoring anything (docs/pr_plan PR01-PR04, all High except PR04 Medium/High).

## Method

Synthetic tests per plan: PR01 (`--max_frames` truncates GT, NaN-gap matching, fewer predicted than GT columns, CPU/GPU normalization match, calculate_accuracy key types, equal-length build_accuracy_dict); PR02 (disconnected nodes never get label 0, new clusters keep IDs, low-confidence nodes get -1, align_all with 0-1 labelings, top-2 fallback, single-object frames); PR03 (window coverage incl. first and last frame, num_frames includes the last frame, resume pickle round-trip, umap_projection path); PR04 (crops clamped at volume edges, empty crops skipped, train_fraction 1.0 and absolute counts, variable N in augmentation tests; 9fbeda5 already done). Then confirm against a known project baseline.

## Results

## Conclusion

## Caveats

Plans written at 9fbeda5; later commits (#139 label propagation, #140 tracking windows, #141 cropping, #143 indices) may already cover some items; the plans are not ticked, so check each before writing tests.
