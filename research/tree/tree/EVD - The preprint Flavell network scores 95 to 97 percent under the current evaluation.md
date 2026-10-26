---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1776f-7d2a-723a-8e55-dbc29f02c6f2
status: done
priority: normal
dateCreated: 2026-10-24T05:44
dateModified: 2026-10-26T09:59:36+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-26
---

## Purpose

Re-score the preprint's Flavell baseline network with the current evaluation, the Flavell part of the planned re-scoring of all preprint baselines.

## Method

Paper checkpoint TrainedBarlow/baseline_flavell/trial_0 through eval_accuracy.py at bc9ff96: --source project (segmentation detections of flavell_data/projects/2025_07_01, as in the paper) against the NWB GT, 25 seeds, tracking seeds 0, 1, 2 (E2: r261023-2327458, r261024-2472680); and the name-aligned NWB path (E3, see the bug Evidence).

## Results

Project path: 0.9518 / 0.9517 / 0.9528 (seeds 0/1/2; misses 3,233, mismatches 7,924 at seed 0). Aligned NWB path: 0.9692. Paper: 96.7%; the paper's stored tracks 0.968.

## Conclusion

With the paper's own network, the current evaluation gives 0.952 on the paper's detections and 0.969 on GT-centred crops, so the preprint's Flavell accuracy holds under the corrected code. The 1.5 pt shortfall on the project path matches the normalization mismatch (about 0.7 pt) and the U*S SVD (about 0.8 pt) measured on saved embeddings.

## Caveats

Flavell only; Zimmer, Leifer and Samuel are not re-scored yet. The label-propagation seed spread (0.001) is far below the effects here.
