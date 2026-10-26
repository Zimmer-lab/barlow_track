---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a3-6149-76c7-8622-62ac30c02a74
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: low
completedDate: 2025-11-12
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Test a younger life stage: L3 larvae are about half the size of young adults, so neurons are denser (Fig. S2i); all published studies used young adults.

## Method

Fig. 4g-i: heatmap, traces and coverage for L3 recordings.

## Results

Good performance by coverage and heatmap quality ("we could still achieve good performance").

## Conclusion

BarlowTrack tracked L3 larvae, a life stage no published freely moving whole-brain pipeline has handled.

## Caveats

As reported in the paper; not rerun. No ground truth: judged by track coverage and heatmap/trace quality only, which Fig. 3d-f shows is a weak proxy on two of three labs. Number of animals per condition not stated (5 animals across all novel conditions, Table 1). No figure script found in the repository. Segmentation losses rise in denser brains (Discussion: 10-40% of head neurons lost).
