---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a3-60e2-7253-94b2-a98358b92798
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

Test a hardware change: higher magnification changes inter-volume motion and object density in pixels.

## Method

Fig. 4d-f (heatmap, traces, coverage) at 63x instead of 40x; Fig. S2g-i compare displacement and nearest-neighbour distance across conditions.

## Results

Clean heatmap, plausible traces and high coverage; displacement and density per condition differ from baseline (S2g-i).

## Conclusion

BarlowTrack, unchanged, tracked 63x recordings with high coverage.

## Caveats

As reported in the paper; not rerun. No ground truth: judged by track coverage and heatmap/trace quality only, which Fig. 3d-f shows is a weak proxy on two of three labs. Number of animals per condition not stated (5 animals across all novel conditions, Table 1). No figure script found in the repository.
