---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a3-6415-7190-bc33-415fd1adf1e5
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

Make correcting tracking errors (and producing ground truth) cheap.

## Method

Section 2.7: tracks are rebuilt from short high-quality tracklets, and the annotator only checks tracklet associations and occasionally splits tracklets; compared with per-time-point correction used in other studies.

## Results

10-100x faster than per-time-point correction, per the authors' earlier tests (ref. 10).

## Conclusion

Correcting tracklet associations instead of single time points speeds manual correction by one to two orders of magnitude.

## Caveats

As reported in the paper; not rerun. The number comes from prior work, not this paper. The tracklet method itself does not generalize (3.5% on Flavell data), so it serves correction, not tracking.
