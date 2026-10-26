---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-1912-738d-aa6e-f9ed04db7df5
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: medium
completedDate: 2025-11-12
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Place BarlowTrack against previously published trackers.

## Method

Table 2, other rows: published accuracies on the Leifer dataset (NeRVE, fDNC, CeNDeR, ZephIR) taken from the literature; the authors' own earlier tracklet method (Fieseler et al. 2025) rerun on Zimmer and Flavell data (Fig. S1d, trained on the Zimmer datasets used for evaluation).

## Results

Leifer: NeRVE 82.9%, fDNC 79.1%, CeNDeR 88.25%*, ZephIR 94.5%* (* needs partial GT annotation) vs BarlowTrack 97.4%. Zimmer tracklet method: Zimmer 91.4%, Flavell 3.5% (near noise, Fig. S1d broken axis), vs BarlowTrack 97.3% and 96.7%.

## Conclusion

On the Leifer benchmark BarlowTrack beats the best published method (ZephIR, which needs partial annotation) by about 3 points and annotation-free methods by 8-18 points; the authors' earlier tracklet method falls from 91.4% on its own lab's data to 3.5% on Flavell data.

## Caveats

As reported in the paper; not rerun. Literature numbers were not rerun on the same segmentation, frames or metric; no other method was run on Zimmer or Flavell data. The Flavell lab's own tracker is the Flavell GT, so it cannot be compared there.
