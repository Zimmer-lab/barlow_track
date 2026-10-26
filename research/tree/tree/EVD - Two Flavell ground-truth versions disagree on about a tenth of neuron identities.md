---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1776f-7da9-7eb4-90aa-033acfee32a1
status: done
priority: normal
dateCreated: 2026-10-24T07:29
dateModified: 2026-10-26T09:59:36+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-26
---

## Purpose

Find which Flavell GT the preprint's 96.7% was measured against, and why the revision sweeps (scored against the project GT) read low.

## Method

Compare the project GT (flavell_data/projects/2025_07_01) and the NWB GT (flavell_data.nwb), matched by segmentation id; score the paper's stored tracks against each; check both against the project's masks on 6 frames.

## Results

Project GT 160 neurons / 216,238 entries; NWB GT 163 / 231,268. Matched by seg id, 89.7% of project entries agree; the median neuron agrees on 93.5% of frames, only 7/160 neurons above 99%; 11,748 cells present in both carry different seg ids. Both are consistent with the masks (99.9%). The 10 NWB-only neurons have more gaps (855-1,475 frames vs median ~1,480). Paper stored tracks: baseline 0.968 against the NWB GT, 0.927 against the project GT; untrained 0.696 vs 0.672.

## Conclusion

There are two annotation versions of the Flavell GT that differ in identity assignment, not only in which neurons are kept; the preprint's 96.7% is against the NWB GT. Revision sweep cells scored against the project GT (round_flavell*, objective accuracy) read about 4 points low relative to the paper.

## Caveats

A gap filter (the user's recollection) explains the extra NWB neurons but not the per-frame identity differences; which version is right is not known, and neither is manual (the Flavell GT is the Flavell lab's own tracking).
