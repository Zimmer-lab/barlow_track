---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-7178-76a3-9655-cb11b6e860e1
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

Find a quality metric computable without ground truth: does the fraction of time points a track is present (coverage, i.e. few gaps) predict its accuracy?

## Method

Fig. 3d-f: per-track accuracy vs track coverage (scatter with linear fit) and histogram of coverage, one panel per lab.

## Results

Most tracks have about 98% coverage (histogram peaks at 98-100%). The coverage-accuracy fit is steep for Zimmer (tracks at 90% coverage near 90% accuracy, at 100% near 98%) and nearly flat for Leifer and Flavell (≈96-98% across the range).

## Conclusion

Track coverage is high (≈98% for most tracks) and correlates positively with per-track accuracy, clearly on Zimmer data and only weakly on Leifer and Flavell data, so it is a rough proxy at best.

## Caveats

As reported in the paper; not rerun. No correlation coefficients or tests reported; slopes read from the figure. The text calls the correlation with "number of gaps"; the figure plots coverage. The co-author markup flags a wrong panel reference (Fig. 3b) in this section.
