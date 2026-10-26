---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-12cc-7ab1-b5ae-e2a7d0415782
status: done
priority: normal
dateCreated: 2026-10-19
dateModified: 2026-10-19
confidence: low
completedDate: 2026-10-19
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Check whether the Barlow test loss can select architectures in the revision sweeps.

## Method

Rank correlation of test loss with full-video tracking accuracy across trials of the Oct 2026 sweep rounds (summary in AGENTS.md).

## Results

Loss rank-correlates with accuracy only within narrow regimes; elsewhere ≈0 and sometimes negative. The sweeps therefore added `objective: accuracy` (346c26c).

## Conclusion

Across architectures and crop sizes, lower validation loss does not reliably mean higher tracking accuracy, which limits Fig. 3g to one narrow preprint-era search.

## Caveats

No correlation values or trial counts in the summary; git-ignored source (AGENTS.md); cluster records not checked. Crop-size changes alter what the loss measures, so part of the breakdown may be by construction.
