---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a165a4-a22d-7191-b0b2-05b2d54b97c4
status: open
priority: high
dateCreated: 2026-10-23T00:04:10+02:00
dateModified: 2026-10-26T09:59:59+01:00
projects:
  - "[[barlowtrack]]"
---

## Question

After the post-preprint fixes to evaluation, label propagation, tracking windows and cropping, do the preprint's default-setting networks still reach 96.7-97.4% on the Zimmer, Flavell and Leifer benchmarks?

## Context

A code review after the preprint (docs/pr_plan PR01-PR04, written at 9fbeda5) found bugs in accuracy evaluation, label-propagation edge cases, tracking windows and crops at volume edges. Revision sweeps then scored Flavell at .62-.73 against the preprint's 96.7%.

As of 2026-10-25, Flavell is answered: [[CLM - Current training and evaluation code reproduce the preprint's Flavell accuracy]], and the sweeps' low Flavell scores were mostly the evaluation's doing ([[CLM - Most of the revision networks' Flavell deficit came from the evaluation, not the networks]]; issue i261026-1). Zimmer and Leifer are open ([[EVD - Preprint baseline networks re-scored with the corrected evaluation]]), as are the regression tests. Reviewer comments were not found; the question is my reading.
