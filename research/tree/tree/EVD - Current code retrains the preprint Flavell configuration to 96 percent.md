---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1776f-7e24-7b2b-99c1-5e0ca04a2763
status: done
priority: normal
dateCreated: 2026-10-25T10:34
dateModified: 2026-10-26T09:59:36+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-26
---

## Purpose

Separate training-code and pipeline drift since the preprint from design changes: do today's training and evaluation code still produce a >95% Flavell network from the preprint's configuration?

## Method

Retrain the preprint Flavell baseline config with the code at bc9ff96 (seed 0, 11,900 steps, as the preprint run), score with eval_accuracy.py --source project against the NWB GT, 25 seeds, tracking seeds 0-2 (runs r261024-2611790, r261025-2443993).

## Results

0.9559 / 0.9572 / 0.9579 (mean 0.957; misses 3,660, mismatches 6,534 of 231,268 at seed 0). Preprint checkpoint on the same path 0.952 (it loses about 0.7 pt to the normalization mismatch, which this network does not have). Best revision Flavell network (round_flavell_more_epochs trial27) 0.924. Final losses vs the preprint run: train 2.8e-5 vs 1.5e-5, val 9.0e-4 vs 5.8e-4.

## Conclusion

Current training and evaluation code reproduce a 95.7% Flavell network from the preprint's configuration, slightly above the preprint checkpoint on the same path; the revision networks' remaining gap is in their design, not in the pipeline or training length.

## Caveats

One training seed. Flavell only. The train/val split differs from the preprint run's, and the loss code changed since v0.1.0, so the losses are not directly comparable.
