---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-7f1b-74e7-a388-f82248c330fb
status: open
priority: normal
dateCreated: 2026-10-23T00:05:07+02:00
dateModified: 2026-10-23T00:05:07+02:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Measure whether volume-level (global) augmentation before cropping beats per-crop augmentation.

## Method

Train matched networks (same seeds, crops, epochs) with global vs local augmentation and with position-first augmentation (global_augmentation branch), score on the four GT sets.

## Results

## Conclusion

## Caveats

06f9558 (#137) sits on performance_tests, not on global_augmentation; check which version is meant.
