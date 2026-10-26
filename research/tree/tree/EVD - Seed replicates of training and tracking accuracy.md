---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-7d01-7057-9b3b-21ce740dc176
status: open
priority: normal
dateCreated: 2026-10-23T00:05:06+02:00
dateModified: 2026-10-23T00:05:06+02:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Give the spread behind Table 2's medians (docs/pr_plan PR06).

## Method

Train the default network with several seeds per lab (seed_all / replicate_seed in utils_seeding.py), track each with several label-propagation seeds, report median and range; test that the same seed gives identical label propagation, seed-time list and eval JSON, and different seeds differ.

## Results

## Conclusion

## Caveats

80a2bbd (#144) and aa42e45 implement part of PR06; verify coverage first.
