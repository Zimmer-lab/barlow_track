---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a165a4-a43b-70d3-9511-7d6d7f3df7fb
status: open
priority: normal
dateCreated: 2026-10-23T00:04:11+02:00
dateModified: 2026-10-23T00:04:11+02:00
projects:
  - "[[barlowtrack]]"
---

## Question

Does the Barlow validation loss rank networks by tracking accuracy across architectures, crop sizes and datasets, or only within one narrow search?

## Context

Fig. 3g supports it within one preprint-era search per lab. The revision sweeps switched to optimizing tracking accuracy directly on the training project (`objective: accuracy`, 346c26c) after finding the loss rank-correlates only within narrow regimes. Matters for users without ground truth, which is the paper's target audience.
