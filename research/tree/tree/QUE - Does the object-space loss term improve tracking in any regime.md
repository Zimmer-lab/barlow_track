---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a165a4-a298-7e44-b912-b4789186902f
status: open
priority: high
dateCreated: 2026-10-23T00:04:10+02:00
dateModified: 2026-10-23T00:04:10+02:00
projects:
  - "[[barlowtrack]]"
---

## Question

The abstract presents the object-space term as the paper's contribution, but the ablations show no gain at the accuracy ceiling. Does the term help where accuracy is not saturated: few training volumes, harder datasets (Flavell), smaller crops, or the newer position-aware encoders?

## Context

Fig. S1e and the loss-balance sweep (whose values are not shown and whose text still has a "??" placeholder). A reviewer will likely ask why the headline contribution has no measured benefit; either show a regime where it helps or reframe the contribution (e.g. that a per-time-point object loss suffices on its own).
