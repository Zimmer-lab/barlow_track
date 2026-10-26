---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a165a4-a792-7516-a4d0-661f7e4d99f5
status: open
priority: low
dateCreated: 2026-10-23T00:04:12+02:00
dateModified: 2026-10-23T00:04:12+02:00
projects:
  - "[[barlowtrack]]"
---

## Question

How long do training, embedding and label-propagation tracking take per recording, on GPU and CPU, and when is reusing a pretrained network worth it?

## Context

The paper calls per-dataset retraining "computationally expensive" without numbers. docs/accuracy_evaluation.md gives CPU figures (≈100-150 neurons/s embedding; label propagation with 25 seeds ≈25-35 min on 120k points, ≈2 h on 283k) and notes full-video tracking does not fit on a 12 GB GPU next to the model.
