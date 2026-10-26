---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a165a4-a4a5-72ef-8bae-2336ab98e56e
status: open
priority: normal
dateCreated: 2026-10-23T00:04:11+02:00
dateModified: 2026-10-23T00:04:11+02:00
projects:
  - "[[barlowtrack]]"
---

## Question

What crop size (z and xy extent) maximizes accuracy per dataset, and is Flavell's accuracy limited by crop and data rather than by the encoder?

## Context

Methods 4.3 states crop size matters and Flavell needs 16 z-slices, without data. Crop size cannot be swept under the loss objective but can under the accuracy objective (AGENTS.md). Current defaults (8, 64, 64), Flavell z 16; Flavell neurons span median 8 / p90 12 / max 16 z-planes. Flavell stays at .62-.73 across all revision architectures. Cropping and shape fixes since the preprint: 726c913 (#141), b28fd39.
