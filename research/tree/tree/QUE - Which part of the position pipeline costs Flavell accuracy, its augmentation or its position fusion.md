---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a17930-f1ba-777d-be30-6f219f30985c
status: in-progress
priority: high
dateCreated: 2026-10-25T07:25
dateModified: 2026-10-26T22:25:30+01:00
projects:
  - "[[barlowtrack]]"
---

## Question

Switching the preprint Flavell configuration from the legacy crop pipeline to the position pipeline (use_position: true) costs 16-20 points. That switch changes two things at once: the data path (global affine before cropping, with only blur and noise per crop) and the model (ViSNet position features added to the image features). Which of the two costs the accuracy, and through what mechanism?

## Context

Tracked across runs since 2026-10-25 before it had a node. As of 2026-10-26 the data path, not fusion: training without the position branch scores 0.819, inside the fused cell's training-seed range 0.77-0.85 (NWB GT), against 0.957 for the legacy pipeline. The position-shortcut answer is refuted as the main cause, and missing per-crop augmentation is ruled out (adding it made things worse). Open: which part of the data path. Validation loss rises in every position-pipeline run and falls in the legacy one, so the global data path's image features generalize poorly from training frames; non-rigid deformation between views (global elastic, 914141c) is the first candidate under test.

Single runs settle little here: the position pipeline's Flavell accuracy spreads by 5-12 points across training seeds (round_flavell_paper_ablation seed pairs), while the legacy pipeline's training-seed spread is unmeasured. Read every comparison against that spread.
