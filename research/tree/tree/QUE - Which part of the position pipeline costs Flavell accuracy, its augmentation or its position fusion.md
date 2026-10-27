---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a17930-f1ba-777d-be30-6f219f30985c
status: in-progress
priority: high
dateCreated: 2026-10-25T07:25
dateModified: 2026-10-27T18:00:46+01:00
projects:
  - "[[barlowtrack]]"
---

## Question

Switching the preprint Flavell configuration from the legacy crop pipeline to the position pipeline (use_position: true) costs 16-20 points. That switch changes two things at once: the data path (global affine before cropping, with only blur and noise per crop) and the model (ViSNet position features added to the image features). Which of the two costs the accuracy, and through what mechanism?

## Context

Tracked across runs since 2026-10-25 before it had a node. As of 2026-10-27 the data path, not fusion: training without the position branch scores 0.819, inside the fused cell's training-seed range 0.77-0.85 (NWB GT), against 0.944-0.957 for the legacy pipeline (four networks). The position-shortcut answer is refuted as the main cause. Ruled out within the data path so far: missing per-crop augmentation (adding it made things worse), global elastic deformation (worse), and the whole-volume rotation (turning it off left fused accuracy at 0.787). Still open: the per-crop rotation (croprot arm running), the global scale and translation, and how batches and crops are built.

Validation loss rises over training in every position-pipeline run, which was read as poor generalization of the image features; the norot run shows it rising while frame-pair matching on held-out frames improves every epoch, so the rise does not by itself mean poor generalization.

Single runs settle little here: the position pipeline's Flavell accuracy spreads by 5-12 points across training seeds (round_flavell_paper_ablation seed pairs), while the legacy pipeline's training-seed spread is unmeasured. Read every comparison against that spread.
