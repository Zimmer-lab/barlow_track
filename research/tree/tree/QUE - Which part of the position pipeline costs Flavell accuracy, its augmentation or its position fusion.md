---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a17930-f1ba-777d-be30-6f219f30985c
status: in-progress
priority: high
dateCreated: 2026-10-25T07:25
dateModified: 2026-10-25T07:25
projects:
  - "[[barlowtrack]]"
---

## Question

Switching the preprint Flavell configuration from the legacy crop pipeline to the position pipeline (use_position: true) costs 16-20 points. That switch changes two things at once: the data path (global affine before cropping, with only blur and noise per crop) and the model (ViSNet position features added to the image features). Which of the two costs the accuracy, and through what mechanism?

## Context

Tracked across runs since 2026-10-25 before it had a node. Candidate answers so far: the augmentation (the position pipeline lacks the preprint's per-crop rotation and elastic deformation) and a position shortcut (within a training pair both views are one volume under a near-rigid transform, so position almost solves the Barlow loss and the image branch need not learn identity). Both point at the same lever: augmentation that moves neurons non-rigidly between views, which the global elastic deformation (914141c) adds.

Single runs settle little here: the position pipeline's Flavell accuracy spreads by about +-4 points across training seeds (round_flavell_paper_ablation seed pairs, round_flavell_more_epochs), while the legacy pipeline's training-seed spread is unmeasured. Read every comparison against that spread.
