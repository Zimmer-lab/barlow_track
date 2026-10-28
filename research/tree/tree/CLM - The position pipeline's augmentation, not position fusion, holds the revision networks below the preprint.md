---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1776f-cb63-73ca-b611-fc8617478ee9
status: open
priority: normal
dateCreated: 2026-10-26T09:59:37+01:00
dateModified: 2026-10-28T09:44:55+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Claim

The revision networks lose Flavell accuracy in their image features, through how the position pipeline makes its training views (global augmentation before cropping, crop-centre jitter and dropout, with only blur and noise per crop), not through position fusion and not through the rest of its data path.

## Rationale

Switching only use_position on the preprint config drops Flavell from 0.957 to 0.77-0.85 across three training seeds (NWB GT). Training the same cell with the position branch removed gives 0.819 ([[EVD - Image features alone in the position pipeline]]), inside that range, so fusion is not where the points go. Keeping the position pipeline's data path but making its views legacy-style (preprint per-crop transforms, no global volume affine, no jitter or dropout, re-drawn views) recovers 0.938 truncated / 0.950 rounded crop centres after 12 of 50 epochs ([[EVD - Legacy-style crop views in the position data path recover preprint Flavell accuracy]]), within a point of the legacy pipeline; its validation loss falls like the legacy pipeline's, where every position run's rose.

## Scope

Flavell only, single training seeds per arm; position-pipeline Flavell accuracy spans up to 12 points across seeds ([[EVD - Position-pipeline Flavell accuracy spans 12 points across training seeds]]), but 0.95 lies above all 16 position runs. Which part of the view generation costs the accuracy is open: the legacy-matched arm changes the global affine, jitter, dropout, per-crop transforms and re-drawing at once. Adding geometric augmentation on top of the revision views made things worse (per-crop rotation and elastic 0.657, [[EVD - Preprint per-crop augmentation in the position pipeline]]; global elastic 0.718, [[EVD - Flavell-matched global elastic deformation in the position pipeline]]; re-drawn views 0.652 at epoch 32), which points at geometry reaching the crops. round_flavell_legacy_crop_aug (fusion x global augmentation on keypoints only or on the volume too, 3 seeds) separates these and tests fusion on legacy-style views. On Zimmer, the training set of most sweeps, the revision pipeline reaches 0.94-0.97, so the cost is Flavell-specific or training-set-specific.
