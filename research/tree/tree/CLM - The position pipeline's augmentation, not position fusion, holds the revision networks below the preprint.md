---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1776f-cb63-73ca-b611-fc8617478ee9
status: open
priority: normal
dateCreated: 2026-10-26T09:59:37+01:00
dateModified: 2026-10-26T22:24:55+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Claim

The revision networks lose Flavell accuracy in their image features, through the position pipeline's data path (global augmentation before cropping, with only blur and noise per crop), not through position fusion.

## Rationale

Switching only use_position on the preprint config drops Flavell from 0.957 to 0.77-0.85 across three training seeds (NWB GT). Training the same cell with the position branch removed gives 0.819 ([[EVD - Image features alone in the position pipeline]]), inside that range, so fusion is not where the points go. In every position-pipeline run, image-only included, validation loss rises over training while the legacy pipeline's falls, so the image features learned under the global data path generalize poorly from training frames to held-out ones.

## Scope

Flavell only; the image-only arm has one training seed. Which part of the data path costs the accuracy is open: the preprint's per-crop rotation and elastic deformation added on top made it worse (0.657, [[EVD - Preprint per-crop augmentation in the position pipeline]]), so it is not simply missing per-crop augmentation. Candidates: no non-rigid deformation between views (tested by [[EVD - Flavell-matched global elastic deformation in the position pipeline]]), global resampling of the volume, and other differences between VolumeCoordsDataModule and the legacy NeuronCropImageDataModule (crop source, frame sampling, batch composition). On Zimmer, the training set of most sweeps, the same pipeline reaches 0.94-0.97, so the cost is Flavell-specific or training-set-specific.
