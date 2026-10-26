---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1776f-cb63-73ca-b611-fc8617478ee9
status: open
priority: normal
dateCreated: 2026-10-26T09:59:37+01:00
dateModified: 2026-10-26T18:11:57+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Claim

The revision networks lose accuracy in their image features, through the position pipeline's data path (global augmentation before cropping, with only blur and noise per crop), while position fusion itself adds about a point.

## Rationale

Switching only use_position on the preprint config drops Flavell from 0.957 to 0.795 (0.756 backbone-only); the best revision network's backbone alone is 0.915 against its fused 0.927 and the preprint's 0.952. The per-crop rotation and elastic deformation of the preprint pipeline are missing from the position pipeline.

## Scope

A working hypothesis, low confidence: one seed, Flavell only, and use_position switches augmentation and fusion together. The simplest version, that the missing per-crop rotation and elastic deformation are the gap, failed: adding them lowered Flavell to 0.657 ([[EVD - Preprint per-crop augmentation in the position pipeline]]). What remains is that the global data path itself (global resampling, no non-rigid deformation between views) costs the accuracy, tested by [[EVD - Image features alone in the position pipeline]] (an image-only score near the position pipeline's supports it) and [[EVD - Flavell-matched global elastic deformation in the position pipeline]]. Competes with [[CLM - Position fusion lets the position pipeline skip learning appearance on Flavell]].
