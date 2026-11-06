---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a1a858-9f67-78b5-9bd6-0450f012aeb6
status: in-progress
priority: high
dateCreated: 2026-11-04T21:55:42+01:00
dateModified: 2026-11-04T21:55:42+01:00
projects:
  - "[[barlowtrack]]"
---

The Leifer NeRVE recording has manual GT clicks, not masks, so every accuracy number on it depends on a segmentation (NeRVE's own, with pseudo masks around clicks it missed, or StarDist's) and on a rule turning clicks into GT objects (centroid within 1.0 or 1.5 um, mask membership, NeRVE's segmentation id). The user wants this tested rigorously and summarised in a supplementary figure: which segmentation and rule, how much accuracy moves with each choice and with labelprop seed, where the errors fall, and how much the traces differ between segmentations and between GT-tracked and tracker-tracked neurons.
