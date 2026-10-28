---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a181e8-d505-75ef-a418-ff7bde19ca29
status: in-progress
priority: high
dateCreated: 2026-10-28T10:48:02+01:00
dateModified: 2026-10-28T11:10:43+01:00
projects:
  - "[[barlowtrack]]"
---

## Question

In the position pipeline, each training view gets its own global rotation (up to 180°), translation, 10% uniform scale, 180° "flip" (a rotation, not a mirror), 1-voxel jitter and 20% object dropout. Keypoints reach the position branch normalized by the fixed volume size only (`center_per_volume` centres embeddings before attention, not keypoints, and is inert without attention). ViSNet is invariant to rotation and translation but not scale, so between the two views it sees the scale jitter, the jitter and the dropout. Real Flavell frames differ by body bending.

If the two views' position inputs are much more alike than real frames, the arrangement of a neuron's neighbours could identify it almost for free in training (a shortcut), letting the image branch learn less and failing to carry across the recording; and a fusion cell trained this way could not show whether position helps. The user raised this on 2026-10-28; the same idea was [[CLM - Position fusion lets the position pipeline skip learning appearance on Flavell]], refuted as the main cause on 2026-10-26.

## Where it stands

Geometry: yes, partly. On Flavell the augmentation changes mid-range neighbour distances about 1.3x less than 100 frames of real motion and the nearest pairs more ([[EVD - The position augmentation deforms mid-range neighbour distances less than real motion does]]). Network: the trained position branch does not exploit it; it matches about 10% of neurons between its own training views, no better than an untrained branch ([[EVD - Trained position branches carry no more neuron identity than an untrained one on Flavell]]). What remains open is why the branch learns so little.
