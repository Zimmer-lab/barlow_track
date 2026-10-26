---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a165a4-a510-7288-8e01-f2396a156fdc
status: open
priority: normal
dateCreated: 2026-10-23T00:04:11+02:00
dateModified: 2026-10-23T00:04:11+02:00
projects:
  - "[[barlowtrack]]"
---

## Question

If the geometric augmentations that carry the training signal are applied to the whole volume (and keypoints) before cropping, rather than to each crop, does accuracy or generalization improve?

## Context

Branch global_augmentation: d0f0c6f (#127), position-based and position-first augmentation (fd1b378, d64394b #133, 1238fb5 #130), 06f9558 (#137, on performance_tests), a napari augmentation viewer (c8813db, #131). Global augmentation keeps neighbouring objects and their positions consistent, which the position encoders need.
