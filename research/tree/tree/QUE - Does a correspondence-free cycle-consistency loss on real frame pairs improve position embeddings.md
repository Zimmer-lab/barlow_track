---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a186ca-11cc-7e08-b09f-dd5ec0fba44c
status: open
priority: low
dateCreated: 2026-10-29T09:32:32+01:00
dateModified: 2026-10-29T09:32:32+01:00
projects:
  - "[[barlowtrack]]"
---

## Question

Can the position (or fused) encoder learn from real non-rigid motion between frames t and t+k without any correspondence, using a cycle-consistency loss: softmax affinities between the two frames' object embeddings, t -> t+k and back, with the round trip required to return each object to itself (contrastive random walk, Jabri et al. 2020; an entropic optimal-transport / Sinkhorn soft assignment is a variant)? It would sit beside the Barlow loss on the synthetic pairs and needs neither GT nor a matcher.

## Context

Deferred by the user until after the revisions are finished (2026-10-29): too large a change for them. It came out of the real-frame-pair augmentation (position_augment pair_p, 2676174, posonly_fast batch 16, r261029-3720770), whose identities come from GT-free nonrigid_matches. Those are pseudo-labels from a locally rigid tracker. They resemble rigid augmentation, which ViSNet is invariant to, but the matched objects still carry the leftover strongly non-rigid motion, at the cost of the objects that do not match (Zimmer lag 1: ~42 of ~160 per frame at 0.96 precision against GT). The cycle loss would use every object and all of the motion.
