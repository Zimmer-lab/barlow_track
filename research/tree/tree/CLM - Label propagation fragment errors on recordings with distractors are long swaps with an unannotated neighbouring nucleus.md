---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1b2e5-7b6b-7c8f-8ab4-f0264cc606a9
status: open
priority: normal
dateCreated: 2026-11-06T23:05:46+01:00
dateModified: 2026-11-07T02:24:47+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Claim

What the error decomposition calls fragmentation on full-length recordings with distractors is mostly not a track broken in time. On Zimmer 1210 a GT neuron's label sits for tens to hundreds of volumes on a neighbouring nucleus of normal size, about 5 um away, that has no GT click of its own, while the neuron itself is held by that neighbour's track; at 76% of these label switches the track itself moves to the other object. Both tracks span almost the whole recording. Tracker settings (batch 1), label merging and chain voting leave these errors unchanged.

## Scope

Zimmer 1210, one network, seed 0 for the object-level analysis; Flavell 0330 shows the same concurrency of the partner tracks (no positions checked). Revised 2026-11-07: Leifer StarDist was first counted here, but there the tracker mostly holds its object while the click's mask flickers between touching masks ([[CLM - Most of the label propagation error left on the StarDist Leifer recording is clicks on the boundary of touching masks]]).
