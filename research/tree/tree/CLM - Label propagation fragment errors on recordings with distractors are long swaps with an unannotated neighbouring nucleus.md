---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1b2e5-7b6b-7c8f-8ab4-f0264cc606a9
status: open
priority: normal
dateCreated: 2026-11-06T23:05:46+01:00
dateModified: 2026-11-06T23:05:46+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Claim

What the error decomposition calls fragmentation on full-length recordings with distractors (Z1210, F0330, Leifer StarDist) is mostly not a track broken in time. A GT neuron's label sits for tens to hundreds of volumes on a neighbouring nucleus of normal size, 4-5 um away, that has no GT click of its own, while the neuron itself is held by that neighbour's track. Both tracks span almost the whole recording. Tracker settings (batch 1), label merging and chain voting leave these errors unchanged.

## Scope

Zimmer 1210 and Leifer StarDist, one network each, seed 0 for the object-level analysis; Flavell 0330 shows the same concurrency of the partner tracks (no positions checked). On Leifer StarDist these are 5.6 of 6.3 pt of error, so they bound what tracker changes can gain there. Whether each one is a real swap or a click on the boundary between touching masks needs the manual check of [[EVD - Half of the base network's Leifer StarDist errors are clicks where every tracker agrees on another object]].
