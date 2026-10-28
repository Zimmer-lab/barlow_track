---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a181fd-4d08-7d24-809c-c0dcab1c2d52
status: done
priority: normal
dateCreated: 2026-10-28T11:10:23+01:00
dateModified: 2026-10-28T11:10:23+01:00
confidence: medium
completedDate: 2026-10-28
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

[[EVD - The position augmentation deforms mid-range neighbour distances less than real motion does]] showed the training views are geometrically easier than real frames, and its caveat left open whether the network uses that. Test it at the network: can the position branch alone identify neurons between the two training views (a shortcut would need it to), and between real frames?

## Method

[[@BarlowTrack position-branch matching between augmented views and real frames]], run r261028-2940337 (barlow_track@a3bab89).

## Results

Fraction of GT neurons matched correctly by position features alone (chance about 0.006 with ~178 detections):

| checkpoint (fused Flavell acc.) | between training views | real frames, lag 1 / 10 / 100 / 500 / 1000 | effective rank |
|---|---|---|---|
| use_position (0.795) | 0.098 | 0.031 / 0.027 / 0.024 / 0.028 / 0.026 | 2.2 |
| norot (0.787) | 0.057 | 0.018 / 0.015 / 0.015 / 0.016 / 0.014 | 2.0 |
| elastic | 0.093 | 0.031 / 0.031 / 0.025 / 0.029 / 0.027 | 1.9 |
| legacyaug (0.657) | 0.098 | 0.031 / 0.030 / 0.024 / 0.026 / 0.023 | 2.0 |
| re-initialized branch | 0.064-0.113 | 0.035 / 0.028 / 0.026 / 0.033 / 0.029 | 2.0 |

Sanity checks on the comparator, one frame: matching a frame to itself 1.000; with 0.1 / 0.3 / 1.0 voxel jitter 0.99 / 0.80 / 0.52. Raw (unstandardized) descriptors have mean cosine 0.99 between different neurons.

## Conclusion

On Flavell the trained ViSNet position branch identifies about 10% of neurons between its own two training views and 2-3% between real frames at any lag, no better than the same branch at random initialization, and its descriptors span about 2 effective dimensions in every checkpoint. The position branch cannot be the identity shortcut the training views would allow: it encodes coarse location in the worm, not which neuron is which, and training did not make it more informative. The open problem in the position branch is that it learns almost nothing, not that it learns too easily.

## Caveats

Flavell only, four checkpoints of one config family (preprint Flavell config through the position pipeline), one training seed each; the better-tuned round_flavell_more_epochs trial27 (fused 0.927, backbone 0.915) was not measured. Position features were scored on their own; in the fused descriptor a 2-dimensional location code can still help disambiguate neurons the image features confuse (fusion added about 1 point on trial27). GT neurons scored, but all detections embedded. Effective rank from one frame. The untrained branch's rank of 2 suggests the encoder setup itself (single node type, lmax 1, cutoff 0.35 on a volume whose short axes span about 0.3, up to 32 neighbours) gives little room for per-neuron detail; not tested.
