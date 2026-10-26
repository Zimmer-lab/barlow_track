---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-7eaf-74b8-abb0-9186bcc1ef64
status: open
priority: normal
dateCreated: 2026-10-23T00:05:07+02:00
dateModified: 2026-10-23T00:05:07+02:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find the crop size each dataset needs, especially Flavell.

## Method

Sweep target_sz_z (4, 8, 12, 16) and target_sz_xy (32, 48, 64) with `objective: accuracy` per dataset, with the ViSNet and the image-only encoder.

## Results

## Conclusion

## Caveats

Not comparable across crop sizes under `objective: loss`.
