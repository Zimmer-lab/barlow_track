---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-7d6a-7914-8735-d32b45a21722
status: open
priority: high
dateCreated: 2026-10-23T00:05:06+02:00
dateModified: 2026-10-23T00:05:06+02:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Look for a regime where the object-space term helps, or establish firmly that it does not.

## Method

Sweep lambd_obj (0, 0.33, 0.67, 1) crossed with training volumes (16, 64, 256) and encoder (image-only, ViSNet), several seeds, on all four GT sets with `objective: accuracy`; report accuracy with spread.

## Results

## Conclusion

## Caveats

Needs the seed replicates to tell small effects from noise.
