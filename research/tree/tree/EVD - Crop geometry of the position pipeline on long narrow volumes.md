---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17cd8-2c77-75d2-8cda-3ec879d1dbb9
status: done
priority: normal
dateCreated: 2026-10-27T11:11:44+01:00
dateModified: 2026-10-27T11:11:44+01:00
confidence: medium
completedDate: 2026-10-27
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find whether the position pipeline's whole-volume rotation gives Flavell training crops that the network never sees at evaluation, and whether Zimmer, where the same pipeline reaches 0.94-0.97, is spared.

## Method

barlow_track@44c0e11:barlow_track/scripts/scratch/crop_geometry_by_pipeline.py, geometry only (volume shape and segmentation-metadata centroids, no image data): 30 random frames, 4 views per frame, default global augmentation (rotation up to 180 deg about z, scale 0.9-1.1, translation (2, 8, 8)). Per object and view: kept after the in-bounds test, in-plane offset of the neuron from the crop centre after clamping the window to the volume, fraction of crop voxels with no source in the raw data. Compared with the evaluation crop and the legacy per-crop rotation.

## Results

Flavell, volume (64, 284, 120), crop 16x64x64: evaluation offset >8 voxels 1.7%, fill 0. Position pipeline: kept 90.4%, offset p90 15.2 voxels, offset >8 voxels 17.6%, fill 0.8%. Legacy: kept 100%, offset >8 voxels 1.8%, fill 16.5% (rotated-crop corners, 67% of crops above 10%). Flavell, crop 16x32x32: position pipeline kept 90.4%, offset >8 voxels 3.9%. Zimmer, volume (22, 650, 900), crop 8x64x64 or 16x64x64: position pipeline kept 99.0%, offset >8 voxels 0.5%, against evaluation 0.2%.

## Conclusion

Rotating the 284 x 120 Flavell volume as a whole drops about a tenth of the neurons from each training view and pushes 18% of the remaining crops' neurons more than 8 voxels off-centre (10 times as many as at evaluation), because the worm's long axis rotates into the narrow one. On Zimmer's 650 x 900 volume neither happens (1% dropped, 0.5% off-centre). The legacy pipeline keeps every neuron centred and pays only padded corners. This is a Flavell-specific training/evaluation mismatch in the position pipeline, consistent with its cost being Flavell-specific.

## Caveats

Geometry only; whether it costs accuracy is tested by the global-rotation-off runs. Elastic field, position jitter and photometric transforms left out. Centroids from segmentation metadata; the legacy pipeline actually centres on the segmentation bounding box. Leifer and Samuel volume shapes not measured.
