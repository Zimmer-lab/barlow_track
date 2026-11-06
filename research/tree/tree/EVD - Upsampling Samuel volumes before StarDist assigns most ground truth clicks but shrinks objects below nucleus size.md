---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1ab99-2f3d-7ef8-b6f3-55b90f555b64
status: in-progress
priority: normal
dateCreated: 2026-11-05T13:05:05+01:00
dateModified: 2026-11-05T13:05:05+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Method

Samuel 153 re-segmented with the wbfm StarDist model students_and_lukas_3d_zarr (trained on Zimmer data) in a new project (wbfm_tracker_baseline/samuel_resegment/153_stardist, run r261105-3423621). GT centroids from samuel_data/153.nwb assigned with Leifer's strict rule (gt_tracks_from_clicks.py --rule mask --edge_um 0 --coincide_um 1.0 --voxel_um 1.75 0.45 0.45; r261105-3532825; written to 3-tracking/manual_annotation/gt_sd_strict.h5). Then, for 8 evenly spaced volumes (665 clicks), the red volume zoomed by (z, xy) before StarDist and the labels mapped back by nearest neighbour (stardist_rescale_test.py, uncommitted; r261105-3535238, -3554388, -3564506). A click counts as assigned if the mask at its voxel holds no other neuron's click.

## Result

Full video, no zoom: 30% of 120,923 clicks assigned, 36% outside every mask, 34% in masks shared by 2+ GT neurons, median neuron coverage 10%; 51 objects per volume (the NWB's GT-seeded masks: ~76).

| zoom z, xy | assigned | in a mask | shared | objects/volume | median object um^3 |
|---|---|---|---|---|---|
| 1, 1 | 32% | 67% | 34% | 56 | 117 |
| 1, 1.4 | 58% | 84% | 26% | 98 | - |
| 1.17, 1.75 | 74% | 91% | 17% | 155 | 45 |
| 2, 2.1 | 88% | 94% | 7% | 236 | 27 |
| 2.5, 2.1 | 89% | 93% | 4% | 278 | 23 |
| 3, 2.5 | 85% | 87% | 2% | 325 | 14 |

Zimmer StarDist objects: median ~97 um^3. Leifer StarDist strict: 95% of clicks assigned.

## Conclusion

Unscaled, StarDist merges Samuel's small (in voxels) nuclei and misses a third of the clicks, so a Samuel StarDist benchmark is not usable as is. Resampling to Zimmer's nucleus size in voxels (z 1.17, xy 1.75) more than doubles the assigned clicks (32% to 74%); stronger zooms reach 88-89% but cut objects to a quarter of a nucleus, i.e. split nuclei, which the click rule counts as assigned.

## Caveats

8 volumes only. The click rule rewards fragments, so assigned fraction overstates usable segmentation at strong zooms; whether fragments hurt tracking needs a re-segmented project and a tracking score. Object volumes assume the stated 0.45 um pixels, themselves in doubt (i261105-2; masks suggest ~0.55). Samuel's GT positions are centroids of GT-seeded masks, not clicks. wbfm's segmentation pipeline has no resampling step yet.
