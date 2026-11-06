---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1a5c0-7735-78bc-935b-6ddaed5903aa
status: open
priority: normal
dateCreated: 2026-11-04T09:50:16+01:00
dateModified: 2026-11-04T09:50:16+01:00
confidence: high
projects:
  - "[[barlowtrack]]"
---

## Conclusion
Of 119,019 manual GT clicks on the Leifer recording, only 700 (0.6%) are objects NeRVE segmented but StarDist did not (no StarDist mask within 1 um), and only about 290 (0.25%) are more than 2 um from any StarDist object. These misses are dim (median 0.51x the brightness of assigned clicks), and the 180-object cap never removed anything (StarDist finds 103-145 objects per volume). Every other click the centroid GT excluded is a merge, coincident annotations, or a click inside or just off a mask whose centroid is far away, so the mask GT can keep 95.4% of the clicks.

## Method
Every click (volume, neuron) put in exactly one category, first match wins, replaying the centroid GT's rule (`gt_click_exclusions.py`, 5db47c7, r261104-885770; it reproduces the centroid GT's 103,052 assignments exactly). Then per volume: StarDist object count, NeRVE object count (every 10th volume), distance from each click to the nearest mask voxel, and red intensity at each click (mean of a 3 x 5 x 5 voxel box, z x y) relative to the median of that volume's assigned clicks (`gt_click_misses_vs_cap.py`, 6481f73, r261104-3278557). Voxel 1.5 x 0.3226 x 0.3226 um.

| Category | Clicks | % | Distance | Brightness vs assigned |
|---|---|---|---|---|
| Assigned by the centroid rule (1.5 um) | 103,052 | 86.6 | - | 1.0 (p10 0.44) |
| Inside a mask holding no other click, centroid > 1.5 um | 10,770 | 9.0 | to centroid median 1.88, p90 2.67, p99 3.91 um | 1.25 |
| Background, nearest mask within 1 um | 1,510 | 1.3 | to mask median 0.46, p90 0.91 um | 1.50 |
| Another neuron's click within 1.5 um | 1,353 | 1.1 | - | 1.34 |
| Mask also holds another neuron's click (merge) | 1,358 | 1.1 | - | 1.42 |
| NeRVE has a mask, StarDist none within 1 um | 700 | 0.6 | to mask median 1.6, p90 7.8 um | 0.51 (p10 0.18) |
| No mask in either segmentation | 68 | 0.06 | median 1.4 um | 1.03 |
| Volume has no objects (4 corrupted volumes) | 208 | 0.2 | - | - |

Clicks with no mask under them (2,278), by distance to the nearest StarDist mask: under 0.5 um 786, 0.5-1 um 724, 1-1.5 um 259, 1.5-2 um 217, 2-3 um 105, 3-5 um 96, over 5 um 91.

Object counts: StarDist 103-145 per volume (median 129, 1,532 volumes with objects), never at the cap `max_number_of_objects: 180` that wbfm's `perform_post_processing_3d` applies (it keeps the brightest 180). NeRVE: median 125 (10th-90th percentile 117-136). The dimmest StarDist object kept per volume is 0.17x the assigned clicks' brightness; the volume median voxel 0.11x.

Also: 939 assigned clicks share their mask with another neuron's click (the centroid rule gave the mask to the nearer one). neuron_070 is an outlier, 35% of its clicks on background and 11% coincident.

## Caveats
- One recording, one StarDist model (`students_and_lukas_3d_zarr`).
- "Missed" is judged against NeRVE's masks and a 1 um tolerance; a NeRVE mask is not proof of a real neuron, and the 1 um tolerance is a choice.
- Brightness uses the red channel in a fixed box, not the object; clicks near a bright neighbour read brighter.
- Clicks just off a mask (1.3%) are brighter than assigned ones, which suggests tight masks on bright, crowded neurons; not checked by eye.
