---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1a960-12b8-7667-80b9-8d6890f42021
status: open
priority: normal
dateCreated: 2026-11-05T02:43:28+01:00
dateModified: 2026-11-05T09:00:43+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
Four networks were tracked on the Leifer project with only NeRVE's detections and scored against the GT on those masks. On the clicks NeRVE segmented they reach 0.887–0.893 (gt_valid). Counting the 11,663 pseudo-mask clicks as segmentation misses, they reach 0.799–0.805. The same base network on StarDist masks scores 0.940 on assigned clicks and 0.920 on all kept clicks (strict GT).

The original NeRVE benchmark tracks the pseudo masks (built from the GT clicks) as objects. Re-run, it gives 0.885–0.886, and its pseudo masks are not what holds it down:
- On NeRVE's own clicks t57 scores 0.895 with the pseudo masks present and 0.891 without them.
- On the pseudo-mask clicks themselves it scores 0.844.

The deficit is about 10k mismatches on clicks NeRVE did segment, against about 2.4k on StarDist. A network trained on NeRVE masks (ref_local_nerve, 0.888) does no better.

The cause is NeRVE over-splitting nuclei. Of the 103,515 clicks assigned on both segmentations, 17.5% sit in a StarDist object that NeRVE splits into two or more masks. On those, base fails 35.6% on NeRVE masks and 10.2% on StarDist. On all other clicks NeRVE masks are slightly easier (4.3% vs 5.0% errors). 72.8% of the clicks base gets right on StarDist and wrong on NeRVE are on split objects. The StarDist objects there are of normal size (median 538 voxels vs 552), so these are NeRVE splitting a nucleus, not StarDist merging two.

## Method
- Project `Leifer_NeRVE_Worm1_reexport_A2_nervegt` (NeRVE masks only).
  - GT variant nerve_only: each click's NWB `raw_segmentation_id`. Pseudo masks (label 1000 and above) count as segmentation misses. Coincident clicks are excluded at 1.0 µm.
  - Built with `gt_tracks_from_clicks.py` (891f275, r261104-2025465): 106,552 clicks assigned of 118,215 kept.
  - Networks were embedded and tracked on CPU (labelprop, seed 0, 25 seeds), saving tracks (r261104-3067468). Scored with `score_gt_variants.py` (ab90a41, which aligns the 1,523 tracked volumes by time index).
- Original-benchmark path:
  - `eval_accuracy --source nwb --nwb_objects all --nwb_gt_mapping segmentation_id`, which tracks the pseudo masks as objects. Re-tracked on the GPU with saved tracks (r261105-3009009).
  - Scored against nerve_pseudo_nwblabels (all 118,215 kept clicks assigned in NWB label space, coincident clicks excluded) with `--split_label_at 1000` (45a04d2).

NeRVE masks only:

| Network | gt_valid, assigned | gt_valid, all kept | nandist, assigned | Tracker misses | Mismatches |
|---|---|---|---|---|---|
| leifer_reseg base | 0.8925 | 0.8045 | 0.8847 | 1,547 | 9,905 |
| wk2 trial 57 | 0.8909 | 0.8030 | 0.8831 | 1,294 | 10,333 |
| wk2 trial 50 | 0.8865 | 0.7991 | 0.8787 | 1,192 | 10,897 |
| ref_local_nerve (trained on NeRVE masks) | 0.8879 | 0.8003 | 0.8800 | 1,701 | 10,244 |

NeRVE masks plus pseudo masks (original benchmark), gt_valid:

| Network | Eval (all 119,019 clicks, nandist) | All kept | NeRVE-segmented clicks (106,552) | Pseudo-mask clicks (11,663) |
|---|---|---|---|---|
| wk2 trial 57 | 0.8859 | 0.8899 | 0.8949 (mismatches 10,461) | 0.8444 (mismatches 1,145) |
| leifer_reseg base | 0.8849 | 0.8887 | 0.8932 (mismatches 10,305) | 0.8478 (mismatches 1,108) |

Per-click comparison, base seed 0 (`nerve_vs_stardist_clicks.py`, cf62da6, r261105-3147438; output `~/.cache/barlow_scratch/train/leifer_reseg/nerve_vs_stardist/`). Clicks assigned on both sd_strict and nerve_only: 89,808 correct on both, 7,568 StarDist only, 3,603 NeRVE only, 2,536 neither. A split means 2 or more NeRVE masks each with at least 20% of their voxels inside the click's StarDist object.

| Class | n | StarDist object (vox) | NeRVE mask (vox) | Nearest NeRVE mask (µm) | NeRVE splits the StarDist object | NeRVE mask spans 2+ StarDist objects |
|---|---|---|---|---|---|---|
| correct on both | 89,808 | 552 | 135 | 4.03 | 12.0% | 1.9% |
| StarDist only | 7,568 | 538 | 110 | 2.55 | 72.8% | 5.0% |
| NeRVE only | 3,603 | 471 | 137 | 3.41 | 25.2% | 14.1% |
| neither | 2,536 | 492 | 122 | 2.76 | 37.1% | 4.1% |

Errors by crowding (quartiles of the distance to the nearest other NeRVE mask): nearest 0.8–3.1 µm, 28.7% on NeRVE vs 12.0% on StarDist; 3.1–3.9 µm, 6.2% vs 5.6%; beyond 3.9 µm, about 2% on both.

## Caveats
- The kept sets differ from StarDist strict (118,215 vs 115,507). NeRVE has no merges, while StarDist strict excludes them. The per-click comparison of which clicks fail on which segmentation is not done.
- The split analysis uses one network (base, seed 0) and a 20% overlap threshold for a split.
- One labelprop seed per network.
