---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a1a5e5-5216-7e3c-8f87-312260cd2752
status: open
priority: normal
dateCreated: 2026-11-04T10:30:31+01:00
dateModified: 2026-11-04T10:59:41+01:00
projects:
  - "[[barlowtrack]]"
---

## Description
The Leifer NeRVE worm1 recording (A2 re-export) as a full wbfm project with NeRVE's own segmentation and the ground truth (GT) as NeRVE assigns it, plus wbfm traces. It is the reference to compare the StarDist projects' traces against, and the nearest the current code comes to the paper's Leifer traces (computed differently: wbfm trace extraction from the re-export's masks).

- Project: `/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline/leifer_reexport/Leifer_NeRVE_Worm1_reexport_A2_nervegt/project_config.yaml`.
  - Made with wbfm `make_project_like` of `.../leifer_reexport/Leifer_NeRVE_Worm1_reexport_A2` (steps kept: preprocessing, segmentation, training_data; suffix `_nervegt`). Configs only; the kept steps point at the A2 project by absolute path.
  - Segmentation: NeRVE's detections only, in the project's own `1-segmentation/masks.zarr` (labels 1-152, consecutive per volume) and a copy of the A2 metadata. The A2 project's masks also hold pseudo masks (label 1000 + GT index, watershed around the 12,019 clicks NeRVE missed, added by `export_leifer_nwb.py` and not in the metadata); `drop_mask_labels.py` (7a84d3f) set them to background here, every other voxel and the metadata unchanged (checked at volumes 0, 700, 1500). The A2 project itself is untouched.
- GT: `3-tracking/manual_annotation/gt_tracks_from_nerve_segmentation_id.h5` (with `_provenance.json`), the project's `final_3d_tracks_df`. 79 neurons, 107,000 of 119,019 clicks (89.9%). Per-neuron coverage p10 0.81, median 0.97; one neuron (minimum 0.0007) NeRVE almost never detects.
- Traces: `4-traces/` (red and green traces of 79 neurons, 11.8% of frames zero or NaN; paper traces 69 neurons; heatmap and grid plots).

## Method (for the paper)
Each click takes the NWB's own `raw_segmentation_id`, NeRVE's assignment of that click to a mask (`gt_tracks_from_clicks.py --rule segmentation_id --max_label 1000`, 7799d00). Excluded:
| Reason | Clicks |
|---|---|
| Label is a pseudo mask (>= 1000), built from the click, not by NeRVE | 10,603 |
| No objects in the volume (NeRVE's 26 empty volumes) | 1,416 |
| Label shared by two neurons | 0 (NeRVE's assignment is one to one) |

Traces: the GT h5 was copied to `3-tracking/postprocessing/combined_3d_tracks.h5`, so the snakemake dry run plans only `extract_full_traces` and `traces`; then `snakemake traces` (r261104-2149802).

## Provenance
- Runs: r261104-1389743 wrote the GT, then traces that came out empty (below); r261104-2149802 traces with a wbfm workaround, since discarded; r261104-1391957 the current traces, stock wbfm on the NeRVE-only masks (79 neurons, 69 in the paper traces, as with the workaround).
- Why the masks were copied: wbfm's `reindex_segmentation` uses a 1000-entry lookup table, and a label >= 1000 raised IndexError, which zeroed that volume's reindexed mask; with pseudo masks in every volume, every trace was empty. Labels >= 1000 are treated as an upstream error (the user's call, 2026-11-04), so instead of padding the table wbfm now raises a clear ValueError naming the volume and label, in both `reindex_segmentation` and `all_matches_to_lookup_tables` (dlc_for_wbfm `dev`, uncommitted on 2026-11-04), and the masks were fixed.
- The GT copy in `3-tracking/postprocessing/combined_3d_tracks.h5` was refreshed after the new masks: older than the masks, it made the snakemake dry run plan the whole tracking chain.

## Caveats
- The traces are not the paper's Leifer traces: NeRVE's masks, our re-export and current wbfm extraction, not the preprint's pipeline. Not yet compared number for number.
- Coverage differs from the StarDist GTs (89.9% here, 86.6% centroid, 95.4% mask), so the neuron sets in the paper traces differ (69, 62, 76).
- Rerunning tracking here would overwrite `combined_3d_tracks.h5` and `final_3d_tracks_df`.
