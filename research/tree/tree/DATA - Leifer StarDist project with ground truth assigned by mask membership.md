---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a1a5c0-be22-7865-9ba2-f4709b1d809f
status: open
priority: normal
dateCreated: 2026-11-04T09:50:34+01:00
dateModified: 2026-11-04T09:50:34+01:00
projects:
  - "[[barlowtrack]]"
---

## Description
The Leifer NeRVE worm1 recording (A2 re-export) segmented with StarDist. It uses the same images, masks and metadata as [[DATA - Leifer NeRVE recording re-segmented with stardist with ground truth assigned from manual clicks]], with a second ground truth (GT) assigned from the same manual clicks by mask membership rather than by centroid distance. It keeps 95.4% of the clicks instead of 86.6%. Benchmark dataset label `leifer_stardist_maskgt`; the centroid GT is `leifer_stardist`.

- Project: `/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline/leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist_maskgt/project_config.yaml`.
  - Made with wbfm `make_project_like(..., steps_to_keep=['preprocessing', 'segmentation', 'training_data'], target_suffix='_maskgt')`. This copies config files only. The kept steps point at the StarDist project's data by absolute path, so nothing was copied.
  - Masks, centroids and red data were checked identical at volumes 0 and 700.
- GT: `3-tracking/manual_annotation/gt_tracks_from_clicks_mask.h5`, with a `_provenance.json`. It is the project's `final_3d_tracks_df`.
  - 79 neurons, 113,499 of 119,019 clicks (95.4%).
  - Per-neuron coverage: minimum 0.47, 10th percentile 0.91, median 0.98. The centroid GT's are 0.43, 0.62 and 0.93.

## Method (for the paper)
The rule is applied per volume over every annotated neuron's clicks. Positions are in um; voxel size z x y is 1.5 x 0.3226 x 0.3226.
1. A click's object is the StarDist mask at the click voxel. If that voxel is background, it is the nearest mask voxel within 1.0 um (about three xy pixels).
2. A click with another neuron's click within 1.5 um (one z-plane) is assigned to none: the annotations themselves do not separate the two neurons.
3. A mask holding the clicks of two or more neurons is assigned to none of them, as a merge of neighbouring neurons.
4. A click matched only through the 1.0 um edge rule is dropped if any other click claims its mask. An edge click never removes a click that lies inside the mask.

No click was moved, interpolated or added. The other track columns come from the segmentation metadata (wbfm `add_metadata_to_df_raw_ind`), as for the centroid GT.

Exclusions, of 119,019 clicks:

| Reason | Clicks |
|---|---|
| No object in the volume (volumes 78, 405, 945, 1516) | 208 |
| Another neuron's click within 1.5 um | 1,541 |
| Mask shared with another neuron's click (about 850 are edge clicks whose nearest mask is another neuron's) | 3,003 |
| No mask within 1.0 um | 768 |

Assigned: 112,838 inside a mask, 661 on a mask edge.

Why not the centroid rule: 10,770 clicks (9.0%) lie inside a mask that holds no other click, but whose centroid is more than 1.5 um away (median 1.88, p90 2.67 um). StarDist objects span about six z-planes, so a click 1.5-3 um from the centroid is still on the object. See [[EVD - StarDist misses under one percent of the Leifer ground truth clicks]].

## Provenance
- Script: `barlow_track/scripts/project_reanalysis/gt_tracks_from_clicks.py --rule mask --edge_um 1.0 --coincide_um 1.5`, at 16a906d.
  - `--rule mask` was added in 3e2fdc7.
  - 16a906d fixed edge clicks removing in-mask clicks; that bug left the first build, r261104-3312389, at only 94.65%.
- Run: r261104-568713.
- Traces:
  - The GT h5 was also copied to `3-tracking/postprocessing/combined_3d_tracks.h5`, so that snakemake plans only `extract_full_traces` and `traces` (checked by dry runs first).
  - r261104-2551987 wrote `4-traces/` from the GT: 79 neurons. Frames without GT are zero intensity (6.5%). The paper traces keep 76 neurons, against 62 with the centroid GT.
  - The same was done in the StarDist project.

## Caveats
- Still a subset of the annotations, chosen by how StarDist segments them (merges and coincident clicks are excluded), though much less of a subset than the centroid GT.
- Networks score about one point lower against it than against the centroid GT, because the added clicks are harder ones. See [[EVD - Networks scored on the Leifer StarDist project against the centroid and mask ground truths]].
- Rerunning wbfm tracking in this project would overwrite `combined_3d_tracks.h5` and `final_3d_tracks_df`. The GT file in `manual_annotation/` is the reference copy.
