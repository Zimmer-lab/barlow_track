---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a1a274-5dba-7559-ab8b-83a4ffda9709
status: open
priority: normal
dateCreated: 2026-11-03T18:28:17+01:00
dateModified: 2026-11-05T05:13:49+01:00
projects:
  - "[[barlowtrack]]"
---

## Description
> **Not the headline Leifer GT (user decision, 2026-11-04).** The headline is "StarDist strict":
> - a click outside every mask counts as a segmentation miss;
> - coincident clicks at 1.0 um and merges are excluded;
> - matching is gt_valid.
> Files are in `3-tracking/manual_annotation/variants/` (sd_strict and the other variants, each as `_status.npz` plus `.h5`) and the `_stardist_strictgt` project.
> Scores: [[EVD - On the strict Leifer StarDist ground truth the base network tracks 92 percent of kept clicks]].
> The 1.5 um and 1.0 um files below are earlier variants, kept for comparison.

The Leifer lab NeRVE worm1 recording (the A2 re-export) as a regular wbfm project segmented with stardist rather than NeRVE, with ground truth (GT) expressed in the stardist objects' ids, so `eval_accuracy.py --source project` scores it like the paper's wbfm projects. Tracking sees every stardist detection.

- Project: `/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline/leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist/project_config.yaml`
- Images: the A2 re-export's preprocessed red and green volumes (in the leifer_reexport project's `dat/`). They match the NWB voxel for voxel (checked on volumes 0, 500 and 1000), 1536 volumes of 32 x 632 x 600, voxel size z x y 1.5 x 0.3226 x 0.3226 um.
- Segmentation: wbfm 3d stardist, model `students_and_lukas_3d_zarr`, run locally (r261103-1181386). Median 129 objects per volume (max 145). Volumes 78, 405, 945 and 1516 have no objects; these are the corrupted volumes NeRVE also blanked.
- GT: `3-tracking/manual_annotation/gt_tracks_from_clicks_1.5um.h5`, the project's `final_3d_tracks_df`. 79 neurons, 103,052 of 119,019 manual clicks assigned (86.6%).
- Stricter variant: `gt_tracks_from_clicks_1.0um.h5`, 89,343 clicks assigned (75.1%). Each file has a `*_provenance.json` next to it.

## Method (for the paper)
GT positions are the Leifer lab's manual annotations (`manualSummary.manualPoints`), stored as `NeuronCentroids` in `Leifer_NeRVE_Worm1_reexport_A2.nwb`.

In each volume, each annotated position was matched to the nearest stardist object centroid, with distances in um (wbfm `calc_nearest_neighbor_matches`). A match was kept only if all of the following held:
1. The centroid lay within 1.5 um of the click, one z-plane spacing; clicks are quantized to z-planes.
2. The click voxel lay inside that object's mask.
3. No other annotated neuron's click was nearest the same object. An object nearest two or more annotated neurons was left unassigned for all of them, as a probable merge of neighbouring neurons, rather than given to one.

Unmatched annotations were left missing. No annotation was moved, interpolated or added. The remaining track columns (`raw_segmentation_id`, position, volume, brightness) were filled from the segmentation metadata with wbfm `add_metadata_to_df_raw_ind`.

Exclusions at 1.5 um, of 119,019 clicks:
| Reason | Clicks |
|---|---|
| No object in the volume (the 4 empty volumes) | 208 |
| No centroid within 1.5 um | 13,487 |
| Click outside the matched mask | 1,020 |
| Object shared with another annotated neuron | 1,252 |

Assigned fraction by cap: 0.75 um 63.8%, 1.0 um 75.1%, 1.5 um 86.6%. Per-neuron coverage at 1.5 um: minimum 0.43, 10th percentile 0.62, median 0.93.

## Provenance
- Script: `barlow_track/scripts/project_reanalysis/gt_tracks_from_clicks.py`. It was uncommitted when the files were written on 2026-11-03. It was committed unchanged in 3e2fdc7, which also added `--rule mask`; the default centroid rule is the code that wrote these files.
- Runs:
  - r261103-3393482: both GT files.
  - r261103-1354985: dry runs at 0.75, 1.0 and 1.5 um.
  - r261103-437335: cross-check.
  - r261103-2200226: smoke test.
- Segmentation fix: the project's preprocessing subfolder config pointed at the source project, so the bounding boxes resolved there. It now points at the project's own `dat/preprocessing_config.yaml`, with absolute paths to the preprocessed zarrs. Backups are `*.bak_20261103`.

## Checks
- Cross-check: 97.9% of clicks fall inside some stardist mask. For 95.3% of them, that mask is also the stardist object overlapping most of the neuron's NeRVE mask. 2.9% of clicks share a mask with another GT click.
- Click-to-centroid offsets show no global bias (median signed offset under 0.06 um on each axis).
- Low-coverage neurons fall into two groups:
  - neurons in large objects, around the 90th percentile of size, the likely merges (neuron_032, neuron_044);
  - neurons consistently about one plane off in z (neuron_076, neuron_015).
- At sampled volumes, the stored `raw_segmentation_id` equals the mask label under the stored centroid in all 234 entries checked.
- Smoke test (r261103-2200226): `eval_accuracy --mode trained --source project` with baseline_leifer/trial_0 on 50 frames, 5 seeds, accuracy 0.860. Pipeline check only.

## Caveats
- The GT is a subset of the annotations, chosen by how well stardist segments each neuron. Scores here are not comparable to the NeRVE-object or GT-only Leifer scores, and they favour cleanly segmented neurons.
- Running wbfm tracking on this project rewrites `final_3d_tracks_df` in `3-tracking/tracking_config.yaml` (backup `.bak_20261103`). The GT h5 itself is not touched.
- Traces: the 1.5 um GT h5 was copied to `3-tracking/postprocessing/combined_3d_tracks.h5`, so that snakemake plans only `extract_full_traces` and `traces` (checked by dry runs first). r261104-2551987 wrote `4-traces/` from the GT: 79 neurons, frames without GT at zero intensity (15.1%), 62 neurons in the paper traces. A rerun of tracking would overwrite that copy.
- A second GT on the same segmentation, by mask membership (95.4% of clicks): [[DATA - Leifer StarDist project with ground truth assigned by mask membership]]. Why each click was excluded: [[EVD - StarDist misses under one percent of the Leifer ground truth clicks]].
- The 1.5 um z spacing is unverified (see the re-export note).
