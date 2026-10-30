---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a18a86-41c0-7c54-83e0-f5f67aaa3b56
status: open
priority: normal
dateCreated: 2026-10-30T02:56:56+01:00
dateModified: 2026-10-30T02:56:56+01:00
---

## Description
The Leifer lab NeRVE worm1 recording (Nguyen et al. 2017), re-exported from the raw camera stream
(`leifer_data/Nguyen2017NeuronTracking/worm1/sCMOS_Frames_U16_1200x600.dat`) into a new NWB with a
segmentation of every detected neuron, so it can be used like the Samuel and Flavell NWBs (wbfm projects,
crops around every detection, `--nwb_objects all`).

- File: `/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/leifer_reexport/Leifer_NeRVE_Worm1_reexport_A.nwb`
  (14.5 GB; benchmark label `leifer_reexport`). A variant with each volume's own frames (34 planes) is
  `..._reexport_B.nwb`.
- Images: red only, 1536 volumes of 632 x 600 x 32, the old NWB's frame-to-plane map (measured plane by
  plane; keyed by frames per volume), one volume per gzip chunk.
- Segmentation: NeRVE's own masks (`outputFiles/CLstraight/pointStats*.mat` warped back to raw),
  `CalciumSeriesSegmentationUntracked`, 117.5 objects per volume (61-152). 26 volumes are empty: 9 corrupt
  pointStats files in the release, 4 that NeRVE blanked (>= 200 objects), 13 ERROR volumes at the end.
- GT: `manualSummary.manualPoints` (79 neurons, 119,019 points). `NeuronCentroids` are snapped to the
  Hungarian-matched mask centroid (<= 2.6 um, 89.9% matched); the raw clicks are `NeuronCentroidsManual`.

## Provenance
Built by `barlow_track@0314776:barlow_track/scripts/scratch/leifer_export/export_leifer_nwb.py`, validated
by `validate_leifer_nwb.py` (logs `~/.cache/barlow_scratch/leifer_export/validate_A.log`).

## Differences from the old NWB
- 1496 of 1536 volumes are pixel-identical. 39 volumes differ in one corrupted 632-735 px line of the
  old file. The old file stores volume 1535 z-flipped; the re-export does not.
- 118,777 of the old 118,811 GT points are identical (the rest are in volume 1535). The re-export keeps
  208 clicks in volumes 78, 405, 945 and 1516 that the old file dropped.
- GT-only scores (`--nwb_objects gt`) are not comparable to the old file's, because only matched GT has a
  segmentation id and positions are snapped; keep GT-only scoring on the old NWB.

## Caveats
Physical z spacing 1.5 um is unverified (Nguyen 2016 report ~1.5-1.6 um). Channel metadata copied from the
old file looks inconsistent. 10% of GT neurons have no NeRVE detection.
