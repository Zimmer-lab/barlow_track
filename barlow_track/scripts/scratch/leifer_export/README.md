# Leifer NeRVE worm1: re-export from raw into a new NWB (with segmentation)

## Built 2026-10-30 (export_leifer_nwb.py, validate_leifer_nwb.py, old_map_check.py)

Output: `/lisc/.../barlow_track_revisions/leifer_reexport/Leifer_NeRVE_Worm1_reexport_A.nwb` (14.5 GB, 25 min,
24 processes). Red only, images (1536, 632, 600, 32, 1) and NeRVE masks as
`CalciumSeriesSegmentationUntracked` (1536, 632, 600, 32), one volume per gzip chunk. GT as per-neuron
`NeuronCentroids` (snapped to the Hungarian-matched mask centroid, <= 2.6 um), `NeuronCentroidsManual`
(clicks), `NeuronSegmentationID`; per-volume frame map and status in `processing/NeRVEExport/VolumeInfo`.

Also `Leifer_NeRVE_Worm1_reexport_B.nwb` (15.5 GB, z rule B: each volume's own frames, nz 34, z = 0 at the
top of the sweep, zero padded; same masks and GT, z shifted by the map). Validation logs:
`~/.cache/barlow_scratch/leifer_export/validate_{A,B}.log`. Scoring (not run here, hours):
`eval_accuracy.py --lab leifer --source nwb --nwb <new>.nwb --nwb_objects all --mode trained --weights
<trial>/resnet50.pth --cluster labelprop --num_seeds 25 --track_device cpu`. With `--nwb_objects gt` only the
matched 90% of GT is cropped (it needs a finite raw_segmentation_id), at the snapped positions, so it is
not the old file's GT-only score; for that, keep scoring the old NWB.

Findings that correct the exploration below:
- Old z rule (measured plane by plane, `old_map_check.py`): keyed by the volume's frame count. Up sweeps
  z = k is start + top - k with top = 29/30/30/31/31 for 30/31/32/33/34 frames; down sweeps start + skip + k
  to the last frame, skip = 1/2/2 for 32/33/34 frames; zero planes after. This explains the "39 unexplained"
  volumes. The old file stores the last volume (1535, a down sweep) z-flipped; the export does not.
- Old file has one corrupted 632-px line in one plane of 39 volumes; the export takes the .dat.
- Old file drops all GT of volumes 78, 405, 945, 1516 (208 clicks, plausible on the image); these are the
  volumes where NeRVE blanked its detections (>= 200 objects). The export keeps them.
- 9 pointStats files are corrupt in the release (tar copies are byte-identical): stacks 13, 189, 536, 625,
  636, 679, 763, 1435, 1470. With the 4 zero-detection and 13 ERROR volumes, 26 volumes have empty masks.
- Mask alignment: row = transformy, col = transformx, frame = start + transformz, all as stored (+-1 shifts
  in each axis lower the in-mask red symmetrically).
- Matching: costs above the cut-off are set constant, else the assignment trades close pairs for far ones.

Status 2026-10-29: exploration only, nothing built. Written by an exploration agent, read-only on
/lisc; checked by me only as far as summarised here. The scripts in `explore/` are the agent's checks;
they expect `/lisc/.../barlow_track_paper/leifer_data/` and a scratch output folder. The NeRVE MATLAB
files the agent streamed from the front of `Nguyen2017NeuronTracking.tar` are in
`~/.cache/barlow_scratch/leifer_explore/tar_extract/` (also public at github.com/leiferlab/NeRVEclustering).

Raw: `leifer_data/Nguyen2017NeuronTracking/worm1/`. Existing NWB: `Leifer_NeRVE_Worm1.nwb` (Haejun Han,
Lu lab, 2025-08-04, from the IEEE DataPort release; red only, no segmentation).

## Formats (agent: verified)

- `sCMOS_Frames_U16_1200x600.dat`: 51,111 frames, `np.memmap(f, '<u2').reshape(-1, 600, 1200)`.
  Red = `frame[:, :632].T` (632, 600), bit-identical to the old NWB; green = `frame[:, 634:1200].T`.
- Frame -> volume: `manualSummary.stackList` (= `hiResData.stackIdx` shifted by 12). 1536 volumes,
  30-34 frames each, alternating up / down sweeps.
- Old NWB z rule: 32 raw frames copied as is; up sweep z=k is frame start+31-k, down sweep start+2+k
  (fits 1493 of 1532 GT volumes; 39 unexplained). Chunks (48, 40, 38, 2, 1), uncompressed: reading
  one volume reads 48 volumes' chunks, probably why Leifer eval cells take hours.
- GT: `manualSummary.manualPoints` (1536 cells of 79x3), equals the NWB CentroidSeries with columns
  0 / 1 swapped; z is a 0-based .dat frame index. Volumes 78, 405, 945, 1516 have no GT.
- Detections for all neurons: `outputFiles/PointsStats*.mat` `rawPoints`, ~117 per volume (stacks
  1-1523), raw coords (x on the 600 axis, y on the 632 axis, z = offset from the volume's first frame
  in acquisition order). 89% of GT neurons have a detection within ~2.6 um.
- Real masks: `outputFiles/CLstraight/pointStatsNNNNN.mat` `baseImg` (straightened binary mask) plus
  `transformx/y/z` lookups back to raw. Labelled in Fortran order they reproduce `pointIdx`; warped
  to raw they give ~100 voxels per neuron with mean red 1452 vs background 100, and +-1 frame / row
  shifts lower it, so the alignment is right. Stacks 1524-1536 have no segmentation (ERROR files).

## Plan

1. Frame map per volume from `stackList`. Option A: the old 32-plane rule (comparable to old scores).
   Option B (agent's pick): only that volume's frames, nz = 34, zero-padded. Image, GT and masks all
   pass through the same per-volume map.
2. Images: one sequential pass over the .dat (73.6 GB), red (optionally green after refitting the
   `alignments.S2AHiRes` control points), chunks (1, 632, 600, nz, 1), gzip.
3. Segmentation: label `baseImg`, scatter through `transform*`, map frames to z, uint16 labels =
   `pointIdx`. Volumes 1524-1536: drop, or watershed from GT centroids.
4. GT from `manualPoints`; Hungarian match to mask labels (~2.6 um cut-off) for NeuronSegmentationID.
   Optionally NeRVE's own `trackIdx` (pointStatsNew) as a free baseline tracker.
5. Writer template: `wbfm/utils/nwb/convert_harvard_data.py` (the Samuel export) plus
   `utils_nwb_export.py` helpers (`df_to_nwb_tracking`, `CustomDataChunkIterator`,
   `MultiChannelVolumeSeries`). `convert_leifer_data.py` is broken (undefined names, opens input r+);
   do not run it.
6. Validate: pixel identity vs the old NWB, `ProjectData.load_final_project_data(new,
   allow_hybrid_loading=True)`, `eval_accuracy --source nwb` with `--nwb_objects gt` (should match old
   scores under option A) and `all`. Note `_nwb_all_objects` maps GT to a mask only within 0.5 voxel
   of the mask centroid: snap GT centroids to matched masks, or use NeuronSegmentationID.

Risks: physical z spacing unknown (old file says 1.5 um; check Nguyen 2016 / 2017 methods); piezo
range drifts, so frame-index z is not a fixed depth; 11% of GT neurons undetected; NeRVE drops objects
touching the stack top / bottom. Disk: ~39 GB uncompressed red; fits the 1.5 TB free on scratch.
