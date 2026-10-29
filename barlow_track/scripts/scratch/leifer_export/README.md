# Leifer NeRVE worm1: re-export from raw into a new NWB (with segmentation)

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
