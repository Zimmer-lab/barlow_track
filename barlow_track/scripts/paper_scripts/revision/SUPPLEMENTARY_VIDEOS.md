# BarlowTrack supplementary videos

Start with the standard MP4 for a recording to watch the neurons through a bend or turn. To examine an incorrect assignment, pause at its frame number and open the same frame in `debug/<recording>_vectors.mp4`. All 12 benchmark recordings have both versions in the shared output directory.

```text
/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/revision_analysis/supplementary_videos/
```

## Reading a frame

The left panel shows the reference-channel microscopy image. The right panel shows the same XY maximum-intensity projection with trial-59 BarlowTrack predictions. Color follows the predicted track across frames. The occasional numeric label identifies that track within the recording. A gap means the tracker made no assignment for that track in that volume.

| Marker | Meaning in this volume |
| --- | --- |
| Open circle | The track is on the detection assigned to its reference identity. |
| X | The reference places the claimed identity elsewhere, or identifies the marked detection as belonging to another neuron. |
| Small filled dot | The available annotations establish neither where the claimed identity belongs nor which identity occupies this detection. |

**Volume accuracy** is the number of scored reference identities assigned to the correct detection divided by all scored reference identities in that volume. A missed prediction counts in the denominator and leaves no marker to count on screen. The denominator can change from frame to frame. `N/A (0 scored)` means the volume has no scored reference identities.

**Debug arrows** start at an X and point to the reference detection for the identity claimed by that track **in the same volume**. They show the XY distance and direction of the assignment error. The footer gives the number of crosses with known targets. A cross can have no known target when the reference identifies the neuron under the cross but supplies no location for the claimed identity. Neurons at different depths can overlap in projection, so even a known target can have zero visible XY displacement.

For example, open `debug/flavell_vectors.mp4` at **15.5 seconds**, original benchmark frame **1518** (Flavell 2022-03-31-01). The header reads **43.4% (23/53)**. The overlay contains 28 crosses, and nine have known targets. The next frame reads **N/A (0 scored)**.

## Clip selection and timing

Each clip spans about 20 seconds of acquisition around a 20-volume window from `revision_analysis/t1_segments.tsv`. Selection used the mean deformation measure (`d_novel_mean`) and visibility of the motion, independently of tracking accuracy. Lead-in and follow-through show the same region before and after the movement.

All movies play at **two original volumes per second**. The frame number and elapsed acquisition time refer to the original recording. Flavell also shows the one-based NRRD frame number. Playback duration varies with acquisition rate. For example, the 10-volumes/s Samuel recordings play at 0.2× speed, so 20 seconds of acquisition takes 100 seconds to watch.

- **Samuel** training, evaluation and visualization use reference detections. BarlowTrack predicts the associations across volumes. Recording 184-004 uses a pixel scale bar because its physical XY calibration is uncertain.
- **Leifer** uses the StarDist recording. Its score covers reference clicks assigned to StarDist detections in that volume.

## Files and reproduction

`captions.txt` lists the recording, frame range and selected window for each clip. `volume_accuracy.tsv` gives the numerator, denominator and accuracy by original frame. To trace a marker, look up its recording, frame and predicted track ID in `assignments.tsv.gz`. The table gives its detection ID and ZXY coordinates, status, and any known arrow target. The `manifest.json` identifies the source images, per-recording coordinate joins, trial-59 tracks and encoded-frame checks. Marker positions come from `pred_all` assignments joined to detections. The saved `pred_matched` and `gt` tables supply the scoring information.

To reproduce both sets, run this from the `barlow_track` repository root with the revision data mounted and FFmpeg on the path.

```bash
/home/charles/anaconda3/envs/wbfm/bin/python \
  barlow_track/scripts/paper_scripts/revision/export_supplementary_videos.py \
  --revision-dir /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions
```
