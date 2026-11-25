---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a21315-491b-729e-bd23-dd0b0ba4cda2
status: done
priority: normal
dateCreated: 2026-11-25T15:21:31+01:00
dateModified: 2026-11-25T15:21:31+01:00
confidence: high
completedDate: 2026-11-25
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Conclusion
Across five revised trial 59 training replicates, the median per-neuron correlation with the reference calcium trace ranged from 0.9954 to 0.9973 on Zimmer 1128, 0.9803 to 0.9828 on Leifer NeRVE worm 1, and 0.9991 to 0.9995 on Samuel 153. The GFP control, Flavell 2022-03-31-01, ranged from 0.9295 to 0.9559. In the first replicate, 3/131, 18/79, 0/96 and 43/163 evaluable neurons, respectively, had processed-trace r below 0.9.

## Method
Used the saved gt and pred_matched tracks from all 20 trial 59 v2 outputs. Extracted integrated red, green and mask volume once for every frame-local object (r261125-5.2) and joined both tracks by frame and mask. The reference identities were Zimmer's curated tracks, retained Leifer manual clicks inside unambiguous StarDist masks, Samuel manual centroids with watershed masks, and the published Flavell tracker. Every assigned mask was found for both reference and prediction in every replicate.

We subtracted the project offsets (75 per voxel in both channels on Zimmer, Leifer and Flavell, 0 on Samuel), fitted a separate exponential per channel and track, divided green by red and normalized to the track's median ratio. After five-standard-deviation outlier rejection and centered three-volume smoothing, missing and rejected observations were masked again. Pearson r required ten shared finite values and varying traces. Mask disagreements stayed in the sample. Coverage was shared observed masks over observed reference masks. The tables also report the raw-ratio check.

Runs r261125-3.3 and r261125-4.4. The script is barlow_track/scripts/paper_scripts/revision/trace_fidelity.py in the working tree (SHA-256 29e16df31ba188a711b5478b84aa938f3e1e805fbd130b1cd39c234ad1ec66ba). The figure, tables, caption and command are in docs/paper/trace_fidelity/. This analysis extends [[EVD - On StarDist masks the tracker changes Leifer traces less than the choice of segmentation does]] to the revised checkpoints and shared observations.

## Caveats
One recording per lab. Replicates measure variation in the trained tracker on the same recording. Samuel uses reference-derived detections. One Samuel identity has no observed reference mask and one has a zero median corrected ratio across all five runs. The latter is evaluable for the raw ratio. Leifer reference coverage is conditional on an assigned click. Flavell GFP is a fluorescence control with algorithmic reference identities.
