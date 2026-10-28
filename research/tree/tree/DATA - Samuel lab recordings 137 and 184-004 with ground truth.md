---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a1845a-8b63-7647-a97c-50425215e866
status: done
priority: normal
dateCreated: 2026-10-28T22:11:28+01:00
dateModified: 2026-10-28T22:11:28+01:00
completedDate: 2026-10-28
projects:
  - "[[barlowtrack]]"
---

## Description

Two further Samuel lab recordings with manual ground truth, beside 153 ([[DATA - Samuel lab recording with ground truth]]). 137: 1326 frames, 92 GT neurons, (320, 192, 20) volumes. 184-004: 3002 frames, 85 GT neurons, (256, 160, 16) volumes. In every Samuel NWB (plain and thresholded_segmentation/, 137, 153 and 184-004) the segmentation holds only GT neurons: labels per frame never exceed the GT count (h5py check, 2026-10-28). Any score on these data is therefore with oracle detections; there is no detector output to test against.

## Provenance

/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data/137.nwb and thresholded_segmentation/184-004.nwb. samuel_data/184-004.nwb (2 MB) has no volumes, so the thresholded file is the one to use. thresholded_segmentation/185-005.nwb does not open with h5py ("bad object header version number"); 184_15GT-001.h5, 184_5GT-006.h5 and 185-005.h5 exist only as h5 and were not used.
