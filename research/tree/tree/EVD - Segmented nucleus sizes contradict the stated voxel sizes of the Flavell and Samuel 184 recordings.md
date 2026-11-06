---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1ab7b-daea-7a8f-86c2-4235a813599a
status: in-progress
priority: normal
dateCreated: 2026-11-05T12:33:03+01:00
dateModified: 2026-11-05T12:33:03+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Method

Median per-object standard deviation of mask voxel coordinates (z, and xy averaged over x and y), over all objects of 6 evenly spaced volumes of each recording's wbfm project segmentation (the training projects of the per-recording rounds). Nuclei are about the same physical size in every lab, so voxel_um(rec) ~= voxel_um(Zimmer) x sigma_vox(Zimmer) / sigma_vox(rec), Zimmer StarDist (0.325 um xy) as the calibration. Script barlow_track/scripts/scratch/voxel_size_from_masks.py (uncommitted as of 2026-11-05), run r261105-3468094. Stated sizes: NWB grid_spacing.

## Result

| Recording (segmenter) | sigma_xy vox | implied xy um | stated xy um |
|---|---|---|---|
| Zimmer 1128 / 1210 / 1123 (StarDist) | 3.02 / 2.96 / 2.96 | 0.325 (calibration) / 0.33 / 0.33 | 0.325 |
| Leifer (StarDist) | 3.04 | 0.322 | 0.3226 |
| Leifer (NeRVE masks) | 2.11 | 0.47 | 0.3226 |
| Samuel 137 / 153 (GT-seeded) | 1.80 / 1.71 | 0.54 / 0.57 | 0.45 |
| Samuel 184 (GT-seeded) | 1.64 | 0.60 | 0.30 |
| Flavell, 5 recordings (Flavell lab) | 1.16-1.24 | 0.79-0.84 | 0.30 (isotropic) |

Flavell objects are also wider in z than in xy in voxels (sigma_z 1.6-1.8 vox).

## Conclusion

With one segmenter the estimate is exact across labs (Leifer StarDist 0.322 vs 0.3226 um); a different segmenter shifts it up to ~1.45x (NeRVE masks). Samuel 184 matches Samuel 137 and 153 within 10%, so its stated 0.3 um xy is very likely wrong (0.45 like the others). Flavell's stated 0.3 um isotropic would make its nuclei ~2.7x smaller than Zimmer's, beyond the segmenter spread; its pixels are probably ~0.6-0.8 um, so the 64-voxel xy crop would span ~40-50 um there (Zimmer: 21 um) and visnet_z_scale 1.0 is likely wrong too.

## Caveats

z is not calibrated (axial blur differs by microscope: Leifer StarDist gives 1.9 vs 1.5 um stated); only xy is usable. Samuel and Flavell masks come from other segmenters, so their absolute values carry the ~1.45x segmenter uncertainty; the within-lab comparison (Samuel 184 vs 137/153) does not. Flavell's true voxel size is not checked against its paper or acquisition metadata yet.
