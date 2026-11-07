---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b39b-72e5-7335-be49-6ae2b4a95522
status: done
priority: normal
dateCreated: 2026-11-07T02:24:31+01:00
dateModified: 2026-11-07T02:24:31+01:00
confidence: medium
completedDate: 2026-11-07
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Tell apart the two readings left open by [[EVD - On Leifer StarDist the fragment errors follow an unannotated touching neighbour mask]]: a real identity swap by the tracker, or a click whose mask assignment flips between two touching masks while the tracker stays on one object.

## Method

diag_switch_jumps.py (barlow_track@9218f4a) on the cached base-tracker labels of diag_merge.py (seed 0). For each GT neuron, its main label holds most of its objects. A switch is a pair of adjacent volumes where the neuron's GT object enters or leaves the main label while the main label has an object at both. At each switch it compares the step of the GT object, of the main label's object, and (Leifer) of the manual click from Leifer_NeRVE_Worm1_reexport_A2.nwb final_tracks, with the neuron's typical step. Positions are in a local frame (minus the mean of the 8 GT neurons nearest on median, clicked in both volumes), because whole-worm motion on Leifer is about 4.3 um per volume, as large as the neighbour distance. A jump is a step above 2× typical. It also measures click-to-centroid distances in the error volumes. Cases: Leifer StarDist t57 (strict click GT, voxels 1.5 × 0.3226 × 0.3226 um) and Zimmer 1210 t36 (1.5 × 0.325 × 0.325 um).

## Results

| | Leifer StarDist | Zimmer 1210 |
|---|---|---|
| switches (neurons) | 6663 (78) | 1144 (85) |
| typical local step | 0.86 um | 0.57 um |
| only the GT object jumps | 68% | 23% |
| only the track jumps | 10% | 30% |
| both jump | 15% | 46% |

Leifer clicks: at 51% of switches the GT object jumps while the click does not (click step within 2× its typical 1.0 um); at 32% the click jumps too (median 1.6 um). In the 6019 error volumes (GT object off the main label), the click is 0.75 / 1.41 / 1.89 um (q25 / median / q75) from its own mask's centroid, against 0.32 / 0.52 / 0.85 um in correctly tracked volumes. It is 1.97 / 2.8 / 3.74 um from the tracked object's centroid, which lies 3.4 / 3.9 / 4.5 um from the GT object. In 21% of error volumes the click is nearer the tracked object than the mask it was assigned to.

## Conclusion

On the paper's Leifer (StarDist masks, strict click GT), most of the tracker's remaining error, about 5 pt, is not the tracker changing objects. The tracker keeps a smooth track on one mask, and the GT object flickers between that mask and a touching neighbour (6663 switches, 68% with only the GT object moving). The errors fall where the click sits off-centre in its mask, about a third of the way toward the tracked neighbour: 1.4 um from its mask centre against 0.5 um in correct volumes. These are clicks near the boundary of two touching StarDist masks. On Zimmer the same test finds mostly real tracker swaps, with the track moving at 76% of switches, so the test does not simply blame the GT.

## Caveats

Base tracker, one network, one seed; c_all_gh leaves these errors in place (distractor fragments 4.6 to 5.0 pt in batch 2, 3 seeds), so the picture should carry over. The 2× threshold and the local frame (8 GT neurons) are choices; Leifer clicks are themselves noisy (typical step 1.0 um). Centroid distances ignore mask shape, and a z voxel is 1.5 um. The test cannot say which of the two touching masks is the right one, only that the click is ambiguous between them; a manual look at a few neurons (e.g. 21, 22, 55, 68, 72) would settle that. Mask-level alternatives to the strict GT (the sd_edge* variants, the clickmasks project) were not scored here.
