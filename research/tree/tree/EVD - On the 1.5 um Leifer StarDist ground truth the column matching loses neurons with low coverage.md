---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1a858-e3c8-7805-8ec1-0803b8c8910a
status: open
priority: normal
dateCreated: 2026-11-04T21:56:00+01:00
dateModified: 2026-11-04T21:56:00+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
`base` scores 0.9476 on the 1.5 um StarDist GT with the default `nandist` column matching but 0.9667 when the same tracks are matched with `gt_valid` (agreement only on frames that have GT). The difference is four neurons the centroid rule covers in only 42-51% of frames (neuron_032, 044, 076, 015; median neuron 92%). Their objects are tracked, but `nandist` counts the track's objects in the other half of the frames as disagreements and gives the track to no neuron, so those neurons score 74-100% errors. On the mask GT (coverage median 97%) the two matchings give the same score (0.9385), so the bias comes from GT coverage, not the tracker.

## Method
`base` (local leifer_reseg sweep, seed 0) re-tracked from its cached embedding with `eval_accuracy.py --save_tracks` (4ae0c07; r261104-4092161 centroid GT, r261104-4092164 mask GT), reproducing the recorded 0.9476 and 0.9385 exactly. `tracking_error_patterns.py` (05d2104) rematches the saved tracks (all 153 predicted tracks) with `match_columns_to_gt` both ways and scores with `calculate_accuracy`.

| GT | nandist | gt_valid |
|---|---|---|
| 1.5 um centroid | 0.9476 (2,971 misses, 2,430 mismatches) | 0.9667 (466, 2,967) |
| mask | 0.9385 (1,183, 5,800) | 0.9385 |

Under `nandist` on the centroid GT, 2,852 of 2,971 misses are GT objects that sit in a predicted track matched to no GT neuron. Errors in runs of 50 or more frames are 39% of all errors (longest 736), almost all from those four neurons.

## Caveats
- One network, one labelprop seed (seeds 1-4 running in r261104-3067468).
- `gt_valid` does not penalise a track that holds an object where the GT is NaN; that is correct when the GT is incomplete (it is: 13% of clicks dropped) but would also hide a track that wanders onto other objects in those frames.
- Every recorded benchmark number so far uses `nandist`.
