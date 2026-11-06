---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1aa76-ed87-7b3d-8f10-8a27e8ccae09
status: open
priority: normal
dateCreated: 2026-11-05T07:48:03+01:00
dateModified: 2026-11-05T07:48:03+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
On the Leifer worm1 recording, scored with the same GT variants and matching as our networks (gt_valid):

- **fDNC:** the published model, with the axis convention chosen GT-free and a 5-template vote, tracks 0.821 of assigned clicks on StarDist masks (strict GT; 0.803 of all kept clicks). The base network tracks 0.940 / 0.920.
- **NeRVE:** its own tracks, scored on its own masks with detections only, track 0.886 of assigned clicks (0.798 of all kept). The base network on the same masks tracks 0.893 / 0.805.
- **fDNC on NeRVE masks:** 0.806 / 0.726.

So BarlowTrack matches NeRVE on NeRVE's segmentation and beats fDNC by 12 points on StarDist. Neither published method is worse than its paper reported on assigned clicks: the fDNC paper gives fDNC 79.1% and NeRVE 82.9% on this recording. Counting the clicks NeRVE never segmented, NeRVE falls to 0.80.

## Method
**fDNC**
- `fdnc_track_and_score.py` (579dea7, `--save_tracks`), r261105-3083686.
- Every detection of the StarDist project and of the NeRVE-masks-only project (`_nervegt`). Positions in µm (0.3226 xy, 1.5 z), one GT-free rescale to fDNC's reference worm size.
- Four axis conventions. Templates at volumes 10, 307, 614, 921 and 1228, plus fDNC's multi-template vote (wbfm `calc_bipartite_from_candidates`, templates mapped to template 10 by fDNC's own match, so no GT).
- The headline is the convention with the highest mean fDNC match probability (GT-free): swap_mirror on both (0.802 on StarDist, 0.876 on NeRVE masks). On Samuel and Zimmer this probability ranked conventions as the GT does.

**NeRVE**
- `nerve_tracks.py` (82c84fb, summary from f8f20c1), r261105-3087916.
- `pointStatsNew.mat` trackIdx: a detection's track holds the re-export mask with label pointIdx. Checks: centroids within 0.36 px of rawPoints, and straightPoints map back to their own mask in 99.75% of cases.
- NeRVE's extra points (rows past its detections: consensus fill-in and replacement of detections more than 3 s.d. off, Nguyen 2017) were placed through NeRVE's own CLstraight lookups and given the mask under them. This is scored separately (below).

**Scoring**
- `score_gt_variants.py` against sd_strict / sd_maha_sqrt5 (StarDist) and nerve_only / nerve_pseudo_nwblabels (NeRVE), with `--split_label_at 1000`.
- Output in `~/.cache/barlow_scratch/fdnc/leifer/` (scores/scores.jsonl, score_all.sh) and `~/.cache/barlow_scratch/train/leifer_reseg/nerve_tracker/`.

StarDist strict GT (gt_valid; nandist identical to three digits):

| Tracker | Assigned | All kept | Tracker misses | Mismatches |
|---|---|---|---|---|
| base network (labelprop) | 0.940 | 0.920 | | |
| fDNC swap_mirror, 5-template vote | 0.821 | 0.803 | 4,728 | 15,499 |
| fDNC swap_mirror, template 10 | 0.762 | 0.745 | 5,432 | 21,468 |
| fDNC swap_mirror, best single template (307, chosen with GT) | 0.856 | 0.837 | | |
| fDNC, other conventions (vote) | 0.43–0.79 | | | |

The Mahalanobis √5 GT gives the same fDNC numbers to within 0.002.

NeRVE masks only (nerve_only; 106,552 assigned of 118,215 kept clicks):

| Tracker | Assigned | All kept | Tracker misses | Mismatches |
|---|---|---|---|---|
| base network | 0.8925 | 0.8045 | 1,547 | 9,905 |
| wk2 trial 57 | 0.8909 | 0.8030 | 1,294 | 10,333 |
| NeRVE, detections only | 0.8857 | 0.7983 | 2,622 | 9,559 |
| NeRVE, detections + extra points | 0.9059 | 0.8165 | 179 | 9,848 |
| fDNC swap_mirror, vote | 0.8058 | 0.7263 | 1,106 | 19,582 |
| fDNC swap, vote | 0.816 | 0.736 | | |

**NeRVE's extra points**
- They are 21,799 of 180,577 tracked points (12%). Of these:
  - 17,830 land on a NeRVE detection that another NeRVE track already holds.
  - 75 land on an untracked detection.
  - 3,195–3,759 land in background.
  - 564 land in pseudo masks.
- They recover only 4.2% of the 11,663 pseudo-mask clicks (clicks NeRVE did not segment).
- Their +2-point gain on assigned clicks is 2,585 clicks gained and 431 lost. 99.6% of the gained clicks (2,574) sit on a mask that another NeRVE track holds by its own detection: two NeRVE identities on one object.

## Caveats
- **Metric differs from the papers.** The paper numbers are per-volume matches to a template: fDNC paper, Yu et al. 2021; Nguyen 2017 counts only neurons "consistently found by both algorithm and human". Ours requires one track identity per GT neuron across the whole video.
- **NeRVE's extra-point score is lenient.** One-to-one assignment, which label propagation obeys, would not allow two identities on one object, so the detection-only number is the comparable one.
- **Convention choice for fDNC.** The axis convention is picked by fDNC's own confidence. The best single template (0.856) was chosen with GT and is an upper bound, not a result.
- **Nine NeRVE CLstraight files failed to decompress** (volume 12 is known to be corrupt; the other 8 may be read errors on the mount). 135 extra points went unmapped. 936 detections in volumes the re-export has no NeRVE masks for are segmentation misses for every tracker.
- **One seed for our networks;** fDNC and NeRVE are deterministic.
- **Not done:** fDNC on the old NWB GT objects, i.e. the original benchmark, where the tracker gets the pseudo masks.
