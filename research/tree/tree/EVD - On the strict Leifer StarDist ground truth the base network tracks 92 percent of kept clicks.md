---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1a960-1186-72b2-af0a-74ab675be4bf
status: in-progress
priority: high
dateCreated: 2026-11-05T02:43:27+01:00
dateModified: 2026-11-05T09:14:46+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
This is the headline Leifer GT the user chose. A click outside every StarDist mask counts as a missed segmentation, i.e. an error; coincident clicks (another neuron within 1.0 µm) and merges are excluded. Under it, the leifer_reseg base network scores 0.940 on the 112,996 assigned clicks and 0.920 on all 115,507 kept clicks (labelprop seed 0, gt_valid matching). 2,511 kept clicks (2.2%) are segmentation misses.

The best benchmarked network, wk2 trial 50, scores 0.922–0.924 on all kept clicks over labelprop seeds 0–4. Five networks have five labelprop seeds each. Their all-kept seed means are wk2 t50 0.923, t74 0.922, lambd0_nf400 0.921, t63 0.920 and t57 0.917, with seed s.d. 0.08–0.17 points. The top 12 networks lie within 0.4 points of each other, about two to three seed s.d., so the top of the ranking is not resolved. Trial 57 drops from second on the 1.5 µm centroid GT to 13th (0.917).

Two alternatives move the number:
- Attaching off-mask clicks to the nearest mask through wider shells raises all-kept to 0.934 at 3 µm. The attached clicks are hard: about 53% are tracked correctly.
- Adding a mask at every missed click (built from the GT) gives 0.935.

## Method
GT variants:
- Built with `gt_tracks_from_clicks.py` (891f275, r261104-2025465). Each variant writes a per-click fate file (`_status.npz`) in `Leifer_NeRVE_Worm1_reexport_A2_stardist/3-tracking/manual_annotation/variants/`.
- The variants are:
  - sd_strict: the click must be inside a mask.
  - sd_edge{0.5,1.0,1.5,2.0,3.0}um: the nearest mask edge is within that distance.
  - sd_maha_sqrt5: Mahalanobis distance to the nearest object is at most √5, the surface of the solid ellipsoid with the object's voxel covariance.
- All variants exclude coincident clicks at 1.0 µm and merges (a mask holding two neurons' clicks).

Scoring:
- Saved predicted tracks (`eval_accuracy.py --save_tracks`, 4ae0c07) are scored offline against every variant with `score_gt_variants.py` (891f275, ab90a41). Label propagation never sees the GT.
- Runs:
  - base seed 0 on CPU, seeds 1–2 (r261104-3067468)
  - GPU labelprop control (r261104-237552), which matches CPU to 4 digits on every variant
  - every benchmarked network re-tracked on the GPU from its cached benchmark embedding, seed 0, plus wk2 t50 and t57 seeds 1–4 (r261105-2927825)
  - wk2 t74, t63 and lambd0_nf400 seeds 1–4 (r261105-3124433)
- Scores: `~/.cache/barlow_scratch/train/leifer_reseg/gt_variants/scores/scores.jsonl` and `.../gpu_retrack/scores/scores.jsonl`.

Click-masks control (r261105-2217669):
- Project `_stardist_clickmasks`, made by `add_click_masks.py` (891f275 and its fix c15cb45): one ellipsoid of median StarDist extent around each of the 2,511 missed clicks.
- base was embedded and tracked there, then scored on that project's strict GT, which assigns all 115,507 kept clicks: 0.9350.

base, gt_valid (nandist agrees to 4 digits at this coverage):

| Variant | Kept | Segmentation misses | Accuracy, assigned | Accuracy, all kept |
|---|---|---|---|---|
| strict (headline) | 115,507 | 2,511 | 0.9404 | 0.9199 |
| Mahalanobis √5 | | | 0.9400 | 0.9219 |
| edge 0.5 µm | | | 0.9393 | 0.9254 |
| edge 1.0 µm | | | 0.9385 | 0.9305 |
| edge 1.5 µm | | | 0.9376 | 0.9324 |
| edge 2.0 µm | | | 0.9371 | 0.9330 |
| edge 3.0 µm | 114,392 | 392 | 0.9368 | 0.9336 |
| click masks added (own project) | 115,507 | 0 | 0.9350 | 0.9350 |

Benchmark on the strict GT (seed 0 unless a range is given). The 1.5 µm column is the earlier benchmark (centroid GT, nandist, CPU).

| Network | 1.5 µm centroid GT | Strict, assigned | Strict, all kept |
|---|---|---|---|
| wk2 t74 (seeds 0–4) | 0.9408 | 0.9407–0.9439 | 0.9203–0.9234 |
| wk2 t50 (seeds 0–4) | 0.9485 | 0.9425–0.9444 | 0.9220–0.9239 |
| lambd0_nf400 (local sweep; seeds 0–4) | 0.9416 | 0.9400–0.9433 | 0.9196–0.9228 |
| wk2 t63 (seeds 0–4) | 0.9416 | 0.9380–0.9422 | 0.9176–0.9217 |
| nf400 (local sweep) | | 0.9411 | 0.9207 |
| wk2 t65 | 0.9402 | 0.9409 | 0.9204 |
| lambd1 (local sweep) | 0.9394 | 0.9405 | 0.9200 |
| base (local sweep; seeds 0–2) | 0.9476 | 0.9381–0.9411 | 0.9180–0.9210 |
| lambd1_nf400 (local sweep) | 0.9414 | 0.9403 | 0.9199 |
| wk2 t76 | 0.9402 | 0.9403 | 0.9198 |
| ref_local_nerve (trained on NeRVE masks) | | 0.9402 | 0.9198 |
| snap1 t0 | 0.9347 | 0.9380 | 0.9176 |
| wk2 t57 (seeds 0–4) | 0.9448 | 0.9366–0.9390 | 0.9162–0.9186 |
| lambd0 (local sweep) | 0.9366 | 0.9364 | 0.9160 |
| wk1 t1 | 0.9276 | 0.9320 | 0.9117 |
| wk2 t14 | 0.9266 | 0.9296 | 0.9094 |
| samuel137 t63 | 0.9388 | 0.9284 | 0.9082 |
| wk2 t37 | 0.9330 | 0.9280 | 0.9078 |
| zimmer1128 t0 | 0.9256 | 0.9229 | 0.9029 |
| wk2 t9 | 0.9247 | 0.9164 | 0.8964 |
| wk2 t24 | 0.8916 | 0.8752 | 0.8562 |

## Caveats
- The shells are biased. The GT then names the nearest object, which the tracker is drawn to as well, so shell accuracies overstate how often the tracker finds the true neuron.
- The click-masks project adds objects exactly where the GT is, which no segmentation could do without the GT.
- The misclick check is unresolved. 8,456 clicks deviate more than 2.8 µm from their neighbours' motion, mostly real non-rigid motion. All are kept, including neuron_038's ~60 clicks on spots no segmentation has, and those count as errors.
- One training seed per network. Five labelprop seeds for wk2 t50, t74, t63, t57 and lambd0_nf400 (s.d. 0.08–0.17 points), base has three, and the others one. The ranking among the top 12 is within labelprop-seed noise.
- One recording and one StarDist model.
- The CPU 1.5 µm benchmark and the GPU strict scores come from different label-propagation runs. On base they agree to 4 digits; for the other networks this was not checked.
