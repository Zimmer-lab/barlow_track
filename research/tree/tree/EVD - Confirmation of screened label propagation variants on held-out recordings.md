---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b113-4cff-7dde-892d-b7ad89e9cfe4
status: done
priority: high
dateCreated: 2026-11-06T14:36:34+01:00
dateModified: 2026-11-07T19:35:21+01:00
confidence: high
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-11-07
---

## Purpose

Confirm the batch 1 winners, singly and in cross-stage combinations, on networks and recordings the screen did not see, before any change goes into the library tracker.

## Method

variants.py `BATCH2` (barlow_track@d8a1335; chain variants b5e20d2; c_all_gh_s150 397c80f; no-temporal variants 0e0a566 and aab0f33): base; the six batch-1 promotions singly (svd100, seeds50, seeds100, seed_strat, temporal5, tsmooth05); neighbours svd150, temporal2, temporal10; combinations c_seeds (50 stratified seeds), c_all (svd100 + 50 stratified seeds + temporal5 + tsmooth05), c_all_s100, c_all_notemp, c_all_gh (+ gauss edges, Hungarian per run), c_all_l30 (+ 30 layers), c_all_gh_s150 (Ax round 3 best: svd 150); chain voting chain05, chain1, c_all_chain, c_all_gh_chain; and, after the held-out Flavell result, c_all_gh_notemp, c_all_gh_s150_notemp (temporal_w 0) and c_all_gh_s150_plain (temporal_w 0 and temporal_smooth 0: no temporal information at all). Cases (CONFIRM_CASES) × seeds 0, 1, 2: S137 t63, S184 t32, Z1210 t52, Leifer re-export t57 (held-out lab, original segmentation, NWB every detection), Flavell 2022-03-24-02 t28 (held-out recording), and Leifer StarDist t57 (the same network on the StarDist re-segmentation, strict click GT; the paper's Leifer; added at 3756753 on the user's request). Runs r261106-595573, r261106-598057, r261106-606174, r261106-673582, r261106-677223 (superseded schedulers; records kept, the sweep is resumable), r261107-743860 (v4, GPU lanes), r261107-807027 (v5, reran the CUDA-OOM gaps after b7a4d69), r261107-807396 (s150 on Samuel, CPU), r261107-867761 and r261107-950766 (no-temporal variants). Output ~/.cache/barlow_scratch/labelprop_opt/batch2_confirm.jsonl (432 records), analysed with analyze.py --per_case. Adopt per PLAN.md section 4 (mean ≥ +0.3 pt, paired t-test p < 0.05 over all pairs, no case mean below -0.5 pt). Then check the adopted recipe at 100 seeds on 2 cases (r261107-969986, restarted for seeds 44-99 as r261107-1126428 after /home filled up; Leifer StarDist and Z1210, seeds 3-99 plus batch 2's 0-2, output batch2_seeds100.jsonl).

## Results

Complete, 2026-11-07 07:46: 6 cases × 3 seeds for every variant. Base: Leifer NWB 0.8789, StarDist 0.9364, S137 0.9346, F0324 0.9731, S184 0.9306, Z1210 0.9382 (seed range up to 1.37 pt on Z1210). Paired gain over base in pt, per-case means over seeds:

| variant | mean | wins | worst case | Leifer NWB | StarDist | S137 | F0324 | S184 | Z1210 |
|---|---|---|---|---|---|---|---|---|---|
| c_all_gh_s150_notemp | +1.04 | 18/18 | +0.27 | +0.33 | +0.31 | +1.43 | +0.27 | +2.75 | +1.18 |
| c_all_gh_notemp | +0.99 | 17/18 | +0.20 | +0.41 | +0.31 | +1.23 | +0.20 | +2.60 | +1.17 |
| c_all_gh_s150 | +0.95 | 15/18 | -0.35 | +0.56 | +0.35 | +1.28 | -0.42 | +2.71 | +1.18 |
| c_all_notemp | +0.89 | 17/18 | +0.19 | +0.38 | +0.28 | +1.03 | +0.19 | +2.16 | +1.30 |
| c_all_gh | +0.89 | 15/18 | -0.37 | +0.49 | +0.39 | +1.02 | -0.37 | +2.62 | +1.18 |
| **c_all_gh_s150_plain** | **+0.88** | 16/18 | +0.11 | +0.11 | +0.23 | +1.24 | +0.24 | +2.33 | +1.13 |
| c_all | +0.74 | 15/18 | -0.27 | +0.40 | +0.41 | +0.77 | -0.27 | +1.92 | +1.20 |
| chain05 | +0.44 | 18/18 | +0.02 | | | | | | |
| tsmooth05 | +0.30 | 16/18 | -0.10 | | | | | | |
| temporal5 | +0.19 | 14/18 | -0.42 | | | | | | |
| seed_strat | +0.07 | 8/18 | -0.13 | | | | | | |

Temporal edges (temporal_w: mutual nearest neighbours in embedding space between adjacent volumes, weight 5) are what cost the held-out Flavell: alone -0.42 pt there (p 0.14 overall, not promoted), and every combination with them is below base on F0324, by label swaps rather than fragments (err_swapped base 1.12, temporal5 1.38, c_all_gh 1.49, c_all_notemp 1.03 pt). Without them nothing is lost elsewhere (Z1210 c_all_notemp vs c_all +0.10). Temporal smoothing (post step) on top of the no-temporal-edge recipe: +0.16 pt (c_all_gh_s150_notemp vs _plain, 17/18 pairs, p < 0.001; per case F0324 +0.02, Leifer NWB +0.22, StarDist +0.08, S137 +0.19, S184 +0.42, Z1210 +0.05). On the Leifer cases nearly all of the gain is fewer split fragments (StarDist err_frag_split 0.98 to 0.19-0.31 pt); the distractor fragments (about 5 pt on StarDist) stay.

100-seed check of c_all_gh_s150_plain against base (paired by RNG seed): Leifer StarDist (n 99; seed 47 hit CUDA OOM) base 0.9368 (sd 0.09 pt, 0.9342-0.9401), recipe 0.9398 (sd 0.11, 0.9353-0.9430), gain +0.31 pt (median +0.32, 5th-95th percentile -0.00 to +0.54), 93/99 seeds, Wilcoxon p 1e-17. Zimmer 1210 (n 100) base 0.9394 (sd 1.30 pt, 0.8780-0.9504), recipe 0.9480 (sd 0.17, 0.9428-0.9507), gain +0.86 pt (median +0.47, 5th-95th -0.13 to +2.69), 88/100, p 7e-16. On Zimmer the recipe mostly removes the bad seeds of base: its spread over seeds falls from 1.30 to 0.17 pt and its worst seed (0.9428) is above base's mean.

## Conclusion

Adopted: c_all_gh_s150_plain (svd 150, 50 stratified seeds, Gaussian edge weights, Hungarian per-run uniqueness; no temporal edges and no temporal smoothing), +0.88 pt over the current tracker over 6 held-out cases × 3 seeds (16/18 wins, worst case +0.11, p 0.001); +1.1 to +2.3 pt on Samuel and Zimmer, +0.1 to +0.2 on Leifer and +0.24 on Flavell. Temporal edges cost a held-out Flavell recording 0.4 pt in swaps and buy nothing once the seed changes are in. Temporal smoothing adds a consistent but small +0.16 pt, under the adoption bar, and the user prefers no post-processing step that does not clearly help. Over 100 RNG seeds the gain holds (+0.31 pt on the paper's Leifer StarDist, 93/99 seeds; +0.86 pt on Zimmer 1210, 88/100), and the recipe is far less seed-dependent (Zimmer seed sd 1.30 to 0.17 pt).

## Caveats

The differences between the top recipes (0.1-0.2 pt) are within Z1210's seed spread at 3 seeds; only the adopted recipe was run at 100 seeds. Six cases from five networks; the Bayesian search tuned temporal_w on Flavell F0330, which did not show the harm, so the dev set alone would have kept temporal edges. The StarDist case scores assigned strict clicks only (multiply by 0.97826 for the all-kept headline), and most of its remaining error is clicks on touching-mask boundaries ([[CLM - Most of the label propagation error left on the StarDist Leifer recording is clicks on the boundary of touching masks]]). Whether temporal edges fail for the reason given (wrong mutual nearest neighbours under Flavell's motion) is untested: the fraction of temporal edges joining different GT neurons per case would show it.
