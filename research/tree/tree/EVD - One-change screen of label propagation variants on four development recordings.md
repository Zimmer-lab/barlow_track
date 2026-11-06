---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b113-4b8e-751e-8541-c015c8b610f0
status: done
priority: high
dateCreated: 2026-11-06T14:36:34+01:00
dateModified: 2026-11-06T20:43:30+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-11-06
---

## Purpose

Find which single changes to the label propagation tracker raise accuracy on saved embeddings, beyond the 0.8 pt re-tracking noise, so that they can be confirmed and combined.

## Method

Harness barlow_track/scripts/scratch/labelprop_opt (sweep_labelprop.py, lp_core.py, variants.py `batch1`; batch 1 at barlow_track@3fb6198 (9a32928 plus a project-load fix that leaves results unchanged), runs r261106-384883 (stopped on CUDA OOM, 32 records kept), r261106-400753 and the GPU queues r261106-473331 (Z1210 and F0330 seeds 1-2); speed batch at 8f22a22, run r261106-361651). 35 one-change variants plus `base` (the tracker as the sweep runs it) on 4 dev cases × RNG seeds 0, 1 and 2 (12 pairs, 432 records):
- S137 t50 (round_bayes_wk2_samuel_137, weak, .877);
- S184 t27 (round_bayes_wk3_samuel_184, .957);
- Z1210 t36 (round_bayes_wk3_zimmer_1210, project segmentation with distractors);
- F0330-02 t61 (round_bayes_wk3_flavell_2022033002, NWB, every detection).

Each variant is paired with `base` at the same seed and device. Graphs, syncs and scoring run on the CPU; propagation on the CPU (CSR) or the local TITAN V. Output: ~/.cache/barlow_scratch/labelprop_opt/batch1_dev.jsonl. Analysis: `analyze.py batch1_dev.jsonl --per_case` (and `--metric err_*`). Screen rule (PLAN.md section 4): promote if mean delta ≥ +0.3 pt and wins in ≥ 75% of pairs.

## Results

Base (mean of 3 seeds, seed range): S137 .8896 (0.37 pt), S184 .9571 (0.11), Z1210 .9338 (1.31), F0330 .9689 (0.60).

Paired deltas, pt (mean, wins/12, per case S137 / S184 / Z1210 / F0330):

| variant | mean | wins | S137 | S184 | Z1210 | F0330 |
|---|---|---|---|---|---|---|
| svd100 | +0.74 | 10 | +2.02 | +0.78 | +0.01 | +0.17 |
| seeds100 | +0.57 | 11 | +0.82 | +0.24 | +0.98 | +0.24 |
| temporal5 | +0.46 | 9 | +0.24 | +0.23 | +1.38 | -0.02 |
| seeds50 | +0.43 | 10 | +0.72 | +0.18 | +0.87 | -0.03 |
| seed_strat | +0.38 | 12 | +0.54 | +0.14 | +0.59 | +0.24 |
| tsmooth05 | +0.34 | 11 | +0.24 | +0.43 | +0.46 | +0.22 |
| whiten | +0.31 | 7 | +1.38 | +0.44 | -0.40 | -0.21 |
| temporal1 | +0.29 | 8 | -0.32 | +0.09 | +1.20 | +0.19 |
| gauss | +0.25 | 11 | +0.59 | +0.17 | +0.01 | +0.23 |
| hungrun | +0.23 | 10 | +0.82 | +0.16 | -0.06 | -0.00 |
| spread97 / spread90 | +0.19 / +0.19 | 9 / 8 | +0.57 / +0.64 | | | |
| layers30 | +0.16 | 8 | +0.67 | +0.02 | -0.01 | -0.02 |
| mutual | -0.44 | 8 | +0.81 | +0.23 | **-2.84** | +0.05 |
| k10 | -0.16 | 7 | +1.21 | +0.31 | **-2.17** | -0.00 |
| seed_complete | -0.53 | 6 | +0.07 | +0.08 | **-2.40** | +0.11 |
| k40 | -0.50 | 4 | -2.02 | -0.68 | +0.86 | -0.15 |
| sym | -0.21 | 4 | -1.40 | -0.98 | +1.52 | +0.00 |
| layers300 | -0.46 | 5 | -1.52 | -0.44 | +0.03 | +0.10 |
| tsmooth2 | -0.41 | 6 | -1.69 | +0.02 | +0.47 | -0.43 |
| reprop09_w4 | -1.72 | 0 | -1.10 | -0.90 | -4.24 | -0.62 |

Within ±0.2 pt (no effect): thr0, classnorm, nosame, pca, l2, runw, drop20, framenorm, exact, kextra40, probw, reprop07, thr03; reprop09 -0.30 (0/12).

Error types (paired, pt; lower is better): svd100 and whiten cut swaps (-0.60, -0.45; S137 -1.6); more seeds and tsmooth05 cut unlabelled objects (seeds50 -0.43, seeds100 -0.30, tsmooth05 -0.26); the three Zimmer failures (mutual, k10, seed_complete) add unlabelled objects (+0.6 to +0.8 mean), i.e. sparser graphs leave Zimmer neurons unreached.

## Conclusion

Six single changes pass the screen rule (mean ≥ +0.3 pt, ≥ 9/12 wins): an SVD of 100 dimensions instead of 50 (+0.74 pt, +2.0 on the weak S137 network), 100 or 50 seed volumes instead of 25 (+0.57, +0.43), temporal edges of weight 0.5 (+0.46, +1.4 on Z1210), seeds spread evenly over time (+0.38, 12/12) and temporal smoothing 0.5 of the sync (+0.34, 11/12). No change does better than +1 pt on average across the four recordings, and the gains are largest on the weakest network. Sparser graphs (mutual kNN, k=10, complete-seed selection) help Samuel but cost Zimmer 2-3 pt by leaving neurons unreached, so graph density should not be tuned on Samuel alone.

## Caveats

One network per lab, n = 3 seeds per case; Leifer and the second recording of each lab are held out for confirmation. Graph and seed variants re-randomise propagation, so pairing removes less noise for them. Gains of single changes need not add up; the combination is tested by the Ax search ([[EVD - Bayesian search over label propagation settings on four development recordings]]) and the confirmation batch. Partial numbers quoted mid-run (e.g. whiten +0.97, mutual +0.62 before the Zimmer seeds were in) were misleading: Zimmer seeds 1-2 came last.
