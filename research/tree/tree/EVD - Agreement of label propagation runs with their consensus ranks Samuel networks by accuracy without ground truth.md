---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1ab7b-a236-7aee-8124-bd3d9258726a
status: in-progress
priority: normal
dateCreated: 2026-11-05T12:32:48+01:00
dateModified: 2026-11-05T22:35:01+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Method

For trials of a GT-tuned per-recording round, re-run label propagation from the trial's cached full-video embedding (<round>/emb_cache) exactly as eval_accuracy --mode trained does (50-component SVD, 25 seed volumes, seed 0, k=20, CSR propagation on the GPU), then spectral synchronization to the consensus. GT-free metrics: agree_mean = mean over the 25 runs of the fraction of objects whose run label, mapped to the consensus by the run's permutation, equals the consensus label; conf_mean = mean consensus confidence; plus assigned fraction, duplicate fraction, complete tracks. Every trial of wk2_samuel_137 (109; r261105-3531457) and wk3_samuel_184 (76; r261105-3581004); 12 trials at evenly spaced accuracy ranks of wk3_zimmer_1210 (r261105-3461106), then every scored trial of wk3_zimmer_1210 (57) and wk3_zimmer_1123 (26; r261105-3650602), wk3_flavell_2022033002 (67) and wk3_flavell_2022032402 (75; r261105-3662972). All with the historical independent sync rounding; the top 5 trials per round were re-scored with the CPQR + refinement default of a5cb2f1 (r261105-3772374, r261105-3795686). Partial correlation: rank residuals after regression on the 8 swept settings (as for val loss), fusion=add trials only. Scripts barlow_track/scripts/scratch/gt_free_proxy/consistency_proxy.py, consistency_corr.py, consistency_partial.py (2511d60; consistency_proxy.py routed through the tracker in 5695edc, identical numbers on samuel_137 trial 63), metrics in barlow_track/utils/utils_consistency.py (5695edc).

## Result

| Round | n | accuracy range (better half from) | agree_mean rho all / better half / partial | conf_mean rho all / better half / partial |
|---|---|---|---|---|
| wk2_samuel_137 | 109 (91 add) | 0.48-0.931 | 0.94 / 0.72 / 0.89 | 0.95 / 0.75 / 0.90 |
| wk3_samuel_184 | 76 (61 add) | 0.481-0.957 | 0.93 / 0.62 / 0.87 | 0.95 / 0.71 / 0.89 |
| wk3_flavell_2022033002 | 67 (54 add) | 0.505-0.970 (0.940) | 0.98 / 0.91 / 0.97 | 0.99 / 0.93 / 0.98 |
| wk3_flavell_2022032402 | 75 (63 add) | 0.796-0.980 | 0.86 / 0.37 / 0.77 | 0.86 / 0.35 / 0.77 |
| wk3_zimmer_1123 | 26 (19 add) | 0.874-0.987 (0.977) | 0.91 / 0.82 / 0.78 | 0.95 / 0.86 / 0.89 |
| wk3_zimmer_1210 | 57 (41 add) | 0.595-0.943 (0.936) | 0.82 / 0.10 / 0.52 | 0.77 / -0.04 / 0.38 |

The settings explain 0.32-0.36 (Samuel), 0.45-0.66 (Zimmer, Flavell) of the accuracy rank variance. The trial with the highest agree_mean is the true best on samuel_137 (trial 63, 0.9311), samuel_184 (trial 27, 0.957), flavell_2022033002 (trial 61, 0.9695) and flavell_2022032402 (trial 47, 0.9798); on zimmer_1123 it is rank 8 of 26 (0.9833 vs best 0.9868), on zimmer_1210 rank 8 of 57 (0.9394 vs best 0.9435). Zimmer 1210's better half spans 0.7 pt (0.936-0.943), within which agreement is flat (~0.83-0.85).

## Conclusion

How well the 25 label-propagation runs agree with their consensus ranks networks by tracking accuracy without ground truth on every recording tested (rho 0.82-0.98 over 26-109 GT-tuned trials of Samuel 137 and 184, Flavell 2022033002 and 2022032402, Zimmer 1123 and 1210) and, unlike val loss (partial at most ~0.5, [[EVD - Validation and training loss rank sweep trials mostly through the training budget]]), keeps most of that ranking among configs of equal training budget (partial 0.77-0.97, except 0.52 on Zimmer 1210). Selecting on it alone finds the best network on 4 of 6 recordings and loses 0.35-0.41 pt on the two Zimmer ones. Where it fails to rank is the saturated top of Zimmer 1210 (better half within 0.7 pt), while on the equally high Zimmer 1123 it still ranks the better half (0.82), so the plateau is saturation, not a Zimmer-specific blind spot. It is the objective of the staged shared-config sweep round_shared_v1.

## Caveats

Leifer untested. Ranking within the better half is weaker on Flavell 2022032402 (0.37) as well. Agreement shares the labelprop machinery with the tracker that is scored, so it cannot see errors every run makes alike (e.g. a consistent swap), and it may favour embeddings that are confidently wrong in the same way. Within-recording only: levels differ between recordings (Samuel ~0.6-0.7, Flavell ~0.79, Zimmer ~0.85), hence the per-recording references in the shared sweep. Depends on the sync: Hungarian refinement of the per-run permutations (opt-in, 711ba82) raises agreement by 0.015-0.035 on the same embedding, so references and trials must use the same sync. Among each round's top 5 the new sync spreads agreement as much or more and ranks them no worse (n=5 per round, noise-level either way); its correlation over whole rounds is not measured. All rounds were trained with frozen view pairs (i261027-1).
Uses 25 seed volumes, as the eval and sweep do; the tracker's default and the paper use 100. With 100 seeds (r261105-3881276, same trials, new sync) the ranking is no better: samuel_137 rho .97 -> .98 (better half .82 -> .88, same pick), flavell_2022032402's better half .47 -> .29 (25 and 100 seeds agree only rho .57 there, so within its 0.7 pt range the ranking is mostly propagation noise), at 2.7-3.2x the time.
