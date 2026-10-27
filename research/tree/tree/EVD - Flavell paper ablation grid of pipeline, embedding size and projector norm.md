---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17770-1f70-745f-a76d-601322f9b596
status: in-progress
priority: high
dateCreated: 2026-10-26T09:59:59+01:00
dateModified: 2026-10-27T13:43:31+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find which of the revision design changes cost or recover Flavell accuracy, starting from the preprint configuration.

## Method

round_flavell_paper_ablation (barlow_track@54690ba:barlow_track/scripts/scratch/round_flavell_paper_ablation/, fixed in d8cbddd so the legacy cells get no global augmentation): use_position {false, true} x embedding_dim {64, 512} x projector_norm {batchnorm, layernorm}, 50 epochs, objective accuracy, --repetitions 2 (16 cluster jobs; trial_0-7 seed 43, replicates seed 921677937). Launched by the user on 2026-10-25, run r261025-j7666018 (first jobs 7666018-7666027). The objective scores against the project GT: every cell is rescored against the NWB GT from its cached embedding on CPU (barlow_track@ee2a06a:barlow_track/scripts/scratch/rescore_flavell_nwb_gt.sh), labelprop seed 0, fused stage for position cells. Local cells already done: (false, 64, bn) 0.957; (true, 64, bn) 0.795 fused / 0.756 backbone.

## Results

NWB GT [project GT], rescored in r261026-1606883 (7 position cells) and r261027-2071685 (legacy trials 0, 1, 3 and position trial 15).

| pipeline | emb, norm | seed 43 | seed 921677937 |
|---|---|---|---|
| legacy | 64, bn | 0.953 [0.917] | not done |
| legacy | 64, ln | 0.950 [0.917] | not done |
| legacy | 512, bn | not done | not done |
| legacy | 512, ln | 0.944 [0.907] | not done |
| position | 64, bn | 0.769 [0.743] | 0.848 [0.811] |
| position | 64, ln | 0.782 [0.754] | 0.904 [0.870] |
| position | 512, bn | 0.826 [0.787] | 0.882 [0.840] |
| position | 512, ln | 0.808 [0.778] | 0.901 [0.873] |

NWB GT minus project GT: +2.6 to +4.2 points (legacy 3.2-3.8). Position seed pairs differ by 5.6-12.2 points (all four favour seed 921677937); the three legacy seed-43 cells span 0.9 points. Legacy trials 2 and 8-11 were not finished on the cluster at 2026-10-27 11:40.

## Conclusion

Partial (11 of 16 cells). At seed 43 every legacy cell matches the preprint (0.944-0.953, n 3) whatever its embedding size or projector norm, while the position cells score 0.77-0.83, so the pipeline switch alone costs 12-18 points and the other two factors barely matter in the legacy pipeline. Position cells are far less stable across training seeds (5.6-12.2 points per pair) than the legacy cells are across architectures (0.9 points), and even their better seed (0.85-0.90) stays below every legacy cell.

## Caveats

use_position switches augmentation and fusion together; [[EVD - Image features alone in the position pipeline]] separates them. The legacy pipeline's own training-seed spread is not yet measured (its replicate cells 8-11 are pending), so "legacy is stable" rests on architecture variation at one seed. One labelprop seed per cell.
