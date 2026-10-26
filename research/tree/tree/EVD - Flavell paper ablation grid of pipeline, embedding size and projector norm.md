---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17770-1f70-745f-a76d-601322f9b596
status: in-progress
priority: high
dateCreated: 2026-10-26T09:59:59+01:00
dateModified: 2026-10-26T22:25:15+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find which of the revision design changes cost or recover Flavell accuracy, starting from the preprint configuration.

## Method

round_flavell_paper_ablation (barlow_track@54690ba:barlow_track/scripts/scratch/round_flavell_paper_ablation/, fixed in d8cbddd so the legacy cells get no global augmentation): use_position {false, true} x embedding_dim {64, 512} x projector_norm {batchnorm, layernorm}, 50 epochs, objective accuracy, --repetitions 2 (16 cluster jobs; trial_0-7 seed 43, replicates seed 921677937). Launched by the user on 2026-10-25, run r261025-j7666018 (first jobs 7666018-7666027). The objective scores against the project GT: every cell is rescored against the NWB GT from its cached embedding on CPU (barlow_track@ee2a06a:barlow_track/scripts/scratch/rescore_flavell_nwb_gt.sh). Local cells already done: (false, 64, bn) 0.957; (true, 64, bn) 0.795 fused / 0.756 backbone.

## Results

Partial, 2026-10-26: the 7 position cells finished by then, rescored against the NWB GT (r261026-1606883; project GT in brackets). Seed 43: (64, bn) 0.769 [0.743], (64, ln) 0.782 [0.754], (512, bn) 0.826 [0.787], (512, ln) 0.808 [0.778]. Seed 921677937: (64, bn) 0.848 [0.811], (64, ln) 0.904 [0.870], (512, bn) 0.882 [0.840]. NWB GT minus project GT: +2.6 to +4.2 points (mean 3.4), larger at higher accuracy. Legacy cells (use_position false) still running.

## Conclusion

## Caveats

use_position switches augmentation and fusion together; [[EVD - Image features alone in the position pipeline]] separates them. Training seed moves a position cell by 5-12 points (seed 921677937 above seed 43 in all three pairs), more than embedding size or projector norm do, so cells need both repetitions before they are compared.
