---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1821c-8994-73ea-a998-f016fe567724
status: done
priority: high
dateCreated: 2026-10-28T11:44:30+01:00
dateModified: 2026-10-29T19:34:25+01:00
confidence: high
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-29
---

## Purpose

Find whether a position-only network can learn neuron identity, and what has kept the ViSNet position branch from learning it ([[EVD - Trained position branches carry no more neuron identity than an untrained one on Flavell]]): the RBF scale problem (issue i261028-2), the neighbourhood size, or too little deformation between the training views. Also the first position-only baseline, which the paper revision needs.

## Method

round_posonly_{zimmer,flavell,samuel} on the cluster ([[@BarlowTrack position-only ViSNet cluster rounds]]): per dataset visnet_rbf_rescale {false, true} x visnet_cutoff_nn {0, 3, 6, 12} x elastic_scale {0, 1}, one seed, 50 epochs, width 64, 100 frames, dropout 0.2; full-video accuracy with label propagation on the trial's own recording (Flavell and Samuel against their NWB GT objects, Zimmer 2022-11-28 project segmentation). Samuel's rbf-on cells of that round have no records yet, so Samuel's rbf-on numbers come from round_posonly2_samuel ([[@BarlowTrack position-only ViSNet second cluster round]]; rbf on, width 512, 600 frames, 60 epochs). The local fast loop ([[@BarlowTrack position-only fast local loop]], runs r261028-3310541 to r261029-3865613) took the fixed network further on Zimmer.

## Results

| dataset | RBF fix off (8 cells) | RBF fix on |
|---|---|---|
| Zimmer 2022-11-28 | 0.09-0.58 | 0.66-0.87 (8 cells, round_posonly_zimmer) |
| Flavell 2022-03-31-01 (GT objects) | 0.11-0.37 | 0.88-0.97 (8 cells, round_posonly_flavell) |
| Samuel 153 (GT objects) | 0.15-0.18 (4 cells) | 0.69-0.94 (7 cells, round_posonly2_samuel) |

Elastic deformation made no consistent difference with the fix on (Zimmer 0.825 / 0.827 at cutoff_nn 0). With the fix on, later local and cluster runs reached 0.961 on Zimmer (posonly_fast long15: width 512, 600 frames, 60 epochs, jitter auto, dropout 0.02; long14 0.958 with dropout 0), 0.975-0.981 on Flavell (round_posonly2_flavell trials 0-5) and 0.936 on Samuel 153 (round_posonly2_samuel trial 6). For comparison the best image-only / fused networks on the same cells: Zimmer 0.923 (trial 27's backbone) / 0.970, Flavell 0.963 / -, Samuel 0.951 / 0.950 (docs/best_of_table.md, ed5e46e).

## Conclusion

The RBF mis-scaling (issue i261028-2) was what kept the ViSNet position encoder from learning neuron identity: with it fixed, a position-only network goes from 0.1-0.6 to 0.66-0.97 on all three datasets in every grid cell, and after the fast-loop changes tracks within a point of the best image or fused network (Zimmer 0.961 vs 0.970 fused; Flavell 0.98 vs 0.963 image-only; Samuel 0.936 vs 0.951).

## Caveats

One seed per cell in round_posonly; the best numbers are single runs (round_posonly2's second seeds were still running). Flavell and Samuel cells track the NWB GT objects (oracle detections), Zimmer the full segmentation, so the datasets are not on one footing. Samuel's rbf-on comparison mixes rounds (different width, frames and epochs from the rbf-off cells), so its gap overstates the fix alone. Every fused network trained before 2026-10-28 had the blind ViSNet (issue i261028-2); notes resting on them are listed by `ygg issues i261028-2`.
