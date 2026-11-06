---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b1ca-6244-7514-85d5-325a36d4a9a0
status: in-progress
priority: high
dateCreated: 2026-11-06T17:56:33+01:00
dateModified: 2026-11-06T20:54:15+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find the best combination of label propagation settings, beyond one-at-a-time changes, using the Ax machinery of the network sweeps. Suggested by the user, 2026-11-06.

## Method

barlow_track/scripts/scratch/labelprop_opt/ax_labelprop.py; experiment directories ~/.cache/barlow_scratch/labelprop_opt/ax/round{1,2}.
- Objective: mean over the 4 dev cases (S137 t50, S184 t27, Z1210 t36, F0330-02 t61) of the mean official accuracy at RNG seeds 0 and 1.
- Persistent workers per case take (trial, case, seed) jobs from a file queue.
- Round 1 (barlow_track@5e40a5b, runs r261106-506551 and r261106-510144): Ax client with Sobol (6) then GPEI; search space svd_dim {50, 100, 150, 200}, svd_mode {scaled, whiten}, graph {knn, mutual}, k 8-30, edge_weight {uniform, gauss}, num_layers {20, 30, 50, 100}, per_run_unique {greedy, hungarian}, num_seeds {25, 50, 100}, seed_select {random, stratified}, temporal_smooth {0, 0.5, 1}. Its 8 attached trials were base, svd100, whiten, mutual, seeds100, layers30, hungarian+gauss, and one combination. Round 1 was stopped after them, before any Sobol trial, because the final batch 1 showed temporal edges (temporal_w) to be the largest Zimmer gain and they were not in the space.
- Round 2 (barlow_track@d8a1335, run r261106-595574): the same space plus temporal_w {0, 1, 2, 5, 10}; round 1's complete trials imported as completed trials (temporal_w 0); attached temporal5, seed_strat, tsmooth05, c_all (svd100 + 50 stratified seeds + temporal5 + tsmooth05) and c_all_gh (+ gauss, hungarian); Sobol (4) then GPEI, 50 trials in total, 3 in parallel.

## Results

Round 1 (objective; per case S137 / S184 / Z1210 / F0330): base .9360 (.8885 / .9569 / .9301 / .9686); svd100 .9429 (.9082 / .9651 / .9288 / .9696); whiten .9398; mutual .9331 (Z1210 .9060). Round 2 running.

## Conclusion

Pending. The winner must still pass the adoption rule against base on the held-out confirmation cases ([[EVD - Confirmation of screened label propagation variants on held-out recordings]]).

## Caveats

The search selects on 4 networks, one per recording, so it can overfit them. The confirmation on held-out networks and recordings (Leifer, Flavell 2022-03-24-02) is the guard.
