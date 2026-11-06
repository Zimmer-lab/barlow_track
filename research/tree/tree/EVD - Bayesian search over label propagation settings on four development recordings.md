---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b1ca-6244-7514-85d5-325a36d4a9a0
status: in-progress
priority: high
dateCreated: 2026-11-06T17:56:33+01:00
dateModified: 2026-11-06T17:56:33+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find the best combination of label propagation settings, beyond one-at-a-time changes, using the Ax machinery of the network sweeps. Suggested by the user, 2026-11-06.

## Method

barlow_track/scripts/scratch/labelprop_opt/ax_labelprop.py (barlow_track@5e40a5b); experiment directory ~/.cache/barlow_scratch/labelprop_opt/ax/round1.
- Ax client: Sobol (6 trials) then GPEI, 40 trials, 3 in parallel.
- Search space: svd_dim {50, 100, 150, 200}, svd_mode {scaled, whiten}, graph {knn, mutual}, k 8-30, edge_weight {uniform, gauss}, num_layers {20, 30, 50, 100}, per_run_unique {greedy, hungarian}, num_seeds {25, 50, 100}, seed_select {random, stratified}, temporal_smooth {0, 0.5, 1}.
- Attached first: base, the batch-1 leaders as single changes, and one combination of them.
- Objective: mean over the 4 dev cases (S137 t50, S184 t27, Z1210 t36, F0330-02 t61) of the mean official accuracy at RNG seeds 0 and 1.
- Persistent workers, one or two per case: GPU for Zimmer and Flavell, CPU for the rest.

## Results

Running since 2026-11-06 ~18:00.

## Conclusion

Pending. The winner must still pass the adoption rule against base on the held-out confirmation cases ([[EVD - Confirmation of screened label propagation variants on held-out recordings]]).

## Caveats

The search selects on 4 networks, one per recording, so it can overfit them. The confirmation on held-out networks and recordings (Leifer, Flavell 2022-03-24-02) is the guard.
