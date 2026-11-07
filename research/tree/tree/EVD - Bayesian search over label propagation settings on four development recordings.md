---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b1ca-6244-7514-85d5-325a36d4a9a0
status: done
priority: high
dateCreated: 2026-11-06T17:56:33+01:00
dateModified: 2026-11-07T07:48:18+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-11-07
---

## Purpose

Find the best combination of label propagation settings, beyond one-at-a-time changes, using the Ax machinery of the network sweeps. Suggested by the user, 2026-11-06.

## Method

barlow_track/scripts/scratch/labelprop_opt/ax_labelprop.py; experiment directories ~/.cache/barlow_scratch/labelprop_opt/ax/round{1,2,3}.
- Objective: mean over the 4 dev cases (S137 t50, S184 t27, Z1210 t36, F0330-02 t61) of the mean official accuracy at two RNG seeds.
- Persistent workers per case take (trial, case, seed) jobs from a file queue.
- Round 1 (barlow_track@5e40a5b, runs r261106-506551 and r261106-510144): Ax client with Sobol (6) then GPEI; search space svd_dim {50, 100, 150, 200}, svd_mode {scaled, whiten}, graph {knn, mutual}, k 8-30, edge_weight {uniform, gauss}, num_layers {20, 30, 50, 100}, per_run_unique {greedy, hungarian}, num_seeds {25, 50, 100}, seed_select {random, stratified}, temporal_smooth {0, 0.5, 1}. 8 attached trials (base, svd100, whiten, mutual, seeds100, layers30, hungarian+gauss, one combination); stopped before any Sobol trial because batch 1 showed temporal edges (temporal_w), not in the space, to be the largest Zimmer gain.
- Round 2 (barlow_track@d8a1335, run r261106-595574, then r261106-633070): plus temporal_w {0, 1, 2, 5, 10}; round 1 imported; attached temporal5, seed_strat, tsmooth05, c_all, c_all_gh; Sobol (4) then GPEI, 3 in parallel. Stopped at 01:33 when round 3 took over.
- Round 3 (barlow_track@73022a2 to b7a4d69, run r261107-759603): plus chain_w {0, 0.25, 0.5, 1}; rounds 1-2 imported; attached c_all_gh_chain, c_all_gh with chain_w 0.25, c_all_gh_svd150; Sobol (2) then GPEI, 45 trials, 3 in parallel. GPEI kept re-suggesting its incumbent on the same seeds (trial 17 repeated trial 12 at exactly 0.9530), so from round 3 each repeat of an arm runs a fresh pair of seeds, and an arm suggested a third time is abandoned for a random one-parameter neighbour (--max_repeats 2).

## Results

Round 1: base .9360 (S137 .8885 / S184 .9569 / Z1210 .9301 / F0330 .9686); svd100 .9429; whiten .9398; mutual .9331 (Z1210 .9060). Round 2: best c_all_gh .9530; c_all_gh across seeds 0-9 .9520-.9530, so differences under ~0.1 pt between arms are seed noise. Round 3 (complete, 45 trials): best observed trial 44 .9556 (svd 150, k 21, 50 layers, 50 stratified seeds, gauss, Hungarian, temporal_w 2, chain_w 0.25, no temporal smoothing); model best the same arm at .9554; trials 23 (c_all_gh with svd 150) .9549, 33 .9553, 29 .9549, 42/43 .9548 on two seed pairs. Every arm above .954 has svd 150, gauss edges, Hungarian, 50 stratified seeds, k 19-21 and graph knn; num_layers 50 and 100 tie, temporal_w 2-5 and chain_w 0-0.25 tie within seed noise.

## Conclusion

The search confirms the combination the screen suggested and adds one change: svd 150 instead of 100, about +0.2 pt over c_all_gh on the four dev cases (.9549 vs .9520-.9530). Beyond that it plateaus at about .955; temporal weight, chain voting, layers and k move the objective by less than the seed noise. svd 150 held up on the held-out cases ([[EVD - Confirmation of screened label propagation variants on held-out recordings]]); the temporal edges the search kept did not (they cost the held-out Flavell 0.4 pt), so the adopted recipe drops them.

## Caveats

The search selects on 4 networks, one per recording, and two seeds per trial; its Flavell dev recording (F0330) did not show the temporal-edge harm that the held-out F0324 did, so its temporal settings are not trustworthy on their own. The objective is a mean over cases, dominated by S137, which has the most room. Ax 0.3.3 GPEI; the plateau could hide a better region that 45 trials did not reach, but the one-parameter neighbours around the incumbent found none.
