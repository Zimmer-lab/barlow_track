---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b113-4b8e-751e-8541-c015c8b610f0
status: in-progress
priority: high
dateCreated: 2026-11-06T14:36:34+01:00
dateModified: 2026-11-06T15:02:14+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find which single changes to the label propagation tracker raise accuracy on saved embeddings, beyond the 0.8 pt re-tracking noise, so that they can be confirmed and combined.

## Method

Harness barlow_track/scripts/scratch/labelprop_opt (sweep_labelprop.py, lp_core.py, variants.py `batch1`; batch 1 at barlow_track@3fb6198 (9a32928 plus a project-load fix that leaves results unchanged), run r261106-369804; speed batch at 8f22a22, run r261106-361651, continuing as pid 361654). 35 one-change variants plus `base` (the tracker as the sweep runs it) on 4 dev cases × RNG seeds 0 and 1:
- S137 t50 (round_bayes_wk2_samuel_137, weak, .877);
- S184 t27 (round_bayes_wk3_samuel_184, .957);
- Z1210 t36 (round_bayes_wk3_zimmer_1210, project segmentation with distractors);
- F0330-02 t61 (round_bayes_wk3_flavell_2022033002, NWB, every detection).

Each variant is paired with `base` at the same seed. Propagation runs on the local TITAN V; graphs, syncs and scoring run on the CPU (8 streams, one per case and seed, 7 threads each). The speed batch runs first, as a GPU equivalence and reproduction check (base, local, exact, batch5, layers50, layers30, seed 0). Output: ~/.cache/barlow_scratch/labelprop_opt/{speed_dev,batch1_dev}.jsonl. Analysis: `analyze.py --per_case`.

## Results

Running since 2026-11-06.

## Conclusion

Pending.

## Caveats

The screen uses one network per lab; Leifer is held out for confirmation. Graph and seed variants re-randomise propagation, so pairing removes less noise for them than for sync or post-processing variants.
