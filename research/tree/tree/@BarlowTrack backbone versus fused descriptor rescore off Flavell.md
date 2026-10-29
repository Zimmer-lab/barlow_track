---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a18538-b231-707f-9b5c-eed26dfbf85a
status: done
priority: normal
dateCreated: 2026-10-29T02:14:07+01:00
dateModified: 2026-10-29T02:14:07+01:00
completedDate: 2026-10-29
run: barlow_track@cdd5761:barlow_track/scripts/scratch/backbone_rescore_off_flavell/
projects:
  - "[[barlowtrack]]"
---

## Summary
Two existing fused ViSNet networks rescored locally at both descriptor stages (`--descriptor_stage backbone` = image backbone only, `auto` = the network's own fused head), same code and ground truth, label propagation with 25 seeds: round_visnet_full trial 27 (trained on Zimmer, concat fusion with 2 self-attention layers) on zimmer_1128, Samuel 153 and Leifer; round_bayes_samuel trial 0 (trained on Samuel, add fusion) on Samuel 153.

## Method
Runs r261028-3159902 (run.sh at c81c641: Samuel, Leifer; its Zimmer cells OOM'd in GPU label propagation next to other training) and r261028-3398874 (run_zimmer_cpu_track.sh at cdd5761, Zimmer with `--track_device cpu`). Samuel on NWB GT objects; Zimmer and Leifer from the projects. Outputs in ~/.cache/barlow_scratch/runs/backbone_rescore_off_flavell/ (jsonl and embedding caches, local only).
