---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1776e-9609-7456-af21-13e794fd0f3c
status: done
priority: normal
dateCreated: 2026-10-23T22:07
dateModified: 2026-10-23T22:07
completedDate: 2026-10-23
run: barlow_track@95b93fb:barlow_track/scripts/scratch/eval_nwb_aligned.py
data:
  - /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132
  - /home/charles/.cache/barlow_scratch/runs/flavell_sweep_aligned_20261024_0032
projects:
  - "[[barlowtrack]]"
---

## Description

eval_accuracy.py (bc9ff96) with `_load_nwb_arrays` monkeypatched to select each NWB field by neuron name, removing the Flavell neuron-order bug (issue i261026-1). Used for E3 (paper Flavell baseline checkpoint, r261023-2330704 / r261023-2333818) and to rescore round_visnet_full trials 27 and 20 (r261024-2391371). 25 seeds, seed 0, GPU embedding, CPU tracking. The unpatched runs on the same checkpoint (E1, E2) are in the same output folder. Uncommitted when run; committed unchanged as 95b93fb.
