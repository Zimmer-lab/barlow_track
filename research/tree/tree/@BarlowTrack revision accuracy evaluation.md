---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a165a3-b265-7236-aae2-f4735a8c23a7
status: done
priority: normal
dateCreated: 2026-09-30
dateModified: 2026-09-30
completedDate: 2026-09-30
run: barlow_track@bc9ff96:barlow_track/scripts/eval_accuracy.py
data:
  - /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/
projects:
  - "[[barlowtrack]]"
---

## Description

Post-preprint accuracy evaluation (first added 0ddb46d, 2026-09-30; updated e71b3b8 #138): embeds a full video with a given or untrained network (`--mode image|position|posonly|attention|trained`), tracks with label propagation (default 25 seeds; the paper used 100) or global clustering, and scores against ground truth with the paper's recipe (rename_columns_using_matching on raw_segmentation_id, then calculate_accuracy; Leifer via `--source nwb`, matching on raw_neuron_ind_in_list). Results and the September 2026 comparison of image-only vs position-aware embeddings are written up in barlow_track@bc9ff96:docs/accuracy_evaluation.md; records went to /tmp/claude/exp_results.jsonl on the machine that ran them.

Cited at the current tip of global_augmentation (bc9ff96); the September results were produced by earlier versions of the script (the standalone leifer script was merged in Oct 2026). Not rerun here.
