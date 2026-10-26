---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1776f-7cac-7ae3-b05d-d675ed47be4f
status: done
priority: normal
dateCreated: 2026-10-24T01:59
dateModified: 2026-10-26T09:59:36+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-26
---

## Purpose

Find why revision sweeps scored Flavell flat at .62-.73 while the preprint reports 96.7%.

## Method

The paper's Flavell baseline checkpoint (TrainedBarlow/baseline_flavell/trial_0) through eval_accuracy.py --source nwb as is (E1, r261023-2312954 / r261023-2321988) and with the NWB fields aligned by neuron name (E3, r261023-2333818); then two round_visnet_full checkpoints rescored with the aligned path (r261024-2391371). 25 seeds, seed 0. Issue i261026-1 describes the bug.

## Results

Paper checkpoint: bugged NWB path 0.8648 (209,890 GT entries embedded, cap 0.908), aligned 0.9692 (all 231,268 entries; misses 4,446, mismatches 2,668). round_visnet_full trial27 0.820 -> 0.912; trial20 0.750 -> 0.831. Even aligned, GT-centred NWB crops change the features (cosine median 0.955 to the paper's segmentation-centred embedding).

## Conclusion

The Flavell NWB path cropped one neuron and labelled it with another's ids; this alone cost a known-good network about 10 points and the revision networks 8-9 points. With the fields aligned, the current evaluation gives the paper network 0.969, matching the paper's 96.7%.

## Caveats

The NWB path gives the tracker only GT objects (an oracle candidate set), so it is not the paper's setting; the user wants the algorithm not to see only GT objects. eval_accuracy.py itself is not fixed yet, so every Flavell `--source nwb` record before the fix is low by this much. Single seed.
