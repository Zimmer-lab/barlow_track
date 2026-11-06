---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b2e5-0b7f-73aa-8318-219c6f27c006
status: done
priority: high
dateCreated: 2026-11-06T23:05:17+01:00
dateModified: 2026-11-06T23:05:17+01:00
confidence: medium
completedDate: 2026-11-06
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find out whether the fragment errors on the paper's Leifer (StarDist masks, strict click GT) are the same kind as on Zimmer ([[EVD - Fragmented neurons on Zimmer swap with a neighbouring nucleus for long stretches]]), and whether merging or chain voting removes them.

## Method

Network wk2 Leifer t57, embedding on the StarDist objects (benchmark cache), harness case `leifer_stardist_strict` (barlow_track@3756753), base tracker at seed 0 on the CPU. diag_merge.py (r261106-606521), diag_chain.py (r261106-647605) and diag_fragment_pairs.py (r261106-674571, voxels 1.5 × 0.3226 × 0.3226 um), as on Zimmer.

## Results

- Base: accuracy .9371 over assigned strict clicks, reproducing the strict-GT objective (.93707, r261105-3536566). Errors: fragments 5.62% of GT points, swaps 0.47%, unlabelled 0.20%; merge bound .9878.
- 98 true fragment pairs among 917 candidates; their median volume overlap is 0.985, so no setting of the merge grid merges anything.
- Chain voting: +0.17 pt at most (chain_w 1). Chains: 99.1% of links join one GT neuron, an object's chain has a median length of 40 volumes.
- Fragment pairs (42 neuron-partner pairs, 5135 volumes): the neuron's object and the main label's other object are 3.0 / 3.9 / 4.7 um apart (q10 / median / q90, weighted by volumes). The nearest other GT neuron is 3.5 / 4.7 / 7.4 um away. 26% of pairs are closer than the 10th percentile of that. Mask volumes are alike (median ratio 1.07; GT masks median 536 voxels). The other object is a GT object in only 3% of these volumes. A swap lasts a median of 38 volumes, up to 746.

## Conclusion

On the paper's Leifer, almost all of the tracker's error (5.6 of 6.3 pt at seed 0) is a GT neuron's label sitting for tens to hundreds of volumes on a touching neighbour mask, about 4 um away and of normal size, that has no click. The neighbour is unannotated: it is not a GT object 97% of the time. Neither merging labels nor chain voting fixes this, so tracker post-processing is not the lever. This matches [[EVD - Half of the base network's Leifer StarDist errors are clicks where every tracker agrees on another object]]: these are the clicks on which trackers agree on another object. Each such case is either a real identity swap with an unannotated neighbour, or a click that lies on the boundary between two touching masks, so that the strict GT itself flips between them. Only a manual check of the clicks (consensus_error_clicks.csv) can tell which.

## Caveats

One network, one seed. Distances use mask centroids, not the clicks. The 26% of pairs closer than nearly all neighbouring GT neurons are the best candidates for click or segmentation problems (e.g. one nucleus split in two, the click in one half).
