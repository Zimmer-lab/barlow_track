---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b2b1-a387-77d2-9aa7-3694919122a8
status: done
priority: high
dateCreated: 2026-11-06T22:09:08+01:00
dateModified: 2026-11-06T23:05:17+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-11-06
---

## Purpose

Tell what the partner tracks of fragmented neurons hold: pieces of the same nucleus (over-segmentation, the GT click in one piece or the other) or a neighbouring nucleus.

## Method

`diag_fragment_pairs.py` (barlow_track@8a1c29e, labelprop_opt), run r261106-640685, on the cached base labels of Z1210 t36 at seed 0 (from [[EVD - Fragment merging of final label propagation tracks on recordings with distractors]]). For every GT neuron, its main label (most of its objects) and each partner label holding at least 20 of its objects; in volumes where the neuron sits in the partner label and the main label holds another object, the physical distance between the two objects (voxels 1.5 × 0.325 × 0.325 um), their mask volumes and whether the other object is GT. Reference: the nearest other GT neuron in the same volume (every 16th volume).

## Results

66 neuron-partner pairs, 5879 volumes. Distance between the neuron's object and the main label's object, weighted by volumes: q10 3.7, median 5.3, q90 11.2 um; nearest other GT neuron q10 3.3, median 4.2, q90 7.0 um. 7.6% of pairs are closer than the 10th percentile of the nearest-neuron distance. Mask volume ratio neuron/other: median 0.98 (GT masks median 644 voxels). The other object is a GT object in 24% of those volumes (weighted), not GT in 76%. Within a pair the distance is tight (e.g. 586 volumes at 5.0-6.1 um), so each partner is one fixed neighbour, and a pair's swap holds for a median of 54 volumes (up to 586).

## Conclusion

On Z1210, fragmented GT neurons are not split nuclei. They swap identity with one neighbouring nucleus of normal size, a median of 5.3 um away, at about the spacing of neighbouring GT neurons. The swap lasts tens to hundreds of volumes, and in three quarters of those volumes the neighbour is a detection that is not in the GT, most likely an unannotated neuron. The "distractor" fragment errors are therefore long swaps with unannotated neighbours, which the error decomposition counts as fragments only because the neighbour's track has no GT neuron of its own.

## Caveats

One recording, one network, one seed; Flavell (NWB source) has no segmentation metadata in the harness, so the same test there needs NWB centroids. Whether the non-GT neighbours are neurons is inferred from their size and spacing, not checked in the images.
