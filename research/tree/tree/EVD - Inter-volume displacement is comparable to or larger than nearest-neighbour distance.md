---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a1-79bd-726d-a504-1010bbbb6a4f
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: high
completedDate: 2025-11-12
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Quantify how far each neuron moves between consecutive volumes relative to how closely neurons are packed, which sets how hard frame-to-frame association is.

## Method

Fig. 1c: displacement vector fields and colour-coded flow for 3 adjacent Zimmer time points. Fig. 1d: per-neuron displacement over time relative to nearest-neighbour distance (capped at 4). Fig. S1a-c: inter-volume displacement of ground-truth neurons per lab, in voxels (a, b) and normalised by nearest-neighbour distance (c).

## Results

Fig. 1c shows rigid rotations and nonrigid nose compression between volumes 0.3 s apart. Median displacement (read from Fig. S1a boxplots): ≈18 voxels Flavell (1.7 Hz), ≈14 Zimmer (3.5 Hz), ≈10 Leifer (6 Hz). Normalised medians (Fig. S1c) ≈2-3 Flavell, ≈1.5 Zimmer, ≈1 Leifer, with tails beyond 10. Displacement is often much larger than nearest-neighbour distance even though the camera follows the worm and removes centroid motion.

## Conclusion

In all three benchmark labs the typical neuron moves about one to three nearest-neighbour distances between volumes, so nearest-position matching and ROI overlap cannot associate neurons reliably.

## Caveats

As reported in the paper; not rerun. Medians read off boxplots, approximate. Flavell positions come from the Flavell lab's tracking algorithm, not manual annotation. Voxel units differ between labs (Flavell z resolution is 0.33 µm vs 1.5 µm), so the raw-voxel panels are not directly comparable; the normalised panel is.
