---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-e8b5-79ee-8aee-12cc4dcf96c7
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: medium
completedDate: 2025-11-12
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find how many volumes a network must be trained on.

## Method

Fig. 3i: accuracy of networks trained on 1, 2, 4, 8, 16, 32, 64, 128, 256 volumes (training_data_sweep_{zimmer,flavell,leifer}), several networks per point, label propagation on the full recording.

## Results

Zimmer stays ≈0.6-0.75 up to 16 volumes and jumps near 32-64; Flavell climbs steadily from ≈0.75; Leifer starts ≈0.85-0.88. All reach ≈0.95-0.97 by 128-256. The text puts the jump around 64 volumes (≈4% of a video) and saturation after 256 (≈16%).

## Conclusion

Training on a few percent of a recording's volumes (about 64) gives most of the accuracy and about 256 volumes (≈16%) saturates it; the dataset with the least distinctive neighbourhoods (Zimmer) needs the most.

## Caveats

As reported in the paper; not rerun. Values read from the figure. The text cites "Fig. 3e" for this panel; it is Fig. 3i (co-author markup). Default training uses 200 frames, below the stated saturation point.
