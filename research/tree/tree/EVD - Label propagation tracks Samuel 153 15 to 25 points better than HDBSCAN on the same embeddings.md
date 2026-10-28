---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1845b-148e-72aa-ac25-e521cb3b6ad1
status: done
priority: normal
dateCreated: 2026-10-28T22:12:03+01:00
dateModified: 2026-10-28T22:12:03+01:00
confidence: medium
completedDate: 2026-10-28
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find why the paper-era Samuel results stayed near 75-80% while revision networks score 95% on Samuel 153: the network, the recording or the tracker.

## Method

[[@BarlowTrack Samuel per-recording accuracy check]]: each network's saved 153 embeddings tracked twice, with label propagation (25 seeds) and with the global clusterer (SVD, UMAP, HDBSCAN), same NWB GT objects and scoring. The 88 stored paper-era track sets of the 153 project (old tracker, HDBSCAN windows) scored with the current metric.

## Results

| network | label propagation | global HDBSCAN |
|---|---|---|
| paper-era harvard_153_flip trial 8 | 0.949 | 0.804 |
| round_bayes_samuel trial 0 | 0.951 | 0.705 |

The stored paper-era tracks score best 0.805, median 0.647 (flip sweep, 28 trials), so the HDBSCAN re-track of trial 8 (0.804) reproduces the old number. About 90% of their errors are misses: HDBSCAN leaves noise points and small clusters unassigned (~88 tracks for 98 neurons).

## Conclusion

On Samuel 153 the tracker, not the network, set the paper-era accuracy: the same paper-era embeddings go from 0.80 with HDBSCAN to 0.95 with label propagation (+14.5 pt), and the revision network gains +24.6 pt. Under label propagation the two networks are tied (0.949 vs 0.951). The gain is far larger than the preprint's +0.6-2.3 pt on Zimmer, Flavell and Leifer.

## Caveats

One recording, which both networks were trained on (generalization across frames, not recordings). Oracle detections: the segmentation holds only GT neurons. One label-propagation seed set (25 seeds) and one HDBSCAN setting (defaults, not tuned for Samuel). Whether label propagation code at the time the user first tried it on Samuel was buggy (docs/pr_plan PR02) was not checked.
