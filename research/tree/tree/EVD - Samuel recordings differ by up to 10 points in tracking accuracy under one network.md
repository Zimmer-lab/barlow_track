---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1845b-150c-7c62-80b1-4db5d219dbd1
status: done
priority: normal
dateCreated: 2026-10-28T22:12:04+01:00
dateModified: 2026-10-28T22:12:04+01:00
confidence: medium
completedDate: 2026-10-28
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find whether Samuel 153 is an easy recording, and how much accuracy varies across Samuel recordings.

## Method

[[@BarlowTrack Samuel per-recording accuracy check]]: label propagation (25 seeds), NWB GT objects, on 137, 153 and 184-004, for round_bayes_samuel trial 0, paper-era harvard_153_flip trial 8 (both trained on 153) and the untrained image baseline.

## Results

| recording | frames | bayes trial 0 | paper-era trial 8 | image baseline |
|---|---|---|---|---|
| 153 | 1331 | 0.951 | 0.949 | 0.821 |
| 137 | 1326 | 0.848 | 0.807 | 0.641 |
| 184-004 | 3002 | 0.913 | 0.899 | 0.801 |

## Conclusion

137 is the hard Samuel recording for every embedding, 10 points below 153 with the revision network and 18 points below with the untrained baseline; 184-004 lies between. Trained networks gain most where the recording is hard (+21 pt over the image baseline on 137, +13 on 153), and the revision network beats the paper-era one on both held-out recordings (+4.1 on 137, +1.4 on 184-004).

## Caveats

Both networks were trained on 153, so 137 and 184-004 mix recording difficulty with cross-recording transfer; the image baseline (no training) separates them only in part. One network per kind, one seed set. Oracle detections throughout. 185-005 could not be read; the h5-only recordings were not scored.
