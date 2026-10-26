---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a165a2-17e7-7192-a2d1-419524ae7296
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: medium
completedDate: 2025-11-12
projects:
  - "[[barlowtrack]]"
---

## Claim

Clustering BarlowTrack embeddings with seeded label propagation that enforces one label per time point, plus spectral alignment of the seeded labelings, is more accurate and less variable than HDBSCAN.

## Rationale

Table 2 and Fig. S1d: +0.6 to +2.3 points on three labs on identical embeddings; HDBSCAN drops low-density neurons as outliers.

## Scope

Default-setting BarlowTrack networks on the Zimmer, Flavell and Leifer benchmarks.
