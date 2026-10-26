---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a3-607c-7a83-8493-d4be52337d5d
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: low
completedDate: 2025-11-12
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Track animals moving faster than in any published whole-brain dataset; earlier work (incl. the authors') used dense agarose to slow animals.

## Method

Fig. 4a-c: Zimmer-lab recordings on lower-density agarose (less viscous, faster animals, more displacement per volume); heatmap of all tracked neurons, traces of identifiable neurons, per-track coverage histogram.

## Results

Clean heatmap and recognisable neuron traces; most tracks with high coverage (values not read off the figure).

## Conclusion

BarlowTrack produced complete-looking tracks and plausible traces for fast animals that the authors' earlier setup avoided.

## Caveats

As reported in the paper; not rerun. No ground truth: judged by track coverage and heatmap/trace quality only, which Fig. 3d-f shows is a weak proxy on two of three labs. Number of animals per condition not stated (5 animals across all novel conditions, Table 1). No figure script found in the repository. The agarose percentage is not explained in the text (co-author markup).
