---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a21343-3707-7365-899d-64ee7e4a0fca
status: done
priority: normal
dateCreated: 2026-11-25T16:11:41+01:00
dateModified: 2026-11-25T16:11:41+01:00
confidence: high
completedDate: 2026-11-25
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Conclusion
Both tracks gave exactly the same raw green/red ratio at all 3,272,220 finite same-mask pairs (all five training replicates counted). The correlations also include 69,087 pairs with different masks. First-replicate median r was 0.9983 on Zimmer 1128, 0.9862 on Leifer NeRVE worm 1, 0.9995 on Samuel 153 and 0.9469 on Flavell 2022-03-31-01 (GFP). Counts below 0.9 were 4/131, 9/79, 3/97 and 52/163, respectively.

## Method
Runs r261125-4.6 and r261125-4.7 checked same-mask ratios element by element across 20 trial 59 tracks_v2 outputs. Panel A shows the first 120 seconds in each recording. Panel B summarizes complete recordings. Source script barlow_track/scripts/paper_scripts/revision/trace_fidelity.py, SHA-256 fae6e0d9e9543cf92f5865853f04407cbe2c7fceddbee1d8219c903a17abaf89. Figures, both trace metrics, coverage and reproduction details are in docs/paper/trace_fidelity. The processed comparison is recorded in [[EVD - Trial 59 preserves the median calcium trace across five training replicates in three recordings]].

## Caveats
Using the same image and mask gives the same raw ratio by construction. Separate per-channel bleach fits can introduce time-varying differences at same-mask frames when the tracks differ elsewhere, as observed for Leifer neuron_060, replicate 1. Median normalization changed the relative scale by a further 0.22%. Full-recording raw r was 0.947 and processed r was 0.856. Replicates use the same animal per recording.
