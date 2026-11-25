---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a2134f-8769-76c3-a4c5-5401bfe96816
status: done
priority: normal
dateCreated: 2026-11-25T16:25:08+01:00
dateModified: 2026-11-25T16:25:08+01:00
confidence: high
completedDate: 2026-11-25
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Conclusion
Across one primary recording per lab, the median full-recording raw green/red Pearson r (fDNC versus BarlowTrack trial 59 first training replicate) was 0.709 versus 0.998 in Zimmer 1128, 0.906 versus 0.986 in Leifer worm 1, 0.875 versus 0.9995 in Samuel 153, and 0.466 versus 0.947 in Flavell 0331 GFP. For fDNC the numbers below r = 0.9 were 125/131, 37/79, 51/95 and 159/163 respectively. Median coverage was at least 95% on each recording.

## Method
Used the confidence-selected five-template outputs from the paired benchmark, recovered its whole-recording identity assignments, and reproduced the benchmark counts of missed and wrong masks and total reference observations for every recording. Used the same per-mask fluorescence, behavior, acquisition time and secondary processed ratio as in the BarlowTrack analysis. Pearson correlations use shared, observed, finite pairs and include tracker disagreements. See docs/paper/trace_fidelity/caption.md for figures, tables, inputs and regeneration. Runs r261125-7.3 and r261125-4.8. Working-tree script barlow_track/scripts/paper_scripts/revision/trace_fidelity.py SHA-256 64d7f61dba42726e3c29419ca0f313a9f72794700eb99be10c12b1592422fe51.

## Caveats
One pretrained deterministic fDNC tracker with five templates, and five training replicates of BarlowTrack. The first BarlowTrack run supplies the example distribution. GFP fluorescence in Flavell is a tracking control. Samuel uses reference-derived detections and only 95 of 98 reference columns had enough fDNC overlap. Mask accuracy includes missed assignments. The correlation denominator requires observations on both tracks.
