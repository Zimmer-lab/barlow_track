---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-18af-7e3f-b250-8592a1dbfffe
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

Benchmark BarlowTrack, trained per dataset without annotation, against ground truth from three independent labs.

## Method

Table 2: median accuracy of networks trained with default settings (Methods 4.3; 200 training volumes, crop 8x64x64, 16 z for Flavell), tracked with label propagation or HDBSCAN. Accuracy: BarlowTrack columns matched to GT neurons on raw segmentation ID, fraction of correctly assigned GT points. Fig. 3a-c show example volumes from each lab.

## Results

Label propagation: Zimmer 97.3%, Flavell 96.7%, Leifer 97.4%. HDBSCAN: 96.7%, 94.4%, 96.2%. Zimmer: 3 recordings, 1667 volumes, 60-130 GT objects; Flavell: 5 recordings (algorithmic GT); Leifer: 1 recording, 1536 volumes, 79 objects.

## Conclusion

Without manual annotation, BarlowTrack with label propagation tracks 96.7-97.4% of ground-truth neuron positions correctly in recordings from three labs that differ in resolution (z 0.33-1.5 µm), volume rate (1.7-6 Hz) and reference promoter.

## Caveats

As reported in the paper; not rerun. Number of networks behind each median not stated. Flavell GT is the Flavell lab's algorithm, and neurons split across segments were dropped, which can favour BarlowTrack. Leifer is a single recording. Accuracy is limited to objects the segmentation found (10-40% of head neurons are lost in dense regions, Discussion). Revision-era re-evaluations score lower (see the open question on corrected evaluation).
