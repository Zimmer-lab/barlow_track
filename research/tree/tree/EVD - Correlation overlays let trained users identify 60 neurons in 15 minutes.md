---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a3-63a6-7c25-8670-4b76cb9aba6c
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

Support neuron identification without NeuroPAL, using position plus activity.

## Method

Fig. 5: napari GUI on NWB data with raw, segmentation, tracking and ID layers (a-c) and a recolouring of the segmentation by correlation with an anchor neuron (d-f): separated but correlated pharyngeal neurons, left-right pairs, a widely distributed signal.

## Results

For trained users, the set of ≈60 neurons routinely identified in the Zimmer lab takes about 15 minutes, previously several hours.

## Conclusion

Correlation overlays cut routine identification of ≈60 neurons from hours to about 15 minutes for trained users.

## Caveats

As reported in the paper; not rerun. Anecdotal timing, no user study, no accuracy of the identifications. The paper itself warns that correlated neighbours need NeuroPAL or single-neuron confirmation. The GUI lives in the wbfm repository; not linked.
