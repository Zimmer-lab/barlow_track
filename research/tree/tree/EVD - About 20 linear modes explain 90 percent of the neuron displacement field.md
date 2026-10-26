---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a1-7a21-7e8f-b47f-265cc11d9faf
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

Measure the dimensionality of the neuron displacement field, in contrast to body posture where 4 PCA modes explain ≈95% of variance.

## Method

Fig. 1e: probabilistic PCA (handles missing ground-truth points) on object displacement vector fields over time; variance explained and single-step "object reconstruction accuracy" (fraction of objects whose reconstruction error is below half the nearest-neighbour distance, Methods 4.2) as a function of the number of modes.

## Results

≈20 modes for 90% variance explained and 90% single-step tracking accuracy; ≈50 modes for 95% accuracy.

## Conclusion

Neuron motion is structured but high dimensional: a linear motion model needs about 20 modes for 90% and about 50 for 95% next-step identification, far more than body posture needs.

## Caveats

As reported in the paper; not rerun. The figure appears to use Zimmer ground truth; the number of recordings is not stated. No script for Fig. 1 was found at barlow_track@987eca2. The "1d red line" call-out in the text points to the wrong panel (co-author markup); the PPCA plot is panel e.
