---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-171a-75dc-a862-96ced0ce6662
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

Test whether the custom time-aware clustering (seeded label propagation, one label per time point, spectral alignment of seeded labelings; Algorithm 1) improves on off-the-shelf HDBSCAN (after PCA to 50 and UMAP to 10 dimensions) applied to the same trained network.

## Method

Table 2 (median accuracy over networks trained with default settings) and Fig. S1d (per-network accuracy, Baseline = label propagation vs HDBSCAN vs the earlier tracklet method), Zimmer, Flavell and Leifer ground truth. Label propagation: n_neighbors 20, 100 layers, top_k 2, 100 seeds, confidence threshold 0.1.

## Results

Label propagation vs HDBSCAN: Zimmer 97.3% vs 96.7%, Flavell 96.7% vs 94.4%, Leifer 97.4% vs 96.2%. In Fig. S1d the label-propagation networks also spread less. HDBSCAN occasionally marks whole low-density neurons as outliers.

## Conclusion

On identical embeddings, time-aware label propagation adds 0.6-2.3 accuracy points over HDBSCAN across three labs and reduces the variance between networks.

## Caveats

As reported in the paper; not rerun. Few networks per lab (Fig. S1d shows a handful of points); no statistical test. HDBSCAN parameters were not tuned per dataset as far as stated. Post-preprint review (docs/pr_plan/PR02) found edge-case bugs in the label-propagation code; their effect on these numbers is not known.
