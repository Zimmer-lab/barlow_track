---
type: project
title: BarlowTrack
created: 2026-10-22
roots:
  - "[[QUE - How complex is neuron motion in freely moving C. elegans]]"
  - "[[QUE - Which loss and clustering design lets a self-supervised embedding track neurons]]"
  - "[[QUE - How accurately does BarlowTrack track neurons on benchmarks from different labs]]"
  - "[[QUE - Can BarlowTrack be used on a new dataset without ground truth]]"
  - "[[QUE - How much training data and which augmentations does BarlowTrack need]]"
  - "[[QUE - Does BarlowTrack track neurons in conditions that were previously untracked]]"
  - "[[QUE - Can a GUI make whole-brain tracking results interpretable and correctable]]"
---

## Summary

BarlowTrack (Fieseler et al., bioRxiv 2025-11-12, [[@BarlowTrack - A Self-Supervised Framework for Zero-Shot Multi-Object Cell Tracking]]) tracks segmented neurons in freely moving *C. elegans* by embedding 3D crops with a CNN trained by a modified Barlow Twins loss (a new object-space term across all objects of one time point) and associating them across time with time-aware label propagation. Without manual annotation it reaches ≈97% accuracy on ground truth from three labs, the first tracker shown to work across independent labs. This tree digests the preprint figure by figure (code at barlow_track@987eca2, tag v0.1.0) and tracks the revision work since (branch global_augmentation: position encoders, global augmentation, corrected evaluation, sweeps on tracking accuracy).

## Findings

- Neurons move about one to three nearest-neighbour distances between volumes in all three benchmark labs ([[CLM - Neurons move further between volumes than the distance to their nearest neighbour]]), and their motion is high dimensional ([[CLM - Neuron motion in the head is high dimensional]]).
- BarlowTrack tracks 96.7-97.4% of ground-truth neurons in Zimmer, Flavell and Leifer recordings without annotation ([[CLM - BarlowTrack tracks neurons above 96 percent accuracy across labs without annotation]]), above every published pipeline's reported number ([[CLM - BarlowTrack outperforms published tracking pipelines on their benchmarks]]), while the authors' earlier tracklet method collapses on Flavell data ([[CLM - The earlier Zimmer tracklet method fails on Flavell lab data]]).
- Time-aware label propagation beats HDBSCAN on the same embeddings ([[CLM - Time-aware label propagation tracks more accurately than HDBSCAN]]) and already tracks most neurons on random-CNN features for some datasets ([[CLM - Label propagation on an untrained encoder already tracks most neurons on some datasets]]).
- The new object-space loss term can replace the original Barlow Twins term but does not improve accuracy ([[CLM - The object-space loss term can replace the redundancy-reduction term without loss of accuracy]]).
- Training needs a few percent of a recording ([[CLM - Accuracy needs a few percent of a recording as training volumes]]); geometric augmentations carry the signal ([[CLM - Affine and flip augmentations carry most of the training signal]]) and no single augmentation is necessary ([[CLM - No single augmentation is necessary for accurate tracking]]); crop size is the hyperparameter that matters ([[CLM - Crops must cover the neuron neighbourhood in physical units]]).
- Trained networks transfer across recordings and labs ([[CLM - BarlowTrack networks generalize across recordings and labs]]); validation loss ([[CLM - Lower validation loss predicts higher tracking accuracy]], opposed by revision sweeps) and track coverage ([[CLM - Track coverage is a rough proxy for tracking accuracy without ground truth]]) are weak proxies without ground truth.
- Fast animals, 63x magnification, L3 larvae and altered reference markers are tracked with high coverage, without ground truth ([[CLM - BarlowTrack tracks fast-moving animals]], [[CLM - BarlowTrack tracks recordings at higher magnification]], [[CLM - BarlowTrack tracks L3 larvae]], [[CLM - BarlowTrack tracks strains with altered reference markers]]).
- The GUI speeds neuron identification ([[CLM - Activity correlation overlays speed up neuron identification]]) and tracklet-based correction ([[CLM - Tracklet-based correction produces ground truth faster than per-timepoint correction]]).

- Revision: on Flavell, today's code reproduces the preprint's accuracy ([[CLM - Current training and evaluation code reproduce the preprint's Flavell accuracy]]), and the revision networks' low Flavell scores were mostly an evaluation bug and a ground-truth mismatch ([[CLM - Most of the revision networks' Flavell deficit came from the evaluation, not the networks]]); the remaining 3-point gap may come from the position pipeline's augmentation ([[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]], low confidence, under test).
- Revision: with the ViSNet RBF fix, a network on neuron centroids alone tracks within a few points of the best image or fused network on Zimmer, Flavell and Samuel ([[CLM - A position-only network tracks within a few points of the best image or fused network]], low confidence: single seeds), helped by keypoint jitter sized from the segmentation's own noise; synthetic segmentation errors, bending and real frame pairs did not help ([[EVD - Synthetic geometry and detection-error augmentations do not help position-only networks]]).

Open for the revision (high priority): whether the preprint accuracies hold under the corrected code on Zimmer and Leifer ([[QUE - Do the preprint accuracies hold under the corrected evaluation and tracking code]]; Flavell holds), whether the object-space term helps anywhere ([[QUE - Does the object-space loss term improve tracking in any regime]]), a same-input comparison with other trackers ([[QUE - Does BarlowTrack outperform other trackers run on the same data and metric]]), and position information ([[QUE - Does adding neuron position information improve tracking]]).
