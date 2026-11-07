# Methods draft: time-aware clustering by label propagation (revision)

Draft for the revision of Fieseler et al., *BarlowTrack: A Self-Supervised Framework for Zero-Shot
Multi-Object Cell Tracking*, bioRxiv 10.1101/2025.11.10.687595 (v1, 2025-11-12; for correspondence:
charles.fieseler@univie.ac.at). It replaces the Label Propagation part of Methods 4.3.2 ("Clustering"),
updates Algorithm 1 ("Time-aware Clustering"), and adds a paragraph on how the settings were chosen. The
main-text description (Results 2.3.2, "Tracking via clustering") still holds and needs only the new
references; a suggested sentence is at the end.

What the preprint says now (4.3.2): the embedding is 64-dimensional and is clustered directly; *"Label
Propagation (step 1: graph formation): n_neighbors=20; (step 2: label propagation): n_layers=100, top
k=2, num_seeds=100; (step 3: label alignment): confidence_threshold=0.1"*. No dimensionality reduction,
edge weighting, seed placement or per-run uniqueness step is described.

Code: `barlow_track/utils/labelprop_recipes.py` (recipe `v2`, the library default since 8903472),
`utils_label_propagation.py`, `utils_spectral_relabeling.py`,
`utils_tracking.WormClusterTracker.track_using_label_propagation_clusterer`. Evidence: research tree,
*EVD - Confirmation of screened label propagation variants on held-out recordings* and the notes it
links to.

---

## 4.3.2 Clustering (Label Propagation; proposed text)

**Overview.** Tracking requires assigning every detected object to one neuron identity across the video.
We cast this as clustering the per-object embeddings under the constraint that the objects of one time
point (volume) are all distinct. Labels are propagated over a nearest-neighbour graph of the embeddings,
starting from the objects of one volume (a *seed volume*), which gives one labelling of the whole video
per seed volume. Because the label indices of different seed volumes are unrelated, the labellings are
then aligned to each other by a time-aware spectral synchronization and combined into a consensus, from
which each volume's objects receive distinct final labels (Algorithm 1).

**Dimensionality reduction.** The embeddings of all objects of the video (N objects × D dimensions) are
reduced to their first 150 singular components with a truncated SVD, keeping the singular-value
scaling (U·Σ). This removes noise dimensions of the learned space while preserving its relative
distances, and makes the nearest-neighbour search cheaper. (For videos of at most 500 volumes, N_t/10
components are kept.)

**Nearest-neighbour graph.** We build a directed k-nearest-neighbour graph (k = 20, Euclidean distance,
approximate search with NNDescent) on the reduced embeddings. Each edge i → j is weighted with a
self-tuning Gaussian kernel, w_ij = exp(−d_ij² / (σ_i σ_j)), where σ_i is the distance from i to its 7th
nearest neighbour (local scaling; Zelnik-Manor and Perona, 2004). Near neighbours therefore count more
than distant ones, while dense and sparse regions of the embedding are treated alike. A self-loop of
weight 1 is added to every node, and each row is normalised to sum to one, so that one propagation step
replaces a node's label scores by the weighted mean over its neighbourhood.

**Seeded, clamped label propagation.** For each of S = 50 seed volumes, every object of that volume
receives its own label (one-hot), and the scores are propagated for 100 steps, H ← W H, re-imposing
("clamping") the seed objects' one-hot rows after every step. Clamping, rather than the more common
averaging with the initial labels, keeps the information of these very sparse seeds from being diluted.
The seed volumes are stratified over the recording: the volumes are split into S consecutive blocks and
one volume is drawn at random from each, so that every part of the video is near a seed. After
propagation, the scores of each object are normalised to sum to one, and each object keeps its two
highest-scoring candidate labels (top k = 2); candidates with a score below 1.1/n_s, where n_s is the
number of objects in the seed volume, are discarded as chance level.

**One label per object and volume within each labelling.** Within each labelling, the objects of a
volume must carry distinct labels. For every volume, objects and their candidate labels form a bipartite
graph weighted by the propagated scores, and an optimal assignment (Hungarian algorithm) gives each
object at most one label and each label at most one object, maximising the total score. Candidates with
a score below the confidence threshold of 0.1 are removed first.

**Aligning the labellings (time-aware spectral synchronization).** Each labelling r is written as a
sparse N × K matrix A_r (row-normalised assignment scores; K is the largest number of labels of any
labelling plus 10). The matrices are concatenated column-wise, S = [A_1 | … | A_R], each column is
weighted by the inverse square root of its total mass, and the top-K right singular vectors of S are
computed (randomized SVD). Labels of different runs that mark the same objects have similar singular
vectors. Because the singular subspace is only defined up to a rotation, we first rotate it so that K
well-separated run labels (chosen by column-pivoted QR) lie on the coordinate axes (CPQR rounding;
Damle, Minden and Ying, 2019). Each run's block of the rotated vectors is then rounded to a partial
permutation from its labels to the K consensus labels (Hungarian assignment; a run label with a weak
best match maps to none). These permutations are refined by alternating re-alignment: each run is
re-assigned (Hungarian) to maximise its overlap with the consensus of all other runs, which cannot
decrease the total agreement, until no permutation changes (at most 10 sweeps).

**Final labels.** The aligned labellings are summed into a consensus score matrix (N × K). For each
volume, an optimal assignment of objects to consensus labels (Hungarian algorithm, with an "unlabelled"
option worth 2 % of the runs) gives distinct final labels; an object's confidence is its consensus score
divided by the number of runs, and objects below 0.02 are left unlabelled. Each final label is one
tracked neuron.

**Parameters.**

| Step | Preprint | Revision |
|---|---|---|
| Dimensionality reduction | none (64-d embedding) | truncated SVD, 150 components, U·Σ |
| Graph | kNN, k = 20, uniform edge weights | kNN, k = 20, self-tuning Gaussian weights (σ: 7th neighbour) |
| Seed volumes | 100, random | 50, stratified over the recording |
| Propagation | 100 steps, clamped, top k = 2 | unchanged |
| Uniqueness within a labelling | greedy, highest score first | optimal assignment (Hungarian) |
| Alignment | spectral ensemble clustering, confidence threshold 0.1 | spectral synchronization with CPQR rounding and alternating refinement; threshold 0.1 |
| Final labels | one label per object and volume | unchanged (Hungarian, minimum confidence 0.02) |

**Choice of the settings.** Starting from the preprint's tracker, we screened 35 single changes to every
stage (dimensionality reduction, graph construction, seed selection, propagation, alignment and
post-processing) on four development recordings (one network each from the Zimmer, Flavell and two
Samuel recordings), at three random seeds per change, and kept the changes that improved accuracy by at
least 0.3 points on average and in at least 75 % of recording-and-seed pairs. Combinations of the kept
changes were then searched with Bayesian optimisation (Ax; 45 trials, mean accuracy over the four
recordings) and confirmed on six held-out cases at three seeds each: networks not used for selection,
three of them on recordings not used either (a second Flavell recording and the Leifer recording in its
original and its StarDist segmentation). A change was adopted
if it raised accuracy by at least 0.3 points on average (paired t-test p < 0.05) and lowered no
recording by more than 0.5 points. The adopted settings raised accuracy on all six held-out cases
(mean +0.88 points, 16 of 18 recording-and-seed pairs; by recording +0.11 to +2.33 points). Over 100
random seeds (seed volumes and kNN graph), accuracy rose from 0.937 to 0.940 on the Leifer StarDist
recording (higher at 93 of 99 seeds) and from 0.939 to 0.948 on Zimmer 2022-12-10 (88 of 100 seeds),
where the seed-to-seed standard deviation fell from 1.30 to 0.17 points. Two changes that helped on the
development recordings were not adopted: edges between mutual nearest neighbours of consecutive volumes
cost a held-out Flavell recording 0.4 points (as label swaps), and smoothing the consensus along such
links gave less than 0.2 points.

---

## Algorithm 1: Time-aware clustering (proposed)

```
Input:  embeddings X (N × D) of all objects; volume t(i) of each object i
Params: d = 150, k = 20, S = 50 seed volumes, L = 100 steps, top-k = 2, τ = 0.1, τ_final = 0.02

1  Z ← first d SVD components of X (U·Σ)
2  G ← kNN graph of Z (k neighbours); w_ij ← exp(−‖z_i − z_j‖² / (σ_i σ_j)), σ_i = distance to 7th neighbour
3  W ← row-normalise(G + I)
4  T ← one volume drawn at random from each of S consecutive blocks of volumes
5  for each seed volume s in T:
6      Y_s ← one-hot label for each object of volume s;  H ← Y_s
7      repeat L times:  H ← W H;  rows of the seed objects ← Y_s          (clamped propagation)
8      keep each object's top-k labels with score ≥ 1.1 / |objects in s|
9      in every volume, assign objects to labels one-to-one (Hungarian), drop scores < τ  → A_s (N × K)
10 S ← [A_1 | … | A_R] with column weights 1/√(column mass);  V ← top-K right singular vectors of S
11 rotate V to anchor rows (column-pivoted QR);  round each run's block to a partial permutation P_r
12 refine: repeat until no P_r changes (≤ 10×): P_r ← Hungarian(A_rᵀ Σ_{q≠r} A_q P_q)
13 C ← Σ_r A_r P_r                                                          (consensus, N × K)
14 in every volume, assign objects to consensus labels one-to-one (Hungarian, unlabelled option τ_final · R)
15 return label of each object (unlabelled if C_i,label / R < τ_final), confidence C_i,label / R
```

---

## Suggested main-text sentence (Results 2.3.2)

After "for this, we use a custom time-aware implementation of spectral ensemble clustering":
*"…in which the labellings are aligned by spectral synchronization with pivoted-QR rounding (Damle et al.,
2019) and refined by alternating Hungarian re-alignment; the graph uses self-tuning Gaussian edge weights
(Zelnik-Manor and Perona, 2004) and the seed volumes are spread evenly over the recording (Methods
4.3.2)."*

## Notes for the author (not for the paper)

- **References to add:** L. Zelnik-Manor, P. Perona, "Self-tuning spectral clustering", NeurIPS 2004;
  A. Damle, V. Minden, L. Ying, "Simple, direct and efficient multi-way spectral clustering", Information
  and Inference 8(1), 2019. Check the exact citations in Zotero (it was not reachable from this session).
  Label propagation itself: Zhu and Ghahramani (2002) is presumably already cited.
- **Numbers to recompute before submission:** the benchmark tables were scored with the preprint tracker
  (`eval_accuracy --labelprop legacy`, the default for recorded scores). Re-scoring with `--labelprop v2`
  writes a separate record set (the benchmark driver keeps the two apart). The gains quoted above are from
  the label-propagation harness on fixed embeddings, not from the benchmark tables.
- **Which networks:** the confirmation used one network per held-out case; the "Choice of the settings"
  paragraph should name them, or move to the supplement with the per-recording table (research tree note
  above).
- **Leifer:** the gain on the paper's Leifer (StarDist re-segmentation) is small (+0.2 to +0.3 points)
  because most of its remaining error is manual clicks on the boundary between touching masks, not
  tracker error (open question a261107-1: strict-click only, or also a boundary-tolerant score?).
- **Preprint wording to fix:** "Psuedocode" → "Pseudocode" (Results 2.3.2).
- **Embedding dimension:** 4.3.1/4.3.2 say the embedding is 64-dimensional and clustered directly; the
  revision networks use larger embeddings (256 to 1024) reduced by the SVD above. Update 4.3.2's first
  paragraph accordingly once the final architecture is fixed.
- **HDBSCAN paragraph** (PCA to 50, UMAP, HDBSCAN) is unchanged by this.
