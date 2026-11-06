# Label-propagation tracker: accuracy plan (2026-11-06)

Scope: improve tracking accuracy of `WormClusterTracker.track_using_label_propagation_clusterer` by
re-tracking cached sweep embeddings (no retraining), within the current runtime and memory. Nothing here
changes library code. Files in this folder:

| file | what |
|---|---|
| `lp_core.py` | the tracker split into stages (embed, graph, prop, sync, post), each a function of one flat variant dict (`DEFAULTS`). With default values every stage calls the library code, so `base` is the tracker. Other values switch to local implementations. |
| `sweep_labelprop.py` | harness: loads a case's embedding, project and GT once, memoises the stages, scores each (case, variant, RNG seed) as the sweep does, and writes one JSON line per triple (resumable) |
| `variants.py` | variant batches (`batch1` screen, `speed`, `batch2` placeholder) |
| `analyze.py` | paired deltas against `base` per (case, seed), with SE, wins, worst case, p-value and a verdict |
| `cases.sh`, `run_local.sh`, `labelprop_opt.sbatch` | case lists and launchers (local, Slurm array) |

## 0. Summary

* **The harness reproduces the sweep.**
  * On the full S137 t63 video (CPU), `base` scores 0.9333 with agreement 0.666, against the recorded 0.9338
    (compare_sync cpqr_refine, GPU) and the reference agreement 0.6654.
  * On 100-150-volume smokes of S137, Zimmer 1210 (project segmentation), Flavell 0330-02 and the Leifer
    re-export (NWB, every detection), the harness labels equal `track_using_label_propagation_clusterer`'s
    exactly (1.000000), with the same official accuracy. All four scoring paths are covered.
  * At full size, `local` (local graph and propagation code) and `batch5` give the same accuracy as `base`.
* **Re-tracking is cheap. CPU is enough, but run on the GPU now that one is available** (TITAN V, 12 GB,
  about 10.5 GB free on 2026-11-06). On the GPU, propagation drops to about 2.5 s per seed at 190k objects.
  Graphs, syncs and scoring stay on the CPU and become the cost (section 5). S137 t63, 85k objects,
  16 CPU threads:
  * SVD 1.7 s; graph 24 s (NNDescent takes 6 s, the rest is its numba JIT);
  * 25-seed propagation 19.6 s (0.8 s per seed with the CSR adjacency); sync 8 s; scoring 0.5 s.
  * Scoring is 5-11 s on Zimmer: `add_metadata_to_df_raw_ind`.
  * Loading the Zimmer GT project takes about 4.5 min, once per case.
  * The old 102 s/seed CPU figure predates CSR.
* **RNG noise comes from the choice of seed volumes, not the graph.** NNDescent recall against exact kNN is
  0.998, and graphs at RNG seeds 0 and 1 share 99.7% of their edges (S137 t63). So the ~0.8 pt
  re-tracking spread is almost all the 25 seed volumes. Seed selection (`seed_complete`, `seed_strat`,
  `runw`, `drop20`, more seeds) is therefore both an accuracy lever and a variance lever. Exact kNN will
  not reduce the variance.
* **Single-seed full-size hints (S137 t63, not significant):** `layers30` +0.14 pt, `layers50` +0.15 pt,
  `tsmooth05` +0.24 pt, `runw` +0.03, `kextra40` -0.01. Fewer layers are at least not worse and 2-3x
  faster.
* **Where the errors are** (base, error decomposition by GT-optimal matching; Samuel is full video, the
  others are the first 100 volumes):

  | case | accuracy | swapped | on unmatched ("fragment") label | unlabelled | perfect-merge bound |
  |---|---|---|---|---|---|
  | S137 t63 (GT objects only) | .933 | 5.8 | 0.0 | 0.8 | - |
  | Zimmer 1210 t36 | .951 | 2.5 | 2.4 | 0.1 | not measured |
  | Flavell 0330-02 t61 | .920 | 2.9 | 1.9 | 2.6 | .925 -> .941 |
  | Leifer t57 | .840 | 2.2 | **9.7** | 3.4 | .847 -> .936 |

  Error columns are pt of GT cells. On recordings with distractors, a large share of the errors are GT
  objects whose track label is matched to no GT neuron: a track split into pieces over time, or a track
  that wanders between a neuron and distractors. **Track merging, or anything else that reduces
  fragmentation, is the largest single lever on Leifer, and a real one on Zimmer and Flavell.** The bound
  is optimistic: every label goes to its majority neuron, and 100-volume labels are small. Re-measure it at
  full length on batch 1's base rows (`acc_merge_bound`) before building the merge step. Records now split
  `err_fragment` into `err_frag_split` (a piece of the object's own neuron; merging fixes it),
  `err_frag_distractor` (a label made mostly of non-GT objects; merging does not fix it) and
  `err_frag_other`. Section 3a explains why the split matters.
* **Highest-value accuracy candidates.** These are hypotheses, ranked by expected effect over cost; the
  100-150-volume smokes are one seed each and say nothing about sign. For example, `mutual` was -7.4 pt on
  Zimmer, +1.4 on Flavell and -0.6 on Leifer, so effects are probably recording-dependent and need the
  4-lab screen.
  0. fragment reduction and merging (see above), via `kextra40`, `reprop*` and `runw`/`drop20` now, and a
     merge step later (batch 3);
  1. the graph: k (k=20 includes the object itself, so it has only 19 real neighbours) and edge weighting;
  2. the depth of propagation (`num_layers`). Hard-clamped propagation for 100 layers approaches the harmonic
     solution and may over-smooth; 30 layers is also 3x faster;
  3. re-propagation from confident consensus labels (seeds in every volume, not one);
  4. a soft cannot-link inside each volume during propagation (`frame_norm`);
  5. seed-volume choice: about 10% of Samuel 137 volumes have under 40% of the objects, and every run seeded
     there starts with few labels;
  6. PCA centring before the SVD, because TruncatedSVD does not centre.
* **First batch.** `batch1` is 33 one-change variants plus `base` on 4 development cases x 2 RNG seeds. It
  takes about 6-8 h on 16 CPU threads, about half that with 3 processes in parallel on this 56-core machine,
  or 8 Slurm array tasks of about 1 h each. The confirmation batch (section 5) costs about the same.

## 1. What the tracker does (verified in code, HEAD d782242)

`eval_accuracy.run_eval` -> `TruncatedSVD(50, random_state=seed)` on the raw embedding (uncentred; `U*S`,
or `U` with `--svd whiten`) -> `WormClusterTracker(..., seed)` ->
`track_using_label_propagation_clusterer(num_seeds=25, device)`:

1. `select_seed_times`: a seeded shuffle of the time keys (volumes with < 2 objects are not keys), taking the
   first 25.
2. `multi_seed_propagation`: `build_knn_graph` = `NNDescent(X, n_neighbors=20, euclidean, random_state=seed)`.
   `neighbor_graph` includes each point itself (measured: `self_in_knn` = 1.0), so each node has 19 real
   neighbours. The graph is directed: row i lists i's neighbours. `normalized_adjacency` adds a second self
   loop and computes `deg` over rows, which is k+1 for every node, so the "symmetric" normalisation is a
   uniform 1/(k+1) per edge. The result is a row mean over i's own neighbours, with self weight 2/(k+1).
   In-degree (hubness) plays no part. The adjacency is one CSR shared by every seed.
3. Per seed volume: one-hot labels for its objects (C = objects in that volume), `H <- A H` for 100 layers
   with hard clamping of the seed rows, `probs = H / rowsum` (`softmax=False`, so `tau` is unused), top-2
   labels, and labels below 1.1/C are dropped.
4. `spectral_sync_from_topk(input_probability_threshold=0.1)`:
   * each run is reduced to top-1 with a greedy one-label-per-volume pass (`greedy_top1_timewise`);
   * the rows are normalised, so every kept entry becomes 1.0 and the probabilities only set the greedy
     order;
   * column weighting (inverse square root of the count), then a randomized SVD with `K_dim = K` = largest
     label + 10, `n_iter=2`;
   * CPQR anchor rotation, Hungarian rounding per run, and alternating Hungarian refinement (up to 10
     sweeps; converges in 2-3);
   * consensus = the sum of mapped indicators;
   * `enforce_temporal_uniqueness_hungarian`: one object per label per volume, with confidence = the
     fraction of runs.
5. `cluster_obj2dataframe`, then the eval scores it: `match_columns_to_gt` ('nandist') and
   `calculate_accuracy`. Every detection is tracked on Zimmer (project segmentation), Flavell and Leifer
   (NWB, every detection). Samuel NWB masks are the GT neurons only, so Samuel has no distractors and its
   errors are identity errors.

## 2. Bugs, traps and inefficiencies found

Correctness and semantics (none changes current numbers on its own):

1. **The kNN list includes the point itself** (NNDescent on its own training data), so `k=20` means 19
   neighbours, and `add_self_loops` doubles the self weight. This is a semantic trap rather than a crash:
   `k` sweeps and comparisons with "k=20" in the paper should say so.
2. **`normalized_adjacency` is not symmetric normalisation on this graph.** The graph is directed with a
   constant out-degree, so `deg^-1/2[row] deg^-1/2[col]` = 1/(k+1) everywhere. The docstring and the
   commented-out D^-1 alternative suggest otherwise. Hubs pull no differently from other nodes. Variant
   `sym` tests a true D^-1/2 (A+A^T) D^-1/2.
3. **Dead parameters.**
   * `run_label_propagation(alpha=0.95)` is never used: propagation is hard-clamped with no label spreading.
   * `tau` only applies when `softmax=True`, and the tracker passes `softmax=False`.
   * `aggregate_consensus(normalize_rows=...)` is unused.
   * The tracker's `return_top_k=2` only feeds the greedy fallback.
4. **The sync discards the propagated probabilities**: `prepare_A_list...` with `normalize_rows=True` after
   top-1 sets every entry to 1. This was already noted in the notebook (3183). `probw` tests keeping them.
5. **Inputs below the 0.1 threshold are zeroed but not removed.** In `greedy_top1_timewise` they still
   occupy a label slot and an object slot at value 0. `A.getnnz` (the `obj_counts` diagnostics in
   `spectral_sync_from_topk`) then counts these explicit zeros as "labelled". This affects diagnostics only.
6. **The consensus label space is capped** at `K = max seed-volume label + 10`. If the video has more
   distinct tracks than the largest seed volume has objects (+10), the excess cannot get a consensus label.
   This is mostly latent: the largest seed volume usually has the most objects. `kextra40` tests it.
7. **`randomized_svd(n_components=K, n_oversamples=20, n_iter=2)`** is loose for the small-label tail. The
   refinement now absorbs that, which is part of why it helps weak networks most.
8. Label names are `int2name_neuron(label + 1)` with labels already 1-based. This is harmless, but names
   are off by one against the label ids.
9. `select_seed_times` can pick volumes with few objects. On Samuel 137, objects per volume have p10 = 36
   and median 83 (GT objects). Runs seeded there have about 40% of the labels, and the per-run agreement
   p10 on the smoke was 0.16.

Inefficiencies:

10. **NNDescent's numba JIT costs about 20 s on every new process.** First call: 25.8 s on 85k objects
    (20.2 s on 10k). A second call in the same process: 6.3 s. A persistent `NUMBA_CACHE_DIR` did not
    remove it (same cache dir across my runs), so in a sweep every eval pays it. Exact chunked kNN
    (`knn_impl='exact'`) is deterministic and needs no JIT. It takes 18 s on CPU at 85k objects and grows
    as N^2, so about 3 min at 260k on CPU but seconds on GPU. Use it on GPU evals only.
11. **One SpMM per seed and layer.** Propagating several seeds as one wide right-hand side
    (`batch_seeds`) gives the same result (checked: same accuracy as `base` at full size). On CPU it is
    *slower*: 34.8 s against 19.6 s for 25 seeds (the Python per-block clamp, and CSR SpMM on CPU already
    saturates). It may still help on GPU, where SpMM is memory-bound; time it with the `speed` batch on GPU
    before patching.
12. **100 fixed layers with no convergence test.** If `layers30` is not worse, propagation costs 3.3x less.
13. **`greedy_top1_timewise` is a Python loop over every object of every run** (25 x N). Together with
    the per-volume Python loop in `enforce_temporal_uniqueness_hungarian`, the sync takes about 8 s on 85k
    objects and should take about 25 s on 260k. It could be vectorised per volume.
14. `cluster_obj2dataframe` runs a Python loop per object. This is minor (0.3 s at 85k objects).
15. compare_sync re-runs `evaluate_trained_checkpoint` (project, GT and embedding load) for each variant;
    the new harness does that once per case.

## 3. Map of tunable choices

Columns: current value, hypothesis, expected effect on accuracy (pt), cost, and where the code would
change. "Variant" is the name in `variants.py`; "lib" means a library change is needed for production use.
Every row is implemented in `lp_core.py`, so it can be tested without changing the library.

| stage | knob (variant) | now | hypothesis | effect | cost | code |
|---|---|---|---|---|---|---|
| embed | whitening (`whiten`) | U*S | Equalising dimensions changes neighbours. Notebook 205: U was +0.8 pt over U*S on the preprint Flavell net (U*S is the eval default; `--svd whiten` exists). | ±1 | 0 | eval `--svd` exists |
| embed | PCA centring (`pca`) | none | Uncentred SVD spends component 1 on the mean direction (s1 >> s2 on the paper net), so norm variation enters the distances. | 0 to +0.5 | 0 | eval / `utils_tracking` |
| embed | dims (`svd100`), L2 rows (`l2`) | 50 | More dims keep finer identity cues; L2 removes brightness/norm effects. | ±0.5 | 0 | eval |
| graph | k (`k10`, `k40`) | 20 (19 real) | Lower k gives purer edges. Smoke edge purity: k10 0.927, k20 0.908, k40 0.877. Too low fragments the graph. Probably the largest single knob. | ±1 | 0 | `num_neighbors` arg exists |
| graph | symmetric / mutual (`sym`, `mutual`) | directed | Mutual kNN cuts hub edges into wrong clusters; sym lets well-connected nodes speak. | ±0.5 | small | `build_knn_graph`, `normalized_adjacency` (lib) |
| graph | cannot-link (`nosame`) | none | Edges inside one volume join two different neurons. Only 0.7% of Samuel edges; likely more with distractors and split masks. | 0 to +0.3 | NNDescent with 3k | lib |
| graph | edge weights (`gauss`) | uniform | Self-tuning Gaussian weights downweight far "neighbours" of isolated or distractor objects. | 0 to +0.5 | 0 | lib |
| graph | exact kNN (`exact`) | NNDescent | Removes one of the three noise sources (graph) and the JIT; exact neighbours are slightly purer. | 0 to +0.2, less variance | GPU: s | lib |
| graph | temporal edges (`temporal1/5`) | none | Mutual-NN links between adjacent volumes. Frame-pair precision is 0.96-0.98 (Zimmer, Samuel) but 0.82 (Flavell), notebook 1376. | ±0.5 | small | lib |
| seeds | count (`seeds50`) | 25 | Measured neutral at 100 on S137 (+0.3/+0.5 pt vs the 25-seed mean); skip unless cheap. | ~0 | 2x | arg exists |
| seeds | choice (`seed_strat`, `seed_complete`) | random | Seeds from volumes with at least the median object count start with nearly all labels; stratified seeds cover posture and time. | 0 to +0.5 | 0 | `select_seed_times` (lib) |
| prop | layers (`layers30`, `layers300`) | 100 | Early stopping regularises: long clamped propagation drifts toward the harmonic solution, where big clusters bleed. | ±0.5; 3.3x faster at 30 | -/+ | arg exists |
| prop | label spreading (`spread99`) | hard clamp | Soft clamping tolerates seeds that are mislabelled in their own volume. | ±0.5 | 0 | `alpha` already plumbed, unused (lib) |
| prop | class-mass norm (`classnorm`) | none | Corrects labels whose mass spreads widely (Zhu et al. 2003). | 0 to +0.3 | 0 | lib |
| prop | soft cannot-link (`framenorm`) | none | Each label's mass within a volume is capped at 1 after every layer, so one seed cannot claim two objects of one volume. This directly attacks swaps. | 0 to +1 | +30% prop | lib |
| sync | probability-weighted input (`probw`) | 1.0 per entry | Uses propagation confidence in the consensus. | ±0.5 | 0 | `normalize_input_rows` arg exists (not plumbed) |
| sync | input threshold (`thr0`, `thr03`) | 0.1 | Fewer or more low-confidence votes. | ±0.3 | 0 | tracker hard-codes 0.1 |
| sync | per-run uniqueness (`hungrun`) | greedy | Optimal assignment inside each run's volume. | 0 to +0.2 | sync +50% | lib |
| sync | label space (`kextra40`) | +10 | Room for tracks that are absent from the biggest seed volume. | ~0 | 0 | lib |
| sync | run weighting (`runw`, `drop20`) | equal | Downweights runs from bad seed volumes (agreement p10 is 0.16). | 0 to +0.3 | 0 | `doweight_runs` exists (not plumbed) |
| post | temporal smoothing (`tsmooth05/2`) | none | Adds adjacent-volume MNN consensus before the per-volume Hungarian, to fix isolated flips. | 0 to +0.3 | 1 s | lib |
| post | re-propagation (`reprop09/07`, `reprop09_w4`) | none | One extra propagation seeded by every confidently labelled object in every volume, mixed into the consensus. Low-confidence objects are then labelled from nearby confident ones rather than from 25 distant seed volumes. | 0 to +1 (weak nets most) | 1 prop | lib |
| post | fragment merging | none | `err_fragment` is 1.9-9.7 pt on distractor recordings (100-volume smokes); perfect merging bound +1.6 (Flavell) / +8.9 pt (Leifer). Merge consensus labels whose time supports barely overlap and whose embedding centroids (or adjacent-volume MNN links) are close, using a Hungarian or greedy on a label x label affinity. Not implemented: build it in batch 3 if the full-length bound in batch 1 holds up (> 0.5 pt). | 0 to several pt (distractor data) | small | lib |

### 3a. Fragmentation: where it comes from, and whether seeds help

"Fragment" errors are GT objects whose final label is matched to no GT neuron. The tracker can produce
one in three ways, and they need different fixes:

1. **A neuron is absent from most seed volumes** (undetected, merged into a neighbouring mask, or out
   of the field of view). Runs seeded in those volumes have no label for it. They give its objects a
   neighbour's or a distractor's label, and the consensus label for the neuron rests on the few runs that
   saw it. *More seeds and better seeds help directly here:* `seeds50`, `seeds100`, and `seed_complete`
   (only volumes with at least the median object count). This is the only cause that seed count targets.
2. **The sync splits one neuron into two consensus labels.** If the embedding drifts over the video (for
   example with posture), runs seeded early and runs seeded late can label the neuron consistently within
   their own half and inconsistently across halves. Each group then forms its own consensus label; the
   per-volume Hungarian picks one or the other, and the neuron is split in time (`err_frag_split`). More
   runs make each group's label more stable but do not bridge the two. *More varied seeds*
   (`seed_strat`, one per time bin) put runs in every regime, which fixes coverage but can make this kind of
   split worse unless something links the regimes. What links them: temporal edges (`temporal1/5`, labels
   flow along time and not only through embedding similarity), re-propagation from confident consensus
   labels (`reprop*`, every volume becomes a seed), and an explicit merge of labels whose time supports
   barely overlap (batch 3).
3. **Distractor tracks.** Seed volumes contain distractors, and each gets a label. A distractor has no
   stable identity, so its label wanders and, in some volumes, takes a GT neuron (`err_frag_distractor`).
   More seeds give more distractor labels, not fewer. What helps: the soft cannot-link `framenorm` (a
   label cannot hold two objects of one volume during propagation), `thr03`, and `runw`/`drop20`. After
   the sync, labels that are mostly low-confidence across volumes could be dropped (not yet a variant).

The existing seed-count measurement (100 seeds neutral on S137) **says nothing about fragmentation**:
Samuel tracks GT objects only, and its `err_fragment` is 0. Judge seed variants on Zimmer and Flavell,
and on the fragment columns as well as accuracy:
`analyze.py ... --metric err_frag_split --cases zimmer flavell` (for `err_*` metrics, lower is better, so
ignore the verdict column). Whichever cause dominates decides the next step:
* `seeds100` or `seed_complete` cuts `err_fragment` on Zimmer/Flavell: cause 1 is real, and seeds are the
  cheap fix.
* mostly `err_frag_split`, and seeds do not move it: cause 2, so the temporal and re-propagation
  variants, then the batch 3 merge.
* mostly `err_frag_distractor`: cause 3, so cannot-link and label pruning, not merging.

Each record carries `err_unlabelled` / `err_swapped` / `err_fragment` / `err_no_detection` (fast
GT-optimal matching; `acc_fast` is its accuracy), plus `acc_merge_bound` (every label assigned to its
majority neuron) and `n_labels` / `n_unmatched_labels`. Batch 1's base rows therefore also say where the errors
are, which bounds each post-processing idea: fragment merging cannot gain more than `err_fragment`, and
thresholds no more than `err_unlabelled`. `err_no_detection` is out of the tracker's reach. See the table
in section 0: on Samuel almost all errors are swaps; on distractor recordings, fragments and unlabelled
objects make up half of the errors or more.

## 4. Noise and decision rules

* Noise: the same network re-tracked with RNG seeds 0/1/2 (SVD, seed volumes and graph all change) spreads
  about 0.8 pt (S137 base 0.920/0.927/0.927). Changing only the sync on fixed propagation moved results by
  ±0.3 pt. Pairing the variant and `base` at the same RNG seed removes the shared part of that noise for
  late-stage variants (sync, post) but less for graph or seed variants, which re-randomise propagation.
  analyze.py reports each case's base seed range as the yardstick.
* Screen (batch 1): 4 cases x 2 seeds = 8 pairs. With a per-pair delta SD of about 0.5-0.6 pt, the SE is
  about 0.2 pt. **Promote** if the mean delta is >= +0.3 pt and the variant wins >= 6/8 pairs (analyze.py
  `promote`). **Drop** if the mean is <= -0.3 pt. A variant that is equal in accuracy and faster
  (`layers30`, `exact`, `batch5`) is kept as a speed win if its mean is >= -0.1 pt.
* Confirmation (batch 2): the promoted variants, singly and combined, plus `base`, on the 5 confirmation
  cases x 3 seeds = 15 pairs (SE about 0.15 pt). **Adopt** if the mean delta is >= +0.3 pt, the paired
  t-test p < 0.05, and no case's mean delta is below -0.5 pt. Two of the confirmation cases are held out
  entirely from the screen: the Leifer lab and Flavell 2022-03-24-02. The other three are other networks of
  the screened recordings.
* Saturated recordings (Zimmer 1123, Flavell 0316, ~.98-.99) are not used for selection. They can serve as
  a final "does no harm" check.
* GT-free check: record whether the winners also raise `agree_mean`. Sync-stage variants raise it partly
  by construction, so do not select on it.
* Reproducibility check before trusting a batch: `base` at seed 0 must reproduce the recorded new-sync
  accuracy of each case to within float noise (compare_sync cpqr_refine values: S137 t63 .9338,
  t50 .8880, S184 t27 .9575, t32 .9241, Z1210 t52 .9406, t36 .9316, F0330-02 t61 .9714).

## 5. Experiments

### Batch 1: one-change screen (ready to run)

* Variants: `batch1` in variants.py, 34 plus `base`. The 13 graph-level ones (`whiten`, `pca`, `l2`,
  `svd100`, `k10`, `k40`, `sym`, `mutual`, `nosame`, `gauss`, `exact`, `temporal1`, `temporal5`) each
  rebuild the graph and propagate again. 8 are propagation-level (`layers30`, `layers300`, `spread99`,
  `classnorm`, `framenorm`, `seed_strat`, `seed_complete`, `seeds50`, `seeds100`). The other 12 reuse the memoised
  propagation (sync and post only).
* Cases (`cases.sh` DEV_CASES): S137 t50 (.877, weak, 85k objects), S184 t27 (.957, 210k), Z1210 t36 (.929,
  261k, distractors), F0330-02 t61 (.969, 228k, every detection). This is one network per lab except
  Leifer, which is held out.
* RNG seeds 0 and 1.
* Cost, CPU at 16 threads, from the S137 measurement scaled by objects x labels:
  * per (case, seed): about 26 propagations (layers300 counts 3, seeds50 2, seeds100 4), 36 syncs and
    15 graphs;
  * S137: about 22 x 20 s + 35 x 9 s + 15 x 10 s, about 15 min;
  * the 210-260k-object cases: about 22 x 100 s + 35 x 25 s + 15 x 40 s, about 60 min;
  * total about 3.3 h per seed and 6.5-7 h for both, or 2.5-3 h wall time with 3 processes in parallel
    here.
  * **On GPU (the default now):** propagation is about 2.5 s per seed for 190k objects (notebook
    2887), so a big (case, seed) is about 35-40 min, dominated by graphs and syncs on the CPU. With two
    GPU streams (one per RNG seed, 24 CPU threads each; GPU memory per stream is under 2 GB, as dense H is
    about 0.8 GB at 260k x 250 labels), batch 1 takes about 2.5 h wall time. That matches three CPU
    streams, leaves CPU cores free, and makes `seeds100` cheap.
  * Run the speed batch first (below), with base at seed 0 on the GPU. It checks GPU equivalence
    (`local`, `batch5` equal `base`), reproduces the recorded GPU accuracies (section 4), and times
    `exact` on the GPU, all in about 1 h.

Launch (local, GPU, two streams; `DEVICE=cpu THREADS=16` and three streams if the GPU is busy):

```bash
cd barlow_track/scripts/scratch/labelprop_opt
export WORK=$HOME/.cache/barlow_scratch/labelprop_opt
mkdir -p $WORK
# 1. speed / reproduction check (about 1 h)
DEVICE=cuda THREADS=24 bash run_local.sh speed dev "0" > $WORK/speed_dev.log 2>&1
# 2. batch 1, one GPU stream per RNG seed (resumable; both append to $WORK/batch1_dev.jsonl)
for s in 0 1; do
  DEVICE=cuda THREADS=24 nohup bash run_local.sh batch1 dev "$s" > $WORK/batch1_dev_s$s.log 2>&1 &
done
# afterwards
PY=/home/charles/anaconda3/envs/wbfm/bin/python
$PY analyze.py $WORK/batch1_dev.jsonl --per_case
$PY analyze.py $WORK/batch1_dev.jsonl --metric err_frag_split --cases zimmer flavell   # section 3a
```

(run_local.sh writes `$WORK/batch1_dev.jsonl`. The two seed streams append to it from different
processes; each line is under 4 KB and is written with one `write`, so the appends do not interleave on a
local disk. On the cluster use the sbatch, which writes one file per task.)

Cluster (one GPU and 16 CPUs per (case, seed), about 1 h each, 6 h walltime set):

```bash
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track/barlow_track/scripts/scratch/labelprop_opt
BATCH=batch1 SET=dev SEEDS="0 1" sbatch --array=0-7 labelprop_opt.sbatch
python analyze.py /lisc/.../barlow_track_revisions/labelprop_opt/batch1_dev_*.jsonl --per_case
```

The sbatch also works as a CPU-only job: drop `--gres` / `--constraint` and pass `--device cpu` (edit the
python line). Per the memory note "round configs on the cluster", copy the launch line into the round's
config_files directory rather than git.

### Batch 2: confirmation and combination

Fill `BATCH2` in variants.py with the promoted variants. Add pairwise combinations of promoted variants
from different stages (graph x prop x post), because effects in different stages are likely to add up.
Add the best value of each promoted knob plus one neighbouring value (for example k10 promoted: add k7 and
k14). Also include the speed winners from the `speed` batch. Run them on `CONFIRM_CASES` x seeds 0 1 2:

```bash
bash run_local.sh batch2 confirm "0 1 2"          # local
BATCH=batch2 SET=confirm SEEDS="0 1 2" sbatch --array=0-14 labelprop_opt.sbatch
```

Cost: about 10 variants x 15 (case, seed), about 6-8 CPU-h, or about 3 h wall time locally in 3 GPU
streams (one per RNG seed). Leifer is in the confirmation set and is the case where fragmentation matters
most (section 3a), so report its fragment columns separately. Adopt
per section 4. Then run the 100-seed check of the adopted recipe on 2 cases (the paper uses 100). The
seed count interacts with run weighting and seed choice.

### Batch 3: structural options (need library changes or more code)

Only where the error decomposition (full-length batch-1 base rows) leaves a gap. Swaps dominate on Samuel;
fragments and unlabelled objects matter on distractor recordings.

* Fragment merging, if the full-length `acc_merge_bound - acc_fast` is > 0.5 pt on distractor cases. The
  Leifer smoke suggests several points. Implement it as a `post` option in lp_core (label x label affinity
  from temporal overlap and centroid distance, then greedy merge with no two merged labels in one volume),
  and evaluate it in the same harness.
* Iterated re-propagation: EM-style rounds of propagate, consensus, re-seed.
* Sinkhorn / Hungarian label assignment per volume *inside* propagation, the hard version of
  `framenorm`.
* GT-free selection of the recipe per recording by the agreement proxy. This is valid only for
  propagation and graph variants, not for sync variants.

### Speed batch (run once; small)

Run it first, before batch 1 (launch line above). `bash run_local.sh speed dev "0"` on GPU (and once
with `DEVICE=cpu` if CPU-only eval jobs matter) compares `base`, `local` (must be
equal), `exact`, `batch5` (must be equal), `layers50` and `layers30` for time and accuracy.

## 6. Speed and memory

Measured on S137 t63 (85k x 512, 16 CPU threads, this machine):

| phase | time |
|---|---|
| SVD | 1.7 s |
| graph | 23.6 s (NNDescent 6.3 s, the rest JIT; exact kNN 18 s) |
| propagation, 25 seeds x 100 layers | 19.6 s (layers50 10.0 s, layers30 5.9 s, batch5 34.8 s) |
| sync | 7.7-9 s |
| per-volume Hungarian | 0.3 s |
| scoring | 0.5 s |

Per-case one-off costs: Zimmer GT/project load about 4.5 min; Flavell NWB every-detection mapping (regionprops on
every volume) cached in `--work` after the first run. GPU (notebook 2887, Leifer 196k objects): 2.3-3.0 s per
seed, 118 s per track.

Cheap wins, in order:

1. **Track on the GPU when the eval has one** (the sweep's default, `objective_track_device: cuda`); this
   harness now defaults to it too. On CPU-only jobs, CSR is enough: the old 102 s/seed CPU figure was torch_sparse spmm, before the CSR
   adjacency. CSR is about 0.8 s/seed at 85k objects on 16 threads. CPU eval jobs (`objective_cpu_job`)
   are therefore no longer slow at label propagation. Re-measure Leifer-size on CPU with the `speed` batch
   before relying on this.
2. **Remove the NNDescent JIT (about 20 s per eval process)** with exact chunked kNN on the eval device
   (`exact`). On GPU, 200k x 200k x 50 takes about a second. On CPU it took 18 s at 85k and grows as
   N^2 (about 3 min at 260k), so keep NNDescent on CPU-only jobs.
3. **`num_layers` 100 -> 30**, if batch 1 shows it is neutral or better: 3.3x less propagation.
4. **`batch_seeds`**: identical results. Measured slower on CPU (1.8x); a GPU-only option, to be timed
   there. Memory is about 3 x N x (sum of C) x 4 B, about 2.4 GB for 5 seeds at 200k x 200.
5. Vectorise `greedy_top1_timewise` (sort per volume by probability, then one pass with numba).
   This saves most of the 8-25 s sync.
6. Memory: the dense N x C float32 H, Y and SpMM output are about 3 x 200k x 200 x 4 B = 0.5 GB per seed on
   GPU, with the adjacency below 100 MB. The OOM next to the trained model is about the model, not
   propagation. `torch.cuda.empty_cache()` between eval stages, and float16 H (to be tested; risk to tails),
   would halve it.

## 7. Proposed library patches (descriptions only; not applied)

1. `utils_label_propagation.build_knn_graph`: optional `exact=True` (chunked matmul + topk on the device)
   and `drop_same_frame` / `time_index` (cannot-link). Document that the kNN list includes the point
   itself.
2. `run_label_propagation` / `_propagate_all_seeds`:
   * `batch_seeds` (several seeds per SpMM);
   * optional `frame_norm`, which needs a frame index tensor;
   * honour `alpha` (label spreading) or remove it;
   * optional class-mass normalisation;
   * early stopping when `max |dH| < tol`.
3. `track_using_label_propagation_clusterer`: plumb `input_probability_threshold`, `normalize_input_rows`
   (probability-weighted sync), `seed_select`, the run weights (`aggregate_consensus` already takes
   `doweight_runs`) and `reprop_conf`. Record them in the eval record so like is compared with like, as
   `consistency['sync']` does now.
4. `spectral_sync_from_topk`:
   * remove the explicit zeros, i.e. drop below-threshold entries instead of zeroing them (fixes item 5 of
     section 2);
   * make `K` extra a parameter;
   * vectorise `greedy_top1_timewise`.
5. `eval_accuracy`: optional `--svd pca` (centring).

Any adopted change alters recorded accuracies and the GT-free proxy references (round_shared_v1 template),
as the sync switch a5cb2f1 did. Bump a `tracker_version` field in the records when it lands.
