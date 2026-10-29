# Drop-in loss variants and the tracking stage: literature for the BarlowTrack revision

Agent report, 2026-10-29. Sources: web (arXiv, proceedings, bioRxiv) and the user's Zotero library (read-only).
Zotero already holds: Barlow Twins (ZGX7ZT26), VICReg (MDK76IP4), Balestriero & LeCun 2022 (P5MX563N),
LeJEPA (MMX87R8R), fDNC (EBZB8JFE webpage), Chaudhary 2021 CRF (KR27RFJG), the BarlowTrack preprint (2Y9YWW3D).
Everything else below was found on the web; links given. I read the PDF text of Garrido et al. (2023) and the HTML of
FroSSL, Agrawal et al. and ASCENT; the others I know from abstracts and listings only, and I say where.

## How our code relates to the literature (read before the ideas)

From `barlow_track/utils/barlow.py` (read-only):

- The batch is the objects of **one volume** (N ~ 100-200 neurons), embedding dim D (256-1024).
- `c_features` (D x D) normalizes each **dimension** over objects: this is Barlow Twins, a *dimension-contrastive*
  criterion in the terms of Garrido et al.
- `c_objects` (N x N) normalizes each **embedding** over its features and pushes the diagonal to 1 and the off-diagonals
  (other neurons in the same frame) to 0. This is a squared-loss *sample-contrastive* criterion with same-frame
  negatives: the same family as SimCLR-sq / DCL-sq / spectral contrastive loss (Garrido et al., Prop. 3.2), and the
  squared-loss version of what ASCENT does with NT-Xent (idea 6).
- So `(1-lambd_obj)*L_feat + lambd_obj*L_obj` is a sum of one dimension-contrastive and one sample-contrastive term, each
  with its own normalization. That is close in spirit to FroSSL, which is built to be both (idea 1).

From `barlow_track/utils/utils_label_propagation.py` (read-only), things that matter for ideas 3-5:

- Unweighted kNN graph (k=20, NNDescent) over **all objects from all frames**, **including edges between different
  neurons in the same frame**. Symmetric normalization, 50 clamped iterations (harmonic-function style,
  Zhu et al. 2003), argmax plus a 1.1/n_seeds threshold.
- `run_label_propagation(..., alpha=0.95)` accepts `alpha`, but `clamped_label_propagation` never uses it, so `alpha` has
  no effect: the method is pure clamped diffusion, not Zhou et al.'s LGC. Worth knowing before tuning it, or describing
  it in the methods text.
- Seeds are every object of one frame (one label per neuron), so the label rate is about 1/T (~1/1600) per class. This is
  the "very low label rate" regime where theory says Laplace learning degenerates (idea 3).
- Multi-seed consensus: rolling Hungarian alignment of the labelings, then per-frame Hungarian on **vote counts**.

---

## Ranked ideas

### 1. Explain the transpose term with the sample-contrastive vs dimension-contrastive duality, and measure its speed effect through the embedding spectrum

- **Summary:** Theory says the transpose term and the Barlow term minimize criteria that are equal up to row/column
  normalization. The optimum (final accuracy) should therefore barely move, while optimization speed can differ.
  That matches both the preprint ("insensitive") and the new result ("~4x fewer epochs").
- **Citations:**
  - Garrido, Chen, Bardes, Najman, LeCun (2023). *On the duality between contrastive and non-contrastive self-supervised
    learning.* ICLR 2023 (oral, top 5%). arXiv:2206.02574, https://arxiv.org/abs/2206.02574. Key facts from the full text:
    Def. 3.2, L_c = ||K^T K - diag||_F^2 (sample-contrastive), L_nc = ||K K^T - diag||_F^2 (dimension-contrastive);
    Theorem 3.3, L_nc + sum_j ||K_j||^4 = L_c + sum_i ||K_i||^4, via ||K^T K||_F = ||K K^T||_F. If rows **and**
    columns are L2-normalized, L_nc = L_c + N - M, which makes them equivalent for optimization. The authors conclude
    that the remaining differences "come from the optimization process as well as the implementation details". With
    tuning, both families reach the same accuracy at 100 epochs.
  - Skean, Dhakal, Jacobs, Sanchez Giraldo (2024). *FroSSL: Frobenius Norm Minimization for Efficient Multiview
    Self-Supervised Learning.* ECCV 2024. arXiv:2310.02903, https://arxiv.org/abs/2310.02903. Code:
    https://github.com/OFSkean/FroSSL. Motivated explicitly by "dimension-contrastive methods ... require more epochs to
    converge". They argue FroSSL is both sample- and dimension-contrastive up to normalization (Props. 1-3) and converges
    faster because its regularizer acts directly on the covariance eigenvalues. STL-10, epochs to 80% top-1: Barlow Twins
    370, VICReg 360, FroSSL 290 (2 views), 144 (4 views), 55 (8 views). At epoch 10: BT 46.6%, FroSSL-2v 56.9%.
  - Simon et al. (2023). *On the stepwise nature of self-supervised learning.* ICML 2023. arXiv:2303.15438,
    https://proceedings.mlr.press/v202/simon23a.html. Barlow Twins learns eigendirections one at a time, at times set by
    their eigenvalues, so small-eigenvalue directions arrive last. This gives the mechanism: anything that lifts the small
    eigenvalues early speeds training.
  - Balestriero & LeCun (2022). *Contrastive and non-contrastive SSL recover global and local spectral embedding
    methods.* NeurIPS 2022, https://openreview.net/forum?id=jQgsZDspz5h (in Zotero). Sample-contrastive maps to global
    spectral methods and dimension-contrastive to local ones, "hinting at the benefits and limitations of each". This is a
    principled reason to combine both.
  - Liu et al. (2022). *Self-Supervised Learning via Maximum Entropy Coding (MEC).* NeurIPS 2022, arXiv:2210.11464, code
    https://github.com/xinliu20/MEC. Per the abstract, batch-wise and feature-wise objectives are low-order Taylor
    approximations of one objective. (Abstract only.)
  - Li et al. (2021). *Contrastive Clustering.* AAAI 2021, DOI 10.1609/aaai.v35i10.17037. A precedent for applying a loss
    to both the rows and the columns of the feature matrix. (Abstract only.)
- **Why it matters here:** It turns an empirical curiosity into a supported claim: "the object-space term is the
  sample-contrastive dual of Barlow Twins; theory predicts the same optimum but different dynamics, and we observe the
  same final accuracy in ~4x fewer epochs."
- **A guess of mine (not from the literature; test before claiming):** with N ~ 150 objects per batch and D >= 256, the
  D x D feature cross-correlation is estimated from fewer samples than dimensions. Its rank is at most N, so the Barlow
  off-diagonal target cannot be met and its gradient is noisy. The N x N object term has no such rank problem.
  Prediction: the speed-up from lambd_obj grows with D/N. Testing it costs one sweep axis (embedding dim x lambd_obj),
  and AGENTS.md says embedding dim barely matters for final accuracy, so it does not confound the main claim.
- **How to plug in:** No code change for the theory. For evidence, log the embedding-covariance eigenvalue spectrum or
  effective rank (the stable rank of z, or the entropy of the normalized eigenvalues) every epoch for lambd_obj in
  {0, 0.25, 0.5}. Then plot accuracy proxy vs epoch, and effective rank vs epoch, as FroSSL Fig. 2 does.
- **Cost:** Writing ~0.5 day. Spectrum logging is a few lines in the train loop. 3 lambd_obj x 3 seeds x 2 datasets of
  image-only training, ~1 local GPU-day, or one small cluster round.
- **Fit for revision: HIGH.** Mostly framing plus one figure. It also handles a likely reviewer question ("why add a
  second term if the final accuracy is the same?").

### 2. More than two views per frame, with a mean-based invariance term (multi-view Barlow / FroSSL-style)

- **Summary:** Generate m = 4-8 augmented views of each volume and penalize each view's deviation from the mean
  embedding, instead of using one pair. The literature reports faster convergence and up to 4x fewer unique training
  samples for the same accuracy, which bears directly on the 30-100-frame regime.
- **Citations:**
  - Ghosh/Agrawal, Ghosh, Oberman, Richards (2024). *Addressing Sample Inefficiency in Multi-View Representation
    Learning.* ICLR 2024 (also NeurIPS 2023 SSL workshop). arXiv:2312.10725, https://arxiv.org/abs/2312.10725. From the
    HTML: the loss for m views uses the mean embedding, O(m) cost. Barlow Twins on CIFAR-10, 100 epochs, pdim 256:
    2 views 86.4%, 4 views 91.7%, 8 views 92.7%. They hold "views x unique samples" fixed and claim the pretraining set
    can shrink by up to 4x (the intro says, more cautiously, down to ~25% of the data), and ~2x faster wall-clock. They
    also find the best off-diagonal weight scales as lambda ~ 1/d. Code: no repository is mentioned on the page.
  - FroSSL (above): 8 views reach the target in 55 vs 290 epochs (2 views).
  - Ermolov et al. (2021). *Whitening for Self-Supervised Representation Learning (W-MSE).* ICML 2021, arXiv:2007.06346,
    https://arxiv.org/abs/2007.06346. An early demonstration that several positives per image help a non-contrastive
    loss. Code on GitHub per the authors. (Abstract and summaries only.)
- **Why it matters here:** Frames are the scarce resource (30-100 training frames). Views are cheap because the global
  augmentation pipeline already makes them. It also fits the efficiency goal: fewer epochs per job. For the
  position-only network, more keypoint-jitter/dropout views per frame is the natural version, and jitter is the
  augmentation that already proved decisive.
- **How to plug in:** Loss: generalize `forward(y1, y2)` to a list of views. For the feature term, compute the
  cross-correlation of each view against the mean of the others, or use the m-view mean form, then average. The object
  term can be done the same way. Augmentation: the dataset returns m views instead of 2.
- **Cost:** Per-step cost grows roughly linearly with m (encoder forward/backward). The literature says total epochs
  fall by more than m in some settings, but not all. Local test: m in {2, 4} x 2 datasets x {30, 100} frames, ~1 GPU-day.
- **Fit for revision: HIGH.** A drop-in loss generalization with strong, recent, directly relevant evidence on sample
  efficiency. Risk: GPU memory with 3D crops at m = 8; start at m = 4.

### 3. Replace clamped Laplace propagation with Poisson learning (low-label-rate theory)

- **Summary:** Our seeds give one label per class per ~1600 frames. In that regime Laplace learning (what
  `clamped_label_propagation` implements) is known to degenerate towards uninformative, near-constant solutions.
  Poisson learning is a drop-in fix designed for very low label rates.
- **Citations:**
  - Calder, Cook, Thorpe, Slepčev (2020). *Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label
    Rates.* ICML 2020, PMLR 119:1306-1316. arXiv:2006.11184, https://proceedings.mlr.press/v119/calder20a.html. Code:
    `pip install graphlearning` (https://pypi.org/project/graphlearning). Also Poisson-MBO, which can use known class
    sizes. In our case every class has exactly one object per frame, a strong prior. (Abstract and README only.)
  - Nadler, Srebro, Zhou (2009). *Statistical Analysis of Semi-Supervised Learning: The Limit of Infinite Unlabelled
    Data.* NeurIPS 2009, https://proceedings.neurips.cc/paper/2009/hash/68ce199ec2c5517597ce0a4d89620f55-Abstract.html.
    For d >= 2, Laplacian regularization with few labels becomes ill-posed and degenerates as unlabelled data grows.
  - Zhu, Ghahramani, Lafferty (2003). *Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions.* ICML 2003,
    https://mlanthology.org/icml/2003/zhu2003icml-semi/. The clamped harmonic method we actually run; cite it in the methods.
  - Zhou et al. (2004). *Learning with Local and Global Consistency.* NIPS 16,
    https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency. The alpha-damped variant our API
    suggests but does not run.
  - Iscen, Tolias, Avrithis, Chum (2019). *Label Propagation for Deep Semi-Supervised Learning.* CVPR 2019,
    https://openaccess.thecvf.com/content_CVPR_2019/html/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.html.
    kNN graph on learned embeddings, then diffusion: our setting. (Abstract only.)
- **Why it matters here:** (a) It explains, with theory, why a single seed frame is fragile and multi-seed consensus helps:
  more labelled nodes means a higher label rate. (b) It may raise accuracy where propagation is weakest, at long temporal
  distances from the seed frame. (c) Fifty iterations on a ~200k-node graph may not have converged; Poisson learning is
  normally solved to convergence with a stopping rule.
- **How to plug in:** Eval/clustering only. Swap `clamped_label_propagation` for a Poisson solve: source term = seed
  one-hots minus their mean, iterated on the same normalized graph, ~20 lines of torch or the `graphlearning` package.
  Also fix or remove the dead `alpha` argument, or implement LGC as a second variant. Then compare Laplace, LGC and
  Poisson on cached embeddings.
- **Cost:** CPU only, cached embeddings, minutes to an hour per recording. ~1 day including the comparison table across
  4 datasets.
- **Fit for revision: HIGH.** A pure tracking-stage tweak, no retraining, with a theory citation that also supports the
  "why label propagation" narrative. Evidence for gains on *our* graphs is untested; the gains reported in the paper are
  on MNIST/CIFAR at 1-5 labels per class.

### 4. Encode "one neuron per frame": drop intra-frame kNN edges and do the per-frame Hungarian on soft scores

- **Summary:** Two objects in the same frame can never be the same neuron (a cannot-link). Today the kNN graph lets labels
  leak along intra-frame edges, and consensus fuses hard votes. Remove same-frame edges (or give them negative/zero
  weight), and run the per-frame Hungarian on summed propagation probabilities across seeds instead of vote counts.
- **Citations:**
  - Wu, Zhang, Hu, Ji (2013). *Constrained Clustering and Its Application to Face Clustering in Videos.* CVPR 2013,
    DOI 10.1109/CVPR.2013.450, https://mlanthology.org/cvpr/2013/wu2013cvpr-constrained. The classic use of "faces in the
    same frame cannot be the same person" as cannot-link constraints. (Abstract only.)
  - Tang, Andriluka, Andres, Schiele (2017). *Multiple People Tracking by Lifted Multicut and Person Re-identification.*
    CVPR 2017, DOI 10.1109/CVPR.2017.394. Graph clustering of detections with cannot-link structure; code public. Cite as
    the principled, but too dramatic, end of this spectrum.
  - Tai, Bailis, Valiant (2021). *Sinkhorn Label Allocation.* ICML 2021, arXiv:2102.08622, code
    https://github.com/stanford-futuredata/sinkhorn-label-allocation. Label assignment under class-capacity constraints by
    optimal transport. A soft per-frame Hungarian is the same idea with capacity 1. (Abstract only.)
- **Why it matters here:** Label leakage between neighbouring neurons is the expected failure mode in dense regions. The
  constraint is GT-free and exact. Soft-score fusion keeps information that vote counting throws away. The repo's
  `enforce_temporal_uniqueness_hungarian` in `utils_spectral_relabeling.py` already does part of this for the spectral
  path.
- **How to plug in:** In `build_knn_graph`, filter edges with `t[row] == t[col]`; query k + a few neighbours so the degree
  stays ~k. In `fuse_labels_per_time`, build the matrix from the summed `probabilities` (already returned by
  `multi_seed_propagation`) instead of counts.
- **Cost:** CPU, cached embeddings, a few hours.
- **Fit for revision: HIGH.** Small, principled, easy to ablate. Gains unknown; they may be small where accuracy is
  already 0.96+.

### 5. A tracking-stage ablation that isolates the label-propagation benefit, with literature to explain it

- **Summary:** To claim a "clear label-propagation benefit", compare, on the same cached embeddings:
  (a) HDBSCAN;
  (b) per-frame Hungarian to a single template frame (fDNC-style);
  (c) serial frame-to-frame Hungarian with EMA track embeddings (ASCENT-style);
  (d) LP from one seed frame;
  (e) LP with multi-seed consensus.
  Then explain the ordering with semi-supervised-learning theory.
- **Citations (how others link over time):**
  - Yu et al. (2021). *Fast deep neural correspondence (fDNC).* eLife 10:e66410, DOI 10.7554/eLife.66410
    (Zotero EBZB8JFE). Per-frame matching to a template; 79.1% within-animal tracking with position only. Code public
    (Leifer lab).
  - Han, Pritchard, Lu (2025). *ASCENT: Annotation-free Self-supervised Contrastive Embeddings for 3D Neuron Tracking.*
    bioRxiv, DOI 10.1101/2025.07.23.666425. Serial Hungarian with cosine costs, a dynamic threshold, and an EMA
    (alpha 0.5) of track embeddings. Code "on publication".
  - Nguyen et al. (2017). *Automatically tracking neurons in a moving and deforming brain (NeRVE).* PLoS Comput Biol
    13:e1005517, DOI 10.1371/journal.pcbi.1005517. Each neuron is matched against many reference volumes, and the vector
    of matches is clustered. That is consensus over references, the same principle as our multi-seed fusion.
  - Atanas et al. (2025). *Deep Neural Networks to Register and Annotate Cells in Moving and Deforming Nervous Systems
    (BrainAlignNet / ANTSUN 2.0).* eLife reviewed preprint 108159, https://elifesciences.org/reviewed-preprints/108159.
    Flavell lab: registers selected frame pairs, builds an N x N ROI linkage matrix, then custom hierarchical clustering.
    Reports 99.6% linking accuracy. A graph-then-cluster design with a hand-built affinity.
  - Campello, Moulavi, Sander (2013). *Density-Based Clustering Based on Hierarchical Density Estimates (HDBSCAN).* PAKDD
    2013, DOI 10.1007/978-3-642-37456-2_14.
  - Zhu et al. 2003; Nadler et al. 2009; Calder et al. 2020 (idea 3).
  - HaoChen, Wei, Gaidon, Ma (2021). *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive
    Loss.* NeurIPS 2021 (oral), arXiv:2106.04156. SSL embeddings are spectral embeddings of the augmentation graph, and
    classes are connected subgraphs. Together with Balestriero & LeCun 2022, this argues that graph diffusion is the
    natural reader for these embeddings: it uses connectivity, not density gaps.
- **Why propagation should beat clustering here (synthesis; partly my reasoning, flagged):**
  1. LP is given the exact number of identities and one exemplar of each (the seed frame). HDBSCAN must discover K,
     drops "noise" points, and merges or splits by density alone.
  2. Neuron embeddings drift with posture over a recording. Each identity then traces an elongated, possibly curved
     manifold. Diffusion on a kNN graph follows connectivity along it (the manifold assumption, Zhu 2003). Density
     clustering needs low-density gaps, which tend to vanish where two neurons' trajectories come close.
  3. Low label rate is the main LP risk (Nadler 2009). Multi-seed consensus raises the label rate, which explains why
     consensus over seeds helps, and the NeRVE design uses the same principle.
  4. Prediction to check: the LP-HDBSCAN gap should be largest on recordings with strong posture variation and
     non-uniform density (e.g. Samuel), and smallest where the embeddings are already well separated.
- **How to plug in:** Eval only. (b) and (c) are ~50 lines each with `linear_sum_assignment`. Plot accuracy vs temporal
  distance from the seed frame for each method: the curve shape shows *where* LP wins.
- **Cost:** CPU, cached embeddings, ~1-2 days for all 4 datasets.
- **Fit for revision: HIGH.** Answers the "why LP" question with both a table and a mechanism.

### 6. Per-frame NT-Xent (InfoNCE) with same-frame negatives as an alternative object-space term; and cite and position against ASCENT

- **Summary:** ASCENT is a 2025 competitor that does almost exactly our object-space term, with LogSumExp instead of
  squares: NT-Xent, positives are the same neuron across two augmented views of a frame, negatives are the other
  neurons of that frame, tau = 0.05, batch of 4 frames.
- **Citations:**
  - ASCENT (above). From the full text: NeRVE (Leifer, 1532 frames) MOTA 95.9% vs ZephIR 94.4%, NeRVE 82.9%,
    fDNC 79.1%. Data efficiency: **8 training frames give 95.7%** (8 consecutive: 95.4%). Augmentation ablation: none
    37.3%, position jitter alone 95.1%, plus rotation 95.7%. BarlowTrack is not cited.
  - Yeh et al. (2022). *Decoupled Contrastive Learning.* ECCV 2022, arXiv:2110.06848. Removes the positive from the
    InfoNCE denominator; much better at small batch sizes. Relevant because our "batch" of negatives is only ~150 neurons.
  - Garrido et al. 2023 (idea 1): VICReg-exp vs VICReg-ctr isolates LogSumExp vs Frobenius. Their conclusion is that
    such choices change optimization, not the attainable optimum.
- **Why it matters here:** (a) A reviewer may ask about ASCENT, which overlaps in data (NeRVE/Leifer) and idea. (b) It is
  the best evidence found for question (b): a same-frame sample-contrastive loss reaches high accuracy from 8 frames.
  Note that it is evaluated with serial Hungarian on one Leifer recording, and its positions-plus-image transformer is
  not our encoder. (c) Its jitter result agrees with our finding that keypoint jitter is the decisive position
  augmentation.
- **How to plug in:** Loss: add `loss_type_obj: barlow | ntxent | dcl`, replacing `loss_from_correlation_matrix(c_objects)`
  with a cross-entropy over the cosine-similarity matrix (z1 rows vs z2 rows, both views), tau in {0.05, 0.1}. Keep the
  feature-space Barlow term.
- **Cost:** ~20 lines. Image-only and position-only, 2 datasets x 3 seeds, at 30 and 100 frames: ~1 GPU-day or one
  cluster round.
- **Fit for revision: MEDIUM-HIGH.** Citing and positioning ASCENT is high priority whatever else is done. Whether the
  loss swap beats the squared transpose term is untested, and theory (idea 1) predicts a speed difference rather than
  an accuracy difference.

### 7. FroSSL loss as a whole-loss alternative (log Frobenius norm of each view's covariance plus MSE)

- **Summary:** L = sum_v log ||Cov_v||_F^2 + gamma ||Z_v - Z_mean||^2, with dimension-normalized embeddings. It has no
  eigendecomposition and works for any number of views.
- **Citation:** FroSSL (idea 1), ECCV 2024, code public. From the HTML: under weak augmentations on Tiny ImageNet,
  FroSSL 39.4% vs BT 36.8% vs VICReg 18.1%.
- **Why it matters here:** The fastest-converging drop-in found, and it has a theory story tied to idea 1. The
  weak-augmentation robustness may matter because our augmentations are deliberately mild (no GT fitting).
- **How to plug in:** Loss swap, one function.
- **Cost:** ~1 GPU-day for a fair comparison (with its own gamma sweep).
- **Fit for revision: MEDIUM.** A real alternative, but the revision story is better served by ideas 1+2 (keep Barlow,
  explain and speed it up). Use it as an extra row in a loss-ablation table if one is wanted.

### 8. Write up the multi-seed consensus as permutation synchronization / cluster ensembles

- **Summary:** The spectral relabeling (`utils_spectral_relabeling.py`) and the rolling Hungarian alignment are
  instances of multi-way matching by permutation synchronization and of cluster-ensemble consensus. Cite them, and say
  why the spectral (joint) version is more robust than rolling (sequential) alignment.
- **Citations:**
  - Pachauri, Kondor, Singh (2013). *Solving the multi-way matching problem by permutation synchronization.* NIPS 2013,
    https://proceedings.neurips.cc/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html. Joint estimation by
    eigenvectors; sequential pairwise chains propagate single errors.
  - Strehl & Ghosh (2002). *Cluster Ensembles: A Knowledge Reuse Framework for Combining Multiple Partitions.* JMLR
    3:583-617, https://www.jmlr.org/papers/v3/strehl02a.html.
  - NeRVE (idea 5) as the C. elegans precedent.
- **Why it matters here:** It turns "consensus over seeds" from a heuristic into a named method, and suggests an
  ablation: rolling vs spectral consensus.
- **How to plug in:** Writing plus an existing code path; the ablation is CPU only.
- **Cost:** 0.5 day writing; a few CPU-hours.
- **Fit for revision: MEDIUM** (mostly citations, and value depends on whether consensus is a claimed contribution).

### 9. Mixed Barlow Twins (mixup regularizer against Barlow overfitting)

- **Summary:** Barlow Twins can overfit "because it lacks strong interaction between samples". The authors add a
  mixup-based term on interpolated samples.
- **Citation:** Bandara, De Melo, Patel (2023). *Guarding Barlow Twins Against Overfitting with Mixed Samples.*
  arXiv:2312.02151, https://arxiv.org/abs/2312.02151. Code https://github.com/wgcban/mix-bt. (Abstract only.)
- **Why it matters here:** Their diagnosis (missing sample interaction) is exactly what our transpose term supplies,
  which gives one more literature reason why the object term helps image-only nets. Overfitting is also the expected
  failure at 30 frames.
- **How to plug in:** Augmentation plus loss: mix crops of two different neurons. This is natural for image crops and
  unclear for point sets, so image-only only.
- **Cost:** ~1 GPU-day.
- **Fit for revision: MEDIUM-LOW.** Mainly useful as a citation that supports idea 1. As a method, it adds a knob
  without clear evidence for 3D crops.

### 10. VICReg (and VICRegL) as the obvious "did you try" loss

- **Summary:** Barlow's standardization is replaced by a variance hinge, plus an invariance MSE and a covariance penalty.
  Each branch can be regularized separately.
- **Citations:**
  - Bardes, Ponce, LeCun (2022). *VICReg.* ICLR 2022, arXiv:2105.04906 (Zotero MDK76IP4).
  - Bardes, Ponce, LeCun (2022). *VICRegL: Self-Supervised Learning of Local Visual Features.* NeurIPS 2022,
    arXiv:2210.01571; code facebookresearch/VICRegL.
  - Kernel VICReg (arXiv:2509.07289), https://arxiv.org/pdf/2509.07289. Reports VICReg collapsing on Tiny ImageNet in
    their setup. Weak, single-source evidence.
- **Why it matters here:** Reviewers often ask for it. Garrido et al. show that, with tuning, VICReg and SimCLR land at
  the same accuracy, so expect no accuracy gain. VICRegL's local-feature term targets dense features, which we do not
  need (we already have object crops).
- **How to plug in:** Loss swap; VICReg's per-branch form also suits the fused image+position heads.
- **Cost:** ~1 GPU-day with a short coefficient sweep.
- **Fit for revision: MEDIUM-LOW.** Low expected value, but cheap insurance if a reviewer asks.

### 11. LeJEPA's SIGReg as the anti-collapse term

- **Summary:** Push embeddings towards an isotropic Gaussian using random 1-D projections and the Epps-Pulley test. It
  has a single trade-off hyperparameter and linear cost.
- **Citation:** Balestriero & LeCun (2025). *LeJEPA.* arXiv:2511.08544 (Zotero MMX87R8R), code
  https://github.com/rbalestr-lab/lejepa. Claims: minibatch bias minimal even at batch 16. In-domain pretraining on
  Galaxy10 beats DINOv2/v3 transfer in most cells.
- **Why it matters here:** Small batches (~150 objects) and small in-domain data are its claimed strengths.
- **How to plug in:** Loss: invariance MSE plus SIGReg in place of the Barlow off-diagonal term.
- **Cost:** ~1 GPU-day.
- **Fit for revision: LOW-MEDIUM.** Very new, little independent replication, and it would be a change of objective
  rather than a variant of Barlow.

### 12. Whitening / log-det family (W-MSE, Shuffled-DBN, CorInfoMax, MEC): cite, do not run

- **Citations:**
  - Ermolov et al. 2021 (ICML, arXiv:2007.06346).
  - Hua et al. (2021). *On Feature Decorrelation in Self-Supervised Learning.* ICCV 2021, arXiv:2105.00470.
    Distinguishes complete from dimensional collapse.
  - Ozsoy et al. (2022). *Self-Supervised Learning with an Information Maximization Criterion (CorInfoMax).* NeurIPS 2022,
    arXiv:2209.07999. Log-det of the feature covariance.
  - MEC (idea 1).
  - Shigeto et al. (2023). *Learning decorrelated representations efficiently using FFT.* CVPR 2023, arXiv:2301.01569.
    Relaxed O(n d log d) Barlow regularizer; a speed-of-computation trick, not a convergence trick.
- **Why it matters here:** Full whitening needs N > D. With ~150 objects per frame and D >= 256, the D x D covariance is
  singular, so W-MSE-style whitening needs sub-batching, and log-det terms are ill-conditioned. The same rank argument
  as in idea 1, in the other direction. Hua et al.'s "dimensional collapse" vocabulary is useful in the text. The
  effective-rank diagnostic from idea 1 measures it.
- **Fit for revision: LOW** as experiments. Useful as related work.

### 13. Object-level dense SSL and prototype methods (DetCon, ORL, DenseCL, SwAV): related work only

- **Citations:**
  - Hénaff et al. (2021). *Efficient Visual Pretraining with Contrastive Detection (DetCon).* ICCV 2021,
    arXiv:2103.10957, code https://github.com/deepmind/detcon. Object-level contrast using unsupervised masks; claims up
    to 10x less pretraining (5x in an earlier version).
  - Xie et al. (2021). *Unsupervised Object-Level Representation Learning from Scene Images (ORL).* NeurIPS 2021,
    arXiv:2106.11952.
  - Caron et al. (2020). *SwAV.* NeurIPS 2020, arXiv:2006.09882.
  - DenseCL (Wang et al., CVPR 2021) was not verified by my searches; cite only after checking.
- **Why it matters here:** These methods solve "find the objects, then contrast them", which segmentation already solves
  for us. DetCon's efficiency claim comes from adding object-level contrast to image-level SSL, which is conceptually our
  object-space term, so it can be cited alongside idea 1. A SwAV-style loss with neuron-identity prototypes and per-frame
  equipartition is the most tempting variant, but it amounts to learning templates: closer to a new paradigm than a
  drop-in.
- **Fit for revision: LOW** (related work only).

---

## Direct answers to the two questions

**(a) Does a transpose / sample-space decorrelation term speed up training?**
There is theory and indirect evidence, but no paper tests exactly our combined term.
- Theory (Garrido et al. 2023) says the two criteria are equivalent up to normalization, so the optimum should be similar
  (consistent with "final accuracy insensitive"), and remaining differences come from optimization.
- FroSSL (ECCV 2024) builds a loss that is both sample- and dimension-contrastive and reports faster convergence than
  Barlow Twins and VICReg (290 vs 370 epochs at 2 views). It attributes the speed to faster saturation of the covariance
  eigenvalues; the stepwise-learning theory of Simon et al. (2023) gives the mechanism.
- Bandara et al. (2023) diagnose Barlow Twins' weakness as missing sample interaction. Balestriero & LeCun (2022) map
  sample- vs dimension-contrastive to global vs local spectral methods.
- My additional, untested hypothesis: the feature-space term is rank-limited when N (objects per frame) < D, and the
  object term is not.
- Why the transpose term slightly hurts position-only networks: I found nothing in the literature, so any explanation
  would be speculation.

**(b) Which losses do best with very few training samples (30-100 frames)?**
No benchmark compares Barlow / VICReg / SimCLR at this scale.
- The most relevant evidence:
  1. Multi-view training (Agrawal/Ghosh et al. ICLR 2024; FroSSL): up to 4x fewer unique samples, faster convergence.
  2. ASCENT: a same-frame NT-Xent loss reaches 95.7% MOTA from 8 frames on a Leifer recording.
  3. LeJEPA: claims small-batch and small-in-domain-data robustness.
- Generic image benchmarks (Tiny ImageNet / STL-10) show 1-2 point spreads between methods, mostly with BYOL slightly
  ahead (arXiv:2403.01549; no error bars). One paper reports VICReg collapsing on Tiny ImageNet (arXiv:2509.07289).
- Recommendation: keep Barlow, and test m-view training at 30/100 frames before any loss swap.

## Suggested minimal revision package (ordered)

1. Idea 1 framing plus an effective-rank/speed figure for lambd_obj (one small round).
2. Idea 5 tracking ablation table and accuracy-vs-distance-from-seed plot (CPU), including ideas 3 and 4 as extra LP rows
   (CPU).
3. Idea 2 multi-view at 30/100 frames (one local GPU-day, then the weekend cluster round if promising).
4. Cite and position ASCENT (idea 6); optionally run the NT-Xent object-term swap as a single ablation row.
