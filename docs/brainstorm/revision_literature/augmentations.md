# Augmentations and deformations: literature scan for the BarlowTrack revision

Topic: drop-in augmentations and deformations for (a) keypoint / point-set views of deforming worms and
(b) 3D fluorescence crops. Search date 2026-10-29. Sources: web (arXiv, eLife, PMC, CVF, bioRxiv) and the
user's Zotero library. Zotero holds fDNC (Yu et al. 2021, item EBZB8JFE), Barlow Twins (ZGX7ZT26),
Dubois et al. 2022 (BUUWPMBE), Balestriero & LeCun 2022 (P5MX563N) and the BarlowTrack preprint
(2Y9YWW3D). It does **not** hold Park et al. 2024, PointWOLF, CPD, Atanas et al. 2025 (BrainAlignNet /
AutoCellLabeler), Nejatbakhsh et al. 2023 or the SSL view-design papers below. Every other citation
was found on the web, and each one has a link.

"Cost" is measured against one baseline training run per dataset. "Revision fit" uses the context
constraints: drop-in, no GT fitting, efficient.

---

## Part 1. Why segmentation-error augmentation and bending probably did not help

The literature does not test our exact setting, so the points below are reasoned from it. They are not
established results.

1. **Independent evidence that bending does not help.** In Atanas et al. 2025 (Flavell lab;
   AutoCellLabeler), the training set had custom augmentations: a B-spline warp with an explicit
   "worm-bend" arc in XY, contrast changes, and blur that grows with z. They report that "a network that
   was trained with only standard image augmentations" had "similar performance". That is a published
   null result that matches ours, from a supervised C. elegans pipeline. fDNC (Yu et al. 2021) uses
   centerline bending, CPD non-rigid warps and 20% missing/spurious neurons in its simulator. The paper
   **does not ablate them** (the only ablation is over network depth and width), so it gives no evidence
   that those components matter. The noise level (0.42 µm) is described as "arbitrarily set". Citable
   line for the revision: "Consistent with prior work, explicit bending models were not needed."
2. **An invariant encoder makes most geometric augmentation a no-op.** ViSNet (Wang et al. 2024) is
   E(3)-invariant in its scalar outputs and only looks at a local cutoff neighbourhood. Global rotation,
   translation and flips therefore do nothing in the position branch. A smooth bend whose radius of
   curvature is much larger than the cutoff is close to rigid inside each neighbourhood, so it also does
   nearly nothing. Only deformation that varies within one cutoff radius gives the position encoder new
   views. Per-object jitter is such a deformation, which fits jitter being the augmentation that mattered
   (+0.2 on Samuel). This also predicts the cutoff result: the scale-normalised cutoff changes what
   "local" means and interacts with which deformations are visible.
3. **InfoMin / idealised-representation view.** Tian et al. 2020: view pairs have a sweet spot in
   shared information. Dubois et al. 2022: augmentations should span exactly the downstream equivalence
   classes. Synthetic merge, split and spurious objects change the object set itself, so in our
   per-object Barlow and transpose terms some rows have no true partner. That is label noise for the
   positive pairs, not a nuisance the network can learn to ignore. If real segmentation errors are rare
   or differently distributed, the synthetic ones cost capacity and buy nothing. Label propagation already
   absorbs sparse per-frame errors through graph consensus, which limits any gain at the evaluation
   level.
4. **Why "real frame pairs as views" failed, while real deformations might still help.**
   Purushwalkam & Gupta 2020 show that temporal (video) views give invariances that synthetic
   augmentations do not. Real frame pairs, however, come with *uncertain correspondences*: a GT-free
   matcher such as ICP mislabels some positives. Idea 1 below separates the two: it takes the
   *deformation* from real pairs and applies it to one frame, so the correspondence stays exact.

---

## Part 2. Ranked ideas

### 1. Library of real deformations from unlabeled nearby frames (CPD or low-frequency Fourier fits), applied to one frame
- **Summary:** For many random frame pairs (t, t+k), fit a smooth non-rigid field to the unlabeled
  centroids (affine ICP, which we already have, then CPD or a low-frequency Fourier field). Store the
  fields, or a PCA of them. During training, apply a sampled field to frame t, so the view shows
  realistic deformation and the correspondence is exactly known.
- **Citations:**
  - Park, Barzegar-Keshteli, Korchagina, Delrocq, Susoy, Jones, Samuel, Rahi (2024), "Automated neuron
    tracking inside moving and deforming C. elegans using deep learning and targeted augmentation",
    *Nature Methods* 21:142-149, DOI [10.1038/s41592-023-02096-3](https://doi.org/10.1038/s41592-023-02096-3);
    preprint [10.1101/2022.03.15.484536](https://www.biorxiv.org/content/10.1101/2022.03.15.484536).
    Code: [github.com/lpbsscientist/targettrack](https://github.com/lpbsscientist/targettrack). They fit
    `D(x) = A Re Σ_{|k|<kmax} θ_k e^{-ik·x}`, keeping only low-frequency Fourier modes. The loss is an
    L1 keypoint fit plus a divergence penalty, and the fitted field warps both image and annotation. With
    5 GT volumes, the result matched training sets 3-5 times larger. Their keypoints come from a
    coarse CNN trained on a few labels. We would use ICP or CPD on unlabeled centroids instead, which
    needs no labels.
  - Myronenko & Song (2010), "Point Set Registration: Coherent Point Drift", *IEEE TPAMI* 32(12):2262,
    DOI [10.1109/TPAMI.2010.46](https://doi.org/10.1109/TPAMI.2010.46), arXiv
    [0905.2635](https://arxiv.org/abs/0905.2635). Its regularised displacement field is the
    "Gaussian-process-like" deformation prior. Public implementations exist, for example pycpd and
    probreg.
  - fDNC (Yu et al. 2021, below) already uses CPD to a random real template to create animal-to-animal
    variation. That is the cross-animal version of the same idea.
- **Why here:** It is the only GT-free way to get deformation statistics that match this recording,
  which is exactly what made jitter work (noise sized from the data's own centroid noise). The fitted
  residual also splits naturally into a *smooth* part (real tissue deformation) and an *independent*
  part (segmentation noise). The independent part refines the jitter we already have, and could give it
  anisotropic, per-axis magnitudes.
- **Plug-in:** Augmentation. Precompute fields on CPU (plentiful) once per recording. At train time,
  apply them to keypoints, which is cheap. For images in the global pipeline, either warp the volume or
  use the per-crop Jacobian trick from idea 2.
- **Cost:** About 1 day of coding. CPD precompute takes CPU-hours and no GPU. Training cost is unchanged
  for keypoints.
- **Revision fit: high.** It is a drop-in augmentation, GT-free, and tied to our existing
  noise-estimation story. Risk: by argument 2 in Part 1, if the real deformation is near-rigid at the
  cutoff scale, the position branch may again see no gain. The image branch is the more likely
  beneficiary.

### 2. Cheap smooth random fields (low-frequency Fourier or GP) applied per crop as a local affine, replacing full elastic deformation
- **Summary:** Sample a smooth random displacement field, with length scale and amplitude calibrated
  from idea 1 or swept. Instead of warping the whole volume (which cost us 4x), evaluate the field and its
  Jacobian at each object centroid. Shift the keypoint by `D(x_i)` and apply the 3x3 local affine
  `I + ∇D(x_i)` to that object's crop sampling grid.
- **Citations:** Park et al. 2024 (Fourier parametrisation, above). Hoffmann et al. (2022), "SynthMorph:
  Learning Contrast-Invariant Registration Without Acquired Images", *IEEE TMI* 41:543-558, DOI
  [10.1109/TMI.2021.3116879](https://doi.org/10.1109/TMI.2021.3116879), arXiv
  [2004.10282](https://arxiv.org/abs/2004.10282): random smooth fields from noise, with a diffeomorphic
  variant. Code is public in VoxelMorph/neurite.
- **Why here:** Elastic deformation "neither helps nor hurts, 4x slower". If the effect is real but
  small, a version close to free is the right one to ship. The local-affine approximation keeps rows,
  keypoints and crops aligned, with no resampling of the whole volume.
- **Plug-in:** Augmentation, in `volume_data.py`, inside the global-augmentation-before-crop path.
- **Cost:** Half a day of coding. Negligible runtime.
- **Revision fit: high** as an efficiency fix, **medium** for accuracy. Prior evidence that it raises
  accuracy is weak: Flavell's B-spline warps were neutral.

### 3. PointWOLF: multi-anchor locally weighted transforms for keypoints (a piecewise-rigid body model)
- **Summary:** Sample M anchors by farthest-point sampling, draw a small random rotation, scale and
  translation per anchor, and blend them per point with Gaussian-kernel (Nadaraya-Watson) weights. The
  result is a smooth non-rigid deformation that is locally rigid around each anchor. With anchors laid
  along the anterior-posterior axis, it is effectively a piecewise-rigid worm-body model.
- **Citations:** Kim, Lee, Hwang, Lee, Hwang, Kim (2021), "Point Cloud Augmentation with Weighted Local
  Transformations", *ICCV* 2021, DOI 10.1109/ICCV48922.2021.00059, arXiv
  [2110.05379](https://arxiv.org/abs/2110.05379). Code:
  [github.com/mlvlab/PointWOLF](https://github.com/mlvlab/PointWOLF). On ScanObjectNN with PointNet++,
  accuracy rose from 86.6 to 89.7, and local translation was the largest single component. Robustness to
  75% point dropout rose from 29.5 to 60.8. Supporting reference for a biologically motivated version:
  Nejatbakhsh, Dey, Venkatachalam, Yemini, Paninski, Varol (2023), "Learning Probabilistic Piecewise
  Rigid Atlases of Model Organisms via Generative Deep Networks", *IPMI* 2023, pp. 332-343. The paper
  argues that fluid, elastic and diffusion models suit C. elegans poorly and uses piecewise-rigid models
  regularised to preserve neighbour distances. Code:
  [github.com/amin-nejat/Deformable-Atlas](https://github.com/amin-nejat/Deformable-Atlas).
- **Why here:** It is a better-founded replacement for our hand-made "keypoint bending", and the
  deformation varies at a controllable scale. Its bandwidth h can be set *below* the ViSNet cutoff,
  which addresses argument 2 in Part 1 directly.
- **Plug-in:** Keypoint augmentation, O(MN). It can be applied to crops through idea 2's local affine.
- **Cost:** Hours to port, since the reference code is NumPy/PyTorch. Negligible runtime.
- **Revision fit: medium-high.** Easy and well cited. Expected gain is modest because bending already
  failed. Its value is that it tests the "deformation at sub-cutoff scale" hypothesis cleanly.

### 4. Microscopy photometric augmentations for crops: per-object intensity, gamma, Poisson-Gaussian noise, z-dependent blur
- **Summary:** For the image branch, add (a) per-object multiplicative intensity scaling and gamma,
  sized GT-free from the frame-to-frame variability of each object's mean red intensity (the same logic
  as the jitter); (b) Poisson-Gaussian noise; (c) Gaussian blur that increases with z, mimicking
  defocus.
- **Citations:** Atanas et al. (2025), "Deep Neural Networks to Register and Annotate Cells in Moving
  and Deforming Nervous Systems", *eLife* reviewed preprint, DOI
  [10.7554/eLife.108159](https://doi.org/10.7554/eLife.108159). They use contrast per channel,
  z-increasing blur, and Gaussian plus Poisson noise; code at
  [github.com/flavell-lab/AutoCellLabeler](https://github.com/flavell-lab/AutoCellLabeler), parameters in
  flavell-lab/pytorch-3dunet. Pérez-García, Sparks, Ourselin (2021), "TorchIO", *Computer Methods and
  Programs in Biomedicine* 208:106236, arXiv [2003.04696](https://arxiv.org/abs/2003.04696): RandomBlur,
  RandomNoise, RandomGamma and RandomBiasField are already in our torchio 0.20.1. Isensee et al.
  nnU-Net (arXiv [1809.10486](https://arxiv.org/abs/1809.10486)): gamma (0.7-1.5) and low-resolution
  simulation are standard 3D defaults.
- **Why here:** Flavell image features are the weak spot. Preprint per-crop augmentation versus the
  global pipeline differs by about 3 points, which suggests image-branch augmentation choices matter
  there. Photobleaching and per-neuron brightness drift are nuisances that tracking must ignore. Caveat:
  Atanas et al. found that dropping contrast and blur (and the bend warp) gave *similar* performance, so
  the prior that this helps is weak.
- **Plug-in:** Image augmentation, applied per crop after cropping so it stays cheap and row-aligned.
- **Cost:** Hours of coding. Negligible runtime. One run per dataset to test, with Flavell first.
- **Revision fit: high.** It is standard, cheap, and easy to justify to reviewers.

### 5. Axial anisotropy and sub-plane z-shift simulation for crops
- **Summary:** Simulate different z-sampling of the same neuron. Shift the crop by a random sub-plane z
  offset with resampling, and/or downsample-then-upsample along z (TorchIO `RandomAnisotropy`), so the
  network becomes invariant to where the nucleus falls relative to the imaging planes.
- **Citations:** Weigert, Royer, Jug, Myers (2017), "Isotropic reconstruction of 3D fluorescence
  microscopy images using convolutional neural networks", *MICCAI* 2017, arXiv
  [1704.01510](https://arxiv.org/abs/1704.01510): an anisotropic blur-and-subsample degradation model
  learned from the data itself. TorchIO, as above. nnU-Net's low-resolution simulation, as above.
- **Why here:** With 8 z-planes (16 for Flavell), a neuron's appearance changes a lot under a sub-plane
  z shift, and real frames differ by such shifts. The Flavell z-extent finding (z=8 truncates neurons)
  says z-handling matters there. Integer-plane translation already exists, but a sub-plane shift
  probably does not.
- **Plug-in:** Image augmentation. Do it in the global path before cropping, or as a per-crop resample.
- **Cost:** Hours of coding. Negligible runtime.
- **Revision fit: medium-high.** Cheap and physically motivated. The evidence is indirect: no SSL
  tracking paper tests it.

### 6. Intra-volume motion artifact: per-plane xy shift (z-dependent shear) for crops and keypoints
- **Summary:** In freely moving recordings, each z-plane is acquired at a different time during the
  sweep, so motion within the sweep shears the volume. Simulate this with a per-plane xy shift that is
  linear or a short random walk in z, and the matching z-dependent keypoint shift.
- **Citations:** Nguyen et al. (2016), "Whole-brain calcium imaging with cellular resolution in freely
  behaving C. elegans", *PNAS*, arXiv [1501.03463](https://arxiv.org/abs/1501.03463): a piezo sweeps the
  objective at 5 head-volumes/s while a feedback loop tracks the worm. That documents the acquisition
  timing; it does not quantify the artifact.
- **Why here:** It is a real nuisance specific to moving-worm data. Its size can be estimated GT-free
  as the z-dependence of the affine ICP residual between consecutive frames.
- **Plug-in:** Augmentation, global path.
- **Cost:** Hours to implement, plus a CPU check of whether the residual depends on z.
- **Revision fit: medium.** **Speculative**: I found no paper that uses it as an augmentation. Do the
  cheap residual check first and drop the idea if no z-dependence shows up.

### 7. Calibrate augmentation strength with an InfoMin-style sweep, and report it
- **Summary:** Express every geometric augmentation as a multiple of the GT-free estimated noise or
  deformation scale, and sweep one multiplier (for example 0.5x, 1x, 2x, 4x). Expect an interior
  optimum.
- **Citations:** Tian, Sun, Poole, Krishnan, Schmid, Isola (2020), "What Makes for Good Views for
  Contrastive Learning?", *NeurIPS* 2020, arXiv [2005.10243](https://arxiv.org/abs/2005.10243): an
  interior sweet spot in shared information between views. Kim et al. 2021's *AugTune* (in PointWOLF)
  tunes strength automatically to a target confidence, but that is designed for supervised training.
- **Why here:** It turns the "jitter sized from segmentation noise" finding into a principled, figure-ready
  claim ("best at about 1x the data's own noise"), using the hyperparameter infrastructure we already
  have.
- **Plug-in:** A config sweep. No new code.
- **Cost:** 4-5 training runs per dataset, all GPU. Could reuse existing posonly sweeps if they covered
  the jitter scale.
- **Revision fit: high** as an analysis or figure. It is not a method change.

### 8. Spatially correlated, brightness-conditioned object dropout (realistic missed detections)
- **Summary:** Replace i.i.d. object dropout with dropout that is (a) spatially clustered, removing a
  region much like an occlusion, and/or (b) more likely for dim or small objects. Set the rates GT-free
  from per-frame detection-count fluctuations and per-object intensity.
- **Citations:** Wang, Liu, Yue, Lasenby, Kusner (2021), "Unsupervised Point Cloud Pre-Training via
  Occlusion Completion" (OcCo), *ICCV* 2021,
  [CVF link](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Unsupervised_Point_Cloud_Pre-Training_via_Occlusion_Completion_ICCV_2021_paper.html),
  code [github.com/hansen7/OcCo](https://github.com/hansen7/OcCo): structured occlusion rather than random
  point removal. fDNC: up to 20% missing plus spurious neurons. Zhu, Fan, Weng (2024) point cloud
  augmentation survey, *Pattern Recognition*, arXiv [2308.12113](https://arxiv.org/abs/2308.12113).
- **Why here:** Real misses are not i.i.d.: dim neurons and the edge of the field of view fail
  together. **Caution:** our plane cuts and synthetic segmentation errors did not help, and plane cuts
  are a form of structured occlusion. The prior is weak.
- **Plug-in:** Augmentation, by modifying the existing dropout.
- **Cost:** Hours. Negligible runtime.
- **Revision fit: medium-low.** Cheap, but it overlaps strongly with things that already failed.

### 9. Separate invariant and variant heads for augmentations that may remove useful information
- **Summary:** Learn several projector heads, each one *not* invariant to one augmentation (for
  example global rotation or flip in the image branch, or intensity). Concatenate their outputs for
  tracking.
- **Citations:** Xiao, Wang, Efros, Darrell (2021), "What Should Not Be Contrastive in Contrastive
  Learning", *ICLR* 2021, arXiv [2008.05659](https://arxiv.org/abs/2008.05659). Dubois et al. (2022),
  "Improving Self-Supervised Learning by Characterizing Idealized Representations", *NeurIPS* 2022
  (Zotero BUUWPMBE).
- **Why here:** Image crops of a fixed-orientation worm carry orientation information that is useful
  for identity. Making the image branch fully invariant to rotation and flip may throw it away, and the
  fused network gets position-only information from ViSNet anyway.
- **Plug-in:** Loss and projector variant: extra Barlow heads.
- **Cost:** About 1 day of coding, plus a modest extra GPU cost for the heads.
- **Revision fit: medium-low.** It is closer to a method change, and its benefit here is untested.

### 10. Data-driven bending from the recording's own centerlines (eigenworm sampling)
- **Summary:** If the old bending used hand-set curvature, sample curvature *changes* between frames
  from the recording's own behavioural centerlines or their eigenworm coefficients, and apply them
  through centerline coordinates, as in fDNC's pose step.
- **Citations:** Yu, Creamer, Randi, Sharma, Linderman, Leifer (2021), "Fast deep neural correspondence
  for tracking and identifying neurons in C. elegans using semi-synthetic training", *eLife* 10:e66410,
  DOI [10.7554/eLife.66410](https://doi.org/10.7554/eLife.66410), code
  [github.com/XinweiYu/fDNC_Neuron_ID](https://github.com/XinweiYu/fDNC_Neuron_ID). Its simulator
  straightens, applies a CPD warp to a random real template, a transverse affine, 20% missing/spurious
  neurons, a head centerline from real recordings, up to 5% rescale, and 0.42 µm noise; no augmentation
  ablation is reported. Stephens, Johnson-Kerner, Bialek, Ryu (2008), "Dimensionality and dynamics in
  the behavior of C. elegans", *PLoS Comput Biol* 4:e1000028,
  [PMC2276863](https://pmc.ncbi.nlm.nih.gov/articles/PMC2276863): four eigenworms capture 95% of shape
  variance.
- **Why here:** It is the most "biomechanically realistic" bending available without GT.
- **Plug-in:** Keypoint and global augmentation. It needs centerlines in head-volume coordinates.
- **Cost:** 1-2 days, mostly for registering behaviour centerlines to volume coordinates.
- **Revision fit: low.** Two independent nulls (ours and Atanas et al.) plus argument 2 in Part 1. Idea
  1 captures the same realism more directly. Cite fDNC in the text rather than implementing this.

### 11. Point-cloud mixing (PointMixup, RSMix) and SSL-style point cloud pretraining
- **Summary:** Mix two point clouds, or rigid subsets of them, into a new sample. PointContrast-style
  point-level contrastive pretraining.
- **Citations:** Zhu et al. 2024 survey (arXiv [2308.12113](https://arxiv.org/abs/2308.12113)) covers
  PointMixup and RSMix and notes the damage to local structure and the boundary discontinuities. Xie,
  Gu, Guo, Qi, Guibas, Litany (2020), "PointContrast", *ECCV* 2020, arXiv
  [2007.10985](https://arxiv.org/abs/2007.10985), code
  [facebookresearch/PointContrast](https://github.com/facebookresearch/PointContrast).
- **Why here:** Mixing destroys the identity of objects inside a frame, which is the one thing our
  per-object loss relies on. PointContrast is in essence what we already do: point-level positives
  across views with known correspondence. It is worth *citing* as related work, but it is not a new
  augmentation.
- **Revision fit: low** as a method. **Medium** as a related-work citation for "per-object Barlow is
  the point-level analogue of PointContrast".

### 12. Models Genesis-style non-linear intensity curves and local voxel shuffling for crops
- **Summary:** Random monotone Bezier intensity remaps and local patch shuffling inside the crop.
- **Citation:** Zhou et al. (2019), "Models Genesis: Generic Autodidactic Models for 3D Medical Image
  Analysis", *MICCAI* 2019, arXiv [1908.06912](https://arxiv.org/abs/1908.06912), journal version arXiv
  [2004.07882](https://arxiv.org/abs/2004.07882), code
  [github.com/MrGiovanni/ModelsGenesis](https://github.com/MrGiovanni/ModelsGenesis).
- **Why here:** It is a broader intensity nuisance family than gamma. Shuffling would probably destroy
  the fine texture that distinguishes neighbouring nuclei, so only the intensity curve is worth testing.
- **Revision fit: low-medium.** Idea 4 covers most of it more simply.

---

## Part 3. Suggested minimal experiment set (fits a revision)

1. Idea 7: a strength sweep of the existing jitter. It gives a figure and reuses the infrastructure.
2. Ideas 1 + 2: a real-deformation library applied cheaply per crop. This is the most novel GT-free
   addition and also fixes the 4x elastic cost.
3. Ideas 4 + 5: photometric and sub-plane-z augmentations on Flavell image-only and fused networks.
4. Idea 3 as a single ablation: PointWOLF with bandwidth below the cutoff, on position-only Samuel or
   Zimmer.
5. In the text: cite Atanas et al. 2025 and fDNC (no ablation) when reporting the bending and
   segmentation-error null results, and give the invariance argument (Part 1, point 2).

## Other sources consulted (context, not ranked)
- Nguyen et al. (2017), "Automatically tracking neurons in a moving and deforming brain", *PLoS Comput
  Biol*, [PMC5436637](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5436637/): non-rigid point-set
  registration plus thin-plate splines across reference volumes.
- Ryu et al. (2024), "Versatile multiple object tracking in sparse 2D/3D videos via deformable image
  registration" (ZephIR), *PLoS Comput Biol* 20:e1012075, DOI
  [10.1371/journal.pcbi.1012075](https://doi.org/10.1371/journal.pcbi.1012075): tunable rigidity and
  temporal priors. A registration method, not an augmentation.
- Wen et al. (2021), 3DeeCellTracker, *eLife* 10:e59187,
  [link](https://elifesciences.org/articles/59187): FFN trained on synthetic point sets made with an
  affine transform plus per-point jitter, i.e. our baseline augmentation family.
- Chaudhary, Lee, Li, Patel, Lu (2021), CRF_ID, *eLife* 10:e60321,
  [link](https://elifesciences.org/articles/60321): relative positional relationships between cells are
  more stable than absolute positions. This supports the case for local, invariant position encoders
  over heavy global deformation.
- Wang et al. (2024), ViSNet, *Nature Communications* 15:313, DOI
  [10.1038/s41467-023-43720-2](https://doi.org/10.1038/s41467-023-43720-2).
- Purushwalkam & Gupta (2020), *NeurIPS*, arXiv [2007.13916](https://arxiv.org/abs/2007.13916).
- Gallusser & Weigert (2024), Trackastra, *ECCV*, arXiv [2405.15700](https://arxiv.org/abs/2405.15700):
  checked for augmentation details, but none were in the abstract-level sources.
