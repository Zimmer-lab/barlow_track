# Encoders: literature ideas for the BarlowTrack revision (2026-10-29)

Scope: (1) point-set / geometric encoders over ~90-210 centroids per frame (to replace or
complement ViSNet); (2) image backbones for 8x64x64 anisotropic crops and image+position fusion;
(3) neuron-tracking-specific encoders. Sources: web (arXiv, publisher pages, GitHub) and the
user's Zotero library (items marked [Zotero]). Read-only; nothing in the repo, tree or Zotero was
changed.

## What I checked in the code first (these facts shape the ranking)

- **Keypoints reach the position encoder in voxel units.** `superglue.normalize_keypoints`
  (called from `volume_data._normalize`) subtracts the volume centre and divides all three axes by
  the same scalar (0.7 x the largest volume dimension). Nothing rescales z to physical units
  (`grep` finds no use of wbfm's `PhysicalUnitConversion` in the training path). On Zimmer data
  z is 1.5 um/px and xy is 0.325 um/px (`wbfm/utils/projects/physical_units.py`), so z is
  squashed about 4.6x relative to physical space. ViSNet's SO(3) invariance and its spherical
  radius ball therefore act in a non-physical metric, and that metric differs by dataset.
  From the notebook's GT-free stats, the per-axis within-frame NN median (z, x, y) in voxels is
  Zimmer (2.9, 4.5, 4.5), Flavell (2.9, 3.9, 2.7), Samuel (1.0, 2.9, 2.7).
- **The cutoff is partly a kNN cap already.** The pure-torch `radius_graph` fallback keeps the
  *nearest* `max_num_neighbors` (32) inside the radius. The code comment says a Zimmer frame has
  about 120 neighbours inside 0.35, so on Zimmer the graph is in effect kNN-32. On sparser
  datasets the radius is what binds. The `cutoff` vs `cutoff_nn` results therefore mix together
  three things: radius, neighbour cap and RBF span.
- **Where the RBF bug comes from.** pyg's `ExpNormalSmearing` uses `alpha = 5/cutoff`, but its
  means span `[exp(-cutoff), 1]`. I confirmed that TorchMD-Net's original (`torchmdnet/models/utils.py`,
  "initialize means and betas according to the default values in PhysNet") has the same form.
  The two only agree at a cutoff of about 5 (Angstrom-scale molecules). By contrast, pyg's SchNet
  `GaussianSmearing` (`linspace(0, cutoff)`) and DimeNet's `BesselBasisLayer` (`sin(n*pi*d/cutoff)`
  with an envelope) are scale-correct by construction. pyg's ViSNet defaults to `trainable_rbf=False`.
- **Available locally:** pyg 2.6.1 (SchNet, DimeNet/DimeNet++, ViSNet, `PointTransformerConv`,
  `EdgeConv`, `DynamicEdgeConv`, `PPFConv`, `PointGNNConv`), torch_scatter and torch_sparse.
  e3nn is **not** installed. The cluster has no torch-cluster: `DynamicEdgeConv` and `knn_graph`
  call `torch_cluster.knn`, which the stub refuses. DimeNet's triplets need torch_sparse, so check
  the cluster before using it. Anything below marked "pure torch" can be written with
  `torch.cdist` + `topk` (N <= 210, so dense N x N is trivial).
- **Image backbone in use:** `ResidualEncoder3D` with f_maps=4 and num_levels=2. It uses isotropic
  3x3x3 kernels and isotropic MaxPool(2), which halves z from 8 to 4. A final 1x1 conv goes to 1
  channel, then flatten (about 4096 values), then Linear, then Sigmoid.
- **Cost reference (notebook):** a local position-only `posonly_fast` cell is about 2 min
  (3-epoch smoke) up to about 27 min (15 epochs, 600 frames). Labelprop scoring runs on CPU.

## Ranked ideas

### 1. Put keypoints in an isotropic (physical or neighbour-normalized) metric before the position encoder

- **Summary:** before ViSNet, rescale z by the voxel aspect ratio (from metadata), or by a GT-free
  per-axis NN-spacing ratio. That makes the radius ball, the RBFs and the rotation invariance
  physically meaningful, and the same across datasets.
- **Literature basis:** anisotropy-aware design is standard in 3D microscopy and medical
  segmentation. nnU-Net derives kernel and pooling choices from the dataset's voxel spacing
  (Isensee et al., arXiv [1809.10486](https://arxiv.org/abs/1809.10486) /
  [1904.08128](https://arxiv.org/abs/1904.08128); code public). StarDist-3D handles anisotropy
  explicitly in its rays and grid (Weigert et al., WACV 2020,
  [10.1109/WACV45572.2020.9093435](https://doi.org/10.1109/WACV45572.2020.9093435) [Zotero];
  code public). Every equivariant GNN below (SchNet, DimeNet, EGNN, ViSNet) assumes a Euclidean
  metric. Applying one to a 4.6x-squashed axis breaks the symmetry it is built around. This
  specific combination (equivariant GNN plus anisotropic voxel centroids) is my inference, not a
  published result.
- **Why it might matter here:** this is the most plausible single explanation I found for the
  `cutoff_nn` sign flip. In voxel units Samuel's z spacing is about 3x smaller than its xy
  spacing (NN 1.0 vs 2.9), so its cloud is nearly flat. Zimmer's is less flat (2.9 vs 4.5). A
  median-NN rescale therefore changes the effective z weight differently per dataset. A cheap
  related signal is already in hand: anisotropic jitter sized GT-free beat isotropic jitter.
- **How it plugs in:** encoder input, one flag. Example: `keypoint_metric: voxel | physical | nn_whiten`,
  applied inside `ViSNetPositionEncoder.forward` before `cutoff_nn` and `rbf_rescale`.
  `physical` multiplies z by `um_per_px_z / um_per_px_xy`. `nn_whiten` divides each axis by its
  within-frame median per-axis NN offset, which is GT-free and already computed in
  `dataset_stats.py`. Retune `visnet_cutoff` after the change, since the units move.
- **Cost:** under 1 h of coding. 3 datasets x {voxel, physical, nn_whiten} x {fixed, cutoff_nn}
  at about 27 min each on the local GPU, so about 8 GPU-h for one seed (labelprop on CPU).
- **Fit for revision: HIGH.** It is a preprocessing correction, not a method change. It uses only
  metadata or GT-free statistics. It may unify the cutoff choice across datasets, which is the
  "all datasets" goal. Caveat: if the per-lab projects have already been resampled, the physical
  variant does nothing. Check each dataset's z/xy spacing first.

### 2. Give the encoder both a scale-normalized and an absolute distance channel (two-scale RBF / multi-scale grouping)

- **Summary:** instead of choosing between a fixed cutoff (wins on Zimmer) and a scale-normalized
  one (wins on Samuel), feed both. Use an RBF of d/s_frame and an RBF of d in absolute units,
  plus the scalar s_frame. Or run two ViSNet branches at the two scales and concatenate them.
- **Literature basis:**
  - 3DeeCellTracker's point-matching FFN does exactly this for neurons. Each cell is the 20
    nearest-neighbour offsets divided by their mean distance d, and "the mean distance d was
    included as the last value" (61-D input). Wen et al., eLife 2021,
    [10.7554/eLife.59187](https://elifesciences.org/articles/59187v1); code public
    (3DeeCellTracker on GitHub/PyPI).
  - PointNet++ multi-scale grouping (MSG) concatenates features from several radii, so the
    network can cope with non-uniform density. Its authors report that networks trained on
    uniform density do much worse on varying density. Qi et al., NeurIPS 2017,
    [arXiv 1706.02413](https://arxiv.org/abs/1706.02413); code public.
- **Why it might matter here:** it targets the dataset-dependent sign of `cutoff_nn` directly.
  The network can learn which scale to trust instead of a hyperparameter being picked per dataset.
- **How it plugs in:** encoder. The cheapest version runs two `ViSNetPositionEncoder` instances,
  one with `cutoff_nn=None` and one with `cutoff_nn=k`, each at hidden/2 so the cost stays about
  the same, then concatenates them and applies `head`. A finer version (patch ExpNormal into two
  RBF banks) means subclassing pyg's `ViSNetBlock` and is more work.
- **Cost:** 1-2 h of coding. About 2x a single ViSNet forward at equal width (it is small, N <= 210).
  Same sweep size as idea 1.
- **Fit for revision: HIGH.** It is a drop-in encoder variant with a precedent in a C. elegans
  neuron tracker. Combine it with idea 1 and test both together.

### 3. Make the graph an explicit kNN graph and sweep k, not the radius

- **Summary:** replace the radius graph with a kNN graph (k in {8, 16, 32}), with distances
  normalized per node or per frame. This removes the hidden interaction between radius and
  neighbour cap.
- **Literature basis:**
  - DGCNN builds kNN graphs, in feature space after the first layer (Wang et al., ACM TOG 2019,
    [10.1145/3326362](https://doi.org/10.1145/3326362) [Zotero]; code public).
  - 3DeeCellTracker uses kNN-20 for neurons (above).
  - PCT uses farthest-point sampling plus nearest-neighbour search for its local embedding (Guo
    et al., Comput. Vis. Media 2021, [10.1007/s41095-021-0229-5](https://doi.org/10.1007/s41095-021-0229-5);
    code public).
  - I found no clean published ablation of kNN vs radius under density shift (searched).
    Evidence for "kNN adapts to density" is general rather than benchmarked.
- **Why it might matter here:** on Zimmer the current graph is already about kNN-32, because the
  cap binds. On Samuel the radius binds. Making k explicit should show whether "cutoff" effects are
  really neighbourhood-size effects. kNN is also invariant to global scale by construction.
- **How it plugs in:** config only, nearly. Set `visnet_cutoff` large and `visnet_neighbors=k`.
  The pure-torch fallback then keeps the k nearest, which is a kNN graph. The RBFs must then span
  the kNN distance range, so pair this with `cutoff_nn` or idea 4. Note: a kNN graph is directed
  (not symmetric). ViSNet's message passing tolerates this, but say so in the methods.
- **Cost:** about 0 h of coding, about 1 GPU-h per dataset per k.
- **Fit for revision: HIGH** as a cheap, interpretable ablation that a reviewer would understand
  ("the neighbourhood is k neighbours, independent of scale").

### 4. Use a radial basis that is scale-correct by construction (Bessel + envelope, or SchNet Gaussians; or `trainable_rbf=True`)

- **Summary:** remove the possibility of RBF mis-scaling. Options: (a) DimeNet's
  `sin(n*pi*d/c)/d` with a polynomial envelope; (b) evenly spaced Gaussians on [0, c];
  (c) keep ExpNormal but set `alpha` consistent with the means, plus `trainable_rbf=True`.
- **Literature basis:**
  - DimeNet: Klicpera, Gross, Guennemann, ICLR 2020, [arXiv 2003.03123](https://arxiv.org/abs/2003.03123).
    It reports that the Bessel basis beats Gaussian RBFs with fewer parameters, that the envelope
    makes features smooth at the cutoff, and that results are insensitive to the envelope exponent.
    DimeNet++: Gasteiger et al. [arXiv 2011.14115](https://arxiv.org/abs/2011.14115). Both in pyg.
  - SchNet: Schuett et al., NeurIPS 2017, [arXiv 1706.08566](https://arxiv.org/abs/1706.08566); in pyg.
  - ExpNormal source: PhysNet (Unke & Meuwly, JCTC 2019, [arXiv 1902.08408](https://arxiv.org/abs/1902.08408)),
    via TorchMD-Net (Thoelke & De Fabritiis, ICLR 2022, [arXiv 2202.02541](https://arxiv.org/abs/2202.02541);
    code [torchmd/torchmd-net](https://github.com/torchmd/torchmd-net), MIT). The 5 Angstrom
    assumption is in both.
- **Why it might matter here:** the RBF fix was decisive. With a scale-correct basis, the paper
  can state the fix as a principled choice ("an RBF whose support equals the cutoff") rather than
  a patch. It also stops the bug from coming back when ideas 1-3 change the units.
- **How it plugs in:** encoder. Replace `self.repr.distance_expansion` after construction with a
  Bessel or Gaussian module of the same `num_rbf`. This is a small subclass and pure torch.
  The `rbf_rescale` path is mathematically one fix; this is the general one.
- **Cost:** about 1 h of coding plus a 3-dataset check against `rbf_rescale` (about 3-4 GPU-h).
- **Fit for revision: MEDIUM-HIGH.** Low risk. The expected accuracy gain over `rbf_rescale` is
  small, but it strengthens the methods text and makes the RBF robust to the unit changes in 1-3.

### 5. A lightweight pure-torch EGNN / PaiNN-style encoder, including an E(2)-about-z variant that matches the worm's symmetry

- **Summary:** a dense, few-layer EGNN (scalar messages from ||x_i - x_j||^2) or PaiNN (scalar
  plus vector channels), written in about 80 lines of pure torch. The variant is invariant only
  to rotations in the imaging plane (about z) and to flips, with Δz kept as a separate invariant
  edge feature. The motivating geometry: the worm lies on its side and the head mostly rotates
  in-plane, which is also all that `sample_global_affine` rotates. Head roll about the body axis
  is mostly a small rotation.
- **Literature basis:**
  - EGNN: Satorras, Hoogeboom, Welling, ICML 2021, [arXiv 2102.09844](https://arxiv.org/abs/2102.09844)
    (authors' code on GitHub, not checked here). Equivariant without spherical harmonics, cheap,
    and works in any dimension, so a 2D-plus-scalar-z version is natural.
  - PaiNN: Schuett, Unke, Gastegger, ICML 2021, [arXiv 2102.03150](https://arxiv.org/abs/2102.03150).
    It argues that equivariant (vector) features fix the expressivity limits of invariant-only
    message passing, with a smaller and faster model.
  - SO(2)-reduced convolutions: Passaro & Zitnick, eSCN, ICML 2023,
    [arXiv 2302.03655](https://arxiv.org/abs/2302.03655). This is the conceptual precedent for
    using SO(2) instead of SO(3); I am not suggesting their code.
- **Why it might matter here:** ViSNet already beats the keypoint MLP everywhere, so invariance
  pays off. But full SO(3) invariance in a squashed voxel metric throws away the z vs xy
  distinction (roughly left/right vs dorsal/ventral). An E(2)xz encoder keeps that distinction
  and is still invariant to the augmentation that is applied. It is also a GPU-efficiency play
  (no SH or vector-norm instabilities). Speculative: no paper tests this on neurons.
- **How it plugs in:** new `position_encoder: egnn | egnn_z` next to `visnet` in
  `BarlowWithPosition`. Same (N, 3) in and (N, D) out. Use a kNN or radius mask from idea 3.
- **Cost:** 3-5 h coding and tests, plus a 3-dataset x 2-variant sweep (about 6 GPU-h).
  Expected forward cost is below ViSNet's.
- **Fit for revision: MEDIUM.** It is a different encoder (allowed), but adds a new component.
  It is worth it only if ideas 1-3 do not already close the dataset gap. The z-only symmetry
  argument is a clean, reviewer-friendly story if it wins.

### 6. Global distance-biased self-attention (no hard cutoff): Uni-Mol / fDNC style

- **Summary:** a small transformer over all neurons in the frame. Attention logits get an
  additive bias MLP(RBF(d_ij)), with learnable Gaussian means and widths over the full distance
  range. This is invariant to rotation and translation, and the cutoff becomes something learned.
  It can stand alone or sit on top of ViSNet as global context.
- **Literature basis:**
  - Uni-Mol: Zhou et al., ICLR 2023 ([OpenReview/ICLR page](https://iclr.cc/virtual/2023/poster/11617)).
    Pairwise distances pass through learnable Gaussian kernels and become attention biases,
    giving SE(3)-invariant attention. Code: dptech-corp/Uni-Mol, per search, not checked.
  - fDNC: Yu et al., eLife 2021, [10.7554/eLife.66410](https://pmc.ncbi.nlm.nih.gov/articles/PMC8367385/)
    [Zotero]; code [XinweiYu/fDNC_Neuron_ID](https://github.com/XinweiYu/fDNC_Neuron_ID). A
    6-layer, 8-head full-attention transformer on about 120 neuron positions alone. It reaches
    79% within-animal tracking accuracy (supervised on semi-synthetic data, so the setting is not
    comparable to ours). The paper does not describe how coordinates are embedded or made
    invariant.
- **Why it might matter here:** it removes the cutoff hyperparameter that flips between datasets.
  At N <= 210, full attention costs next to nothing. The repo's `AttentionalGNN` self-layers
  already exist, and the change is a pairwise bias term.
- **How it plugs in:** encoder or contextualizer. Add an optional `dist_bias` to the self-attention
  in `BarlowVolumeAttention` (logits += bias_ij). For position-only, a 2-4 layer stack over
  learned per-node tokens (or ViSNet outputs).
- **Cost:** 3-4 h of coding, about 1.5x a position-only cell. Attention was noted as "still open"
  in the sweeps, so give it 2 seeds.
- **Fit for revision: MEDIUM.** It is architecture-adjacent, but it is an extension of the
  existing attention layer. Risk: attention fusion has been boom-or-bust. Initialise it gated, as
  the current `attn_gate` does.

### 7. Relative (rotary) positional encoding inside the existing self-attention, as in LightGlue

- **Summary:** instead of fusing positions once at the input (SuperGlue-style), apply rotary
  encodings of relative positions to the queries and keys in every self-attention layer.
- **Literature basis:**
  - LightGlue: Lindenberger, Sarlin, Pollefeys, ICCV 2023, [arXiv 2306.13643](https://arxiv.org/abs/2306.13643);
    code [cvg/LightGlue](https://github.com/cvg/LightGlue). They observed that SuperGlue's
    absolute MLP encoding is "forgotten" through the layers. In their ablation, rotary relative
    encoding gives +2.6 precision and +1.6 recall over absolute positions, at about 4% runtime cost.
  - SuperGlue: Sarlin et al., CVPR 2020, [arXiv 1911.11763](https://arxiv.org/abs/1911.11763) [Zotero].
  - DeepMatcher (Xie et al. 2023, [arXiv 2301.02993](https://arxiv.org/abs/2301.02993) [Zotero])
    also adds relative position encodings in every attention layer.
- **Why it might matter here:** fused networks reach 0.97-0.99, and "attention/fusion still open"
  is in the notes. This is the documented fix for the failure mode of the current design (one
  early position fusion, then self-attention). Rotary encoding is translation-invariant but not
  rotation-invariant. That is fine with z-rotation augmentation, or use ViSNet features as the
  invariant part.
- **How it plugs in:** `superglue.AttentionalGNN` / `MultiHeadedAttention`. Rotate q and k by
  learned Fourier projections of p_j - p_i (as in LightGlue). Pure torch.
- **Cost:** 2-3 h of coding and one fused sweep on 3 datasets (GPU, about 1 h per cell with images).
- **Fit for revision: MEDIUM.** Drop-in and well cited. Gains come from homography matching and
  may not transfer to our setting.

### 8. Fusion: FiLM or gated-add instead of concat

- **Summary:** position descriptors predict a per-channel scale and shift (FiLM) for the image
  descriptor, or a learned scalar or vector gate blends `add`. This sits between `add` (never
  collapses) and `concat` (boom-or-bust).
- **Literature basis:** FiLM: Perez et al., AAAI 2018, [arXiv 1709.07871](https://arxiv.org/abs/1709.07871)
  (code public, per listing). Evidence for our setting is generic and weak; I found no
  tracking-specific comparison of fusion operators.
- **Why it might matter here:** concat's instability suggests that one branch can capture the
  other's LayerNorm statistics. FiLM with an identity init (gamma=1, beta=0) starts out as image-only
  and then lets position condition it. It also helps "position optional": a position-dropout
  switch is trivial.
- **How it plugs in:** `fusion: film | gated_add` in `BarlowWithPosition.fused_descriptors`
  (about 15 lines).
- **Cost:** under 1 h of coding, one fused sweep.
- **Fit for revision: MEDIUM.** Cheap and clearly a drop-in. Expected value is modest.

### 9. Anisotropy-aware image backbone: in-plane-first kernels and pooling, (2+1)D factorization, and a less lossy head

- **Summary:** use 1x3x3 kernels and (1,2,2) pooling until the effective spacing is roughly
  isotropic (nnU-Net rule). Optionally factorize 3x3x3 convs into 1x3x3 followed by 3x1x1
  (R(2+1)D). Raise f_maps from 4, and replace the 1-channel flatten + Linear + Sigmoid head with
  more channels plus global or attention pooling, or keep flatten but with more channels.
- **Literature basis:**
  - nnU-Net anisotropy heuristics (arXiv [1904.08128](https://arxiv.org/abs/1904.08128)): pool the
    high-resolution axes first, use 1x3x3 kernels on anisotropic data.
  - R(2+1)D: Tran et al., CVPR 2018, [arXiv 1711.11248](https://arxiv.org/abs/1711.11248). At the
    same parameter count it has more nonlinearities and is easier to optimize; code in pytorchvideo.
  - MedNeXt (Roy et al., 2023, [arXiv 2303.09975](https://arxiv.org/abs/2303.09975); code
    MIC-DKFZ/MedNeXt) is a ConvNeXt-style 3D option. It is probably overkill for 8x64x64 crops.
- **Why it might matter here:** the current backbone pools z from 8 to 4 immediately, with
  isotropic kernels, on data whose z voxel is 4.6x coarser. That spends z resolution where it is
  scarcest. Flavell needed 16 z-planes, and its features are the weak point (0.62-0.73 flat across
  architectures; the bottleneck is "data/crop"). An anisotropy-matched backbone is the encoder-side
  test of that diagnosis. f_maps=4 and a 1-channel bottleneck are also very narrow; widening is cheap.
- **How it plugs in:** `backbone_kwargs` (`conv_kernel_size`, `pool_kernel_size` are already
  arguments of `Abstract3DEncoder`/`create_encoders`; check that per-level tuples are supported).
  A pooled head is a small subclass.
- **Cost:** about 1-2 h, then image-only cells (GPU, slower than position-only). Fewer epochs are
  possible with lambd_obj>0, per the early results.
- **Fit for revision: MEDIUM.** It is "a different encoder", well within scope. The gain is
  uncertain; the Flavell result is the most likely to move.

### 10. Non-invariant encoders made invariant by canonicalization: PCA frame averaging, plus Fourier-feature keypoint MLP

- **Summary:** keep a plain MLP or transformer over coordinates, but (a) lift them with random
  Fourier features and (b) make the result exactly invariant by averaging over the PCA frames of
  the frame's point cloud (4 sign-flip frames for SE(3)), or with a small learned canonicalizer.
- **Literature basis:**
  - Frame Averaging: Puny et al., ICLR 2022, [arXiv 2110.03336](https://arxiv.org/abs/2110.03336);
    code [omri1348/Frame-Averaging](https://github.com/omri1348/Frame-Averaging). The PCA frame is
    centroid plus covariance eigenvectors with sign ambiguity, giving 2^(d-1) frames for SE(d). It
    is undefined or unstable when eigenvalues are nearly equal.
  - Learned canonicalization: Kaba et al., ICML 2023, [arXiv 2211.06489](https://arxiv.org/abs/2211.06489).
  - Fourier features: Tancik et al., NeurIPS 2020, [arXiv 2006.10739](https://arxiv.org/abs/2006.10739).
    Plain coordinate MLPs suffer from spectral bias; Fourier features fix it.
- **Why it might matter here:** this tests whether ViSNet wins because of message passing or only
  because of invariance, which is a natural reviewer question. The worm head's long axis gives a
  well-separated first PCA axis. The two transverse axes may be close to degenerate; that is exactly
  where frame averaging is fragile, and it should be reported if it fails.
- **How it plugs in:** encoder input. `canonicalize_keypoints` already removes translation and
  scale; add PCA rotation and the 4-frame average around `KeypointEncoder`.
- **Cost:** 2 h, position-only cells.
- **Fit for revision: LOW-MEDIUM.** Mostly a diagnostic or ablation for the paper, not an
  expected accuracy gain.

### 11. A 3DeeCellTracker-style kNN descriptor MLP as a cheap position-only baseline

- **Summary:** a per-neuron descriptor from the sorted k nearest-neighbour offsets, normalized by
  their mean distance (plus that distance), passed through an MLP. Train it with the same Barlow loss.
- **Literature basis:** Wen et al., eLife 2021 (above). This is the published neuron-tracking
  position encoder closest to "no GNN".
- **Why it might matter here:** it gives a published, reviewer-recognizable lower bound for the
  position-only result (0.94-0.98), showing what ViSNet's message passing adds. It is not
  rotation-invariant (offsets are in voxel axes). Rely on z-rotation augmentation, or combine with
  idea 10.
- **How it plugs in:** new `position_encoder: knn_mlp`, pure torch (cdist + topk).
- **Cost:** about 1 h, 3 position-only cells.
- **Fit for revision: MEDIUM** as an ablation or baseline. LOW as a replacement.

### 12. Higher-order equivariant GNNs (MACE, Equiformer, SEGNN, TFN): not recommended for the revision

- **Summary and citations:**
  - TFN: Thomas et al. 2018, [arXiv 1802.08219](https://arxiv.org/abs/1802.08219).
  - SEGNN: Brandstetter et al., ICLR 2022, [arXiv 2110.02905](https://arxiv.org/abs/2110.02905).
  - MACE: Batatia et al., NeurIPS 2022, [arXiv 2206.07697](https://arxiv.org/abs/2206.07697);
    code [ACEsuit/mace](https://github.com/ACEsuit/mace). Many-body messages, 1-2 layers.
  - Equiformer: Liao & Smidt, ICLR 2023, [arXiv 2206.11990](https://arxiv.org/abs/2206.11990).
- **Why not:** they need e3nn (not installed; another cluster dependency) and cost more per edge.
  Their gains come on energy and force regression, where higher angular order matters. ViSNet with
  lmax=1 already encodes angles and dihedrals at runtime. The notes say embedding dimension barely
  matters once the encoder is right, which also argues against chasing expressivity. MACE's
  "fewer layers with many-body messages" is the one idea worth keeping in mind for efficiency.
- **Fit for revision: LOW.**

### 13. Point transformers and PointNet-family backbones (PTv1/v3, PCT, PointNet++, DGCNN): low value as replacements

- **Citations:**
  - Point Transformer: Zhao et al., ICCV 2021, [arXiv 2012.09164](https://arxiv.org/abs/2012.09164).
  - PTv3: Wu et al., [arXiv 2312.10035](https://arxiv.org/abs/2312.10035).
  - PCT (above). PointNet++ (above). DGCNN (above).
  - pyg has `PointTransformerConv`, `EdgeConv`, `PPFConv`. `DynamicEdgeConv` needs torch_cluster's
    knn, so it would need a pure-torch kNN on the cluster.
- **Why low:** they are designed for 10^4-10^6 points. PTv3's serialization, pooling hierarchies
  and FPS are pointless at N of about 170. Most are not rotation-invariant, and ViSNet's win over
  the MLP says that invariance matters here. Their transferable ideas are already extracted above:
  multi-scale grouping (idea 2), kNN graphs (idea 3), relative and offset attention (ideas 6-7).
  `PPFConv` (point-pair features, rotation-invariant) is the one pyg layer that would be a fair
  quick comparison if a reviewer asks for a non-molecular invariant baseline.
- **Fit for revision: LOW.**

### 14. Neuron-tracking-specific encoders: positioning, not adoption

- **fDNC** (Yu et al., eLife 2021, above; code public): supervised, semi-synthetic position-only
  transformer. Its augmentations (CPD warp to other templates, transverse-plane affine for head
  rotation, about 20% spurious or missing neurons, centerline deformation, 5% xy scaling, 0.42 um
  noise) are a useful checklist for the augmentation agent.
- **3DeeCellTracker** (Wen et al., eLife 2021; code public): kNN-20 normalized FFN plus PR-GLS
  registration (ideas 2, 3, 11).
- **ZephIR** (Ryu et al., PLoS Comput Biol 2024, [10.1371/journal.pcbi.1012075](https://doi.org/10.1371/journal.pcbi.1012075);
  open-source package): registration with spatial and temporal priors. A different paradigm (out of scope).
- **BrainAlignNet / AutoCellLabeler** (Atanas et al., eLife 2025, [elifesciences.org/articles/108159](https://elifesciences.org/articles/108159)):
  registration network, 99.6% linking on Flavell data. Relevant as a Flavell comparator, not as an encoder.
- **Trackastra** (Gallusser & Weigert, ECCV 2024, [arXiv 2405.15700](https://arxiv.org/abs/2405.15700);
  code weigertlab/trackastra): transformer over detections in a time window for linking. Supervised.
- **NeRVE** (Nguyen et al., PLoS Comput Biol 2017, [PMC5436637](https://pmc.ncbi.nlm.nih.gov/articles/PMC5436637/)):
  point-set registration against reference volumes.
- **For the revision:** cite these as the landscape. None of them is a self-supervised,
  per-recording encoder. That is BarlowTrack's distinguishing point, and fDNC is the closest
  "position-only learned encoder" comparator.
- **Fit for revision:** citations only.

## Suggested order of work (cheapest to most informative)

1. Ideas **1 + 3** together, which are config-level: metric {voxel, physical, nn_whiten} x graph
   {radius 0.35 cap 32, kNN 16, kNN 32} x {Zimmer, Flavell, Samuel}, position-only, 1 seed (about
   8-10 local GPU-h). If one setting wins on all three, the `cutoff_nn` flip is explained and the
   paper gets a single rule.
2. Idea **2** (two-scale) if no single setting wins. Idea **4** (Bessel RBF) as a robustness swap
   alongside.
3. For fused networks: idea **9** (anisotropic backbone) on Flavell first, then **8/7**.
4. Ideas **5/6** only if 1-3 leave a dataset gap. Idea **10/11** as paper ablations if reviewers
   ask "why ViSNet?".

## Honesty notes

- Ideas 1-3 rest on my reading of this repo plus general literature principles. No paper studies
  equivariant GNNs on anisotropic voxel centroids, and "the cutoff flip is caused by anisotropy" is
  a hypothesis.
- I did not verify EGNN's or Uni-Mol's code repositories directly. Citation details for those two
  come from conference and arXiv pages only.
- The nnU-Net kernel rule is cited via its arXiv preprints and a secondary description. The Nature
  Methods version was not opened.
