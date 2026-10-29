# Competing trackers a reviewer would expect on the BarlowTrack benchmark

Literature and code survey, 2026-10-29. Sources: web (journal pages, bioRxiv, GitHub API) and Zotero (the
library holds only the BarlowTrack preprint and fDNC on this topic; most papers below are **not yet in Zotero**).
Repo metadata (licence, last push) comes from the GitHub API on 2026-10-29. Accuracy numbers are copied from
the papers; every paper uses its own metric, so they are not directly comparable to each other or to ours.

## Headline findings

1. **ASCENT (Han & Lu, ICCV 2025) is the closest competitor and is missing from the preprint.** It is a
   self-supervised contrastive method (SimCLR / NT-Xent) on two augmented views of a frame. A transformer takes
   image patches plus centroid positions, and it is trained per recording. It reports **95.90% on the same
   Leifer NeRVE recording** (BarlowTrack preprint: 97.4%; revision best 0.972). The code is public (MIT) and
   takes a centroid CSV plus an HDF5 movie, so it can run on our segmentations with little effort.
   Expect a reviewer to ask what is new compared with it. **Must compare.**
2. **The Leifer numbers in the literature are on our exact recording.** NeRVE Worm1 has 79 GT neurons and about
   1530 volumes. ZephIR, ASCENT, fDNC, NeRVE and CeNDeR all report on it. This makes a literature table on
   Leifer legitimate. Each paper uses a different metric, though, so a rerun with our metric is still needed for
   the top two or three.
3. **Samuel overlap: our Samuel recording 184 is Park et al.'s own data.** The repo note lists
   `184_15GT-001.h5` and `184_5GT-006.h5`, and the targettrack README's point-annotation demo file is
   `184_15GT.h5`. That method (targettrack, Nature Methods 2024) is supervised per recording from 5-25
   annotated frames, and its data format is the one our Samuel h5 files already use.
4. **Flavell GT is ANTSUN's own output, so ANTSUN/BrainAlignNet cannot be compared on Flavell.** Running them
   on other labs' data needs their Julia/nrrd pipeline and centroid-supervised retraining. **Skip; explain why
   in the text.**
5. **What the field does:** recent papers compare mostly against numbers from the literature on NeRVE, and rerun
   at most one to three methods. See the last section. Rerunning fDNC + ZephIR + ASCENT on all four of our
   datasets with one metric would already go beyond the norm.

## Ranked recommendations

| Rank | Method | Recommendation | Supervision | Runs on our NWB with | Est. cost |
|---|---|---|---|---|---|
| 1 | ASCENT / NETr (Han & Lu 2025) | **Must compare** | self-supervised, per recording | centroid CSV + h5 movie | ~0.5-1 d setup, ~1 GPU-h per video |
| 2 | fDNC (Yu et al. 2021) | **Must compare** (wrapper exists) | supervised on semi-synthetic Leifer data; no labels at use | centroids only + 1 template frame | hours; ms per frame |
| 3 | ZephIR (Ryu et al. 2024) | **Must compare** (flag annotated frames) | registration; ≥1 annotated reference frame | raw volume + GT points at k frames | ~1 d; ~35 min per video on GPU |
| 4 | CPD / frame-to-frame Hungarian baseline | **Should** (cheap sanity floor) | none | centroids | < half a day, CPU |
| 5 | targettrack (Park et al. 2024) | Nice to have (Samuel; flag annotated frames) | supervised per recording, 5-25 GT frames | h5 volumes + GT points | 1-2 d; GPU ≥ 6 GB |
| 6 | Ultrack / Trackastra (generic cell trackers) | Nice to have (one run each) | Ultrack: none; Trackastra: pretrained | our segmentation masks | ~half a day each, CPU/GPU |
| 7 | 3DeeCellTracker (Wen et al. 2021) | Nice to have / low priority | ≥1 annotated volume; pretrained FFN | images + segmentation | 1-3 d; slow (min per volume) |
| 8 | NeRVE (Nguyen et al. 2017) | Literature number only | none (100 reference frames) | not practical | skip running |
| 9 | CeNDeR (Wu et al. 2022) | Literature number only | supervised, annotated volumes | not practical | skip running |
| 10 | BrainAlignNet + ANTSUN 2.0 (Atanas et al. 2026) | Skip running; discuss | semi-supervised registration | Flavell GT *is* its output | prohibitive |
| 11 | CRF_ID, NeuroPAL_ID, statistical atlas, fDNC cross-animal ID | Skip (different task) | atlas-based | n/a | n/a |
| 12 | Graph matching (Jones et al. 2023), Deng/Venkatachalam GNN (SPIE 2024) | Skip (no code) | 5 labelled frames / unknown | n/a | n/a |

---

## 1. ASCENT: Annotation-free Self-supervised Contrastive Embeddings (NETr)

- **Citation:** Haejun Han, Hang Lu (Georgia Tech). ICCV 2025, pp. 14676-14687.
  - Preprint: bioRxiv 10.1101/2025.07.23.666425 (posted 2025-07-28), https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1
  - Later version: "Annotation-free whole-brain neuron tracking via transformer-based self-supervised learning",
    Research Square 10.21203/rs.3.rs-8080516/v1 (https://sciety.org/articles/activity/10.21203/rs.3.rs-8080516/v1)
  - Proceedings: https://openaccess.thecvf.com/content/ICCV2025/html/Han_ASCENT_Annotation-free_Self-supervised_Contrastive_Embeddings_for_3D_Neuron_Tracking_in_ICCV_2025_paper.html
- **Code:** https://github.com/lu-lab/ascent. MIT, Python/PyTorch, Python 3.10-3.12 (not our 3.8 `wbfm` env;
  needs its own env), last push 2026-07-17. Pretrained checkpoints exist for NeRVE, a light-sheet dataset and
  their in-house data. CLI: `ascent train`, `ascent run`.
- **Method:**
  - Detections come from a per-dataset StarDist-3D model trained on two proofread frames. This step is
    optional: the README says you can supply your own centroids.
  - Encoder input: a 64x64x3 patch around each centroid plus the centroid coordinates.
  - Image features come from a DINO-pretrained ChannelViT; positions go through an MLP. The two are summed
    and refined by a 4-layer transformer over all neurons in the frame, giving 256-d embeddings.
  - Loss: SimCLR NT-Xent between two augmented views of the same frame (temperature 0.05, batch of 4 frames).
  - Tracks are linked frame to frame with a Hungarian tracker on cosine similarity. Track embeddings are
    updated by an EMA with momentum 0.5; a dynamic threshold applies, and unmatched detections start new
    tracks.
- **Inputs:** HDF5 movie (`t{frame}/c{channel}`, ZYX) plus a CSV `object_id,t,z,y,x` in voxels. Our NWB red
  volumes and segmentation centroids convert directly.
- **Reported accuracy:**
  - NeRVE (1532 vol, 79 neurons): **95.90%**. The same paper quotes ZephIR 94.43%, NeRVE 82.9% and fDNC 79.1%.
  - In-house immobilized/microfluidic data: 98.87% end to end.
  - Cross-dataset: 98.80%.
  - Data efficiency: 95.66% with 8 training frames.
  - Augmentation ablation: no augmentation 37.3%, position jitter alone 95.1%, removing position jitter 85.1%.
    This matches our finding that keypoint jitter is decisive.
  - Metric: modified MOTA without false positives; mutually closest match within 5 µm.
- **Runtime:** training converges in under 15 min on one H100/H200 (the paper contradicts itself on which GPU).
  No inference time is reported; we estimate minutes per video.
- **Why it matters here:** it is conceptually the same recipe as BarlowTrack (two augmented views of one frame,
  image + position encoder with context, per recording, self-supervised). The key differences are the loss
  (contrastive vs Barlow redundancy reduction, plus our transpose/object term), the encoder (ViT patch +
  transformer vs 3D CNN + ViSNet), and the linking (online Hungarian with EMA vs offline label propagation).
  - A reviewer who knows ICCV 2025 will ask for this comparison.
  - It also gives a clean ablation axis: run their Hungarian/EMA tracker on our embeddings, or our label
    propagation on theirs.
  - It does not cite BarlowTrack (it predates our preprint), and our preprint does not cite it either.
- **Plug-in:** export the NWB red channel to their HDF5 layout and our segmentation centroids to the CSV. Run
  `ascent train` then `ascent run` per recording, map `TrackID` to our segmentation IDs by nearest centroid
  per frame, and score with `eval_accuracy.py`.
  - Also score their output with their own metric on Leifer, as a check that the run reproduces ~95.9%.
  - Caveat: 3-plane patches may suit Flavell's isotropic 0.33 µm voxels poorly (our Flavell crops need z=16).
    Run with their defaults, then at most one patch-depth variant.
- **Cost:** about 0.5-1 day of setup (separate env), then about 1 GPU-hour per video; feasible on the cluster.
- **Fit for revision: HIGH.** It needs no method change and closes the most likely "novelty vs prior work"
  objection. Risk: if ASCENT matches us on some dataset, the framing must rest on label propagation,
  efficiency or the multi-lab benchmark.

## 2. fDNC: fast deep neural correspondence

- **Citation:** Yu X, Creamer MS, Randi F, Sharma AK, Linderman SW, Leifer AM. eLife 10:e66410 (2021).
  https://doi.org/10.7554/eLife.66410 (in Zotero as a webpage item, EBZB8JFE).
- **Code:** https://github.com/XinweiYu/fDNC_Neuron_ID. **No licence file** (all rights reserved by default;
  fine for running, not for redistributing). Python/PyTorch, last push 2022-01-04, unmaintained.
- **Method and input:** a transformer on point clouds (centroids only; optional NeuroPAL colour), trained on
  2.3x10^5 semi-synthetic deformed point clouds from 12 Leifer AML32 recordings. At use time it needs only
  centroids and one template frame; there is no per-recording training. It is supervised on synthetic data and
  label-free in use.
- **Reported accuracy:**
  - Nguyen 2017 recording, 1514 volumes, position only: **79.1%**; CPD 62.7%; NeRVE with 1 template 73.1%;
    NeRVE with 100 templates 82.9%.
  - Across animals (NeuroPAL ID): 64-78%.
  - Speed: 10 ms per volume on GPU, ~50 ms on CPU.
- **Data-overlap note:** fDNC was trained on 12 other AML32 recordings and tested on the Nguyen 2017 set. That
  is probably disjoint from our Leifer Worm1, but it is the same lab, strain and microscope, so fDNC is
  favoured on Leifer and disadvantaged on Zimmer, Flavell and Samuel (different voxel sizes and neuron
  densities). The wrapper's `PhysicalUnitConversion.zimmer2leifer` rescales coordinates, but distribution shift
  remains.
- **Existing wrapper:** `/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/utils/nn_utils/fdnc_predict.py`.
  - `track_using_fdnc_from_config` reads project centroids, converts them to Leifer units, and matches every
    frame to one template frame (`template_time_point`). The alternative is multiple templates, one per
    training frame, combined by summed-confidence bipartite matching (`combine_multiple_template_matches`).
  - It writes a tracks dataframe via `template_matches_to_dataframe`, so the output is already in our
    track-dataframe convention.
  - The model path defaults to the cluster checkout `/lisc/.../github_repos/fDNC_Neuron_ID`.
  - Config keys: `leifer_params.{use_zimmer_template, use_multiple_templates, match_confidence_threshold,
    core_options, output_df_fname}`.
  - Gap: the Leifer project lost its `1-segmentation/metadata.pickle`. For Leifer, build the centroids from
    the NWB, or call `predict_matches` directly on NWB centroids.
- **Plug-in:** run the wrapper per project, with a single template (middle frame) and with multiple templates;
  report both. Score with `eval_accuracy.py`.
- **Cost:** a few hours, mostly configuration; compute is negligible.
- **Fit: HIGH.** It is the standard annotation-free baseline, cited by every paper below, and nearly free given
  the wrapper. Use it as the main "same segmentation, same metric" comparison on all four labs.

## 3. ZephIR: multiple-object tracking via deformable image registration

- **Citation:** Ryu J, Nejatbakhsh A, Torkashvand M, Gangadharan S, Seyedolmohadesin M, Kim J, Paninski L,
  Venkatachalam V. "Versatile multiple object tracking in sparse 2D/3D videos via deformable image
  registration." PLoS Comput Biol 20(5): e1012075 (2024). https://doi.org/10.1371/journal.pcbi.1012075
  (preprint bioRxiv 10.1101/2022.07.18.500485, first author then given as James Yu).
- **Code:** https://github.com/venkatachalamlab/ZephIR. MIT, Python/PyTorch, `pip install zephir`, v1.0.6,
  last push 2025-11-10, maintained. It has an annotator GUI. Data: https://zenodo.org/records/10818810.
- **Method and input:** it registers image descriptors sampled around user keypoints, with spatial (spring) and
  temporal priors, and propagates frames in order of similarity to the reference frames. It works on raw
  volumes with no segmentation and no training. It needs **at least one manually annotated reference frame**;
  the authors list this as a limitation. Optionally it uses partial annotations elsewhere.
- **Reported accuracy:**
  - Own 178-neuron, 1060-frame dataset: 89.3% with 3 reference frames, 91.5% with 10.
  - **Nguyen 2017 / NeRVE (79 neurons): 84.0% with few reference frames, 94.48% with extra partial
    annotations.**
  - Metric: the tracked point falls inside the annotated neuron's volume.
  - Runtime: ~1.24 s per volume on a GTX 1080Ti, <1 GB VRAM. NeRVE needs ~50 s per volume on 200 cores;
    3DeeCellTracker ~1 min per volume.
  - The paper benchmarks against NeRVE, fDNC, 3DeeCellTracker and CeNDeR on NeRVE (Table 1, an image; values
    not extracted).
- **Why it matters here:** it is the best published number on Leifer before ASCENT, and our preprint Table 2
  already quotes 94.5%*. It comes from the Venkatachalam lab, which co-authored the Samuel-lab mating work, so
  Samuel reviewers will know it.
- **Plug-in:** convert the NWB volume to ZephIR's input; give it GT positions at k = 1 and k = 3 reference
  frames (chosen by their own frame-sorting or evenly spaced); read predicted coordinates for all frames.
  Score by assigning each prediction to the nearest segmentation object and running `eval_accuracy.py`, and
  also with a distance threshold. Report as "uses k GT frames".
- **Cost:** about 1 day of setup per data format; ~35 min per 1600-frame video on GPU; 4 datasets x 2 settings
  is under 5 GPU-hours.
- **Fit: HIGH.** Strongest registration-based method, maintained code, cheap compute. Flag the supervision
  clearly (our method uses zero GT frames).

## 4. CPD / nearest-neighbour Hungarian baselines

- **Citation:** Myronenko & Song, Coherent Point Drift (TPAMI 2010). We did not re-fetch it. It is used as a
  baseline in fDNC (62.7% on NeRVE), Park et al. (<20% on male mating) and WormND (identification task).
- **Code:** CPD ships inside the fDNC repo; WormND reuses it (https://github.com/focolab/WormND, `run_CPD.ipynb`).
  `pycpd` and `probreg` also exist (not checked).
- **Plug-in:** match centroids of every frame to a template, or frame to frame, with Hungarian assignment on
  Euclidean distance after rigid/affine/CPD alignment. CPU only.
- **Why:** it is a cheap floor that shows the task is not trivial. It also helps interpret the existing claim
  that label propagation on an *untrained* encoder already tracks most neurons on some datasets: a reviewer may
  ask how far pure geometry gets.
- **Fit: MEDIUM-HIGH** (cheap, pre-empts an obvious question; not a published competitor in its own right).

## 5. targettrack (Park et al. 2024, targeted augmentation)

- **Citation:** Park CF, Barzegar-Keshteli M, Korchagina K, Delrocq A, Susoy V, Jones CL, Samuel ADT, Rahi SJ.
  "Automated neuron tracking inside moving and deforming C. elegans using deep learning and targeted
  augmentation." Nat Methods 21:142-149 (2024). https://doi.org/10.1038/s41592-023-02096-3 (preprint bioRxiv
  10.1101/2022.03.15.484536).
- **Code:** https://github.com/rahi-lab/targettrack (README also points to lpbsscientist/targettrack). MIT,
  Python/PyTorch 1.7-1.11, PyQt GUI, last push 2025-05-26.
- **Method and input:** a 3D CNN (3DCN) segments and tracks jointly from raw volumes and is **trained per
  recording** on 5-25 manually annotated frames. "Targeted augmentation" synthesises more training frames by
  deforming the GT frames into other postures. It works with points or masks. GPU ≥ 6 GB; ~10 min per 100
  epochs on a 2080 Ti. Input h5 layout: `i/frame`, attributes C/W/H/D/N_neurons/T; W and H must be multiples
  of 32.
- **Reported accuracy:** on Samuel-lab male mating recordings (1500-3000 volumes, 3 held-out recordings),
  5-25 GT frames reach the same fraction of tracked neurons as an ordinary CNN trained on 100 GT frames, with
  ">80% correctly tracked" where CPD gets <20%. Exact per-recording numbers are in Table 1 and Fig. 3 of the
  paper; we did not extract them from the PDF.
- **Overlap:** the Samuel 184 recordings (`184_15GT`, `184_5GT`) appear to be this paper's data. Their numbers
  on 184 may be directly quotable for our Samuel benchmark (check which recording they held out). Our main
  Samuel benchmark is 153; 184-004 is in the repo note.
- **Plug-in:** the Samuel h5 files are already in targettrack format. Train with 5 and 15 GT frames, export
  points, score with `eval_accuracy.py`. Other labs need h5 conversion and padding to multiples of 32.
- **Cost:** 1-2 days (GUI-centric code, old PyTorch pin); training ~1 GPU-hour per video.
- **Fit: MEDIUM.** It is the method of the Samuel lab, whose data we use; it needs GT frames and so is not like
  for like. Running it on Samuel only (and quoting their numbers) is a reasonable middle ground.

## 6. Generic cell trackers: Ultrack and Trackastra

- **Ultrack:**
  - Citation: Bragantini J, ..., Royer LA. Nature Methods (online 2025-08-25); preprint arXiv 2308.04526.
  - Code: https://github.com/royerlab/ultrack. BSD-3, Python, very active (last push 2026-10-09).
  - Method: ILP over multiple segmentation hypotheses with temporal consistency. CPU/HPC.
- **Trackastra:**
  - Citation: Gallusser B, Weigert M. ECCV 2024, arXiv 2405.15700 (https://arxiv.org/abs/2405.15700).
  - Code: https://github.com/weigertlab/trackastra. BSD-3, last push 2026-09-11.
  - Method: a transformer predicts associations from images plus instance masks. The pretrained `ctc` model
    supports 3D; the others are 2D only.
- **Evidence they fail here:** WormND / WormID-Bench (below) ran Ultrack and 3DeeCellTracker on C. elegans
  calcium videos and reported "considerable room for improvement". Ultrack beat 3DeeCellTracker at both 3 and
  6 µm thresholds. We did not get the numeric table. Run scripts: https://github.com/focolab/WormND
  (`neuron_tracking/`).
- **Why:** a methods reviewer from the cell-tracking community may ask why off-the-shelf trackers are not
  enough. Overlap- or proximity-based linking breaks under the large frame-to-frame motion of freely moving
  worms. One run each on Zimmer and Leifer with our segmentation masks would show that cheaply.
- **Cost:** about half a day each; CPU (Ultrack) or a small GPU (Trackastra).
- **Fit: MEDIUM-LOW.** A nice-to-have supplementary row. Expect low numbers; their tracks fragment rather than
  keeping one identity per neuron, so define how fragmented tracks are scored (e.g. longest-matching track per
  GT neuron).

## 7. 3DeeCellTracker

- **Citation:** Wen C, Miura T, Voleti V, ..., Kimura KD. eLife 10:e59187 (2021).
  https://elifesciences.org/articles/59187. Protocols: Bio-protocol 12(4):e4319 (2022); Methods Mol Biol 2800
  (2024), doi 10.1007/978-1-0716-3834-7_14.
- **Code:** https://github.com/WenChentao/3DeeCellTracker. MIT, Python with **TensorFlow 2.x** (tested 2.5),
  PyPI `3DeeCellTracker` v1.0.1 (2025-11-13). Maintained.
- **Method and input:** segmentation by StarDist or a 3D U-Net trained on ≥1 annotated volume (2-3 h of manual
  work), then an FFN pretrained on synthetic point sets plus PR-GLS point-set registration. "Ensemble mode" is
  for freely moving animals.
- **Reported accuracy:** 98-100% on semi-immobilized worms. On a freely moving worm (Nguyen data,
  **straightened** first): 73% of cells fully correct in single mode; ensemble mode gets 99.8% of movements
  correct over 500 volumes for 90 inspected cells.
- **Runtime:** ~2 min per volume (single mode) and ~10 min (ensemble mode) on a GTX 1080, i.e. tens of GPU-hours
  per 1600-frame video unless v1.0 is much faster (not verified). WormND found it under-detects (~70 of ~150
  neurons per frame) with pretrained StarDist.
- **Plug-in:** feed our segmentation as label images (supported for FFN training in v1.0) to sidestep
  detection, and use single mode.
- **Fit: LOW-MEDIUM.** Widely cited, so a reviewer may name it. Its freely-moving results need straightening and
  are slow; quoting its published or WormND results is probably enough. Run it only if a reviewer insists.

## 8. NeRVE (Leifer lab)

- **Citation:** Nguyen JP, Linder AN, Plummer GS, Shaevitz JW, Leifer AM. "Automatically tracking neurons in a
  moving and deforming brain." PLoS Comput Biol 13(5):e1005517 (2017).
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005517
- **Code:** https://github.com/leiferlab/NeRVEclustering. MATLAB, no licence file, last push 2018. Needs
  straightening from low-magnification behaviour images and ~200 cluster cores.
- **Accuracy:** 82.9% (100 templates) on our exact recording (from fDNC Table/text; ASCENT quotes the same).
- **Fit:** quote only (already in preprint Table 2). The Leifer GT itself came from NeRVE-assisted
  semi-automatic annotation, which deserves a footnote.

## 9. CeNDeR (Wen Quan lab)

- **Citation:** Wu Y, Wu S, Wang X, Lang C, Zhang Q, Wen Q, Xu T. "Rapid detection and recognition of whole
  brain activity in a freely behaving Caenorhabditis elegans." PLoS Comput Biol 18(10):e1010594 (2022).
  https://doi.org/10.1371/journal.pcbi.1010594 (arXiv 2109.10474).
- **Code:** https://github.com/Wenlab/CeNDeR. MIT, PyTorch, last push 2023-08-11. Includes a NeRVE training
  experiment (`e2_wa_nerve.py`).
- **Method:** detection, then a supervised multi-class neuron-recognition network trained on human-annotated
  volumes of the same recording.
- **Accuracy:** "above 80%" tracking on their own data. Our preprint quotes 88.25%* on NeRVE (needs annotation);
  we did not re-verify that number in this search.
- **Fit:** quote only. It is supervised per recording and tuned to their 1024x1024x18 data.

## 10. BrainAlignNet + ANTSUN 2.0 (Flavell lab)

- **Citation:** Atanas AA, Lu AK-Y, Goodell B, Kim J, Baskoylu S, Kang D, Kramer TS, Bueno E, Wan FK,
  Cunningham KA, Weissbourd B, Flavell SW. "Deep neural networks to register and annotate cells in moving and
  deforming nervous systems." eLife 14:RP108159 (VOR 2026-02-17), doi 10.7554/eLife.108159.2; preprint bioRxiv
  10.1101/2024.07.18.601886. ANTSUN 1.x: Atanas et al., Cell 2023.
- **Code:**
  - https://github.com/flavell-lab/BrainAlignNet: GPL-3.0, Python with a custom DeepReg (TensorFlow), last
    push 2026-02-11.
  - https://github.com/flavell-lab/ANTSUN: GPL-3.0, Julia notebooks, last push 2026-02-11.
- **Method and input:**
  - BrainAlignNet is a LocalNet-style DDF registration network on Euler-pre-aligned tagRFP volume pairs.
    Training is semi-supervised with centroid-alignment loss on 5176 pairs from 57 animals; the fully
    unsupervised variant failed.
  - ANTSUN 2.0 segments with a U-Net, registers many frame pairs, builds an NxN ROI-match matrix, then applies
    custom hierarchical clustering. That structure is close in spirit to our clustering/label propagation over
    a similarity graph.
- **Reported accuracy:** 99.6% linking accuracy, estimated from an eat-4::GFP mixing control. Registration
  takes ~9 GPU-hours per dataset with ANTSUN 2.0, compared with ~250 CPU-days for ANTSUN 1.4.
- **Why skip:** our Flavell GT *is* ANTSUN output, so on Flavell our accuracy already measures agreement with
  ANTSUN. On other labs, the network needs data at Flavell resolution (crop 284x120x64, isotropic 0.33 µm),
  matched centroids for retraining (which would leak our GT) and the Julia/nrrd pipeline.
- **Fit:** discuss in text; do not run. The 9 GPU-hours per dataset is a useful efficiency comparison point if
  our training plus label propagation is cheaper.

## 11. Identification tools (not tracking): CRF_ID, NeuroPAL_ID, statistical atlas, fDNC cross-animal

- **CRF_ID:** Chaudhary S, Lee SA, Li Y, Patel DS, Lu H. eLife 10:e60321 (2021), https://elifesciences.org/articles/60321.
  Code: https://github.com/shiveshc/CRF_Cell_ID (MATLAB, no licence file, last push 2021).
- **NWB/WormID corpus and atlas:** Sprague DY, ..., Kato S, Yemini E. Cell Reports Methods (2025), bioRxiv
  10.1101/2024.04.28.591397. 118 worms from 5 labs in NWB; https://github.com/focolab/NWBelegans.
- These assign atlas names across animals (NeuroPAL), not identities over time within one video. **Skip**, but
  one sentence in the Discussion should separate tracking from identification. Our NWB-based benchmark is
  compatible with the WormID NWB extension, which is worth saying.

## 12. Methods with no usable code

- **Graph matching for neurons and neurites:** Jones C, Barzegar-Keshteli M, Gross A, Obozinski G, Rahi SJ.
  "A Graph Matching Approach to Tracking Neurons in Freely-Moving C. elegans." bioRxiv
  10.1101/2023.11.30.569341 (2023). It uses 5 labelled frames and reports ">75%". Code was promised upon
  publication and was not found.
- **Anchor neuron + GNN:** Deng H, Yu J, Venkatachalam V. "Neuron tracking in C. elegans through automated
  anchor neuron localization and segmentation." Proc SPIE 12857, 128570H (2024). No code found. **Skip; cite
  at most.**

## Not found

- **"TWT":** no C. elegans tracker under this name turned up on the web. It may be a lab-internal name or a
  misremembered acronym; check with the user.
- **"eat-it":** not found as a named method. If it means simple greedy or registration baselines, section 4
  covers them.
- **No 2026 foundation-model C. elegans tracker was found.** ASCENT proposes a pretrained "foundation" NETr as
  future work and ships cross-dataset checkpoints. Running its pretrained NeRVE checkpoint zero-shot on
  Zimmer/Samuel would answer a "foundation model" question cheaply.

## What comparisons recent papers made (a guide to reviewer expectations)

| Paper | Compared against | Rerun or quoted |
|---|---|---|
| fDNC (eLife 2021) | CPD, NeRVE(1), NeRVE(100); CRF_ID for ID | rerun on NeRVE |
| 3DeeCellTracker (eLife 2021) | Toyoshima 2016, DeepCell 2.0 | rerun on own data |
| CeNDeR (PLoS CB 2022) | fDNC on NeRVE and own data | rerun (repo has the fDNC benchmark scripts) |
| Park / targettrack (Nat Methods 2024) | plain U-Net/iCNN, CPD | rerun on own data |
| ZephIR (PLoS CB 2024) | NeRVE, fDNC, 3DeeCellTracker, CeNDeR on NeRVE; DeepLabCut (mouse) | mostly quoted (Table 1) |
| WormND / WormID-Bench (NeurIPS 2024 D&B, bioRxiv 10.1101/2025.01.06.631621) | Ultrack, 3DeeCellTracker (tracking); CPD, atlas, fDNC, CRF_ID (ID) | rerun on 36 videos |
| ASCENT (ICCV 2025) | ZephIR, NeRVE, fDNC, CeNDeR on NeRVE + in-house | mostly quoted |
| BrainAlignNet (eLife 2026) | own ANTSUN 1.4, human labellers | own pipeline only |

The field norm is a NeRVE literature table plus one to three reruns. BarlowTrack's claim (multi-lab, zero-shot)
invites a stricter reviewer request: same segmentation, same frames, same metric. A credible minimal answer:

1. **fDNC on all four labs** (centroids only, no annotation; wrapper exists).
2. **ASCENT on all four labs** (self-supervised, same inputs as ours).
3. **ZephIR on all four labs with k = 1 and 3 GT reference frames** (marked as using annotation).
4. A CPD/Hungarian floor.
5. A literature table on Leifer (NeRVE 82.9, fDNC 79.1, CeNDeR 88.25*, ZephIR 84.0 / 94.48*, ASCENT 95.90),
   with a note on metric differences.
6. Optionally: targettrack on Samuel (their own data) and one generic tracker (Ultrack) as a supplementary row.

The total compute is small (tens of GPU-hours), well within the revision budget. Almost all the effort is
format conversion (NWB to h5/CSV) and mapping each method's output back to our segmentation-ID track dataframe
for `eval_accuracy.py`. A shared exporter script (NWB, red volume, our segmentation centroids → h5 + CSV) plus a
shared importer (coordinates/TrackIDs → wbfm track dataframe via nearest centroid) would serve ASCENT, ZephIR,
targettrack and Trackastra at once.

### Fairness notes to state in the paper

- **Detections:** Samuel NWB segmentations contain only GT neurons (oracle detections, per the repo note). For
  Leifer, the current eval crops around GT positions. Every method fed those centroids gets the same oracle
  advantage; say so.
- **Metrics:** ZephIR counts a point inside the neuron volume; ASCENT uses mutual-closest within 5 µm; ours
  matches segmentation IDs. Rescore every rerun with our metric, and for the Leifer sanity check also with the
  method's own metric.
- **Leifer GT provenance:** NeRVE-assisted annotation could favour NeRVE-like methods. **Flavell GT** is ANTSUN
  output.
