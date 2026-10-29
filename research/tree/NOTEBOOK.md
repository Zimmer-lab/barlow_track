# Notebook

The lab notebook of this project: what was tried, decided, abandoned or guessed, as it happens, before it
is settled enough to become a node in tree/. Entries are appended, never edited, newest last; add one
with `ygg log`. A `handoff` entry says where things stand for whoever picks the work up next.

## 2026-10-22 23:55 — decision

Code commit for the preprint: tag v0.1.0 (987eca2, 2025-10-29, Zimmer-lab/barlow_track PR #109). The last commit before 2025-11-12 is 7a555d5 (main, 2025-11-10), but git diff v0.1.0..7a555d5 touches only README.md, the project-template README and conda/*.yaml, so the code is identical; the tag is cited as the release. It is still uncertain whether the paper's figures were produced from exactly this code (the training runs predate it); said in each run Source's Caveats.

## 2026-10-22 23:57 — decision

Run Sources cite Charles-Fieseler-Vienna/barlow_track@987eca2 (the local origin, where the commit is verified). The paper cites github.com/Zimmer-lab/barlow_track; the tag merges PR #109 from Zimmer-lab/dev, so the same commit should exist there, but that was not checked.

## 2026-10-22 23:57 — decision

Figure-to-script links. Linked only where the script's content names the analysis: Fig. 3h -> scripts/paper_scripts/paper_make_generalization_projects.py (cross-lab baseline networks on zimmer/flavell/leifer GT); Fig. 3i -> paper_run_parameter_sweeps.py (training_data_sweep_*) and paper_make_projects_and_run_pipeline.py (tracks every sweep folder); Fig. S1d -> paper_run_old_analysis_methods.py and paper_update_config_files.py (baseline_old_tracklets, baseline_hdbscan with tracking_mode global); Fig. S1e-f -> alternative_project_templates/augmentation_ablation/hyperparameter_search_template.yaml plus paper_make_projects_and_run_pipeline.py (augmentation_sweep, inverse_augmentation_sweep, only_new_loss, only_original_loss, untrained); Fig. 3g -> scripts/multiproject_scripts/analyze_project_with_all_barlow_trials.py (val loss vs accuracy scatter, single project; the paper's per-lab version is not in the repo); Fig. 3d-f and S1a-c -> utils/utils_paper.py (plot_hist_and_accuracy, calculate_distance_diff, calculate_nearest_neighbor_distance; the driving notebook is not in the repo); Table 2 / Fig. 2 -> train_barlow_clusterer.py with barlow_project_template/train_config.yaml (matches Methods 4.3 defaults exactly), utils/barlow.py (loss), utils/utils_label_propagation.py (Alg. 1). Left unlinked (no script found at v0.1.0 nor in dlc_for_wbfm@924a08e): Fig. 1 (motion, PPCA), Fig. 4 and S2 (novel conditions), Fig. 5 (GUI, lives in wbfm), Table 2 numbers of other published methods.

## 2026-10-22 23:57 — decision

Reviewer comments: none found. No ICLR/OpenReview reviews, rebuttal or revision plan in the repo, Zotero (item 2Y9YWW3D has no notes) or ~/Current_work/barlow_networks (only iclr_initial_projects checkpoints and configs from May 2024). The only reviewer-style material is a co-author markup, ~/Downloads/Summary of Comments on BarlowTrack__Zero_shot_multi_object_cell_tracking_using_modified_Barlow_Twins-1_jr.pdf (Julia Riedl, 4-18 Nov 2025): wrong panel references (Fig. 1d red line, Fig. 3b, 3h; text cites Fig. 3e for the training-volume panel which is 3i), agarose percentage unexplained, the augmentation-ablation sentence unclear, and an unresolved placeholder ('a balance of ?? is optimal'). Open revision Questions are therefore my reviewer-style reading of the claims, the Discussion's limitations, and docs/pr_plan + git log since 2025-11-12.

## 2026-10-22 23:58 — decision

Correction to the earlier entry: ygg's run field does not accept 'owner/name', so run Sources cite barlow_track@987eca2:PATH (local repo at /home/charles/Current_work/repos/barlow_track, origin github.com/Charles-Fieseler-Vienna/barlow_track; the paper cites github.com/Zimmer-lab/barlow_track).

## 2026-10-23 00:03 — decision

Post-preprint results are recorded as low/medium Evidence from two Sources cited at bc9ff96 (eval_accuracy.py, run_trials_on_all_ground_truth.py), not as paper Evidence: the numbers come from docs/accuracy_evaluation.md (committed) and the 'Empirical state (Oct 2026)' section of AGENTS.md (git-ignored, so unversioned). I did not open the cluster exp_results files; they should be checked before any of these is raised in confidence.

## 2026-10-23 00:05 — decision

Checked a latent bug named in docs/accuracy_evaluation.md (training silently ignored custom backbone_kwargs): at 987eca2 train_barlow_clusterer.py calls vars() on the YAML dict, which raises TypeError and falls back to num_levels=2, f_maps=4. The template train_config.yaml sets exactly those values, so the preprint's default networks are unaffected; only sweeps varying backbone_kwargs would be (none are reported in Methods 4.3). Not logged as an issue.

## 2026-10-23 00:05 — issue

Post-preprint code review (docs/pr_plan PR01-PR04, written at 9fbeda5) found bugs in the code behind the preprint numbers: accuracy evaluation (time range scored, NaN-gap and duplicate-label handling, CPU/GPU mismatch), label-propagation edge cases (disconnected nodes labelled 0, alignment with 0-1 labelings), tracking windows (final frame omitted, window coverage), and crops at volume edges. Not known whether any of them changes the Table 2 / Fig. 3 / S1 numbers; the preprint's untrained Flavell number reproduces exactly under the new pipeline.

- [ ] issue i261023-1 · affects [[@BarlowTrack time-aware label propagation clustering]], [[@BarlowTrack baseline network training]], [[@BarlowTrack track coverage and displacement helpers]]

## 2026-10-23 00:05 — issue

The Leifer analysed project's 1-segmentation/metadata.pickle no longer exists, so the preprint's Leifer tracks cannot be re-embedded through the standard path; revision evaluations crop around GT positions from the NWB (--source nwb) and match on raw_neuron_ind_in_list. Leifer numbers before and after are therefore not like for like.

- [ ] issue i261023-2 · affects [[DATA - Leifer lab NeRVE recording with ground truth]]

## 2026-10-23 00:05 — todo

Manuscript text fixes from the co-author markup (~/Downloads/Summary of Comments on BarlowTrack__...-1_jr.pdf) and my reading.

- [ ] fix wrong panel references: '1d red line' (PPCA is Fig. 1e), Fig. 3b in section 2.4, training volumes cited as Fig. 3e (is 3i), Fig. 3h wording
- [ ] replace the 'a balance of ?? is optimal' placeholder in section 2.5 with the loss-balance sweep values
- [ ] explain agarose percentage / density for non-experimental readers (section 2.6)
- [ ] rewrite the augmentation-ablation sentence in section 2.5 that the co-author found unclear
- [ ] state n networks per median in Table 2 and add spread
- [ ] release the NWB benchmark datasets (paper: 'Data will be made available upon publication')

## 2026-10-23 00:06 — handoff

**Where things stand.** New tree for BarlowTrack (preprint 2025-11-12, Zotero 2Y9YWW3D), digested figure by figure: 7 root Questions (Figs. 1, 2, Table 2/Fig. 3a-c, 3d-h, 3i/S1e-f, 4/S2, 5), 25 Claims with paper Evidence, 5 Datasets, 10 run Sources at barlow_track@987eca2 (tag v0.1.0; code identical to 7a555d5, the last commit before the preprint), plus 14 open Questions, 5 done post-preprint Evidence and 10 planned Evidence for the revision. Nothing in the repository outside research/ was changed; nothing committed.

**Digested.** Main text, Tables 1-2, Figs. 1-5 (Fig. 3 and S1 read as images for numbers); Figs. S1-S2 to main points. Linked scripts: training (+ template config), loss, label propagation, coverage/displacement helpers (Fig. 3d-f, S1a-c), val-loss scatter (3g), generalization projects (3h), training-data sweeps and ablation tracking runs (3i, S1e-f), augmentation-ablation template (S1e-f), tracklet/HDBSCAN comparison (S1d).

**Left out.** Fig. 1, Fig. 4/S2 and Fig. 5 are unlinked (no script at the tag or in dlc_for_wbfm@924a08e; GUI lives in wbfm). The literature rows of Table 2 have no code. NWB/data-format section (2.2), time-series computation (4.4) and the Discussion's feature-space-as-starting-point remark not made into nodes. Fig. 4 coverage values not read off the figure. Cluster records (exp_results*.json) not opened; post-preprint numbers come from docs/accuracy_evaluation.md and the git-ignored AGENTS.md summary.

**Reviewer comments.** None found (repo, Zotero notes, ~/Current_work/barlow_networks). Only a co-author markup: ~/Downloads/Summary of Comments on BarlowTrack__Zero_shot_multi_object_cell_tracking_using_modified_Barlow_Twins-1_jr.pdf (J. Riedl, Nov 2025); its points are the open text to-dos. Revision Questions are my reviewer-style reading plus docs/pr_plan and git log.

**Open revision work by priority.**
- High: [[QUE - Do the preprint accuracies hold under the corrected evaluation and tracking code]] (first [[EVD - Regression tests for evaluation, label propagation, tracking windows and crops]], which blocks [[EVD - Preprint baseline networks re-scored with the corrected evaluation]]; revision networks so far score lower, Flavell .62-.73 vs 96.7%); [[QUE - Does the object-space loss term improve tracking in any regime]] ([[EVD - Loss-term ablation where accuracy is not saturated]]); [[QUE - Does BarlowTrack outperform other trackers run on the same data and metric]] ([[EVD - Published trackers run on the shared NWB benchmark]], no work started); [[QUE - Does adding neuron position information improve tracking]] ([[EVD - Attention and fusion sweep with the ViSNet encoder]]).
- Normal: seed reproducibility (PR06), validation loss as selection criterion, crop size per dataset (Flavell), global augmentation, GT for novel conditions, Flavell GT validity, fourth lab / multi-animal.
- Low: segmentation of dense regions, other species, compute cost.

**Issues to judge.** i261023-1 (PR01-PR04 bugs vs preprint numbers; 23 notes) and i261023-2 (Leifer segmentation metadata gone; 44 notes) are open; nothing judged yet. My read: neither invalidates the preprint Evidence as recorded, but i261023-1 is exactly what the high-priority re-scoring answers.

**Loose state.** research/ is untracked in git. Six manuscript to-dos are open in the notebook. docs/pr_plan items are unticked although later commits (#139-#144) touch them; check which are done before writing tests.

## 2026-10-23 09:57 — agenda order

What a session looks at first, in this order; the rest follow in the default order:

1. [[QUE - Do the preprint accuracies hold under the corrected evaluation and tracking code]]
2. [[EVD - Loss-term ablation where accuracy is not saturated]]
3. [[EVD - Published trackers run on the shared NWB benchmark]]
4. [[EVD - Regression tests for evaluation, label propagation, tracking windows and crops]]
5. [[QUE - Does BarlowTrack outperform other trackers run on the same data and metric]]
6. [[QUE - Does adding neuron position information improve tracking]]
7. [[QUE - Does the object-space loss term improve tracking in any regime]]
8. [[EVD - Attention and fusion sweep with the ViSNet encoder]]

## 2026-10-23 09:57 — agenda order

The order of what a session looks at first is back to the default.

## 2026-10-23 09:57 — agenda order

What a session looks at first, in this order; the rest follow in the default order:

1. [[QUE - Do the preprint accuracies hold under the corrected evaluation and tracking code]]
2. [[EVD - Loss-term ablation where accuracy is not saturated]]
3. [[EVD - Published trackers run on the shared NWB benchmark]]
4. [[EVD - Regression tests for evaluation, label propagation, tracking windows and crops]]
5. [[QUE - Does BarlowTrack outperform other trackers run on the same data and metric]]
6. [[QUE - Does adding neuron position information improve tracking]]
7. [[QUE - Does the object-space loss term improve tracking in any regime]]
8. [[EVD - Attention and fusion sweep with the ViSNet encoder]]

## 2026-10-23 15:21 — run

Flavell paper-reproduction factorial (local CPU, seed 0), re-tracking the paper's saved untrained flavell embedding (analyzed_projects/flavell/untrained/2025_07_01trial_0) with barlow_track/scripts/scratch/paper_factorial_check.py (uncommitted, on bc9ff96). Verified first: paper features = u[:, :50] of the raw 64-d embedding (unit columns, matches saved X_svd to 1.7e-7; s1=2140 vs s2=11, so eval's TruncatedSVD U*S is dominated by one direction); paper num_seeds = default 100, tracker seed None; spectral relabeling changed behaviour since v0.1.0 (top-k fallback, single-object frames, null cost scaled by R). Stored control reproduces 0.6961. Runs: A = v0.1.0 code + saved X_svd + 100 seeds (exact reproduction); B = HEAD code + saved X_svd + 100 (code-fix effect); C = HEAD + TruncatedSVD + 100 (SVD effect); D = HEAD + TruncatedSVD + 25 (eval settings on paper detections). Detection source (GT-centered NWB crops) not in this round: needs an untrained GT-centered embedding.

run r261023-2158169 · pids 2158169, 2158171, 2158170, 2158172 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_factorial_20261023_1520/A_papercode_saved_s100.log, /home/charles/.cache/barlow_scratch/runs/flavell_factorial_20261023_1520/B_head_saved_s100.log, /home/charles/.cache/barlow_scratch/runs/flavell_factorial_20261023_1520/C_head_truncated_s100.log, /home/charles/.cache/barlow_scratch/runs/flavell_factorial_20261023_1520/D_head_truncated_s25.log

## 2026-10-23 15:33 — run

Flavell TRAINED paper-reproduction factorial (local CPU, seed 0): same four configs as r261023-2158169 (A v0.1.0+saved X_svd+100 seeds; B HEAD+saved+100; C HEAD+TruncatedSVD+100; D HEAD+TruncatedSVD+25) on the paper's trained baseline analyzed_projects/flavell/baseline/2025_07_01trial_0 (checkpoint TrainedBarlow/baseline_flavell/trial_0/resnet50.pth; same 286,473 segmentation detections; saved X_svd is u[:, :50], seed None). Stored tracks of the five baseline trials score 0.9683/0.9678/0.9682/0.9627/0.9665 with the current scorer, matching the paper's 96.7%. Script barlow_track/scripts/scratch/paper_factorial_check.py, uncommitted on bc9ff96.

run r261023-2165243 · pids 2165243, 2165244, 2165245, 2165246 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_trained_factorial_20261023_1533/A_papercode_saved_s100.log, /home/charles/.cache/barlow_scratch/runs/flavell_trained_factorial_20261023_1533/B_head_saved_s100.log, /home/charles/.cache/barlow_scratch/runs/flavell_trained_factorial_20261023_1533/C_head_truncated_s100.log, /home/charles/.cache/barlow_scratch/runs/flavell_trained_factorial_20261023_1533/D_head_truncated_s25.log

## 2026-10-23 21:21 — surprise

Factorial D (HEAD code, TruncatedSVD U*S, 25 seeds, seed 0; eval_accuracy's tracking settings) on the paper's saved flavell embeddings: trained baseline trial_0 0.9584 (stored paper tracks 0.9683); untrained trial_0 0.7246 (stored 0.6961). Eval tracking settings cost about 1 point on trained embeddings, so they do not explain the sweep's flavell .62-.73; the gap is upstream (embedding input/network). A-C (100 seeds) still running.

## 2026-10-23 21:34 — run

E1: eval_accuracy.py --source nwb (GT-centered crops) --mode trained with the paper's flavell baseline checkpoint (TrainedBarlow/baseline_flavell/trial_0), 25 seeds, cuda. Tests whether the NWB input path explains the sweep's low flavell scores (paper tracks 0.968).

run r261023-2312954 · pids 2312954 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E1_nwb.log

## 2026-10-23 21:34 — run

E2: eval_accuracy.py --source project (segmentation detections, flavell_data/projects/2025_07_01) with the same paper baseline checkpoint, 25 seeds, cuda. Embedding should equal the paper's saved embedding.zarr (286,473 rows); score should be near factorial D 0.958.

run r261023-2312979 · pids 2312979 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E2_project.log

## 2026-10-23 21:52 — run

E1 tracking hit CUDA OOM (both evals tracking on the GPU at once); embedding cache kept (209,890 rows, GT-centered). Resumed with --skip_embed after E2 exits (queued script resume_E1.sh).

run r261023-2321988 · pids 2321988 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E1_nwb.log

## 2026-10-23 21:52 — surprise

E2 embedding (eval project path, paper flavell baseline ckpt, GPU crops) vs the paper's saved embedding.zarr, matched on (t, seg): all 286,473 paper rows found (eval adds frame 1599, +182 rows); raw indices agree; not identical: cosine median 0.986 (p5 0.976, min 0.933), median norm 5.72 vs 5.27. So the eval's crop/normalize path is close to but not the paper's. Tracking score will show whether it matters.

## 2026-10-23 21:52 — surprise

Crop intensity normalization changed after the preprint: v0.1.0 used tio.RescaleIntensity(percentiles=(5, 100)) per frame's crop stack (Transform.final_normalization, training and embedding); d1ff47a (#136) moved training, embedding and eval_accuracy to (5, 99.5). New checkpoints are consistent with eval; preprint checkpoints scored by eval_accuracy get a train/eval normalization mismatch. Likely why E2's embedding is cos 0.986 (not 1.0) to the paper's, norms 5.72 vs 5.27.

## 2026-10-23 21:53 — run

E1 and E2 both hit CUDA OOM in labelprop when tracking concurrently (E2's 286k-row graph needs ~3.8 GB extra). Embeddings cached. E1 resumed alone (r261023-2321988); E2 queued to track after it with --skip_embed (resume_E2.sh).

run r261023-2323712 · pids 2323712 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E2_project.log

## 2026-10-23 21:53 — surprise

E1 embedding (eval NWB path, GT-centered crops, paper flavell baseline ckpt) vs paper's saved embedding on shared (t, seg) rows: 209,751 of 209,890 rows match; cosine median 0.955 (p5 0.914, min 0.832), lower than E2's 0.986. Beyond the candidate set, GT-centered crops change the features themselves (centre offset vs segmentation centroid, plus the 99.5 normalization).

## 2026-10-23 22:01 — surprise

E1 result (r261023-2321988): eval_accuracy --source nwb, paper flavell baseline ckpt, 25 seeds: accuracy 0.8648 (misses 26,054, mismatches 5,205, total 231,268). Same ckpt on the paper's saved segmentation features (factorial D) gave 0.958, paper stored 0.968. So the NWB input path alone costs ~9 points with a known-good network, mostly as misses. E2 OOM'd again alone on GPU; rerun tracking on CPU (--track_device cpu).

run r261023-2327458 · pids 2327458 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E2_project.log

## 2026-10-23 22:01 — surprise

Normalization confirmed (barlow_track/scripts/scratch/normalization_check.py, uncommitted; 30 flavell frames, 5,264 rows, paper baseline ckpt, eval's _iter_project_frames on CPU): percentiles (5,100) reproduce the paper's embedding.zarr (cosine median 1.00000, min 0.903); (5,99.5) give median 0.986. So eval_accuracy's project path equals the paper's embedding up to the post-#136 normalization.

## 2026-10-23 22:07 — surprise

BUG in eval_accuracy._load_nwb_arrays (flavell only): df.loc[:, (slice(None), field)] returns each field's columns in frame order, and in flavell_data.nwb the id fields (raw_neuron_ind_in_list, raw_segmentation_id) are in a different neuron order than x/y/z. _iter_nwb_frames therefore crops at neuron A's xyz and labels with neuron B's ids, and keeps only entries where both are present: 209,896 of 231,268 GT entries cropped, accuracy cap 0.908 for any network. Leifer and Samuel NWBs are aligned (checked with barlow_track/scripts/scratch/nwb_gt_coverage.py, uncommitted). Likely a main reason flavell sweep cells are flat .62-.73. Fix: select each field by neuron name. Not applied (sweep running).

## 2026-10-23 22:07 — run

E3: eval_accuracy with name-aligned NWB GT arrays (scratch/eval_nwb_aligned.py monkeypatches _load_nwb_arrays), flavell nwb, paper baseline ckpt, 25 seeds, cuda. Measures the NWB path once the alignment bug is removed (E1 with the bug: 0.865).

run r261023-2330704 · pids 2330704 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E3_nwb_aligned.log

## 2026-10-23 22:13 — run

E3 embedded 231,266 rows (all GT entries, vs 209,890 with the bug) then OOM'd in GPU labelprop (r261023-2330704). Earlier OOMs: E1 r261023-2312954, E2 r261023-2312979 and r261023-2323712 (all GPU labelprop OOM; embeddings kept). E3 tracking rerun on CPU with --skip_embed.

run r261023-2333818 · pids 2333818 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E3_nwb_aligned.log

## 2026-10-23 23:58 — surprise

Untrained flavell factorial done (r261023-2158169, seed 0, paper's saved untrained trial_0 embedding, scratch/paper_factorial_check.py uncommitted). Stored paper tracks 0.6961. A paper code v0.1.0 + saved u-SVD + 100 seeds 0.6994 (harness reproduces the paper within 0.3 pt); B HEAD code 0.7070 (+0.8); C HEAD + TruncatedSVD U*S 0.7249 (+1.8); D C with 25 seeds 0.7246 (seed count ~0). None of code version, SVD or seed count lowers accuracy; eval's tracking settings slightly raise it.

## 2026-10-24 00:32 — surprise

E2 result (r261023-2327458): eval_accuracy --source project (segmentation detections), paper flavell baseline ckpt, (5,99.5) normalization, 25 seeds, seed 0: 0.9518 (misses 3,233, mismatches 7,924). vs factorial D on the paper's own (5,100) features 0.958: normalization mismatch costs ~0.7 pt (single seed).

## 2026-10-24 00:32 — surprise

E3 result (r261023-2333818): eval_accuracy --source nwb with name-aligned GT arrays (scratch/eval_nwb_aligned.py), paper flavell baseline ckpt, 25 seeds, seed 0: 0.9692 (misses 4,446, mismatches 2,668, total 231,268). Bugged NWB path (E1) 0.8648. Paper 96.7% / stored 0.968. So with the alignment fix the current eval reproduces the paper's flavell accuracy with the paper's network; the bug costs ~10 pt. Caveat: NWB path is GT-only candidates (oracle set).

## 2026-10-24 00:32 — run

Rescore round_visnet_full trial27 (recorded flavell 0.820) and trial20 (0.750) with the name-aligned NWB path (scratch/eval_nwb_aligned.py; GPU embed, CPU track, 25 seeds, seed 0). Tests whether the post-preprint networks also recover on flavell once the alignment bug is removed.

run r261024-2391371 · pids 2391371, 2391399 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_sweep_aligned_20261024_0032/trial27_aligned.log, /home/charles/.cache/barlow_scratch/runs/flavell_sweep_aligned_20261024_0032/trial20_aligned.log

## 2026-10-24 01:13 — surprise

Trained flavell factorial done (r261023-2165243, seed 0, paper's saved baseline trial_0 embedding, scratch/paper_factorial_check.py uncommitted). Stored paper tracks 0.9683. A paper code v0.1.0 + saved u-SVD + 100 seeds 0.9680 (harness reproduces the paper); B HEAD code 0.9668 (-0.1); C HEAD + eval's TruncatedSVD U*S 0.9586 (-0.8); D C with 25 seeds 0.9584 (seed count ~0). On trained features the U*S SVD costs ~0.8 pt (untrained: +1.8), single seed. Label-propagation fixes and seed count are neutral.

## 2026-10-24 01:59 — surprise

Sweep rescoring with name-aligned NWB path (r261024-2391371, 25 seeds, seed 0): round_visnet_full trial27 0.820 -> 0.9118 (misses 8,454, mismatches 11,955); trial20 0.750 -> 0.8307. The alignment bug cost the new networks 8-9 pt too. Remaining gap to the paper baseline (0.969 on the same path) is not like-for-like: these checkpoints were trained on zimmer (ZIM2165 2022_11_28; cross-dataset transfer to flavell) with z=8 crops, 100 frames, 25 epochs, whereas the paper's flavell baseline was trained on flavell itself with z=16, 300 frames, 50 epochs.

## 2026-10-24 02:07 — surprise

GT source matters for flavell: _load_project_gt(project 2025_07_01) has 160 neurons / 216,238 entries; _load_project_gt(flavell_data.nwb) 163 / 231,268; matched by seg id only 89.7% of project entries agree (scratch/compare_gt_sources.py). Paper stored tracks: baseline 0.968 vs NWB GT, 0.927 vs project GT; untrained 0.696 vs 0.672. The paper's 96.7% corresponds to the NWB GT. round_flavell* sweep cells (objective: accuracy, --source project, gt = training project) therefore score ~4 pt low vs the paper's GT. Rescoring round_flavell_more_epochs trial27 (0.880 vs project GT) against NWB GT from its cached embedding.

run r261024-2431943 · pids 2431943 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_sweep_aligned_20261024_0032/flavell_more_epochs_trial27_nwbgt.log

## 2026-10-24 03:44 — surprise

round_flavell_more_epochs trial27 (best flavell-trained revision net; recorded 0.880 vs project GT) rescored against the NWB GT from its own cached embedding (r261024-2431943, eval_accuracy --source project, 25 seeds, seed 0): 0.9271 (misses 5,594, mismatches 11,269). Same path with the paper's flavell baseline ckpt (E2) 0.9518. Like-for-like, the best revision flavell net is ~2.5 pt below the preprint network, not ~9-20 pt as the sweep records suggest.

## 2026-10-24 03:44 — run

Seed repeats (seeds 1, 2; cached embeddings, --source project, NWB GT, 25 seeds, CPU tracking) for the paper flavell baseline ckpt (E2, seed 0: 0.9518) and round_flavell_more_epochs trial27 (seed 0: 0.9271). Tests whether the ~2.5 pt gap exceeds labelprop seed noise.

run r261024-2472680 · pids 2472680, 2472681, 2472682, 2472683 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E2_project_seed1.log, /home/charles/.cache/barlow_scratch/runs/flavell_sweep_aligned_20261024_0032/flavell_more_epochs_trial27_nwbgt_seed1.log, /home/charles/.cache/barlow_scratch/runs/flavell_eval_paperckpt_20261023_2132/E2_project_seed2.log, /home/charles/.cache/barlow_scratch/runs/flavell_sweep_aligned_20261024_0032/flavell_more_epochs_trial27_nwbgt_seed2.log

## 2026-10-24 05:44 — surprise

Seed repeats (r261024-2472680; --source project, NWB GT, 25 seeds): paper flavell baseline ckpt 0.9518 / 0.9517 / 0.9528 (seeds 0/1/2, mean 0.952); round_flavell_more_epochs trial27 0.9271 / 0.9209 / 0.9250 (mean 0.924). Gap 2.8 pt, ~10x the labelprop seed spread: the best revision flavell network is genuinely below the preprint network on this path.

## 2026-10-24 05:45 — run

round_flavell_more_epochs trial27 descriptor variants on --source project, NWB GT, seed 0 (auto/contextual baseline 0.9271): (a) --descriptor_stage backbone (image-only, as the preprint used); (b) --center_per_volume at eval (per-frame mean removed). Fresh GPU embeds, CPU tracking. Tests whether the 2.8 pt gap to the preprint network comes from the position-fused descriptor.

run r261024-2523199 · pids 2523199, 2523200 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/flavell_sweep_aligned_20261024_0032/trial27_backbone.log, /home/charles/.cache/barlow_scratch/runs/flavell_sweep_aligned_20261024_0032/trial27_centered.log

## 2026-10-24 07:29 — surprise

Flavell GT versions: both project-2025_07_01 GT and flavell_data.nwb GT are self-consistent with the project's masks (mask at GT xyz == GT seg id for 99.9% of sampled entries; scratch/gt_mask_consistency.py, 6 frames). They differ in identity assignment: matched by seg id, median neuron agrees on 93.5% of frames, only 7/160 neurons >99%; 11,748 cells present in both with different seg ids. The 10 NWB-only neurons have more gaps (855-1,475 frames vs median ~1,480), consistent with the user's recollection of a gap filter, but the filter cannot explain the per-frame identity differences: two annotation versions.

## 2026-10-24 07:30 — run

Retrain the preprint flavell baseline config (TrainedBarlow/baseline_flavell/trial_0/train_config.yaml: 64-dim, no position, 50 epochs, 300 frames, z=16) with HEAD code on the local Titan V, seed 0, wandb offline; project_path repointed to flavell_data/projects/2025_07_01 (original path gone). Then eval like E2 (--source project, NWB GT, 25 seeds). Separates training-code/pipeline drift from design: preprint ckpt scores 0.952 on this path, best revision net 0.924; trial27 trained 17,900 steps vs the baseline's 11,900, so training length is not the cause for it. Queued after the trial27 descriptor variants free the GPU.

run r261024-2567328 · pids 2567328 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/run.log

## 2026-10-24 07:35 — surprise

round_flavell_more_epochs trial27 descriptor variants (r261024-2523199; --source project, NWB GT, 25 seeds, seed 0): backbone (image-only) 0.9154, eval-time per-frame centering 0.9130, vs default auto/fused 0.9271. Position fusion adds ~1 pt; centering hurts. The revision network's image backbone alone (0.915) is ~3.7 pt below the preprint backbone (0.952): the gap is in the image features, i.e. training pipeline or backbone-side design (embedding 512, global augmentation, projector norm), not the position branch.

## 2026-10-24 07:38 — run

Retrain run r261024-2567328 failed at first stats write: train_barlow_clusterer.py needs project_dir/log and checkpoints/ to exist (template provides them). Created both and relaunched the same run.sh.

run r261024-2611790 · pids 2611790 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/run.log

## 2026-10-24 08:42 — decision

Committed the eval-reproduction scratch scripts as barlow_track@95b93fb (barlow_track/scripts/scratch/: paper_factorial_check, normalization_check, nwb_gt_coverage, eval_nwb_aligned, compare_gt_sources, gt_mask_consistency, retrain_paper_baseline_flavell/). Results logged earlier in this notebook were produced with these scripts. Decision: commit each new scratch script as it is written so run entries can cite a hash. Retrain r261024-2611790 is training at ~500 steps/h (epoch 2 at step 500); 11,900 steps puts it near 24 h, then a CPU labelprop eval of ~2 h.

## 2026-10-24 08:52 — decision

Staged sweep round_flavell_paper_ablation (barlow_track@54690ba, copied to config_files/round_flavell_paper_ablation/, not launched; the user launches cluster jobs when they have time). Direct grid from the preprint flavell config: use_position {false=legacy crop pipeline, true=global aug + ViSNet add} x embedding_dim {64,512} x projector_norm {batchnorm,layernorm}, 50 epochs, objective accuracy, --repetitions 2 (16 jobs). Corners: preprint design and ~trial27. use_position cannot separate pipeline from position without a visual-only fusion mode (code change, deferred). All 8 models build; objective preflight passes. Rescore vs NWB GT from emb_cache per README.

## 2026-10-25 05:37 — result

Retrain r261024-2611790 finished training (11,900 steps, same as preprint run; 23 h on the local Titan V). Final losses vs preprint run (TrainedBarlow/baseline_flavell/trial_0): train 2.8e-5 vs 1.5e-5, val 9.0e-4 vs 5.8e-4, test 2.9e-3 vs 6.4e-4. Split and seed differ (seed 0 here) and the loss code changed since v0.1.0, so not decisive; val loss does not predict accuracy anyway. eval_accuracy (project path, NWB GT, CPU labelprop) started 05:37.

## 2026-10-25 07:24 — result

RESULT: preprint flavell baseline config retrained with current code (HEAD bc9ff96, seed 0, 11,900 steps) scores 0.9559 on eval_accuracy --source project vs NWB GT (25 seeds, seed 0; misses 3,660, mismatches 6,534 of 231,268). Preprint checkpoint on the same path 0.952 (it loses ~0.7 pt to the 5,99.5 vs 5,100 normalization mismatch; this net is trained and evaluated with the same one). trial27 0.924. So the current training+eval code reproduces >95%; the ~3 pt gap of the revision networks is their design, not the pipeline or training length.

## 2026-10-25 07:24 — run

Seed repeats 1,2 of the retrained preprint baseline from its cached embedding (labelprop noise; preprint ckpt spread 0.951-0.953).

run r261025-2443993 · pids 2443993 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/seeds.log

## 2026-10-25 07:25 — run

Local single cell of round_flavell_paper_ablation: preprint flavell config with use_position true (global augmentation + ViSNet add fusion; emb 64, batchnorm, 50 epochs, seed 0). Then eval_accuracy project path vs NWB GT at descriptor_stage auto (fused) and backbone. Compare to the retrained preprint config 0.9559: tests whether the augmentation/position switch alone explains the revision networks' ~3 pt gap.

run r261025-2444559 · pids 2444559 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position/run.log

## 2026-10-25 10:34 — result

Seed repeats r261025-2443993 of the retrained preprint flavell baseline (current code; project path, NWB GT, 25 seeds): seeds 0/1/2 = 0.9559 / 0.9572 / 0.9579, mean 0.957. Preprint checkpoint 0.952 (0.9518/0.9517/0.9528); round_flavell_more_epochs trial27 0.924. Reproduction of >95% confirmed beyond labelprop noise.

## 2026-10-25 15:10 — result

RESULT r261025-2444559: preprint flavell config with ONLY use_position switched on (global augmentation + ViSNet add fusion; emb 64, batchnorm, 11,900 steps like the preprint) scores 0.7953 fused / 0.7559 backbone-only (project path, NWB GT, seed 0) vs 0.957 with use_position off. So switching the data pipeline costs ~16-20 pt at the preprint's size; emb 512 + layernorm + 75 epochs (trial27) recover most of it (0.924 / 0.915). Val loss rose over training (0.045 -> 0.054). Zero-placeholder crops 4,206 in training (~0.2%), not the cause. Training ~6x faster than legacy (3.5 h vs 23 h). Candidate cause: the position pipeline's per-crop transform is blur+noise only; the preprint applied per-crop RandomAffine (180 in-plane), a fixed 180 rotation and RandomElasticDeformation (1,3,3) with p=1.

## 2026-10-25 15:10 — decision

Bug in staged round_flavell_paper_ablation (fixed before launch): barlow.Transform applies global_augment on the legacy (use_position false) path too, so the baseline config's global_augment block would have changed the preprint cells. Removed it; use_position true cells now use DEFAULT_GLOBAL_ARGS (p_flip 0.0 vs 0.25). Cluster copy resynced.

## 2026-10-25 15:17 — run

Position pipeline with the preprint's per-crop augmentation (barlow_track@d8cbddd, scratch/train_position_with_legacy_crop_aug.py: alternating legacy view transforms replace build_photometric_transform). Otherwise identical to r261025-2444559 (0.795 fused / 0.756 backbone). If it recovers to ~0.95, the missing per-crop rotation/elastic augmentation is the revision networks' gap; eval at auto and backbone.

run r261025-2666148 · pids 2666148 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_legacyaug/run.log

## 2026-10-25 18:25 — run

User launched round_flavell_paper_ablation on the cluster (16 jobs: 8-config grid use_position x embedding_dim x projector_norm, --repetitions 2; first slurm jobs 7666018-7666027 submitted, 4 running at first check, GPUs busy). Trial configs verified: trial_0-7 seed 43, trial_8+ replicate seed 921677937; no global_augment. Objective scores vs project GT; rescore vs NWB GT per README. Local cells already done: (false,64,bn) 0.957, (true,64,bn) 0.795/0.756.

run r261025-j7666018 · jobs 7666018, 7666019, 7666020, 7666021, 7666022, 7666023, 7666024, 7666025, 7666026, 7666027 · host charles-Precision-7820-Tower

## 2026-10-26 09:57 — issue

Flavell NWB path of eval_accuracy.py has a neuron-order bug (found 2026-10-23 22:07, recorded then only as a surprise entry). `_load_nwb_arrays` selects each field with `df.loc[:, (slice(None), field)]`, which returns the columns in frame order, and in flavell_data.nwb the id fields (raw_neuron_ind_in_list, raw_segmentation_id) are in a different neuron order than x/y/z. `--source nwb` on Flavell therefore crops at one neuron's position and labels it with another's ids, keeping only 209,896 of 231,268 GT entries (accuracy cap 0.908 for any network). Leifer and Samuel NWBs are aligned (scratch/nwb_gt_coverage.py). Fix: select each field by neuron name (scratch/eval_nwb_aligned.py monkeypatches it); eval_accuracy.py itself is not fixed yet.

- [x] issue i261026-1 · affects [[@BarlowTrack revision accuracy evaluation]]

## 2026-10-26 10:00 — judged

Judging i261026-1 (Flavell NWB neuron-order bug), which touches only Flavell scored with `--source nwb`; Leifer and Samuel NWBs are aligned, and the project path takes positions from the segmentation.

- i261026-1: [[@BarlowTrack revision accuracy evaluation]] describes the script at bc9ff96, which has the bug; the bug and its size are recorded in [[EVD - A neuron-order bug in the Flavell NWB evaluation cost about ten points]], and the note stays as the record of what was run.
- i261026-1: [[EVD - Re-evaluation reproduces the preprint's untrained-network accuracies]] scored Flavell from eval_projects/flavell (project path, docs/accuracy_evaluation.md section 2) and Leifer from its aligned NWB; unaffected, and so is [[CLM - Label propagation on an untrained encoder already tracks most neurons on some datasets]].
- i261026-1: [[EVD - Label propagation is seed-fragile on weak embeddings]] rests on Samuel, whose NWB is aligned; unaffected.
- i261026-1: [[EVD - ViSNet position encoder beats a keypoint MLP on every dataset]] ranks the encoders on Zimmer, Leifer and Samuel, which the bug does not touch; its "Flavell flat .62-.73" result was the bug's doing and is replaced by [[EVD - Revision networks trail the preprint by 1 to 3 points once Flavell is scored correctly]].
- i261026-1: [[@BarlowTrack name-aligned NWB evaluation]] is the fix; [[EVD - A neuron-order bug in the Flavell NWB evaluation cost about ten points]] measures the bug; [[EVD - The preprint Flavell network scores 95 to 97 percent under the current evaluation]] uses the project path and the aligned NWB path. All written for this issue, backdated to their runs.

## 2026-10-26 10:00 — resolved

All notes judged; superseded where the bug changed a result. The code fix in eval_accuracy.py is a to-do.

resolved i261026-1

## 2026-10-26 10:00 — todo

From the Flavell reproduction (2026-10-23 to 25).

- [x] fix eval_accuracy._load_nwb_arrays: select each NWB field by neuron name (i261026-1), then rescore NWB-path Flavell records
- [ ] rescore round_flavell* cells against the NWB GT (the accuracy objective scores against the project GT)
- [x] make the NWB path not hand the tracker only GT objects (user: the algorithm should not see only GT objects)

## 2026-10-26 10:00 — note

i261023-1, partial: on Flavell, the tracking code at HEAD (label-propagation fixes since v0.1.0) moves the preprint's trained accuracy by -0.1 pt and untrained by +0.8 pt on the same saved embeddings ([[EVD - Re-tracking the preprint's saved Flavell embeddings reproduces its stored accuracies]]). Not a judgment of its notes: the other labs, tracking windows and crops are untested.

## 2026-10-26 10:00 — judged

i261026-1: [[EVD - Revision benchmarks score below the preprint, most on Flavell]] is redone: superseded by [[EVD - Revision networks trail the preprint by 1 to 3 points once Flavell is scored correctly]] (dated to its runs, so before the issue entry).

## 2026-10-26 10:00 — handoff

**Where things stand.** Written from a review session in the yggdrasil repo, distilling the 2026-10-23 to 25 notebook (the working session never touched the notes). On Flavell the preprint holds: [[CLM - Current training and evaluation code reproduce the preprint's Flavell accuracy]] (saved embeddings re-track to 0.968, the preprint network scores 0.952-0.969 today, its config retrained today 0.957). The revision sweeps' .62-.73 was mostly the evaluation: [[CLM - Most of the revision networks' Flavell deficit came from the evaluation, not the networks]] (NWB neuron-order bug, issue i261026-1, judged and resolved; and the project vs NWB GT versions). The real gap is 2.8 pt on Flavell, in the image features; working hypothesis, low confidence: [[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]].

**Runs.**
- r261025-2666148 running locally (preprint per-crop augmentation in the position pipeline, training since 2026-10-25 15:17, ~23 h like the legacy pipeline): for [[EVD - Preprint per-crop augmentation in the position pipeline]]; output /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_legacyaug/.
- r261025-j7666018 on the cluster (16 jobs of round_flavell_paper_ablation, launched by the user): for [[EVD - Flavell paper ablation grid of pipeline, embedding size and projector norm]]; rescore against the NWB GT from emb_cache per its README. Not checked from here.
- Every run since the last handoff is recorded in a node's Method or a later entry; the OOM'd first attempts (r261023-2312954, r261023-2312979, r261023-2323712, r261023-2330704) and the failed first retrain (r261024-2567328) were rerun.

**Next.**
1. Read r261025-2666148 when it ends and fill [[EVD - Preprint per-crop augmentation in the position pipeline]]: near 0.95 supports the augmentation claim, near 0.80 opposes it.
2. Rescore the cluster grid against the NWB GT and fill [[EVD - Flavell paper ablation grid of pipeline, embedding size and projector norm]].
3. Zimmer and Leifer in [[EVD - Preprint baseline networks re-scored with the corrected evaluation]] (Flavell done; still blocked-by the regression tests in the tree, though the Flavell reproduction went ahead without them).
4. Three new to-dos: fix `_load_nwb_arrays` in eval_accuracy.py, rescore round_flavell* against the NWB GT, and stop the NWB path giving the tracker only GT objects.

**Loose state.** research/ is git-ignored (the uncommitted `research/*` line in .gitignore), so this tree has no history or backup. Scratch scripts are committed through d8cbddd on global_augmentation; eval_accuracy.py is unchanged. The working session ba25cb4a may still be open and planned to check r261025-2666148 around midday 2026-10-26. i261023-1 and i261023-2 remain open; a partial Flavell finding for i261023-1 is in the entry of 10:00.

## 2026-10-26 14:11 — result

RESULT r261025-2666148: position pipeline + preprint per-crop augmentation (rotation, 180 rotation, elastic) scores 0.6571 fused / 0.4870 backbone (project path, NWB GT, seed 0), WORSE than without it (0.795 / 0.756) and backbone below an untrained net (~0.70). Per-crop augmentation hypothesis refuted. No collapse (fused eff rank 30 in both); both reach train loss ~1.2e-4 and val loss rises in both. New hypothesis: position shortcut. Within a pair the views are one volume under a rigid transform, so position nearly solves the Barlow loss and the image branch need not learn identity; heavier crop augmentation pushes the net further onto position, which is a weak cue across time.

## 2026-10-26 14:11 — dead-end

Restoring the preprint per-crop augmentation in the position pipeline does not recover accuracy (0.657 vs 0.795 without it); it makes it worse.

## 2026-10-26 14:11 — run

Position pipeline with the position branch cut out (scratch/train_position_pipeline_visual_only.py: fused_descriptors returns backbone(y)); otherwise identical to r261025-2444559. Eval backbone stage only. ~0.95 => position shortcut explains the gap; ~0.75 => the global-augmentation pipeline itself.

run r261026-1482855 · pids 1482855 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_visual_only/run.log

## 2026-10-26 14:30 — result

Flavell inter-frame non-rigid motion (barlow_track@7c41a37, scratch/position_deformation_vs_augmentation.py; NWB GT, 200 pairs per lag, similarity fit = rotation+translation+uniform scale): residual per neuron median 3.09 / p90 7.66 voxels at lag 1; 5.14 / 12.31 at lag 10; 5.72 / 13.30 at lag 100; 5.54 / 13.03 at lag 500. Nearest-neighbor distance median 6.76. 71-77% of neurons move more than half a neighbor distance non-rigidly at lag >= 10. Fitted scale p5-p95 ~0.86-1.09 (training scale_jitter 0.1 matches). Training position augmentation: jitter 1 voxel iid (|d| median ~1.5), dropout 0.2, nothing non-rigid; ViSNet is rotation/translation invariant. So the two training views' position features differ far less than real frames do: supports the position-shortcut reading (user's point: non-rigid deformation is needed for the position views to differ).

## 2026-10-26 14:55 — decision

Implemented global elastic deformation (barlow_track@914141c): global_augment elastic_std / elastic_length, one smooth field per view applied to volume and keypoints (backward map for the volume, fixed-point inversion at points). Off by default; tests added; full suite 137 passed, 3 failures pre-existing on HEAD (test_projector_uses_layernorm_not_batchnorm, test_old_batchnorm_checkpoint_loads_leniently, test_embed_single_volume_subset). Real flavell frame: centroid intensity corr 0.999 after warp; 0.85 s/view vs 0.25 without. User lifted the no-repo-changes constraint (cluster sweep runs from its own checkout).

## 2026-10-26 14:55 — result

Neighbor-distance change, real flavell frames vs training position augmentation (scratch/local_geometry_change.py; std of log distance ratio by initial distance 0-8/8-12/12-20/20-30/30-50/50-70/70-140 voxels): real lag 100 0.16/0.19/0.17/0.15/0.13/0.13/0.12; current augmentation (jitter 1, scale 0.1, no elastic) 0.28/0.22/0.15/0.11/0.09/0.08/0.08. So the current position augmentation is NOT much weaker than real deformation: stronger for close pairs, ~35% weaker at ViSNet range (cutoff 0.35 normalized = ~70 voxels). Matched: jitter 0.3 + elastic (1,4,4) length 40 gives 0.18/0.18/0.16/0.15/0.14/0.13/0.11. Displacement-std matching (2.4,5.2,7.0 at length 15) folds the field; real motion beyond a similarity fit is mostly near-rigid bending. Weakens the 'views too similar' version of the position-shortcut hypothesis; the image-only run r261026-1482855 is the decisive test.

## 2026-10-26 15:28 — result

Neighbor-distance change on all four GT datasets (barlow_track@0ad34c2, scratch/local_geometry_change.py; std log distance ratio, bins in NN units 0-1.5/1.5-2.5/2.5-4/4-6/6-10/10-cutoff). NN distance (voxels): zimmer 9.5, flavell 6.8, leifer 10.5, samuel 5.1; ViSNet cutoff 23/10/15/15 NN. Real lag 100 vs current augmentation (jitter 1 voxel, scale 0.1): zimmer 0.26/0.22/0.18/0.14/0.11/0.10 vs 0.21/0.14/0.11/0.10/0.09/0.09; flavell 0.18/0.18/0.16/0.13/0.13/0.12 vs 0.27/0.17/0.12/0.10/0.09/0.08; leifer 0.36/0.24/0.18/0.13/0.11/0.10 vs 0.22/0.13/0.11/0.10/0.10/0.10; samuel 0.33/0.29/0.25/0.20/0.14/0.10 vs 0.35/0.22/0.15/0.11/0.09/0.09. Common: current augmentation under-perturbs the mid range (2.5-10 NN) by ~1.3-2x in every dataset; far range ~right (scale jitter); near range dataset-dependent because jitter is in voxels (0.1-0.2 NN). Real scale p5-p95 ~0.82-1.10, matches scale_jitter 0.1. RMS log mismatch current 0.34-0.40; best elastic match 0.06-0.11, xy std ~3-4 voxels in all four (0.3-0.6 NN), length 15-41 voxels (3-6 NN), z std small. Extended grid running (some best matches at grid edge).

## 2026-10-26 15:31 — result

Extended elastic-match grid (local_geometry_change.py --lags 100, finer grid). Best per dataset (elastic_std z,x,y voxels / elastic_length / jitter_std; RMS log mismatch vs real lag 100, current augmentation in brackets): zimmer_1128 (0.2,2.9,2.9)/19/0.5: 0.068 [0.35]; flavell (1.1,4.1,4.1)/41/0.14-0.34: 0.075 [0.35]; leifer (0.3,3.1,3.1)/42/2.1: 0.040 [0.59]; samuel (0.2,2.3,2.3)/10/0.26: 0.031 [0.38]. xy std 2.3-4.1 voxels everywhere; length 10-42 and jitter 0.1-2.1 dataset-specific. Leifer estimates are sampling-noisy: its current-augmentation mismatch was 0.39 in the first run and 0.59 here with a different frame sample (100 pairs, 79 neurons).

## 2026-10-26 15:54 — run

Rerun of r261026-1482855 (image-only descriptor in the position pipeline, preprint flavell config, 50 epochs, then backbone eval vs NWB GT). Attempt 1 died at epoch 15 on OSError EIO reading the flavell zarr from /lisc (mount glitch); no resnet50.pth saved, no resume path, so restarted from scratch. Decides whether the position pipeline's augmentation (~0.75) or position fusion (~0.95) holds the revision networks below the preprint.

run r261026-1549224 · pids 1549224 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_visual_only/run.log · for [[EVD - Preprint per-crop augmentation in the position pipeline]]

## 2026-10-26 16:15 — dead-end

r261026-1549224 died again on EIO reading the flavell zarr over /lisc (19 min in, epoch 1). The /lisc CIFS mount is 'soft', so server hiccups surface as EIO rather than retries. Mirroring the flavell project (~19 GB) and GT NWB to ~/.cache/barlow_scratch/local_data/flavell and rerunning the image-only run from there.

## 2026-10-26 16:21 — run

Third attempt of the image-only position-pipeline run (r261026-1482855, r261026-1549224 died on /lisc EIO), now reading the local mirror of the flavell project and GT NWB. Same config otherwise: preprint flavell config, use_position pipeline with fused_descriptors cut to the backbone, 50 epochs, backbone eval vs NWB GT. ~0.95 means position fusion costs the accuracy; ~0.75-0.80 means the position pipeline's augmentation does.

run r261026-1561729 · pids 1561729 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_visual_only/run.log · for [[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]]

## 2026-10-26 17:50 — surprise

Old revision sweep rounds read back (scratch/old_sweep_records.py at the commit above). (1) Training-seed spread of the position pipeline on flavell is large: round_flavell_paper_ablation same config, seed 43 vs 921677937: 0.743/0.811, 0.754/0.870, 0.787/0.840 (project GT); round_flavell_more_epochs 16x64 cells span 0.769-0.880 (std ~0.03). Every seed-43 ablation cell is at the low end, as was the local paper_flavell_use_position (0.795 NWB GT, seed 43). Single-run comparisons of the position pipeline are worth ~+-4 pt; the legacy pipeline's training-seed spread is unmeasured (one retrain, 0.957, plus the preprint checkpoint 0.952). (2) Training length barely matters for flavell in the position pipeline: 2,000 steps mean 0.77, 12,000 0.80, 18,000 0.80 (project GT). Crop 16x64 beats 16x32 by ~3-5 pt; z=8 is worse. Attention (self_layers 0) does nothing. (3) The only legacy-pipeline arm in any old round is round_2x2 (2 trials, 400 steps): zimmer 0.787 vs 0.89 for position arms, leifer 0.919 vs 0.94, samuel ~equal; too short to say anything about the 12k-step preprint design. (4) On zimmer (training set), the position pipeline reaches 0.94-0.97 (visnet_full; preprint 97.3%), and zimmer-trained visnet_full trial27 transfers to flavell at 0.912 (aligned NWB) - about what flavell-trained position models reach (more_epochs trial27 0.924). The position pipeline's deficit is flavell-specific, consistent with flavell having the largest non-rigid motion (elastic match std (1.1,4.1,4.1) vs ~(0.2-0.3, 2.3-3.1) elsewhere).

## 2026-10-26 18:01 — run

Rescore flavell sweep cells against the NWB GT from their cached embeddings, CPU only (scratch/rescore_flavell_nwb_gt.sh; local copies of embeddings, checkpoints, project and GT; 4 in parallel, ~1.5 h each). Batch 1: the 7 finished round_flavell_paper_ablation cells (trials 4-7, 12-14; includes the seed-43 vs seed-921677937 pairs). Batch 2 (queued after batch 1): the 12 unrescored 16x64 cells of round_flavell_more_epochs (17-26, 28, 29; trial27 done, 0.927). Tests whether the project-to-NWB GT offset is constant (+4.1 preprint stored tracks, +4.7 trial27) and gives the position pipeline's training-seed spread on the preprint's metric.

run r261026-1606883 · pids 1606883, 1608277 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/rescore/round_flavell_paper_ablation.log, /home/charles/.cache/barlow_scratch/rescore/round_flavell_more_epochs.log · for [[EVD - Flavell paper ablation grid of pipeline, embedding size and projector norm]]

## 2026-10-26 18:05 — decision

Correction to the 17:50 entry: the local paper_flavell_use_position run (0.795 NWB GT) and all other local runs (baseline retrain, legacyaug, image-only) use training seed 0, not 43. The local cell is a third training seed of the position design that also landed low, not a replicate of the cluster's seed-43 cells.

## 2026-10-26 18:12 — decision

Added [[QUE - Which part of the position pipeline costs Flavell accuracy, its augmentation or its position fusion]] (backdated to 2026-10-25 07:25, when the use_position switch run raised it; it had been tracked across runs without a node). Under it: the two competing answers ([[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]], scope revised after the per-crop result; new [[CLM - Position fusion lets the position pipeline skip learning appearance on Flavell]], the position-shortcut idea of the 2026-10-26 entry), the filled per-crop evidence (failed: 0.657/0.487), the neighbour-distance evidence, the in-progress image-only run and the planned elastic run.

## 2026-10-26 22:24 — run

Fused position pipeline with Flavell-matched global elastic deformation (elastic_std (1.1,4.1,4.1), length 41, jitter_std 0.3 instead of 1.0; otherwise the preprint Flavell use_position cell, seed 0, num_workers 6), barlow_track@803401c, scored against the NWB GT at auto and backbone. Fused first because the fused cell has three training seeds to compare against (0.769/0.795/0.848).

run r261026-1722467 · pids 1722467 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_elastic/run.log · for [[EVD - Flavell-matched global elastic deformation in the position pipeline]]

## 2026-10-26 22:25 — decision

r261026-1561729: image features alone in the position pipeline score 0.8186 on Flavell (NWB GT, seed 0), inside the fused cell's training-seed range 0.769-0.848; the data path, not fusion, costs the 14 points. Recorded in [[EVD - Image features alone in the position pipeline]]; [[CLM - Position fusion lets the position pipeline skip learning appearance on Flavell]] refuted as main cause, [[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]] raised to medium.

## 2026-10-26 22:25

r261026-1606883 batch 1 (7 round_flavell_paper_ablation position cells vs NWB GT): offsets NWB minus project GT +2.6 to +4.2 (mean 3.4); seed 921677937 beats seed 43 in all three pairs by 5-12 points. Recorded in [[EVD - Flavell paper ablation grid of pipeline, embedding size and projector norm]]. Batch 2 (12 round_flavell_more_epochs cells) running.

## 2026-10-26 22:25 — surprise

Validation loss: the legacy pipeline's falls throughout training (0.0044 -> 0.0009, train 2.8e-5), while every position-pipeline run's falls for a few epochs and then rises (fused 0.036 -> 0.054, image-only 0.040 -> 0.064, per-crop aug 0.044 -> 0.069; train ~1.2e-4), 40-70x higher. With fusion ruled out, the global data path seems to let the image features fit the training frames without generalizing to held-out ones. Worth checking what the views look like (napari viewer) and how the two data modules differ beyond augmentation.

## 2026-10-26 22:38 — run

r261026-1722467 crashed at its first batch: BadZipFile (Bad CRC-32) in a DataLoader worker. The local zip passes testzip; forked workers share the zarr ZipStore's file handle, so num_workers > 0 is unsafe with zip-stored projects (fails loudly; no sweep config sets num_workers). Relaunched with num_workers 0 (barlow_track@01e6be3), otherwise identical; ~2.35 s per item, so ~8 h of training.

- [ ] make VolumeCoordsDataModule worker-safe with zip-stored zarr (reopen red_data per worker in worker_init_fn) if data loading becomes the bottleneck

run r261026-1730274 · pids 1730274 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_elastic/run.log · for [[EVD - Flavell-matched global elastic deformation in the position pipeline]]

## 2026-10-27 04:00 — surprise

r261026-1606883 batch 2 (round_flavell_more_epochs 17-26, 28, 29, NWB GT): 0.802-0.886; with trial27 (0.927) 13 seeds of one config span 0.80-0.93, median 0.865. trial27 was the best seed, so the 2.8-pt gap in [[EVD - Revision networks trail the preprint by 1 to 3 points once Flavell is scored correctly]] is best-of-13; typical gap ~9 pt. Recorded in [[EVD - Position-pipeline Flavell accuracy spans 12 points across training seeds]].

## 2026-10-27 04:01

r261026-1730274 at epoch 30 (7,200 steps, 2.57 s/step, ends ~07:00): val loss falls to 0.0182 at epoch 8, rises to 0.0316 at epoch 25, back to 0.023 by epoch 29. Its minimum is half that of the same cell without elastic (0.036) and of image-only (0.040), although its val views are deformed more, so the rise over training is smaller but not gone. Val loss tracks accuracy weakly; wait for the eval.

## 2026-10-27 10:31 — todo

r261026-1730274 ended: elastic position cell fused 0.7175, backbone 0.5982 vs 0.7953 / 0.7559 without elastic at the same seed. Elastic is not what the position pipeline lacks; added geometric augmentation keeps making its image features worse (per-crop: backbone 0.487), unlike in the legacy pipeline. Filled [[EVD - Flavell-matched global elastic deformation in the position pipeline]].

- [x] extract_crops zero-fills points at y≈119.5 that in_bounds_mask keeps (Flavell volume 120 wide): align the bounds test with the crop rounding

## 2026-10-27 11:11 — run

Train + eval the flavell use_position cell with the global rotation and flip off (norot; scale/translation kept), seed 0, local mirror, NWB GT at auto and backbone. Tests whether rotating the 284x120 volume (10% objects lost, 18% crops off-centre) costs the position pipeline's Flavell accuracy.

run r261027-2053863 · pids 2053863 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_norot/run.log · for [[EVD - Global rotation off in the position pipeline]]

## 2026-10-27 11:26 — dropped

stopped at step ~200 to restart with the per-epoch matching proxy (val_match_frames 30), which does not change training

dropped r261027-2053863

## 2026-10-27 11:26 — run

Restart of r261027-2053863 (norot cell) with val_match_frames 30: the per-epoch frame-pair matching proxy gives a learning curve on the tracking task to set against val loss.

run r261027-2062947 · pids 2062947 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_norot/run.log · for [[EVD - Global rotation off in the position pipeline]]

## 2026-10-27 11:27 — decision

Val loss vs accuracy: val loss scores the training augmentation's pretext task on ~30 held-out frames, re-augmented every epoch (shared dataset, set_epoch), in train mode (no model.eval(); batchnorm projector uses and updates batch stats on val). Not comparable across augmentation settings. Added a GT-based proxy (frame-pair matching on never-sampled frames, val_match_frames; db105ba): Spearman 0.74 with full accuracy over 24 checkpoints, recorded in [[EVD - Frame-pair matching on held-out frames ranks Flavell networks by tracking accuracy]].

- [x] Validation runs in train mode: add model.eval() around the val loss (changes logged val numbers, not training)
- [ ] GT-free variant of the matching proxy (cycle consistency over frame triples) for users without ground truth

## 2026-10-27 11:41

r261027-2062947 epoch 0: proxy works in the loop (30 held-out frames, 435 pairs, ~1 min): match auto 0.390, backbone 0.361 (the rotating use_position cell's final checkpoint: 0.373 / 0.366 on whole-video frames); val loss 0.0091.

## 2026-10-27 11:42 — run

Rescore round_flavell_paper_ablation legacy cells 0, 1, 3 (seed 43) and position cell 15 (seed 921677937) against the NWB GT from cached embeddings, CPU. Gives the legacy pipeline's first same-code seed points for the seed-spread comparison.

run r261027-2071685 · pids 2071685 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/rescore/round_flavell_paper_ablation_batch2.log · for [[EVD - Flavell paper ablation grid of pipeline, embedding size and projector norm]]

## 2026-10-27 13:43 — run

r261027-2071685: legacy seed-43 cells 0.953 (64 bn), 0.950 (64 ln), 0.944 (512 ln) against the NWB GT, position trial15 (512 ln, seed 921677937) 0.901. Legacy cells all at preprint level; recorded in [[EVD - Flavell paper ablation grid of pipeline, embedding size and projector norm]]. Legacy training-seed spread waits on cluster cells 8-11.

## 2026-10-27 15:54 — run

r261027-2062947 (norot) training done. Per epoch, val loss and the frame-pair matching proxy move together, the wrong way: val loss bottoms at epoch 10 (0.0061) and climbs to 0.0110 at epoch 49 (Spearman with epoch +0.80), while the proxy rises almost monotonically 0.390 -> 0.474 fused, 0.361 -> 0.469 backbone (Spearman with epoch +0.96). Spearman(val loss, proxy) = +0.80: early stopping on val loss would keep epoch 10 (proxy 0.42). Proxy at the end is above the rotating cell's 0.373 / 0.366. Full evals (auto, backbone) now running.

## 2026-10-27 15:56 — run

Norot evals handed off: eval_then_croprot.sh (barlow_track@b39f823) stops run.sh once each stage's embedding is cached, starts the croprot training, then tracks both norot stages from cache on CPU. Replaces the eval half of r261027-2062947.

run r261027-2180262 · pids 2180262 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_norot/eval_then_croprot.log · for [[EVD - Global rotation off in the position pipeline]]

## 2026-10-27 16:19 — run

Croprot arm: preprint Flavell config with use_position on, global rotation and flip off, per-crop rotation up to 180 deg about each crop's own z axis (barlow_track@44c0e11 config, proxy from db105ba); compares with norot to test whether image features need rotation diversity. Started by the handoff r261027-2180262.

run r261027-2190583 · pids 2190583 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_croprot/run.log · for [[EVD - Global rotation off in the position pipeline]]

## 2026-10-27 16:44 — run

r261027-2190583 (croprot) epoch 0: proxy 0.463 fused / 0.391 backbone, against norot's 0.390 / 0.361 at epoch 0 (val loss 0.0144, not comparable across augmentations). Runs at 8.2 GB GPU beside the two norot CPU evals.

## 2026-10-27 18:00 — decision

Correction to the r261027-2062947 epoch-0 entry: the rotating use_position cell's final checkpoint scored 0.477 fused / 0.366 backbone on the proxy (proxy_30.jsonl), not 0.373 / 0.366.

## 2026-10-27 18:00 — run

Proxy on the norot final checkpoint with the same 30 whole-video frames as the 24-checkpoint batch (CPU), so it compares with the rotating cell's 0.477 / 0.366.

run r261027-2876617 · pids 2876617 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/val_matching/proxy_30_norot.log

## 2026-10-27 18:15 — issue

Position-pipeline training repeats the same augmented view pairs every epoch. VolumeCoordsDataset seeds each item from (seed, epoch, index) and its epoch advances only via a Lightning trainer; train_barlow_clusterer.py builds the loader once and runs its own epoch loop, so the epoch stays 0 and all 50 epochs see one fixed pair per training frame (~240 pairs). Confirmed on the local Flavell project (crops and keypoints identical across two passes of the loader). In since barlow_track@80a2bbd (2026-10-05, Fix determinism #144), so it covers every use_position network of the Oct 2026 rounds; the legacy crop pipeline draws from the global RNG and is not affected. Opt-in fix reaugment_each_epoch in barlow_track@7f9f7bc. Fits the rising validation loss of every position run and the finding that stronger augmentation made things worse.

- [ ] issue i261027-1 · affects [[@BarlowTrack revision benchmark sweeps]], [[@BarlowTrack preprint Flavell configuration with the position pipeline]], [[@BarlowTrack preprint Flavell configuration in the position pipeline with image features only]], [[@BarlowTrack preprint Flavell configuration with the position pipeline and preprint per-crop augmentation]], [[EVD - Flavell-matched global elastic deformation in the position pipeline]], [[EVD - Global rotation off in the position pipeline]]

## 2026-10-27 18:15 — dropped

Stopped at about epoch 25 to free the GPU: like every position run it trained on frozen view pairs (issue i261027-1), so the per-crop rotation question waits on the re-augmentation test.

Stopped PID 2190583.

dropped r261027-2190583

## 2026-10-27 18:16 — decision

Correction to the r261027-2190583 drop: croprot had finished 8 epochs, not 25.

## 2026-10-27 18:16 — run

Re-augmentation test: the use_position cell (paper_flavell_use_position, fused 0.795 / backbone 0.756, seed 0, global rotation on) with only reaugment_each_epoch true (barlow_track@aa2dbea:barlow_track/scripts/scratch/paper_flavell_use_position_reaug/run.sh). Near 0.95 fused means the frozen view pairs (i261027-1) are what costs the position pipeline its Flavell accuracy.

run r261027-562757 · pids 562757 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_reaug/run.log

## 2026-10-27 18:16 — run

r261027-2876617: norot final checkpoint on the proxy's 30 whole-video frames: 0.499 fused / 0.497 backbone, against the rotating cell's 0.477 / 0.366. Accuracy moved -0.8 fused and +2.4 backbone points, so the proxy ranked the fused pair the wrong way (within noise) and overstated the backbone gain (+13 proxy points for +2.4 accuracy). It orders networks roughly, not by the few points that separate close arms.

## 2026-10-27 18:16 — ask

Every use_position network since 2026-10-05 trained on one fixed augmented view pair per frame (i261027-1). barlow_track@7f9f7bc adds reaugment_each_epoch, off by default per the keep-new-behaviour-off rule, so cluster rounds still train frozen unless their configs set it. r261027-562757 tests whether it closes the Flavell gap (result about 23:00).

- [ ] ask a261027-1 · on [[QUE - Which part of the position pipeline costs Flavell accuracy, its augmentation or its position fusion]] · Once the re-augmentation test is in, should reaugment_each_epoch default to true, or stay opt-in and be set per round?

## 2026-10-27 18:36 — dropped

Crashed in the epoch-0 validation plot: a fresh view with 0 objects reached _correlation_for_plot, which guarded only view 1 (fixed in barlow_track@0de29c9). Epoch 0 had val loss 0.0447, proxy 0.428 / 0.325. Log moved to run_attempt1_crash.log.

dropped r261027-562757

## 2026-10-27 18:36 — run

Re-augmentation test, attempt 2 (barlow_track@0de29c9; config unchanged from aa2dbea): the use_position cell with only reaugment_each_epoch true. Near 0.95 fused means the frozen view pairs (i261027-1) cost the position pipeline its Flavell accuracy.

run r261027-572595 · pids 572595 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_reaug/run.log

## 2026-10-27 19:01 — run

r261027-572595 (reaug) past the crash point. Val loss 0.0446, 0.0248, 0.0224 over epochs 0-2, against the frozen comparator's 0.0446, 0.0381, 0.0348 (its val views are now re-drawn each epoch too, so the curves compare loosely). Proxy 0.429 -> 0.444 fused, 0.324 -> 0.343 backbone.

## 2026-10-27 19:08 — run

Review finding (subagent, verified by reading): with --device cuda, eval_accuracy's _extract_crops_gpu (eval_accuracy.py:147-192) still clips and zero-pads edge crops, while the CPU path (data_loading.get_3d_crop_using_bbox_or_centroid) and training slide the window inside the volume since barlow_track@726c913 (2026-10-05). About 7% of Flavell crops differ. Every --device cuda eval since then, including all of this project's Flavell scores, embedded edge neurons on crops unlike training. This run measures the size: norot auto re-embedded on CPU, against 0.787 from GPU crops.

run r261027-589796 · pids 589796 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_norot/eval_auto_cpucrops.log

## 2026-10-27 19:08 — issue

GPU eval crops differ from training crops at volume edges: _extract_crops_gpu clips and zero-pads while training and the CPU eval path slide the window inside the volume (since barlow_track@726c913, 2026-10-05). About 7% of Flavell crops, more on thin-z Zimmer volumes; zeros also shift the 5th-percentile normalisation of the frame. Size being measured in r261027-589796.

- [ ] issue i261027-2 · affects [[@BarlowTrack revision accuracy evaluation]], [[@BarlowTrack Flavell sweep cells rescored against the NWB GT]]

## 2026-10-27 19:08 — idea

Code review of training, eval and sweep code (subagent, 2026-10-27; top two verified by reading). Also found: 1) the benchmark driver passes no --gt, so zimmer_1123 and zimmer_1210 cells score against the 1128 GT; 2) reusing a train dir pairs files from different runs (get_sequential_filename writes stats-1.json, args-1.pickle, resnet50-1.pth while readers use the fixed names; live in paper_flavell_use_position_reaug, whose log/stats.json is from the crashed attempt); 3) center_per_volume defaults True in training but False on load when the key is missing (latent for attention configs); 4) an OOM mid-training saves a partial model that counts as a finished trial; 5) the NWB-source eval embeds only GT neurons, so position features see a sparser point set; 6) training rounds crop centres, eval truncates; 7) the embedding cache is keyed only by lab, stage and tag; 8) a resumed direct grid resubmits the whole grid with repeated seeds; 9) val and test losses are sums, in train and eval mode respectively; 10) RandomNoise std 0.25 is added before rescaling, so it is about 0.1% of range and does nothing.

- [x] Fix _extract_crops_gpu to slide windows like the CPU path, behind a flag, after r261027-589796 sizes the effect
- [x] Pass --gt for zimmer_1123 and zimmer_1210 in run_trials_on_all_ground_truth.py
- [x] Make train_barlow_clusterer refuse or clean a train dir that already holds a run
- [x] Round, not truncate, centroids in the eval and val_matching crop paths

## 2026-10-27 19:47 — decision

Fixes in tonight (user asked for them before going to sleep): benchmark driver passes --gt per dataset (barlow_track@98a157b); training moves an earlier run's outputs to superseded/<time>/ (f847c81); eval_accuracy --crop_window slide, val_match_crop_window and val_matching_on_checkpoints --crop_window give GPU crops identical to the CPU path and training (c4a4221, default pad so recorded scores reproduce). Each has a synthetic regression test.

- [ ] Decide with the user whether crop_window slide and reaugment_each_epoch become the defaults (see a261027-1)

## 2026-10-27 19:47 — done

barlow_track@98a157b

Done: Pass --gt for zimmer_1123 and zimmer_1210 in run_trials_on_all_ground_truth.py

## 2026-10-27 19:47 — done

moves earlier outputs aside, barlow_track@f847c81

Done: Make train_barlow_clusterer refuse or clean a train dir that already holds a run

## 2026-10-27 19:51 — decision

Review fixes, all opt-in or bookkeeping. 5d2beb6: args.pickle records the effective center_per_volume of BarlowVolumeAttention models (it was the config value, not what the model used). 36872fe: a trial that runs out of GPU memory is now a failure: partial model to resnet50_oom.pth, stats.json ends with an out_of_memory entry so check_training_finished reads it unfinished, error re-raised after saving (py_compile only, not exercised at runtime). 29d4c76: --skip_completed_grid_jobs in optimize_hyperparameters.py leaves out grid jobs a finished prior trial already ran (same swept values and seed); without it a resumed grid warns, since it retrains exact duplicates with repeated seeds. Tests: test_sweep_accuracy_objective.py 55 passed.

## 2026-10-27 19:51 — done

c4a4221, --crop_window slide (default pad); r261027-589796 still sizing the effect

Done: Fix _extract_crops_gpu to slide windows like the CPU path, behind a flag, after r261027-589796 sizes the effect

## 2026-10-27 19:55 — run

r261027-572595 (reaug) at epoch 12 of 50: re-augmentation is not lifting the proxy so far. Fused proxy 0.429 -> 0.458 (ep 1) -> 0.433, backbone 0.324 -> 0.372 (ep 4) -> 0.311; fused effective rank falls 23 -> 14, where the frozen norot run's rose 31 -> 48 over training. Val loss noisy (0.149 spike at ep 7; val views are re-drawn each epoch). Calibration from proxy_30.jsonl (whole-video frames): legacy baseline_flavell_head_rerun (0.95+ accuracy) scores 0.600 / 0.600, position-pipeline networks 0.36-0.55. Not judging before the end-of-training eval; if fused accuracy stays near 0.80, frozen pairs are not the cause of the gap.

## 2026-10-27 19:59 — done

979ec48: val_eval_mode (default off) plus val_mean_batch_loss; projector-only BN, so tracking descriptors are unaffected either way

Done: Validation runs in train mode: add model.eval() around the val loss (changes logged val numbers, not training)

## 2026-10-27 19:59 — done

b4b0a6a: --round_crop_centers / val_match_round_crop_centers (default off); NWB path already rounded

Done: Round, not truncate, centroids in the eval and val_matching crop paths

## 2026-10-27 19:59 — decision

36872fe (OOM is a failure) now exercised at runtime: slow smoke test with a forced OutOfMemoryError raises, leaves resnet50_oom.pth and no resnet50.pth, check_training_finished False (979ec48). Side note from reading the legacy loader: legacy crops come from regionprops bbox midpoints (truncated) of all segmented objects, on frames with at most 200 objects; the eval crops at metadata centroids. Legacy still scores 0.95 under that mismatch, so crop-center precision is unlikely to explain the position pipeline's gap.

## 2026-10-27 20:07 — run

Check of --nwb_objects (barlow_track@850e8f4, script be9460e): legacy baseline_flavell_head_rerun on the first 60 Flavell NWB frames, CPU, GT objects only vs every detection of the NWB raw_segmentation. Expect all <= gt and both plausible. Found while writing it: each Flavell NWB GT position is exactly one mask label's regionprops centroid (5 frames checked, distance 0.00, one to one), with 20-40 non-GT detections per frame; the NWB raw_segmentation_id is not that label.

run r261027-625851 · pids 625851 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/nwb_objects_check/run.log

## 2026-10-27 20:09 — done

barlow_track@<see log>: round_crop_bounds (default off)

Done: extract_crops zero-fills points at y≈119.5 that in_bounds_mask keeps (Flavell volume 120 wide): align the bounds test with the crop rounding

## 2026-10-27 20:09 — decision

Review fixes, cont. b4b0a6a --round_crop_centers (project-path eval crops at the rounded centroid as training does; default truncates). 979ec48 val_eval_mode + val_mean_batch_loss. 6acc0ba embedding cache keyed on sha1 of the checkpoint plus source (a retrain into the same trial folder no longer reuses stale embeddings; keyless old caches still reused). 850e8f4 --nwb_objects all (the NWB source tracks every raw_segmentation detection; default gt). e225a55 round_crop_bounds (in_bounds_mask on the rounded centre; default off). All opt-in; tests pass (test_sweep_accuracy_objective, test_eval_accuracy_correctness, test_benchmark_driver, test_step1_global_augment, test_train_position --runslow).

## 2026-10-27 20:09 — dropped

Compromised: _load_nwb_arrays still has the Flavell neuron-order bug (i261026-1), so the gt arm crops one neuron and labels it with another's ids, and the all arm's GT relabelling indexes columns in arrays['neurons'] order. Rerun after fixing the loader.

Stopped PID 625851.

dropped r261027-625851

## 2026-10-27 20:12 — run

Rerun of the --nwb_objects check on barlow_track@79abd66 (NWB arrays selected by neuron name): legacy baseline_flavell_head_rerun, first 60 Flavell NWB frames, CPU, gt vs all. 79abd66 fixes _load_nwb_arrays (i261026-1): with it every Flavell GT position is the centroid of the mask label equal to its raw_segmentation_id (= raw_neuron_ind_in_list), so my 20:07 entry's 'NWB raw_segmentation_id is not that label' was the bug talking.

run r261027-630225 · pids 630225 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/nwb_objects_check/run.log

## 2026-10-27 20:12 — done

79abd66; NWB-path Flavell records before it carry no nwb_arrays field and are suspect; rescore still to do

Done: fix eval_accuracy._load_nwb_arrays: select each NWB field by neuron name (i261026-1), then rescore NWB-path Flavell records

## 2026-10-27 20:12 — done

850e8f4 --nwb_objects all (default gt); check r261027-630225 running

Done: make the NWB path not hand the tracker only GT objects (user: the algorithm should not see only GT objects)

## 2026-10-27 20:12

The 20:09 tick of the in_bounds_mask to-do says 'barlow_track@<see log>': the commit is barlow_track@e225a55 (round_crop_bounds, default off).

## 2026-10-27 20:14 — run

Per-frame centring test from cached embeddings (barlow_track@bcb80f8, centered_check.sh): comparator paper_flavell_use_position auto (0.795 uncentred) and legacy baseline_flavell_head_rerun (0.956) re-tracked after subtracting each frame's descriptor mean, CPU, NWB GT. Motivation: every Flavell network's descriptors are dominated by a per-frame mean (mean_frac legacy 0.986, position 0.990-1.000); the position nets' residuals are smaller (mean pairwise cosine 0.998-1.000 vs legacy 0.972) and the per-frame mean drifts more relative to them (|m_t - m| / within-frame residual 0.07-0.10 vs legacy 0.056, every 40th frame). If centring lifts the comparator toward 0.95, mean drift costs the position nets.

run r261027-631777 · pids 631777 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/centered_check/run.log

## 2026-10-27 20:58 — run

r261027-630225: legacy baseline_flavell_head_rerun, first 60 Flavell NWB frames, CPU, barlow_track@79abd66. GT objects only 0.944 (259 misses, 235 mismatches of 8882); every raw_segmentation detection (--nwb_objects all) 0.932 (217 misses, 385 mismatches, same 8882 GT entries, so the label mapping keeps the GT intact). Distractors cost 1.2 points here, mostly as mismatches. The dropped run r261027-625851's gt arm finished after the drop and appended to the same results.jsonl (its process outlived --kill): 0.856 with the old _load_nwb_arrays on the same 60 frames, i.e. i261026-1 cost 8.8 points on the NWB path. results: ~/.cache/barlow_scratch/nwb_objects_check/results.jsonl.

## 2026-10-27 21:28 — run

r261027-572595 (reaug) crashed at epoch 31 in the validation correlation plot: a 1-object view reached the train-mode batch-norm projector ('Expected more than 1 value per channel'); no resnet50.pth saved. Fixed in barlow_track@33094f9 (_correlation_for_plot skips views with <2 objects, as the training step already does). Proxy over epochs 13-30 stayed flat: fused 0.39-0.47 (0.460 at ep 30), backbone 0.23-0.38 (0.357), against frozen norot's 0.463 / 0.453 at ep 30. Val loss spiked at ep 7, 18, 19, 23 (0.08-0.16).

## 2026-10-27 21:28 — run

Eval of the partial reaug model (last per-epoch checkpoint, 32 of 50 epochs; barlow_track@33094f9 eval_partial.sh): auto then backbone, Flavell project source, NWB GT, embed cuda, track cpu. Compares with the frozen comparator's 0.795 / 0.756 (50 epochs); near 0.95 would make frozen pairs the cause even at 32 epochs.

run r261027-665732 · pids 665732 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_reaug/eval_partial.log

## 2026-10-27 21:56 — run

r261027-631777 first half: comparator paper_flavell_use_position auto re-tracked after per-frame centring of its cached embeddings: 0.802 vs 0.795 uncentred (+0.7 points, NWB GT, 25 seeds). The per-frame mean drift is not what holds the position nets ~15 points below legacy. Legacy centred control still running.

## 2026-10-27 22:55 — run

r261027-589796: norot auto re-embedded with CPU (sliding-window) crops scores 0.782 on Flavell NWB GT (1600 frames), against 0.787 with the GPU padded crops. The edge-crop difference of i261027-2 is worth about half a point at most, in the other direction, so it does not explain the 15-point gap to legacy. GPU eval numbers stand.

## 2026-10-27 22:55 — decision

i261027-2 judged from r261027-589796: edge-crop padding moved one network's Flavell accuracy by 0.5 pt (0.787 GPU pad, 0.782 CPU slide). None of the 15 notes rests on a difference that small, so none needs a rerun on Flavell. Not resolved yet: only Flavell and one network were measured, and Zimmer and Leifer have more neurons near the volume edge in z; c4a4221's --crop_window slide lets a later eval check.

## 2026-10-27 23:55 — run

r261027-631777 second half: legacy baseline_flavell_head_rerun re-tracked after per-frame centring: 0.958 vs 0.956 uncentred (NWB GT, 1600 frames). Centring moves neither network (comparator +0.7, legacy +0.2): the per-frame mean is not part of the gap.

## 2026-10-27 23:55 — run

r261027-665732 auto: the re-augmenting use_position cell at epoch 32 of 50 (crashed run's last checkpoint) scores 0.652 fused on Flavell (NWB GT), against the frozen comparator's 0.795 at 50 epochs, same seed and config otherwise. Its proxy at epoch 30 (0.460) matched the frozen norot run's (0.463), so the proxy missed a 14-point drop. Re-drawing views each epoch does not close the gap; like per-crop augmentation (0.657) and global elastic (0.718), more augmentation diversity makes the position pipeline worse, where the legacy pipeline re-draws every epoch and scores 0.95. Caveat: 32 epochs, one seed; backbone stage still tracking.

## 2026-10-27 23:55 — decision

i261027-1 judged from r261027-665732: frozen view pairs are not what holds the position networks below legacy (re-augmenting scored 0.652, frozen 0.795), so the notes resting on frozen-pair runs keep their direction; their numbers describe frozen training, which is what the cluster rounds ran. Not resolved: whether reaug hurts at 50 epochs and other seeds is unmeasured.

## 2026-10-27 23:56 — run

Two runs from one script (barlow_track@61894ab round_centers_check.sh): (1) comparator paper_flavell_use_position auto re-embedded with --round_crop_centers (training crops at rounded centroids, the eval truncates; legacy truncates in both), tracked on CPU, vs 0.795 truncated. (2) Once the reaug backbone eval frees the GPU, the legacy-matched bisection (barlow_track@b3f816b): position data path with image features only, preprint per-crop transforms, no global volume affine, no crop-centre jitter or dropout, re-drawn views; backbone eval vs NWB GT. Near 0.95 puts the gap in the position pipeline's geometric augmentation (global affine, jitter); near 0.80 in the rest of its data path. About 23 h.

run r261027-808962 · pids 808962 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/round_centers_check/run.log, /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_legacy_matched/run.log · for [[QUE - Which part of the position pipeline costs Flavell accuracy, its augmentation or its position fusion]]

## 2026-10-28 01:53 — run

r261027-665732 backbone: re-augmenting use_position cell at epoch 32 scores 0.481 backbone (frozen comparator 0.756). Both stages fall with re-augmentation (fused -14, backbone -28 points).

## 2026-10-28 01:53 — run

Legacy-matched training started by r261027-808962 at 01:04 (bash run.sh pid 838680; trains then evals backbone). Epoch 0 proxy 0.482 (auto = backbone here), above every earlier position run's epoch 0 (0.39-0.43); legacy final checkpoint scores 0.600.

run r261028-838680 · pids 838680 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_legacy_matched/run.log · for [[QUE - Which part of the position pipeline costs Flavell accuracy, its augmentation or its position fusion]]

## 2026-10-28 01:54 — surprise

r261028-838680 epoch 0: val loss 0.0043, the legacy pipeline's starting level (0.0044) and a tenth of every position run's (0.036-0.045), so the position runs' 'hard' views come from what this run removes (global volume affine, crop-centre jitter, dropout), not from the crop path itself. 7 s/step: about 25 h for 50 epochs.

## 2026-10-28 02:52 — run

r261027-808962: comparator paper_flavell_use_position auto embedded at rounded crop centres (as it trained) scores 0.814 on Flavell NWB GT, against 0.795 at the truncated centres the project-path eval uses by default (+1.9 points, same checkpoint, 25 seeds). A real eval-side mismatch for position networks, but a small part of the ~15-point gap. Legacy networks train and eval at truncated centres, so the default suits them; a legacy control with --round_crop_centers is not run (it needs the GPU the legacy-matched training holds).

- [ ] Rescore position-pipeline Flavell cells with --round_crop_centers, and run the legacy control, before quoting position-vs-legacy gaps

## 2026-10-28 02:52 — run

r261028-838680 epochs 0-2: val loss 0.0043, 0.0030, 0.0022 (falling, as legacy's does; every position run's rose after a few epochs), proxy 0.482, 0.496, 0.504.

## 2026-10-28 02:54 — run

Early eval of r261028-838680 (barlow_track@2683e5c eval_partial.sh 12): waits for the epoch-12 checkpoint (~07:30), then scores its backbone at truncated and rounded crop centres on Flavell NWB GT. Against the image-only position run's 0.819 (truncated) and legacy's 0.956.

run r261028-3899580 · pids 3899580 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_legacy_matched/eval_partial.log

## 2026-10-28 05:48 — decision

For cluster runs (user offered to launch today): new flags, default off, barlow_track@8ae7898: crop_augment legacy (preprint per-crop view transforms in the position pipeline), global_augment.apply_to_volume false (affine, elastic, jitter, dropout move only the keypoints; crops from the untransformed volume; answers the user's point that per-crop transforms never move positions, which is why global augmentation was introduced), fusion visual_only (replaces the scratch monkeypatch). d02573b lets sweeps name nested keys (global_augment.apply_to_volume). Staged round_flavell_legacy_crop_aug (barlow_track/scripts/scratch/round_flavell_legacy_crop_aug): fusion {visual_only, add} x apply_to_volume {false, true}, 3 seeds, 12 jobs, objective accuracy on flavell; README has launch and NWB-GT rescore (truncated and rounded centres).

## 2026-10-28 07:43 — surprise

Cluster round_flavell_paper_ablation status (read from /lisc): position cells 4-7, 12-15 and legacy 0, 1, 3 finished. Legacy trials 2, 8, 9 hit the 41 h walltime at epochs 38, 40, 48 (legacy crop transforms take 37-41 h for 50 epochs there) and saved no model; at the timeout signal submitit's handler crashed on our string slurm_time (TypeError). Legacy 10, 11 were cancelled at the same minute (SIGTERM at epochs 23, 25, the fail-fast pattern). So the legacy training-seed spread on the cluster still has only the seed-43 cells 0, 1, 3. Fixed in barlow_track@5ca086f (integer --time, train_hours template key); round_flavell_legacy_crop_aug sets train_hours 60.

- [ ] Rerun round_flavell_paper_ablation legacy cells 2 and 8-11 with train_hours 60 for the legacy seed spread

## 2026-10-28 09:45 — run

r261028-3899580: legacy-matched run at epoch 12 scores 0.938 (truncated) / 0.950 (rounded centres) on Flavell NWB GT, against 0.819 for the image-only position run and 0.956 legacy. The position data path recovers preprint accuracy once its views are made the legacy way. Recorded in [[EVD - Legacy-style crop views in the position data path recover preprint Flavell accuracy]] (open until the 50-epoch eval); [[CLM - The position pipeline's augmentation, not position fusion, holds the revision networks below the preprint]] revised.

## 2026-10-28 10:41 — dropped

Stopped at epoch 18 of 50 by decision: the epoch-12 result (0.950 rounded) already answers the bisection; the cluster round carries the replicates. checkpoints/checkpoint.pth (epoch 18) kept; training does not resume from it.

Stopped PID 838680.

dropped r261028-838680

## 2026-10-28 10:48 — decision

Neither pipeline has a real mirror flip. Legacy view 2's tio.RandomFlip(axes=(1,2)) is commented out; p_RandomAffine_flip is a fixed 180° in-plane rotation ('like a flip'), and the position pipeline's p_flip copies that (adds 180° to the angle, redundant with the ±180° range). The preprint trained without a mirror flip. Pointed out by the user, who expected a real flip.

- [ ] Rename p_flip / p_RandomAffine_flip to say rotation, and add a real mirror-flip option (default off)

## 2026-10-28 10:58 — run

Position-branch matching (a3bab89): Hungarian matching on per-frame-standardized position descriptors alone, augmented views vs real frames at lags 1-1000, 4 fused Flavell checkpoints plus re-initialized branches. Smoke test on the comparator: ~10% between views, ~3% at lag 1, effective rank ~2.4 (reinit 2.3) — the position branch encodes coarse location, not a per-neuron fingerprint.

run r261028-2940337 · pids 2940337 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/position_branch_matching/run.log · for [[EVD - Position change between augmented training views against real frame-to-frame change on Flavell]]

## 2026-10-28 11:10 — decision

Cross-dataset check of the user's model (2026-10-28), from the cluster exp_results.jsonl of every round, latest record per tag. (1) Global augmentation weakens image features only on Flavell: consistent but not checked directly. Best fused ViSNet position networks reach the preprint on Zimmer (0.970, round_visnet_full trial27, vs 0.973) and Leifer (0.972 vs 0.974); the only image-only position-pipeline records are round_2x2 (keypoint-MLP era, trained on Zimmer: Zimmer 0.79-0.89, Leifer 0.92-0.95, Samuel 0.89-0.92). On Flavell the tuned round_flavell_more_epochs trial27 reaches 0.915 backbone / 0.927 fused against 0.952, so the local preprint-config runs (0.79-0.82) are the extreme case. (2) Position helps on Samuel: not shown matched. Best Samuel: ViSNet fused 0.950 (round_bayes_samuel, trained on Samuel), keypoint-MLP 0.942 (round_bayes_zimmer trial27, trained on Zimmer), image-only 0.921 (round_2x2). Within round_2x2 (same era and training data) position on 0.894-0.925 vs off 0.888-0.921. Flavell rows before the NWB fix are capped by i261026-1 and were ignored.

- [x] Rescore the best Zimmer, Leifer and Samuel position networks at descriptor_stage backbone (cluster) to check whether global augmentation costs image features off Flavell
- [ ] Find why the ViSNet position branch learns about 2 effective dimensions (cutoff 0.35 vs a narrow worm, lmax 1, add fusion scale)

## 2026-10-28 11:16 — issue

Hungarian matching broke ties in index order, so near-constant descriptors scored as matches whenever both sets share an order (self-matching, augmented views). Fixed in a2ae044 (shuffle one side). Rerunning the four-checkpoint measurement. Also: randomly initialized ViSNet gives near-constant outputs (rank 1.0-1.2 at cutoff 0.07 and 0.35), so an untrained probe cannot rank encoder settings; that needs training.

- [x] issue i261028-1 · affects [[EVD - Trained position branches carry no more neuron identity than an untrained one on Flavell]]

## 2026-10-28 11:16 — run

Rerun of position-branch matching with shuffled assignment (a2ae044)

run r261028-2951846 · pids 2951846 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/position_branch_matching/run_shuffled.log · for [[EVD - Trained position branches carry no more neuron identity than an untrained one on Flavell]]

## 2026-10-28 11:17 — run

Position-only pilot on Flavell (c431d7c, scratch/flavell_position_only_pilot): the comparator config with fusion position_only, keypoint-only global augmentation, 10 epochs, val_match_frames 30; then position-branch matching and a tracking eval. Checks that a position-only network can learn neuron identity before a cluster sweep over the encoder settings.

run r261028-2952895 · pids 2952895 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/flavell_position_only_pilot/run.log · for [[QUE - Do the two training views differ in neuron positions as much as real frames do]]

## 2026-10-28 11:28

i261028-1: [[EVD - Trained position branches carry no more neuron identity than an untrained one on Flavell]] rerun with shuffled assignment (r261028-2951846, a2ae044): every number identical to 3 decimals; the trained descriptors were not degenerate enough for ties to matter. The note stands.

## 2026-10-28 11:28 — resolved

Rerun with the fixed metric reproduced every number; no note changes.

resolved i261028-1

## 2026-10-28 11:39 — issue

ViSNet position encoder is blind to neighbour distances in every run so far. pyg's ExpNormalSmearing places its RBF means on [exp(-cutoff), 1] but evaluates exp(-(5/cutoff) d); these agree only at cutoff 5 (its Angstrom default). At our cutoff 0.35 every distance above ~0.05 (0.14 x cutoff; Flavell NN distance ~0.034 normalized, Zimmer ~0.015) gets RBF features of ~0, so the distance channel is nearly empty and an untrained encoder gives identical outputs for every node (synthetic test: spread 3e-8). Real Flavell frame, untrained: effective rank 4.0 default vs 14.7 with the fix. The encoder kept only direction information (spherical harmonics through the vector channel), which fits the ~2 effective dimensions of trained branches. Opt-in fix visnet_rbf_rescale (barlow_track@480b76c, default off) builds ViSNet at cutoff 5 and rescales keypoints; also visnet_cutoff_nn (radius in median NN distances, scale-invariant).

- [ ] issue i261028-2 · affects [[EVD - ViSNet position encoder beats a keypoint MLP on every dataset]], [[EVD - Attention and fusion sweep with the ViSNet encoder]], [[EVD - Trained position branches carry no more neuron identity than an untrained one on Flavell]]

## 2026-10-28 11:43 — run

Local smoke test of one round_posonly_flavell cell (rbf_rescale, cutoff_nn 12, elastic on): 1 epoch on 10 frames, then the tracking eval on 50 frames; proves the new flags train and score before the cluster rounds.

run r261028-2979125 · pids 2979125 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_smoke/run.log

## 2026-10-28 11:57

r261028-2979125 (smoke, 7641f83): the round_posonly cell with rbf_rescale, cutoff_nn 12 and elastic on trains, logs the proxy and scores end to end. After 1 epoch on 8 training frames: frame-pair proxy 0.318 at the position stage (6 far-apart pairs) and tracking accuracy 0.527 on the first 50 frames, against 0.12 for an untrained position-only network (full video) and 2-3% position-only matching for the fused networks' branches (a different metric). Tiny and not comparable as a result; it only says the fixed branch learns something at once.

## 2026-10-28 11:57 — dropped

Crashed at epoch 0 writing log/stats.json: run.sh did not create log/ (sweep trials create it). Superseded by the smoke test r261028-2979125 and round_posonly.

dropped r261028-2952895

## 2026-10-28 12:06 — run

User launched round_posonly_{zimmer,flavell,samuel} on the cluster (2026-10-28, behind the round_flavell_legacy_crop_aug jobs in the queue), for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]. The README's launch loop ran the dispatchers one after another (each stays in the foreground until its sweep ends); fixed in 629b0a5 to background each.

## 2026-10-28 12:48 — decision

round_flavell_legacy_crop_aug early read (2026-10-28 ~12:15, epochs 3-6 of 50, jobs 7747886-95 = trials 0-9; trials 10-11, add x seed 3, not yet submitted). Frame-pair proxy (30 held-out frames): apply_to_volume false 0.52-0.56 and rising (trials 0, 2, 4, 6), true 0.38-0.43 and flat (1, 3, 5, 7); seed 3 (8, 9) low in both (0.30-0.38), still climbing. Val loss false 0.003 (legacy-like), true 0.02-0.05 (as every earlier position run). add vs visual_only: auto equals backbone to 3 decimals with keypoints only, a few points above with the volume transformed; consistent with the distance-blind ViSNet (i261028-2) adding nothing. The add cells now test a known-broken encoder; the visual_only cells stay the image-side test. Proxy is noisy (missed a 14-point drop before); wait for the eval.

## 2026-10-28 16:09 — decision

round_flavell_legacy_crop_aug: user cancelled the four add cells (jobs 7747888/89/92/93 = trials 2, 3, 6, 7; SIGTERM 16:07) to free GPUs for round_posonly; trials 10-11 (add, seed 3) will not run. The add cells used the distance-blind ViSNet (i261028-2) and are rerun with the fix after round_posonly. Image-only trials 0, 1, 4, 5, 8, 9 still training (18-30 h left at 29-48 min/epoch), done about Friday 30 Oct morning with eval.

## 2026-10-28 16:09 — run

round_posonly_zimmer early read (16:10): position-only trains fast, 50 epochs in 0.5-0.7 h. First three cells, all rbf_rescale false: trial 0 (cutoff_nn 0, elastic off) proxy 0.30 at epoch 49, effective rank 32; trial 1 (cutoff_nn 0, elastic on) 0.23, rank 36; trial 2 (cutoff_nn 3, elastic off) 0.09, rank 2-3, stuck from epoch 0 (the expected blind control, and the same ~2 dims as the fused Flavell position branches). So the unfixed encoder at absolute radius 0.35 still learns something on Zimmer. 13 Zimmer and 10 Flavell jobs submitted; round_posonly_samuel has no trials or submissions yet. Accuracy evals still to come.

## 2026-10-28 16:28 — decision

round_posonly_samuel did not launch: the accuracy objective's preflight found no GT in samuel_data/projects/2025_07_30_gt_traces (samuel GT is only in 153.nwb; earlier samuel rounds used objective: loss). Switched the template to objective: loss; trials are scored afterwards with run_trials_on_all_ground_truth.py --labs samuel (NWB), as every earlier samuel number was.

## 2026-10-28 16:31 — decision

Correction to the 16:28 decision: the samuel project does have GT. Its tracks live only in its NWB (nwb/nwb_config.yaml: nwb_filename nwb/153.nwb, read by hybrid loading), and nwb/153.nwb is missing from 2025_07_30_gt_traces, so final_tracks loads as None. samuel_data/153.nwb is the same recording: 98 neurons x 1331 frames, and at frame 100 all 87 tracked centroids fall on project masks. Fix is to link it in (project/nwb/153.nwb -> ../../../153.nwb); template reverted to objective: accuracy. Earlier samuel rounds were unaffected (objective: loss for training, NWB for scoring).

## 2026-10-28 16:34 — dead-end

Why the Flavell project GT differs from the NWB GT (the ~3.4-4 pt gap): in flavell_data/projects/2025_07_01, 3-tracking/tracking_config.yaml (modified 2025-10-07) sets final_3d_tracks_df to 3-tracking/barlow_tracker/df_barlow_tracks.h5, a Barlow tracker output from 2025-07-18 (next to its embedding.zarr and worm_tracker_barlow_full.pickle). Hybrid loading fills final_tracks from the NWB only when it is None, so _load_project_gt scores against those tracker tracks, filtered to the finished neurons, not against the NWB GT. Fits the 89.7% seg-id agreement. Not checked whether df_barlow_tracks.h5 was manually corrected. Flavell project's NWB is copied in (nwb/flavell_data.nwb, relative path in nwb_config); samuel's config has the same relative form but the copy is missing.

## 2026-10-28 16:43 — decision

Samuel projects: switched round_posonly_samuel to samuel_data/projects/153 (user's call). It passes the sweep preflight: 98 neurons x 1331 frames, metadata for all frames, nwb/153.nwb byte-identical to samuel_data/153.nwb, project GT identical to the NWB GT. 2025_07_30_gt_traces lacks its NWB copy (no GT); projects/137 (= 137.nwb, 92 neurons x 1326 frames) has GT but no 1-segmentation/metadata.pickle, so it cannot train as is. Earlier rounds (round_bayes_samuel, initial_position_embedding_samuel) trained on 2025_07_30_gt_traces: same recording, its own segmentation metadata (14.30 MB vs 14.37 MB in projects/153), not checked for equality.

## 2026-10-28 16:46 — issue

Flavell project 2025_07_01's 'project GT' is a Barlow tracker output, not GT. tracking_config.yaml (modified 2025-10-07) sets final_3d_tracks_df to 3-tracking/barlow_tracker/df_barlow_tracks.h5 (written 2025-07-18 next to that run's embedding.zarr); manual_annotation.xlsx marks no neuron finished, so _load_project_gt returns that whole table (160 neurons). By position (barlow_track@c087fa5:barlow_track/scripts/scratch/flavell_gt_identity.py) it is the same recording as images_for_charlie/flavell_data.nwb (151 of 160 neurons pair at median 0.12 voxels) but agrees on only 91.9% of paired entries (89.1% of all), i.e. a ~90%-accurate tracking. Every --source project / in-job objective: accuracy Flavell score is accuracy against that tracker, not GT; the preprint's 96.7% is against the NWB GT. The four forCharlie_2 projects and 2025_08_01 have final_3d_tracks_df null (GT from their NWBs).

- [ ] issue i261028-3 · affects [[@BarlowTrack revision benchmark sweeps]], [[@BarlowTrack Flavell ground-truth version comparison]]

## 2026-10-28 16:47

i261028-3 follow-up (flavell_gt_identity.py @c087fa5): the 2025_07_01 project's own NWB copy (nwb/flavell_data.nwb) holds the real GT. All 163 neurons pair with images_for_charlie/flavell_data.nwb at median 0.000 voxels, and every entry the reference has sits at the identical position (97.3% of the copy's 237,777 entries; the other 6,509 are entries the reference lacks; 1601 vs 1600 frames). Its raw_segmentation_id values are in a different id space (0.6% equal), which fits ids remapped to the project's segmentation. So setting final_3d_tracks_df: null in 2025_07_01 would make hybrid loading, and the in-job objective, score against GT. Against that copy, the tracker file agrees on 93.1% of paired entries.

## 2026-10-28 16:50

i261028-3 check against NaN filtering (user's question; flavell_gt_identity.py @c23e47d, tracks file vs the project's NWB copy): filtering only removes entries, but of the tracks file's 209,490 entries on paired neurons, 9,448 sit >= 2 voxels from their GT neuron while that neuron is present (median 8.2 voxels, about one Flavell NN distance; p90 79), 40% of them on another GT neuron; 5,016 more have data in a frame where the GT neuron has none; and 9 neurons (6,748 entries) pair with no GT neuron. ~21k of 216k entries cannot come from dropping data. Not ruled out: an older GT version that the Flavell lab later revised; the file's location (3-tracking/barlow_tracker/, next to embedding.zarr and worm_tracker_barlow_full.pickle) points to a tracker run.

## 2026-10-28 17:06

i261028-3, which GT the paper loaded (user asked). Paper notebooks (barlow_track/notebooks/paper): fig_ground_truth (Fig. 3 Flavell accuracy), fig_ground_truth_accuracy_vs_val_loss and fig_ground_truth_plot_training_data_sweep all use gt = images_for_charlie/flavell_data.nwb; the 2025_07_01 project path is there only commented out. So the preprint's Flavell numbers are against real GT. Exception: fig_ground_truth_generalization (Fig. 3h) takes each analyzed project's remote project as GT; for zimmer/ and leifer/baseline_generalization_flavell_2025_07_01 (made 2025-10-10, after the 2025-10-07 config change) that is projects/2025_07_01, i.e. the tracker file, so the Zimmer->Flavell and Leifer->Flavell cells on 2025_07_01 were likely scored against tracker output. The 4 forCharlie_2 projects resolve to their own NWB GT. Training in every sweep (paper TrainedBarlow and revision rounds) uses 2025_07_01, which reads no tracks. CORRECTION to the 16:47 entry (flavell_gt_identity.py @97d324b): the tracks file's raw_segmentation_id equals images_for_charlie's on 100% of agreeing entries, but the project's own NWB copy's on 0.6%. So images_for_charlie is in the project's mask-id space and the copy is not: setting final_3d_tracks_df: null would load GT with the wrong ids. The fix is to make the project's GT the images_for_charlie NWB (nwb_config or replace the copy), not to null the tracks.

## 2026-10-28 17:11 — run

Local GPU: backbone vs auto rescore of the best existing fused ViSNet networks off Flavell (barlow_track@c81c641:barlow_track/scripts/scratch/backbone_rescore_off_flavell/run.sh): round_visnet_full trial27 on zimmer_1128, samuel, leifer and round_bayes_samuel trial0 on samuel, both stages with the same code and GT, local jsonl/emb cache only. Answers the 2026-10-28 to-do (does global augmentation cost image features off Flavell, and does the blind position branch add anything there). Leifer cells last (hours).

run r261028-3159902 · pids 3159902 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/backbone_rescore_off_flavell/run.log

## 2026-10-28 17:11 — done

running locally instead (see the run entry of 2026-10-28 evening)

Done: Rescore the best Zimmer, Leifer and Samuel position networks at descriptor_stage backbone (cluster) to check whether global augmentation costs image features off Flavell

## 2026-10-28 17:11 — todo

Flavell GT follow-ups (i261028-3), left open by the user's choice; the user wants paper figures off notebooks anyway.

- [ ] Rescore the Fig. 3h Zimmer->Flavell and Leifer->Flavell cells on 2025_07_01 against images_for_charlie/flavell_data.nwb
- [ ] Make 2025_07_01's GT the images_for_charlie NWB (its own nwb/ copy has different segmentation ids), so --source project and the in-job objective score against GT
- [ ] Find out why the 2025_07_01 nwb/ copy's raw_segmentation_id differs from images_for_charlie's
- [ ] Move the paper's accuracy figures from notebooks into committed scripts

## 2026-10-28 17:36 — issue

Code review (subagent, read-only, 2026-10-28) findings that bear on recorded numbers: (a) record tags are only <round>_trial<N>_<label>, so resume/skip treats any record as done: a --max_frames smoke run poisons the later full run, a benchmark run skips flavell/samuel cells already written by the in-sweep objective (source project, project GT) under the same tag as the NWB-source benchmark cell (all 81 flavell records in round_flavell, round_flavell_more_epochs, round_flavell_paper_ablation are source project), and a --rerun_completed with another descriptor_stage deletes the other stage's record. (b) The benchmark's NWB path defaults to --nwb_objects gt: only GT neurons are candidates (oracle detections, no distractors); for position networks the ViSNet neighbourhoods at eval are also not the trained ones (flavell ~155 GT neurons vs ~188 detections).

- [ ] issue i261028-4 · affects [[@BarlowTrack revision benchmark sweeps]]

## 2026-10-28 17:36 — issue

Code review findings on the evaluation: (a) GT-to-prediction column matching (wbfm calc_bipartite_from_ids via rename_columns_using_matching, eval_accuracy.py:801) counts 'GT NaN, prediction present' as a mismatch, so it prefers sparse predicted columns; affects --source project cells where the tracker labels every detection (zimmer 1128: 16 of 131 GT neurons under 90% coverage) and --nwb_objects all; synthetic case 0.50 instead of 1.00, real effect unmeasured. (b) Local runs used the compiled torch_cluster radius_graph (arbitrary subset when capped at max_num_neighbors) while the cluster uses the pure-torch nearest-k fallback: different ViSNet graphs for cluster-trained checkpoints rescored locally (synthetic flavell-like box: 3,784 of 5,758 edges differ). Fixed for new runs in barlow_track@86eedc1 (fallback everywhere, BARLOW_COMPILED_RADIUS_GRAPH=1 to opt out); local ViSNet results before that used the compiled graph.

- [ ] issue i261028-5 · affects [[@BarlowTrack revision accuracy evaluation]]

## 2026-10-28 17:36 — todo

Other code-review findings, not yet affecting recorded results: production tracking (track_using_barlow embed_using_barlow) runs model.backbone only, so fused/ViSNet checkpoints track image-only through wbfm; track_using_barlow overwrites a project's final_3d_tracks_df with tracker output (root cause of i261028-3); production dask SVD whitens (u[:, :50]) while eval uses TruncatedSVD U*S; pretrained_model_path copies config args over the checkpoint's (visnet_* not compared); load_barlow_model strict=False leaves fuse_mlp/attn_gate at init for old checkpoints; embedding cache key lacks data path and code version. Unread config keys in the current rounds: training_fraction (code reads train_fraction), p_RandomNoise_base; batch_size is fixed to 1.

- [x] Production tracking: embed fused/ViSNet checkpoints with the eval's stage selection, not backbone only
- [x] Stop track_using_barlow overwriting final_3d_tracks_df on GT projects
- [x] Put source, stage, max_frames and crop flags into record tags (or check them before skipping)
- [ ] GT-valid-frames-only column matching in the eval; measure its effect on zimmer project cells
- [ ] Consider --nwb_objects all for flavell/samuel benchmark cells of position networks

## 2026-10-28 17:36 — run

Fast local position-only loop (barlow_track@86eedc1, 9943f40): keypoints_only (default off) skips volumes and crops in VolumeCoordsDataset and the frame-pair proxy; get_centroids_for_volume reads segmentation_metadata[t] directly (wbfm's accessor re-scans all frames per call, 2.3 s/frame on Zimmer; identical output on Zimmer/Flavell/Samuel, scratch/check_fast_centroids.py). scratch/posonly_fast/posonly_fast.py runs one cell. Smoke (3 epochs, rbf_rescale on, Zimmer updated-format project): 1 min 47 s, proxy 0.22/0.25/0.27 at epochs 0-2 vs 0.07/0.08/0.09 for the unfixed cluster cell round_posonly_zimmer trial 0. Cluster round_posonly_zimmer so far (unfixed encoder): trial 0 (radius 0.35, elastic off) 0.583, trial 1 (elastic on) 0.449, trial 2 (3 NN) 0.130 tracking accuracy; rbf_rescale cells still queued.

## 2026-10-28 17:36

r261028-3159902 first cell: round_bayes_samuel trial 0 on samuel (153.nwb, nwb_objects gt) scores 0.9508 at descriptor_stage backbone vs 0.9503 recorded fused: its (distance-blind) position branch adds nothing on Samuel. User recalls 75-80% on Samuel in the preprint era; a subagent is checking recordings and detection sets (oracle GT-only candidates vs all detections).

## 2026-10-28 17:36 — run

Local fast batch 1 (barlow_track@eee490f): posonly_fast batch1.sh, Zimmer position-only, 15 epochs keypoints_only: unfixed encoder reference + rbf_rescale x cutoff_nn {0,3,6,12} x elastic {0,1}; proxy per epoch. Direction finding ahead of round_posonly on the cluster.

run r261028-3190461 · pids 3190461 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch1.log

## 2026-10-28 17:49 — surprise

posonly_fast batch1 (r261028-3190461): the frame-pair proxy failed in every cell so far: the keypoints_only proxy also requested the 'backbone' stage, which cannot run on placeholder crops, and training's except swallowed it. Fixed in f4be66f (keypoints_only scores 'auto' only; val_matching refuses 'backbone' loudly). Cells from b1_nn3_el1.0 onward started before or after the fix; the first four (rbf_off, nn0 x2, nn3_el0) have val loss only and need rerunning for the proxy.

- [x] Rerun posonly_fast batch1 cells b1_rbf_off, b1_nn0_el0.0, b1_nn0_el1.0, b1_nn3_el0.0 (and nn3_el1.0 if it started before f4be66f) for the proxy

## 2026-10-28 17:49 — decision

Review finding #1 fixed in 3cb571a: production tracking (track_using_barlow) embeds position checkpoints as eval --mode trained does when the tracking config sets descriptor_stage (unset keeps the legacy backbone path, with a warning). final_3d_tracks_df is no longer lost: the replaced path goes to final_3d_tracks_df_history, and protect_final_3d_tracks_df: true leaves it unchanged. Defaults keep current behaviour.

## 2026-10-28 18:02 — run

eval --gt_matching gt_valid and --svd whiten vs the defaults on cached Samuel 153 backbone embeddings (round_bayes_samuel trial 0, round_visnet_full trial 27), CPU tracking; measures how much review findings #2 and #4 move a score (barlow_track@230c297:barlow_track/scripts/scratch/svd_gt_matching_check/run.sh)

run r261028-3231600 · pids 3231600 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/svd_gt_matching_check/run.log

## 2026-10-28 18:02 — result

r261028-3159902 partial: Samuel 153 (nwb, gt objects), backbone vs auto. round_bayes_samuel trial 0: backbone 0.9508, auto (add fusion) 0.9425 (cluster record 0.9503 at auto; local vs cluster differ by 0.8 pt, not chased). round_visnet_full trial 27 (trained on Zimmer, concat + 2 self layers): backbone 0.9489, auto (contextual) 0.8371, so the position and attention head costs 11 pt when transferred to Samuel while its image backbone transfers fine. Zimmer_1128 cells OOM'd in GPU label propagation next to batch1 training (both stages); rerun with --track_device cpu. Leifer still running.

## 2026-10-28 18:02 — done

3cb571a: tracking config descriptor_stage (opt-in; unset warns for position checkpoints)

Done: Production tracking: embed fused/ViSNet checkpoints with the eval's stage selection, not backbone only

## 2026-10-28 18:02 — done

3cb571a: replaced path kept in final_3d_tracks_df_history; protect_final_3d_tracks_df opt-in

Done: Stop track_using_barlow overwriting final_3d_tracks_df on GT projects

## 2026-10-28 18:02 — done

b98330d: resume/skip/dedup by tag and RECORD_SETTINGS; 482dcfb adds svd and gt_matching; crop_window/round_crop_centers are still only in the file name and record, not the settings

Done: Put source, stage, max_frames and crop flags into record tags (or check them before skipping)

## 2026-10-28 18:02 — decision

Review findings #7-#9 and unread config keys fixed: 4290d19 (visnet args compared as architecture; pre-gating attention checkpoints load with ~2% of their attention, now recorded as partial_load in eval records, BARLOW_STRICT_LOAD=1 refuses; embedding cache key gains the data path and a version, compared field by field so existing caches still hit) and 1ceded4 (training warns about training_fraction, p_RandomNoise_base, batch_size != 1). #2/#4 as opt-in eval options in 482dcfb (--svd whiten, --gt_matching gt_valid; production svd_whiten), measured by r261028-3231600. Left alone: #6 (--nwb_objects all exists, opt-in), #10 (legacy bbox-midpoint crop centres, num_frames+1).

## 2026-10-28 18:07 — result

r261028-3190461 (batch1) ended. Proxy (frame-pair matching, 30 held-out Zimmer frames, position-only ViSNet, RBF fix, 15 epochs keypoints_only): cutoff_nn 6 / elastic 0: 0.380; 6/1: 0.342; 12/0: 0.343; 12/1: 0.292. Still rising at epoch 14 in every cell (not converged). Elastic costs 4-5 pt on the proxy; nn6 > nn12. The rbf_off, nn0 and nn3 cells lost their proxy (f4be66f) and are rerun in batch1b.

## 2026-10-28 18:07 — run

posonly_fast batch1b: rerun batch1's proxy-less cells (rbf_off reference, cutoff_nn 0 and 3 x elastic 0/1) with the fixed proxy (barlow_track@cdd5761:barlow_track/scripts/scratch/posonly_fast/batch1b.sh)

run r261028-3236040 · pids 3236040 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch1b.log

## 2026-10-28 18:23 — result

r261028-3236040 (batch1b) partial, proxy after 15 epochs: rbf_off (unfixed encoder) 0.242; cutoff_nn 0 / elastic 0: 0.381; 0/1: 0.400 (best cell so far). With nn0, elastic does not cost proxy, unlike nn6 (0.380 -> 0.342) and nn12 (0.343 -> 0.292). The RBF fix is worth ~+14-16 pt on the proxy. nn3 cells running.

## 2026-10-28 18:32 — result

Samuel double-check (subagent; barlow_track@32fb862:barlow_track/scripts/scratch/samuel_check/). On 153 with NWB GT objects and labelprop-25: round_bayes_samuel trial 0 0.951, paper-era harvard_153_flip trial_8 0.949. The same paper-era sweeps' stored tracks (old WormTsneTracker, HDBSCAN windows, Aug 2025, before label propagation entered in 82f4425) score best 0.805 / median 0.647 (flip, 28 trials), 0.777 (embed), 0.428 (plain); ~90% of their errors are misses (HDBSCAN noise and small clusters unassigned, ~88 tracks for 98 neurons). So the paper-era 75-80% on Samuel was the tracker, not the network or an easy recording. Per the subagent, Samuel's NWB masks and projects/153 segmentation contain only GT neurons, so every Samuel score (nwb gt, nwb all, project) is an oracle-detection score, not comparable to Zimmer's real-segmentation benchmark. Unanswered: whether 137 and 184-004 are harder (queue still running: global clustering, nwb all, project source, 137, image baseline, 184-004; outputs ~/.cache/barlow_scratch/runs/samuel_check/).

- [x] Read the rest of the samuel_check queue (global clustering, nwb all, project, 137, image baseline, 184-004) and judge whether 153 is easier

## 2026-10-28 18:36 — result

Samuel check, tracker swap: round_bayes_samuel trial 0's cached 153 embeddings (NWB GT objects) tracked with --cluster global score 0.705 vs 0.951 with labelprop-25. Network and detections fixed, so the paper-era 70-80% on Samuel is the tracker.

## 2026-10-28 18:40 — result

Samuel check: the paper-era harvard_153_flip trial_8, re-tracked from cached 153 embeddings with --cluster global, scores 0.804, reproducing its stored paper-era tracks (0.805). With labelprop-25 it scores 0.949. round_bayes_samuel trial 0: 0.951 labelprop, 0.705 global. Today's eval reproduces the paper-era Samuel number once the old tracker is used; label propagation lifts both networks to ~0.95 (oracle GT-only detections throughout).

## 2026-10-28 19:47 — decision

Samuel results so far (labelprop-25, oracle detections, round_bayes_samuel trial 0): 153 0.951, 137 0.848; image baseline on 153 0.821. --nwb_objects all on 153 0.898 is not comparable: 76% of GT positions sit off a mask centroid and are left out of the GT. --source project on 153 crashed (NaN centroid in _centroid_boxes). Checked with h5py: every Samuel NWB segmentation, plain and thresholded, holds only GT neurons (labels per frame <= GT count); thresholded_segmentation/185-005.nwb will not open (bad object header version). User: no avoiding GT-only segmentation on these data.

## 2026-10-28 19:47 — run

Samuel follow-up queue: paper-era trial 8 on 137 and 184-004, image baseline on 184-004, after the first queue ends (d2c59c3), to see whether the recording-to-recording spread is the network or the recording

run r261028-3286550 · pids 3286550 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/samuel_check/queue_more.log

## 2026-10-28 19:47 — run

r261028-3231600 partial (Samuel 153, backbone embeddings): gt_valid matching equals nandist on both networks (0.9508, 0.9484), as expected with GT-only segmentation. Whitened SVD (production's path) costs bayes0 1.4 pt (0.9508 -> 0.9365); trial 27 whiten cell still running.

## 2026-10-28 20:37 — run

r261028-3231600 done: whitened SVD (production's path) costs both networks on Samuel 153: bayes0 0.9508 -> 0.9365, trial 27 0.9484 -> 0.9311 (1.4, 1.7 pt). gt_valid matching = nandist on both. Production tracking should use svd_whiten: false (opt-in key exists since 482dcfb).

## 2026-10-28 20:37 — run

Samuel per recording (labelprop-25, oracle detections): bayes0 153 0.951 / 137 0.848 / 184-004 0.913; paper-era trial 8 153 0.949 / 137 0.807 (184-004 running); image baseline 153 0.821 / 137 0.641. 137 is the hard recording for every embedding; bayes0 beats paper-era there by 4 pt and image by 21 pt. r261028-3286550 partial.

## 2026-10-28 20:40 — run

r261028-3236040 done (batch 1b, Zimmer position-only, 15 ep, final proxy): rbf_off 0.242 and nn0 0.381/0.400 reproduce batch 1; cutoff_nn 3 is worst, el0 0.200 / el1 0.203 (cutoff = 3 NN spacings leaves too few neighbours). Ranking: nn0 (fixed cutoff 0.35) >= nn6 > nn12 > nn3. Every cell still rising at epoch 14.

## 2026-10-28 20:40 — done

batch 1b rerun gave proxies for every cell; nn0 el1 best (0.400)

Done: Rerun posonly_fast batch1 cells b1_rbf_off, b1_nn0_el0.0, b1_nn0_el1.0, b1_nn3_el0.0 (and nn3_el1.0 if it started before f4be66f) for the proxy

## 2026-10-28 20:40 — run

r261028-3159902 Leifer done: trial 27 backbone 0.958, auto 0.972. Auto (fused) helps on Leifer by 1.4 pt but costs 11 pt on Samuel (0.949 -> 0.837); bayes0 on Samuel backbone 0.951 / auto 0.943. Zimmer cells OOM'd; CPU-tracking rerun staged.

## 2026-10-28 20:40 — run

Batch 2: best position-only cell (cutoff_nn 0, elastic 1.0, RBF fix on) at 60 epochs with full-video zimmer_1128 eval (--eval), queued behind the Samuel follow-up queue (r261028-3286550). Asks whether the proxy keeps rising past epoch 14 and what accuracy a 0.40-proxy network tracks at. Code 095e9e2.

run r261028-3310541 · pids 3310541 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/b2_nn0_el1.0_ep60.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-28 21:20 — decision

dd0e82e: get_bbox_data_for_volume_with_label skips objects with non-finite centroids (Samuel 153 project: 96 empty-mask objects in 93 frames, NaN centroids). int(NaN) raised, so this crashed eval --source project and production embedding (legacy and position paths) on that project; no recorded result can change. Test fails without the fix. Project-source Samuel cell to rerun once the GPU frees.

- [ ] Rerun sc_bayes0_153_project (Samuel 153 --source project) after dd0e82e

## 2026-10-28 21:20 — run

r261028-3286550 partial: paper-era trial 8 on 184-004 0.899 (bayes0 0.913).

## 2026-10-28 22:10 — run

Batch 3: six 15-epoch position-only screens on Zimmer, one change each from batch 1's best (proxy 0.400 at ep 14): reaugment_each_epoch, num_frames 300, visnet_hidden 128, visnet_layers 5, lr 3e-4, visnet_cutoff 0.25. Runs beside batch 2's CPU label propagation. Code bcca61b.

run r261028-3359025 · pids 3359025 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch3.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-28 22:10 — run

r261028-3310541 training done: proxy 0.401 at ep 14 (reproduces batch 1's 0.400), 0.45 at 25, 0.48 at 40, 0.50 from ep 50 to 59 (plateau). Full-video zimmer_1128 eval running. Every Zimmer project load warns metadata.pickle 'likely corrupted' and loads backup_metadata.pickle; this happens in every posonly run, not new.

## 2026-10-28 22:10 — run

Samuel per recording complete (labelprop-25, oracle detections). bayes0 / paper-era trial 8 / image: 153 0.951/0.949/0.821, 137 0.848/0.807/0.641, 184-004 0.913/0.899/0.801. r261028-3286550 done.

## 2026-10-28 22:12 — done

153 is the easiest of three; 137 hardest ([[EVD - Samuel recordings differ by up to 10 points in tracking accuracy under one network]])

Done: Read the rest of the samuel_check queue (global clustering, nwb all, project, 137, image baseline, 184-004) and judge whether 153 is easier

## 2026-10-28 23:09 — run

r261028-3359025 done (batch 3, Zimmer position-only, 15 ep, proxy at ep 14; baseline nn0 el1 0.400, 60-ep plateau 0.50): num_frames 300 0.521 (4500 steps, beats 6000 steps on 100 frames); visnet_hidden 128 0.451; lr 3e-4 0.436; reaugment_each_epoch 0.403 (no effect, so i261027-1's repeated view pairs are not what limits position-only); visnet_cutoff 0.25 0.310; visnet_layers 5 non-finite loss at epoch 0 step 0 (no checkpoint). Training data is the strongest lever.

## 2026-10-28 23:09 — run

Batch 4: num_frames 300 + visnet_hidden 128 (+ lr 3e-4), num_frames 600 and 1000, 15 epochs each, from batch 1's best. Code 668e517.

run r261028-3386058 · pids 3386058 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch4.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-28 23:15 — decision

Surprise: with visnet_rbf_rescale (the RBF fix, every posonly run since batch 1) ViSNet's per-layer outputs explode on real Zimmer frames: max |dx| 77 -> 2.8e4 -> 2.9e10 over 3 layers (seed 43; 80 -> 1.7e3 -> 4e8 seed 0), NaN at 5 layers (b3_layers5 crash, reproduced with seeds 43 and 44). Without the rescale they stay O(5) (the RBFs see nothing, issue i261028-2). Cause: unnormalized vector features in dense frames (~120 neighbours within 0.35, capped at 32). The output LayerNorm hides it, so batch 1-4 trained on 1e10-scale internals. pyg's vecnorm_type max_min holds every layer at O(80), also at 5 layers. Added as opt-in visnet_vecnorm (6a.. see git log), default off.

## 2026-10-28 23:15

visnet_vecnorm is commit dfa4a92 (the previous entry's commit reference was left unfilled).

## 2026-10-28 23:15 — run

r261028-3310541 done: position-only ViSNet (RBF fix, cutoff 0.35 abs, elastic 1.0, 100 frames, 60 epochs, keypoints only, no image) tracks zimmer_1128 full video at 0.843 (labelprop-25, auto stage; misses 11112, mismatches 21286 of 206701). Same encoder unfixed on the cluster (round_posonly_zimmer trial 1, elastic on, 50 ep): 0.449; trial 0 (elastic off) 0.583. Untrained image encoder on Zimmer 0.62; image-only position pipeline 0.79-0.89. Proxy 0.50 <-> accuracy 0.84 on this one point. Trained with exploding internals (no vecnorm).

## 2026-10-28 23:15 — run

Zimmer backbone vs auto rescore of round_visnet_full trial 27 with CPU tracking (the zimmer_1128 cells of r261028-3159902 OOM'd); runs beside batch 4's keypoint-only training. Code 89531d7.

run r261028-3398874 · pids 3398874 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/backbone_rescore_off_flavell/zimmer_cpu_track.log

## 2026-10-29 00:14 — run

r261028-3386058 partial, proxy at epoch 14 (batch 1 baseline 0.400, b3 frames300 0.521, b3 h128 0.451): frames300+h128 0.548, frames300+h128+lr3e-4 0.518, frames600 0.552 (27 min per cell). More frames is the main lever and stacks a little with width; lr 3e-4 no longer helps once frames are up. frames1000 still training.

## 2026-10-29 00:14 — run

Batch 5 (df9b045): ViSNet vector norm max_min on frames300+h128 (0.548 without), plus layers 5 and cutoff 0.5 with the norm. Runs beside batch 4's frames1000 cell; 56 cores, GPU 4.5/12 GB, so contention only affects timings.

run r261029-3424998 · pids 3424998 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch5.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 00:15 — run

Rerun of sc_bayes0_153_project (Samuel 153, --source project, bayes0, auto stage) now that dd0e82e skips NaN-centroid objects; it crashed before. Gives the project-segmentation number beside the NWB-GT 0.951.

- [x] read r261029 sc_bayes0_153_project result

run r261029-3425729 · pids 3425729 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/runs/samuel_check/queue_project.log

## 2026-10-29 01:14 — run

r261028-3386058 done: b4_frames1000 0.502 at epoch 14 (below frames600 0.552; still climbing, 49 min, so 15 epochs underserves it, or the held-out proxy frames thin out). r261029-3424998 (batch 5, all from frames300+h128 at 0.548): vecnorm max_min 0.564; vecnorm + layers 5 0.569 (trains now, NaN without the norm); vecnorm + cutoff 0.5 0.589, best so far and still rising 0.005/epoch. The norm lets a larger neighbourhood help where cutoff 0.25 hurt. Seed noise is not measured yet; batch 6 adds a seed repeat.

## 2026-10-29 01:14 — run

r261029-3425729: Samuel 153 bayes0 with --source project (project segmentation, auto stage, labelprop 25 seeds) 0.920, against 0.951 on NWB GT objects. With dd0e82e the project path runs; the 3-point gap is the segmentation's detections versus oracle ones.

## 2026-10-29 01:14 — done

0.920 on project segmentation vs 0.951 NWB GT

Done: read r261029 sc_bayes0_153_project result

## 2026-10-29 01:14 — run

r261028-3398874 partial: trial 27 on zimmer_1128 with CPU tracking, backbone stage 0.923; auto stage still tracking.

## 2026-10-29 01:14 — run

Batch 6 (b859d44): around b5 best (vecnorm, cutoff 0.5, frames300, h128; 0.589): seed 44 repeat, cutoff 0.75, 64 neighbours, layers 5.

run r261029-3452046 · pids 3452046 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch6.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 01:14 — run

long6 (b859d44): b5 best with frames 600, 60 epochs, then full-video zimmer_1128 eval; against b2's 0.843.

run r261029-3452047 · pids 3452047 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/long6.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 02:13 — run

r261028-3398874 done: trial 27 on zimmer_1128, labelprop 25 seeds, CPU tracking: backbone 0.923, auto 0.968. So the auto (fused) stage helps on Zimmer (+4.5) and Leifer (+1.4, 0.958 to 0.972) and hurts on Samuel (-11, 0.949 to 0.837): descriptor_stage choice is dataset dependent for this network.

## 2026-10-29 02:13 — decision

r261029-3452046 partial: b6_seed44 (b5 best with seed 44) 0.553 against 0.589 at seed 43, so seed noise in the epoch-14 proxy is ~0.04. Batch 5's vecnorm (+0.016) and layers-5 (+0.005) gains are inside it; only the cumulative climb from 0.400 is clearly real. cutoff 0.75 0.585, 64 neighbours 0.539: no gain. Next batch pairs cells across seeds before ranking.

## 2026-10-29 02:13 — run

Batch 7 (99fbb84), queued behind batch 6: seeds 44 and 45 of no-vecnorm c0.35, vecnorm c0.35 and vecnorm c0.5 (frames300, h128), three seeds each with batch 5/6.

run r261029-3477540 · pids 3477540 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch7.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 03:12 — decision

r261029-3452046 done: b6_layers5 (vecnorm c0.5 + 5 layers) 0.578. r261029-3477540 done. Epoch-14 proxy over seeds 43/44/45 (frames300, h128): no vecnorm c0.35 0.548/0.553/0.560 (mean 0.554); vecnorm c0.35 0.564/0.549/0.543 (0.552); vecnorm c0.5 0.589/0.553/0.531 (0.558). So vecnorm and cutoff 0.5 give nothing at 15 epochs; batch 5's ranking was seed 43 luck. Vecnorm's use is that deeper ViSNets train at all (5 layers: 0.569, 0.578, single seeds). Single-seed gaps under ~0.04 are noise; architecture knobs look saturated at this budget, frames and epochs are what moved the proxy. Batch 1's elastic 1.0 vs 0 (0.400 vs 0.381) was also inside the noise.

## 2026-10-29 03:12 — run

Batch 8 (8bb578e): augmentation strength from frames300/h128/c0.35 no vecnorm, seeds 43+44: elastic 0 and 2.0; with elastic 1.0, jitter_std 2.0 and dropout_p 0.4.

run r261029-3504096 · pids 3504096 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch8.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 04:11 — run

r261029-3504096 done (frames300/h128/c0.35, epoch-14 proxy, seeds 43/44): elastic 0 0.560/0.568 (mean 0.564, vs 0.554 at elastic 1.0 over three seeds) and 4x faster (220 s vs 850 s per cell); elastic 2.0 0.497/0.505; jitter_std 2.0 0.519/0.528; dropout_p 0.4 0.336/0.422. Stronger augmentation hurts the proxy on every axis, dropout most; elastic deformation buys nothing at this budget. Next: weaker augmentation (dropout 0.1 and 0, jitter 0.5) with elastic off, and more frames now that cells are cheap.

## 2026-10-29 04:11 — run

Batch 9: weaker augmentation with elastic off (dropout 0.1, 0; jitter 0.5) and frames 600, seeds 43+44, from frames300/h128/c0.35.

run r261029-3530683 · pids 3530683 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch9.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 04:40 — run

r261029-3452047 done: long6 (frames 600, h128, vecnorm, cutoff 0.5, elastic 1.0, 60 epochs; proxy 0.682) tracks zimmer_1128 at 0.893 full video (labelprop 25 seeds, project segmentation, misses 7069, mismatches 15019 of 206701), up from b2's 0.843 (proxy 0.50). Position only, no image features; trial 27's image backbone alone gets 0.923 on the same recording. The proxy has ranked both full evals correctly so far. Fused dims went 39% dead by epoch 59 without hurting this.

## 2026-10-29 04:40 — run

r261029-3530683 done (elastic 0, frames300/h128/c0.35, epoch-14 proxy, seeds 43/44): dropout_p 0 0.657/0.629 (mean 0.643), 0.1 0.625/0.607 (0.616), 0.2 (b8) 0.564; jitter_std 0.5 0.577/0.552 (0.565, no change from 1.0); frames 600 at dropout 0.2 0.610/0.591 (0.600). Object dropout is the largest lever found: +0.08 from 0.2 to 0. The proxy matches GT objects in both frames, so it cannot see robustness to missed detections; the full eval on project segmentation decides.

## 2026-10-29 04:40 — run

long10: dropout 0 vs 0.2 at elastic 0, frames 600, h128, c0.35, 60 epochs, each with a full-video zimmer_1128 eval (vs long6 0.893).

run r261029-3544674 · pids 3544674 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/long10.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 04:40 — run

Batch 11: hidden 256, 5 layers + vecnorm, lr 3e-4, frames 1000 on the dropout-0/elastic-0 base (frames300, h128), seeds 43+44.

run r261029-3545344 · pids 3545344 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch11.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 05:39 — run

Partial: long10 training done, proxy at epoch 59: dropout 0 0.792, dropout 0.2 0.692 (long6 0.682); full evals running. Batch 11 so far (dropout-0 base 0.657/0.629): hidden 256 0.719/0.645, 5 layers + vecnorm 0.636 (s43), lr 3e-4 0.656 (s43), frames 1000 0.622 (s43).

## 2026-10-29 06:29 — run

Cluster round_posonly (user-launched, read from exp_results.jsonl and trial train_configs, 2026-10-29 06:00): Zimmer RBF fix on, cutoff_nn 0: trial 8 (elastic 0) 0.825, trial 9 (elastic 1.0) 0.827 full-video zimmer_1128; RBF off, best is nn0 el0 0.583, nn3/6/12 0.09-0.23. RBF-on nn 3/6/12 (trials 10-15) not scored yet. Flavell: RBF-off trials 0-5 only, 0.09-0.30. Samuel: no results yet. The cluster's 0.825 matches local b2's 0.843 (similar config, 60 vs 50 epochs) and confirms the RBF fix, ~+0.24 on Zimmer; elastic makes no difference there either.

## 2026-10-29 06:37 — run

r261029-3545344 done (dropout-0 base 0.657/0.629, mean 0.643, epoch-14 proxy): hidden 256 0.719/0.645 (0.682); 5 layers + vecnorm 0.636/0.593 (0.615); lr 3e-4 0.656/0.620 (0.638); frames 1000 0.622/0.628 (0.625). Only width may help (+0.04, at the noise edge); depth, lr and frames beyond 300 do not at 15 epochs.
