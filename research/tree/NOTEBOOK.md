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

- [ ] fix eval_accuracy._load_nwb_arrays: select each NWB field by neuron name (i261026-1), then rescore NWB-path Flavell records
- [ ] rescore round_flavell* cells against the NWB GT (the accuracy objective scores against the project GT)
- [ ] make the NWB path not hand the tracker only GT objects (user: the algorithm should not see only GT objects)

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

- [ ] extract_crops zero-fills points at y≈119.5 that in_bounds_mask keeps (Flavell volume 120 wide): align the bounds test with the crop rounding

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

- [ ] Validation runs in train mode: add model.eval() around the val loss (changes logged val numbers, not training)
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
