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

- [x] ask a261027-1 · on [[QUE - Which part of the position pipeline costs Flavell accuracy, its augmentation or its position fusion]] · Once the re-augmentation test is in, should reaugment_each_epoch default to true, or stay opt-in and be set per round?

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

## 2026-10-29 06:37 — run

Batch 12: hidden 128 and 256 seed 45, hidden 512 seeds 43+44, dropout-0 base.

run r261029-3596459 · pids 3596459 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch12.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 06:42 — run

long13: object dropout 0.02 and 0.05 (user's suggestion of 1-5%), otherwise as long10, 60 epochs + full-video zimmer_1128 eval; brackets long10's 0 and 0.2.

run r261029-3598842 · pids 3598842 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/long13.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 07:42 — run

r261029-3544674 done, full-video zimmer_1128 (labelprop 25 seeds, project segmentation), elastic 0, frames 600, h128, c0.35, 60 epochs: dropout 0 0.932 (misses 5761, mismatches 8312 of 206701), dropout 0.2 0.888 (8096, 15152). Dropout 0 wins on the full eval too, so independent object dropout at 0.2 hurts even with real missed detections. A position-only network now beats trial 27's image backbone on its own recording (0.923, r261028-3398874); fused trial 27 0.968. Correction to what I told the user: the proxy already embeds every detection (untracked objects included) and scores GT neurons present in both frames, so it does see real missing and extra detections as context; only the missed neurons themselves go unscored.

## 2026-10-29 07:42 — run

r261029-3596459 done (dropout-0 base, frames300, epoch-14 proxy): hidden 128 0.657/0.629/0.605 (mean 0.630, seeds 43-45), 256 0.719/0.645/0.686 (0.683), 512 0.737/0.720 (0.729, seeds 43-44). Width is a real lever, +0.05 per doubling, beyond the seed noise.

## 2026-10-29 07:42 — decision

d05dec1: opt-in position_augment spurious_p / merge_p / split_p (synthetic detections, id -1, never paired by the loss), tests in test_detection_errors.py; suite 209 passed, same 3 pre-existing failures. Anisotropic jitter needed no code (jitter_std takes (z, x, y)). Real centroid noise on Zimmer from GT tracks (lag-1 displacement minus the mean of the 6 nearest neighbours', robust std / sqrt 2): (0.17, 1.05, 1.11) voxels, so the isotropic jitter_std 1.0 is ~6x too large in z (z voxels are the coarse axis). Real frames: 169 detections per frame (79-210), 136 GT-named, 14% of GT neurons missing per frame.

## 2026-10-29 07:42 — run

Batch 14: jitter (0.2, 1, 1), spurious 0.2, merge 0.05, split 0.05, and all combined, on the h256 dropout-0 base, seeds 43+44.

run r261029-3633726 · pids 3633726 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch14.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 07:42 — run

long14: long10_drop0 (0.932) at hidden 512, 60 epochs + full-video zimmer_1128 eval.

run r261029-3633727 · pids 3633727 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/long14.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 07:45 — dropped

two streams beside long13 and long14 ran out of GPU memory within a minute; relaunched as one stream

dropped r261029-3633726

## 2026-10-29 07:45 — run

Batch 14 relaunched (one stream, waits for long13's training): same cells as r261029-3633726.

run r261029-3636490 · pids 3636490 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch14.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 08:01 — run

Cluster state 2026-10-29 08:05 (from round folders; no squeue here): round_posonly_zimmer 12 of 16 scored, 4 running; round_posonly_flavell 7 of 16 finished, 3 running, 6 not started; round_posonly_samuel 12 submitted, 3 finished but all crashed in the eval (_centroid_boxes, NaN centroid: the bug dd0e82e fixes), 2 running, 7 not started. Its trained checkpoints are on disk, so they can be rescored with --skip-trained once the cluster code has the fix. The cluster checkout is at d0f45b2, without dd0e82e; local is 48 commits ahead of origin. round_flavell_paper_ablation: 14 of 16 results; the last 2 logs stopped on 2026-10-27, presumably dead.

- [ ] push and pull dd0e82e on the cluster, then rescore round_posonly_samuel's crashed trials

## 2026-10-29 08:01

Correction to the 08:01 entry: there is no --skip-trained flag. Rescore the Samuel checkpoints with run_trials_on_all_ground_truth.py --trials N (it scores existing trial checkpoints and skips recorded tags).

## 2026-10-29 08:07 — result

Centroid noise per axis from GT tracks (barlow_track@bfde2bf:barlow_track/scripts/scratch/centroid_noise.py; upper bound, includes local non-rigid motion), voxels (z, x, y): Zimmer (0.17, 1.05, 1.11), Flavell (0.64, 0.81, 0.90), Samuel 153 (0.11, 0.57, 0.54). Every position run so far used jitter_std 1.0 on every axis: 6-9x the real z noise on Zimmer and Samuel, ~2x in xy on Samuel.

## 2026-10-29 08:11 — decision

Staged (barlow_track@1977e9a): round_posonly2_{zimmer,flavell,samuel}, 16 jobs each (spurious_p {0, 0.2} x merge_p {0, 0.03} x split_p {0, 0.03}, two seeds; base RBF fix, width 512, 600 frames, 60 epochs, dropout 0, elastic 0, keypoints only, jitter matched per dataset) and round_fused_fix_zimmer, 8 jobs (trial 27 + that position branch; visnet_rbf_rescale {false, true} x fusion {concat, add}, two seeds). README and launch in scratch/round_posonly2_zimmer/README.md; needs push + pull on the cluster first (cluster at d0f45b2). Smoke-tested 1 epoch on CPU. User will launch from the office.

## 2026-10-29 08:11 — run

Batch 14 first cell: b14_jitz_s43 (jitter (0.2, 1, 1), h256 dropout-0 base) 0.785 at epoch 14 against the isotropic-jitter base's 0.719 at seed 43: +0.066, above the ~0.04 seed noise but one seed so far.

## 2026-10-29 08:13 — decision

User (2026-10-29 ~08:15): cluster has many jobs pending, so no rush; gather as much information as reasonable before the next round. Plan before launching round_posonly2/round_fused_fix: (1) finish batch 14 and the dropout 0.02/0.05 and h512 full evals; (2) full-video eval on Zimmer of the batch-14 winner (matched jitter + any error augmentation that helps) at h512; (3) the same settings on Samuel 153 and Flavell (8741c78 adds --lab to posonly_fast), old base vs new, since every setting so far was found on Zimmer only; (4) trim the staged rounds by what transfers. GPU is full (11.7 GB) until long13's evals release it.

## 2026-10-29 08:14 — decision

User (2026-10-29): avoid jitter matched to the ground truth; the methods must generalize to unseen datasets, so no setting is fitted to the GT at hand. Reverted the staged rounds (round_posonly2_*, round_fused_fix_zimmer) to isotropic jitter_std 1.0 (306fca2). Batch 14's z-light cells (sized from Zimmer GT tracks) stay as a diagnostic of whether anisotropy matters; if it does, a production rule must be GT-free (e.g. the same residual estimate from unlabelled mutual-nearest-neighbour matches between consecutive frames, or the voxel aspect ratio). Same principle already holds elsewhere: elastic_std was matched to GT motion but elastic is off; spurious_p 0.2 came from the GT-named fraction but is gridded against 0 and chosen by accuracy across datasets.

## 2026-10-29 08:20 — result

GT-free dataset statistics (barlow_track@207994d:barlow_track/scripts/scratch/dataset_stats.py; segmentation only: affine-ICP alignment of consecutive frames, mutual NN matches within half the NN spacing, residual vs neighbours' displacement). Centroid noise (z, x, y) voxels, GT-free vs GT-track estimate (centroid_noise.py): Zimmer (0.30, 0.91, 0.94) vs (0.17, 1.05, 1.11); Flavell (0.80, 0.87, 0.85) vs (0.64, 0.81, 0.90); Samuel (0.10, 0.30, 0.30) vs (0.11, 0.57, 0.54). It recovers the anisotropy and the scale within ~2x (Samuel xy low, likely the match gate truncating its small spacing). Unmatched fraction after ICP: Zimmer 0.57, Flavell 0.74, Samuel 0.10: an affine does not absorb Zimmer/Flavell lag-1 bending, so it is no miss-rate estimate there. Also: detections per frame 170/178/91 (CV 0.12/0.07/0.12), within-frame NN median (z, x, y) Zimmer (2.9, 4.5, 4.5), Flavell (2.9, 3.9, 2.7), Samuel (1.0, 2.9, 2.7). The user allows dataset statistics available on new data, nothing needing GT; the GT estimate here only validates the GT-free one.

## 2026-10-29 08:29 — decision

c434ceb: barlow_track/scripts/best_of_table.py writes docs/best_of_table.md, the best image-only / fused / position-only accuracy per dataset from every cluster round's exp_results.jsonl and the local scratch records, comparable records only (trained checkpoints, labelprop, default eval, full videos; Flavell only against the NWB GT and, for --source nwb, after the i261026-1 fix). cb4e869: opt-in jitter_std 'auto' (GT-free centroid noise, utils/dataset_stats.py; 1 s at setup on Zimmer, (0.31, 0.99, 1.03)); off by default, and the user wants it off in the fast iterations.

## 2026-10-29 08:47 — decision

a14918e: best-of table now one row per recording, all scorable ones shown (Zimmer 1128/1123/1210, five Flavell NWBs, Leifer, Samuel 137/153/184-004); dropped the Zimmer original-format and Samuel project-segmentation rows (user: redundant). Benchmark driver gains flavell_2022031603/0324-02/0330-01/0330-02, samuel_137, samuel_184; NWB datasets now pass --gt (record names its own NWB, resume matches); --device cpu requests no Slurm GPU. Staged scratch/benchmark_more_datasets/run.sh (best checkpoints of each kind x 9 further recordings, CPU only, 63+18 cells) for the user to launch. Smoke: baseline_flavell_head_rerun on flavell 2022-03-16-03, 30 frames, CPU: loads, 173 GT neurons, 0.809. Suite on CPU: 221 passed, 2 pre-existing failures; on the busy GPU 4 more tests OOM (not code).

## 2026-10-29 08:57 — run

posonly_fast batch 15 (831f6a5, augmentations 62f6943): from the batch-14 base (h256, dropout 0, elastic 0), seeds 43/44, epoch-14 proxy: jitter_std auto (GT-free estimate) and 0, axis_scale_jitter 0.1, bend_max 0.3, cut_p 0.5 / cut_max 0.15, global scale_jitter 0 and 0.2. Queued behind batch 14.

run r261029-3707539 · pids 3707539 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch15.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 08:57 — result

r261029-3636490 (batch 14, seed 43, epoch-14 proxy; base 0.719): jitter (0.2,1,1) 0.785, spurious 0.2 0.666, merge 0.05 0.700, split 0.05 0.713, all combined (anisotropic jitter + spurious 0.2 + merge/split 0.03) 0.691. The segmentation-error augmentations cost the proxy even on top of the jitter gain; the jitter gain is GT-matched, so batch 15 tests the GT-free auto estimate instead. Seed 44 still running.

## 2026-10-29 09:08 — run

posonly_fast batch 16 (a524744): real frame pairs, the second view is frame t+lag with GT-free identities (nonrigid_matches, unmatched id -1). Cells pair_p 0.5 / 1.0 (lag 1), pair_p 0.5 lag 1-3, pair_p 0.5 resid 1.5; seeds 43/44, batch-14 base, queued behind batch 15. Caveat: a training frame's t+1 can be a held-out proxy frame (positions only, no labels).

run r261029-3720770 · pids 3720770 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch16.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 09:08 — result

GT-free frame-pair matching scored against GT tracks (scratch/pair_pseudolabels.py @2676174, diagnostic only; 40 start frames per dataset). Lag 1, resid filter 2.0: Zimmer 42 matches/frame at precision 0.964, Flavell 26 at 0.82 (its GT is itself a tracker output), Samuel lag 1 unfiltered 76 at 0.98. Plain affine ICP + mutual NN was 0.75 (Zimmer) / 0.64 (Flavell); the local-median refinement and the residual filter carry it. Chaining to longer lags keeps precision but recall collapses (Zimmer lag 3: 14/frame). So real frame pairs are usable at lag 1 on Zimmer and Samuel, doubtful on Flavell.

## 2026-10-29 09:08 — idea

Augmentation ideas still untested after batch 16, if real pairs help: (1) pseudo-label mining from the network itself (match embeddings between t and t+k, keep confident mutual matches, retrain), a self-training loop that needs no GT; (2) lag-k pairs from nonrigid matching on a coarse-to-fine schedule; (3) z-dependent dropout (dim planes at the volume edges).

## 2026-10-29 09:27 — decision

Real frame pairs (batch 16, r261029-3720770): training reads no GT; identities come from nonrigid_matches on the segmentation alone, and GT was only used to score them (pair_pseudolabels.py). One GT-informed choice remains: pair_resid 2.0 was picked from the GT precision on Zimmer. To be fully GT-free it should be tied to the GT-free noise estimate (e.g. a multiple of the auto jitter) or fixed in advance and not tuned. The user's concern: matching-based pairs are pseudo-labels from a second tracker, biased to easy neurons.

## 2026-10-29 09:27 — idea

Correspondence-free loss for real frame pairs: a contrastive random walk (Jabri et al. 2020, space-time correspondence) between the embeddings of frames t and t+k. Softmax affinities A(t->t+k) and A(t+k->t), and the loss is the cross-entropy of the round trip returning each object to itself (palindrome t->t+k->t). No matching, no GT; it can sit beside the Barlow loss on the synthetic pairs. Also possible: an entropic optimal-transport (Sinkhorn) soft assignment as the latent correspondence.

## 2026-10-29 09:32 — decision

Decision (user, 2026-10-29): keep batch 16. GT-free locally-rigid pseudo-labels are close to rigid augmentation, which is invisible to ViSNet, but the matched objects keep the residual strongly non-rigid motion, which may help even with many objects lost. The cycle-consistency loss went into the tree as a low-priority question for after the revisions.

- [ ] If batch 16 helps: make pair_resid GT-free (a multiple of the auto centroid-noise estimate), since 2.0 was chosen from GT precision

## 2026-10-29 09:38 — dropped

superseded by queue15_16 (same cells, 4 parallel workers)

dropped r261029-3707539

## 2026-10-29 09:38 — dropped

superseded by queue15_16 (same cells, 4 parallel workers)

dropped r261029-3720770

## 2026-10-29 09:38 — run

posonly_fast batches 15+16 as one queue (577b209): the same 22 cells, 4 GPU workers, seed 43 first, after batch 14; aiming for results within ~3 h to stage the next cluster round. Per-cell logs: posonly_fast/<name>.cell.log.

run r261029-3738930 · pids 3738930 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/queue15_16.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 09:54 — dropped

stopped: 4 workers beside long14's eval ran out of GPU memory (12 of 15 started cells OOM'd, moved to posonly_fast/_oom_failed) and maxed the GPU during the user's call. 3 cells finished (results logged).

dropped r261029-3738930

## 2026-10-29 09:54 — run

Rest of the batch 15+16 queue (19 cells), 1 GPU worker while long14's eval holds the GPU, then 2 (the user asked for at most 2 GPU jobs during a call); finished cells skipped.

run r261029-3751540 · pids 3751540 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/queue15_16b.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 09:54 — result

First batch 15/16 cells, seed 43, epoch-14 proxy (base 0.719, GT-matched jitter (0.2,1,1) 0.785): jitter_std auto (GT-free) 0.768; axis_scale_jitter 0.1 0.695; real pairs pair_p 1.0 0.577 (well below base: with every second view a real pair, only the ~35 matched objects per frame are in the loss). Batch 14 seed 44 last cell (all combined) 0.625.

## 2026-10-29 09:58 — run

Rescore round_posonly_flavell trials 8/9 (RBF fix) on the Flavell project segmentation against the NWB GT, CPU only, local records (f4f1ff1).

run r261029-3754534 · pids 3754534 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/rescore_posonly_flavell.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 09:58 — result

Cluster position-only rounds (trained-mode labelprop, full video, own training project as GT): round_posonly_zimmer, all 16 trials scored on zimmer_1128 (original-format project): RBF fix off 0.09-0.58, on 0.66-0.87. With the fix, cutoff_nn 6 is best (0.868 elastic 0, 0.829 elastic 1), then 0 (0.825 / 0.827), 12 (0.801 / 0.727), 3 (0.691 / 0.657); elastic 1 never helps (-0.04 mean). Trial 8 cross-dataset: zimmer_1210 0.808, Flavell NWB 0.558, Flavell 2022-03-16-03 0.634, samuel_137 0.204. round_posonly_flavell: fix off 0.09-0.30, on (cutoff_nn 0) 0.870 / 0.881 against the project GT (tracker output, i261028-3; NWB rescore running); trials 10-15 not trained yet. round_posonly_samuel: 4 checkpoints, no scores (evals crash on the NaN-centroid bug, cluster checkout lacks dd0e82e). Local best for comparison: long10 0.932 on Zimmer (h128, dropout 0, 600 frames).

## 2026-10-29 09:58 — decision

Local loop has not tried visnet_cutoff_nn 6 on the current base (cluster: 0.868 vs 0.825 at 0); add it to the next local queue and to the next cluster round's base candidates.

## 2026-10-29 09:59 — run

posonly_fast batch 17: visnet_cutoff_nn 6 / 12 (and 6 with jitter auto) on the batch-14 base, seeds 43/44, after the 15+16 queue.

run r261029-3755323 · pids 3755323 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch17.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 10:58 — result

r261029-3598842 (long13, h128, 600 frames, elastic 0, 60 epochs, full-video zimmer_1128 labelprop): dropout 0.02 0.945, dropout 0.05 0.931 (epoch-59 proxy 0.782 / 0.797), against long10 dropout 0 0.932 and 0.2 0.888. A little dropout (the user's 1-5%) is at least as good as none; 0.945 is the best position-only Zimmer number so far, above trial 27's image backbone (0.923), below its fused 0.968. One seed each, so 0.932 vs 0.945 is within plausible noise.

## 2026-10-29 10:58 — result

r261029-3751540 seed 43 so far (epoch-14 proxy, batch-14 base 0.719; base seeds 43/44/45 0.719/0.645/0.686): jitter auto 0.768 (seed 44 0.693, base s44 0.645: +0.05 both seeds); jitter 0 0.631; bend 0.3 0.707; axis scale 0.1 0.695; plane cut 0.5 0.664; global scale_jitter 0 0.686, 0.2 0.659 (0.1 base best). Real frame pairs all below base: pair_p 0.5 0.580, 1.0 0.577, 0.5 lag 1-3 0.612, 0.5 resid 1.5 0.647. Reading: positional jitter matters and the GT-free anisotropic estimate beats isotropic 1.0; the new geometry augmentations are neutral to harmful; GT-free real pairs hurt the proxy (stricter filter hurts less, so pseudo-label errors or the easy-object bias are likely the cause).

## 2026-10-29 11:36 — result

r261029-3633727 (long14, width 512, dropout 0, 600 frames, elastic 0, 60 epochs): full-video zimmer_1128 0.958 (epoch-59 proxy 0.903). Best position-only Zimmer network so far, above trial 27's image backbone (0.923), 0.012 below trial 27 fused (0.970).

## 2026-10-29 11:36 — result

r261029-3751540 done, seed 44 (base s44 0.645): jitter auto 0.693, jitter 0 0.595, bend 0.3 0.665, cut 0.5 0.623, global scale 0 / 0.2 0.651 / 0.630, real pairs 0.5 / 1.0 / lag3 / r1.5 0.573 / 0.562 / 0.566 / 0.531. Seed 44 agrees with seed 43 everywhere: jitter auto +0.05 on both seeds is the only gain; bend is neutral; cut, jitter 0, scale 0.2 and every real-pair setting hurt.

## 2026-10-29 11:36 — result

Cluster CPU benchmark (benchmark_more_datasets/run.sh, run by the user; cluster at 7d9970b) on 9 further GT recordings. Image-only round_flavell_paper_ablation t0 is >=0.957 on all five Flavell recordings and 0.942 / 0.879 on Zimmer 1123 / 1210, but 0.468 / 0.432 on Samuel 137 / 184. Trial 27 fused: Zimmer 0.979 / 0.920, Flavell 03-16 / 03-24 0.977 / 0.957 but 03-30-01 / 03-30-02 0.817 / 0.800, Samuel 0.662 / 0.832; its backbone alone collapses on 03-30-01 / 03-30-02 (0.477 / 0.660) and is better on Samuel (0.868 / 0.934). Every Zimmer- or Samuel-trained network collapses on Flavell 03-30-01 / 03-30-02 (0.47-0.63 image-only). Position-only t8 (old base) transfers poorly (Flavell 0.56-0.76, Samuel 0.20 / 0.38). Three Flavell 2022-03-31-01 NWB cells were skipped as already recorded, but their records predate the neuron-order fix (nwb_arrays None): t27 auto 0.820, bayes_samuel t0 auto 0.709, 2x2 t6 0.703; they need --rerun_completed.

- [ ] Rerun Flavell 2022-03-31-01 NWB cells predating the neuron-order fix with --rerun_completed: round_visnet_full t27, round_bayes_samuel t0, round_2x2 t6 (auto stage)

## 2026-10-29 11:48 — issue

Accuracy-objective sweeps on Flavell and Samuel tagged their in-job record with the benchmark's NWB cell label ('flavell', 'samuel') while scoring the project GT (Flavell's is the lab's tracker output); the benchmark then skips the real GT-object cell as done. round_posonly_flavell's 0.870 / 0.881 are such records. Fixed for new rounds by objective_gt (ecd3617); old rounds rescored by cpu_rescores/run.sh with --rerun_completed.

- [x] issue i261029-1 · affects [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 11:48 — decision

Decisions (user, 2026-10-29): Flavell objective must use the NWB GT (objective_gt added, ecd3617; Samuel round too, for the same tag reason); priority is getting position-only to work cleanly, so round_fused_fix_zimmer stays a clean RBF-fix test at trial 27's position settings; CPU rescores staged to wait for the round_posonly jobs to leave the queue. Waiting on the user's push, pull and launch.

## 2026-10-29 11:48 — resolved

logged with a broken --affects (the note title's comma split it); relogged with the note named by a unique part of its title

resolved i261029-1

## 2026-10-29 11:48 — issue

Accuracy-objective sweeps on Flavell and Samuel tagged their in-job record with the benchmark's NWB cell label ('flavell', 'samuel') while scoring the project GT (Flavell's is the lab's tracker output); the benchmark then skips the real GT-object cell as done. round_posonly_flavell's 0.870 / 0.881 are such records. Fixed for new rounds by objective_gt (ecd3617); old rounds rescored by cpu_rescores/run.sh with --rerun_completed.

- [ ] issue i261029-2 · affects [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 11:59 — result

r261029-3755323 batch 17 seed 43 (epoch-14 proxy, base 0.719, base + jitter auto 0.768): cutoff_nn 6 0.670, 12 0.694, 6 + jitter auto 0.709. Locally cutoff_nn 6 trails the fixed cutoff by ~0.05 (one seed, noise ~0.04), the opposite of round_posonly_zimmer's full video (0.868 vs 0.825). The round_posonly2 grid tests both, so no change.

## 2026-10-29 12:14 — decision

Pre-launch check of the four staged rounds (2443724): objective preflight passes for all (round_posonly2_zimmer moved to the updated-format project; the original-format one fails with the current wbfm), jitter_std auto runs on each training project: Zimmer (0.30, 0.99, 0.95), Flavell (0.81, 0.84, 0.83), Samuel (0.10, 0.30, 0.31). Ready for the user's push and launch.

## 2026-10-29 12:42 — result

r261029-3754534: round_posonly_flavell trials 8 / 9 (RBF fix, cutoff_nn 0, elastic 0 / 1) on the Flavell 2022-03-31-01 project segmentation scored against the NWB GT: 0.916 / 0.934 (1600 frames), against 0.870 / 0.881 on the project GT. On that row the best image-only is 0.958 (baseline_flavell_head_rerun) and the best fused 0.927, so a Flavell-trained position-only network is within 2.4 points of the best image network there. Records local (~/.cache/barlow_scratch/rescore_posonly_flavell).

## 2026-10-29 12:42 — result

r261029-3755323 batch 17 done, seed 44 (base 0.645, base + jitter auto 0.693): cutoff_nn 6 0.618, 12 0.652, 6 + auto 0.645. Both seeds put cutoff_nn 6 about 0.05 below the fixed cutoff on the epoch-14 proxy, against the cluster's full-video +0.04 (round_posonly_zimmer). Proxy and full video disagree here; round_posonly2 tests both on full video.

## 2026-10-29 12:42 — run

long15: visnet_cutoff_nn 0 vs 6, full-video zimmer_1128 eval, at round_posonly2's base (h512, 600 frames, 60 epochs, elastic 0, dropout 0.02, jitter auto); settles the proxy-vs-full-video disagreement of batch 17 a day before the cluster round.

run r261029-3865613 · pids 3865613 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/long15.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 13:21 — decision

Pre-launch code review (subagent, read-only) of 7d9970b..HEAD: one blocker, the three round_posonly2 grids crashed at Ax experiment creation (choice mixing 1.0 and 'auto'; confirmed), fixed with value_type str and numeric strings read as floats (resume checked). Follow-ups fixed: auto jitter raises instead of silently using 1.0; cpu_rescores squeue-failure guard; relaunch needs --skip_completed_grid_jobs. Checked fine: default-off augmentations bitwise identical to 7d9970b, id/row alignment, objective_gt records match the benchmark's 'flavell'/'samuel' settings and tags, budgets (posonly2 53 h per job).

## 2026-10-29 13:49 — run

posonly_fast batch 18 (queue18.sh): do the Zimmer fast-loop settings transfer? Samuel 153 and Flavell projects, batch-14 base, cells base / jitter auto / cutoff_nn 6 / dropout 0.02, seeds 43/44, 3 GPU workers after long15 trains. Proxy GT: Samuel's final_tracks (95 named per frame, loaded with allow_hybrid_loading as training does); Flavell's is the project GT (the lab's tracker output), so Flavell ranks settings only.

run r261029-3914302 · pids 3914302 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/queue18.log · for [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]]

## 2026-10-29 14:12 — decision

User's goal for the next ~2 weeks (2026-10-29): a finalized architecture that (a) works on ALL GT recordings, not one per lab (Samuel varies most), (b) runs efficiently (hundreds of jobs to reproduce the paper figures), (c) shows a label-propagation benefit (Samuel), (d) shows a benefit of the transpose / object-space loss (lambd_obj), e.g. faster training or data efficiency. Position helping is optional; position-only and image-only quality are means, not ends: if a fused network beats them, that is the result. The position-only sweep continues but is judged against the best fused numbers.

## 2026-10-29 14:12 — run

posonly_fast batch 19 (queue19.sh): transpose loss where accuracy is not saturated, lambd_obj 0 / 0.67 / 1 x 30 / 100 / 300 training frames, Zimmer position-only, seeds 43/44, after batch 18; per-epoch proxy curves give training speed too.

run r261029-3931397 · pids 3931397 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/queue19.log · for [[EVD - Loss-term ablation where accuracy is not saturated]]

## 2026-10-29 14:21 — decision

objective_cpu_job (default off, 3e014a0): GPU jobs only train, each eval (embedding, label propagation, scoring) is a CPU-only Slurm job with its own cap. The user (2026-10-29): move everything possible to CPU so GPU slots are used efficiently; cross-lab generalization is not needed (self-supervised, a network per recording), so 'works on all datasets' means one recipe trained per recording. Use objective_cpu_job in the weekend round; the running rounds keep in-job evals.

## 2026-10-29 14:22 — decision

Correction to the previous entry: objective_cpu_job is commit d37ac74, not 3e014a0 (AGENTS.md, which also documents it, is git-ignored and local only).

## 2026-10-29 14:50 — run

CPU-training probe: position-only cell at round_posonly2's base (h512, 600 frames, 3 epochs) on CPU with 16 threads, to see whether position-only training can leave the GPU entirely (the user: CPU jobs are plentiful, GPU slots are the budget). Local load ~55 of 56 cores, so the timing is pessimistic.

run r261029-3966658 · pids 3966658 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/cpu_probe.log

## 2026-10-29 14:52 — decision

objective_cpu_parallel now defaults to 1000 (was num_parallel_jobs=10): every trained checkpoint is scored at once on CPU nodes, since only GPU slots are scarce (0884732)

## 2026-10-29 14:57 — run

r261029-3966658: CPU-only training of the h512 / 600-frame position-only network (16 threads, local load ~69 on 56 cores): 100 steps in 246 s, so ~1180 s per epoch against ~94 s on the GPU (12.5x). On a dedicated 16-CPU node maybe half that: 60 epochs in ~10-20 h. Viable for position-only rounds given the generous walltime; stopped after epoch-0 timing

## 2026-10-29 14:58 — decision

train_cpu_job (template key, off by default; 22c6f0c): training jobs ask for no GPU (train_cpu_job_cpus 16, train_cpu_job_mem 64G), for keypoints-only rounds. With objective_cpu_job a position-only round then uses no GPU at all. Needs train_hours sized for ~12x GPU time (r261029-3966658)

## 2026-10-29 14:58 — ask

Per-recording training for the weekend round: 9 GT recordings have a trainable wbfm project with segmentation metadata (zimmer 1128/1123/1210, flavell 2025_07_01 + the four forCharlie_2, samuel 153). Leifer (leifer_project_barlow/_gt), samuel 137 and samuel 184 have no 1-segmentation/metadata.pickle, and loading their NWBs as projects fails (no segmentation metadata), so they cannot be trained on without rebuilding metadata (a write to /lisc, or a local copy) or training on the NWB's GT objects only (which leaks which objects are tracked).

- [ ] ask a261029-1 · on [[QUE - Do the preprint accuracies hold under the corrected evaluation and tracking code]] · For Leifer, Samuel 137 and Samuel 184: rebuild segmentation metadata into a new project copy (where?), train on the NWB objects, or leave them out of the per-recording round?

## 2026-10-29 15:09 — run

r261029-3966658 correction: the first kill stopped only the wrapper shell; the trainer ran on to step 400 (1034 s, a steady ~262 s per 100 steps, so ~1260 s an epoch of 480 steps, 13x the GPU) before I stopped it (PID 3966671)

## 2026-10-29 15:15 — decision

make_per_recording_rounds.py (6e88c46): one round per GT recording from one base config and grid (project_path, target_sz_z 16 for Flavell, objective_dataset = benchmark label, NWB objective_gt). The objective preflight passes on all 9 trainable recordings (zimmer 1123/1128/1210, flavell + 4 forCharlie_2, samuel 153). The weekend round will use it.

## 2026-10-29 15:54 — run

round_posonly2 first cluster records (GT objects, label propagation, seed 43): Flavell position-only 0.975-0.980 (trials 0-2, jitter 1.0), above the best image-only 0.963 on that recording. Samuel 153: jitter 1.0 0.694-0.727 (trials 0-3), jitter auto 0.768 / 0.904 / 0.936 (trials 5 / 4 / 6; best auto + cutoff_nn 6 + dropout 0), so the GT-free per-axis jitter is worth ~+0.2 on Samuel, where the segmentation's centroid noise is ~(0.1, 0.3, 0.3) voxels. No Zimmer or fused_fix records yet. best_of_table regenerated.

## 2026-10-29 16:09 — run

r261029-3931397 (batch 19, position-only Zimmer, epoch-14 proxy, seeds 43/44): lambd_obj 0 > 0.67 > 1.0 at every training-frame count. 30 frames 0.496/0.467 vs 0.490/0.463 vs 0.477/0.452; 100 frames 0.604/0.608 vs 0.594/0.592 vs 0.585/0.577; 300 frames 0.768/0.676 vs 0.714/0.648 vs 0.682/0.636. So in the position-only network the transpose loss gives no data-efficiency gain and costs 0.01-0.05 on the proxy (the 300-frame gap is beyond seed noise). Only the proxy at epoch 14; untested in fused / image networks or on the full video.

## 2026-10-29 16:09 — run

r261029-3914302 (batch 18, proxy, seeds 43/44): Samuel base 0.504/0.568, jitter auto 0.549/0.609 (+0.04 both seeds), cutoff_nn 6 0.459/0.516 (-0.05), dropout 0.02 0.499/0.566 (0). Flavell base 0.781/0.722, auto 0.783/0.728 (0), cnn6 0.738/0.674 (-0.045), dropout 0.793/0.722 (0). The proxy again penalises cutoff_nn 6 while the full video favours it (round_posonly2_samuel 0.727 vs 0.706 at jitter 1.0, 0.936 vs 0.904 at auto; round_posonly_zimmer 0.868 vs 0.825): on cutoff_nn the proxy cannot be trusted; long15 checks it locally on full video.

## 2026-10-29 16:10 — run

Batch 20 (4af7fc0): lambd_obj 0 vs 0.67 in image-only (fusion visual_only) and fused (add) networks from the round_fused_fix_zimmer config, 100 frames, 15 epochs, seeds 43/44, 2 workers. Does the transpose loss help where it was meant to (image features), after batch 19 found it costs position-only networks a little?

run r261029-4011812 · pids 4011812 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch20.log · for [[EVD - Loss-term ablation where accuracy is not saturated]]

## 2026-10-29 16:20 — run

r261029-3865613 (long15, position-only Zimmer at the posonly2 base with jitter auto and dropout 0.02, 60 epochs, seed 43, full video with label propagation): cutoff_nn 0 0.961, cutoff_nn 6 0.943. 0.961 is the best position-only Zimmer score so far (long14, jitter 1.0 and dropout 0: 0.958). So on Zimmer at this base the fixed cutoff wins on full video too, while on Samuel cutoff_nn 6 wins (round_posonly2_samuel 0.936 vs 0.904 at jitter auto): the cutoff choice looks dataset-dependent (one seed each), and it should stay in the grid rather than be fixed.

## 2026-10-29 17:40 — run

r261029-4011812 seed 43, image-only (fusion visual_only, trial 27 image settings, 100 frames, 15 epochs, Zimmer), epoch-14 proxy: lambd_obj 0.67 0.617 vs lambd_obj 0 0.482. The transpose loss trains faster too: 0.67 passes 0.48 at epoch 3, lambd_obj 0 only reaches it at epoch 14 (~4x fewer epochs). The opposite of the position-only networks (batch 19). One seed so far; s44 and the fused cells are running. Fused add lobj0.67 s43 ran out of GPU memory beside another cell; dropped batch 20 to 1 worker and queued its rerun after the batch

## 2026-10-29 17:40 — run

batch 20 retry of the out-of-memory cell (b20_add_lobj0.67_s43), one worker, after batch 20 ends

run r261029-4053272 · pids 4053272 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch20_retry.log · for [[EVD - Loss-term ablation where accuracy is not saturated]]

## 2026-10-29 17:52 — decision

User (2026-10-29): in the preprint, varying lambd_obj didn't matter much; either loss trained well. Training dynamics and data efficiency were not checked then. Test it on the Samuel recordings too, and anywhere else it might help. So batch 20's gap at epoch 14 may be speed rather than final accuracy: run it long enough to see whether lambd_obj 0 catches up, and at few frames.

## 2026-10-29 17:53 — run

Batch 21 (11a58ec), after the batch 20 retry, one worker: image-only lambd_obj 0 / 0.67 on Samuel 153 (30 / 100 frames, 30 epochs, seeds 43/44: does 0 catch up, as the preprint's final accuracies suggest?) and on Zimmer (100 frames, 40 epochs, seed 43, with full-video eval)

run r261029-4059380 · pids 4059380 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch21.log · for [[EVD - Loss-term ablation where accuracy is not saturated]]

## 2026-10-29 18:32 — decision

Ax (Bayesian) mode already scores trials on full-video tracking accuracy in CPU jobs: objective: accuracy + objective_cpu_job: true, without --direct_parameter_sweep. The train job frees its GPU slot, the eval runs as a CPU job, and its accuracy completes the Ax trial (the GP fits that, never val loss); Ax keeps proposing new trials into free GPU slots meanwhile. New test (ef21c16) runs a 3-trial Ax sweep through the real dispatcher with faked train/eval. The user (2026-10-29) suggests using Ax over a promising range rather than fixed grids.

## 2026-10-29 19:35 — decision

Distilled the 2026-10-28/29 position-only work into the tree: filled and finished [[EVD - Position-only ViSNet networks with and without the RBF fix on Zimmer, Flavell and Samuel]] (high); new evidence on GT-free jitter, augmentation dead ends, lambd_obj in position-only networks, the ViSNet cutoff and CPU training cost; new low-confidence claim [[CLM - A position-only network tracks within a few points of the best image or fused network]]; Findings line in ideas/barlowtrack.md. Issue i261028-2's three notes still need judging with the user.

## 2026-10-29 19:42 — run

fDNC baseline on zimmer_1128 (started by the fDNC subagent, CPU only): fdnc_track_and_score.py, 5 template frames, four axis orientations (default / swap / mirror / swap_mirror) in parallel, to find which coordinate convention the pretrained fDNC model expects

run r261029-4117569 · pids 4117569, 4117570, 4117571, 4117572 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/fdnc/logs/zimmer_1128_default.log, /home/charles/.cache/barlow_scratch/fdnc/logs/zimmer_1128_mirror.log, /home/charles/.cache/barlow_scratch/fdnc/logs/zimmer_1128_swap.log, /home/charles/.cache/barlow_scratch/fdnc/logs/zimmer_1128_swap_mirror.log · for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-10-29 19:45 — decision

Keypoints reach ViSNet in voxel units scaled by one scalar (superglue.normalize_keypoints), so on Zimmer z is compressed 4.6x relative to physical units (1.5 um z planes, 0.325 um xy pixels; the other labs' project physical_units are 'CHANGE ME' placeholders). Every ViSNet network so far, fused and position-only, saw this metric. Found by the encoder literature agent, confirmed in the code. New opt-in visnet_z_scale (d9cfd1a; off by default, checked in load mismatches). Possible explanation for the dataset-dependent cutoff_nn result.

## 2026-10-29 19:45 — run

Batch 22 (d152ab3): position-only visnet_z_scale sweep (Zimmer 1 / 1.55 / 3 / 4.6, Samuel 1 / 2.9 / 5, Flavell 1 / 2 / 4) at the batch-18 base with jitter auto, seeds 43 / 44, epoch-14 proxy, one worker beside batch 20

run r261029-4126504 · pids 4126504 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch22.log · for [[QUE - Does adding neuron position information improve tracking]]

## 2026-10-29 19:46 — idea

Revision literature brainstorm in docs/brainstorm/revision_literature/ (688ea7b): four agent reports plus a ranked shortlist. Kept out of the tree as the user asked; ideas become planned notes only if picked. Confirmed in code: voxel-unit keypoints (z squashed 4.6x on Zimmer; visnet_z_scale d9cfd1a) and the unused alpha in run_label_propagation.

## 2026-10-29 19:48 — run

fDNC baseline follow-ups (fDNC subagent, CPU only): zimmer_1128 with rescaled coordinates in four orientations, and samuel 153 (NWB) in four orientations, after the first zimmer_1128 orientation runs

run r261029-4128796 · pids 4128796, 4128797, 4128798, 4128799, 4129942, 4129943 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/fdnc/logs/zimmer_1128_default_rescaled.log, /home/charles/.cache/barlow_scratch/fdnc/logs/samuel_default_rescaled.log · for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-10-29 19:49 — run

r261029-4117569 interim (cde97bb, fdnc_baseline/): the published fDNC model (model.bin) runs on CPU with PYTHONPATH only, ~60 ms per frame per template. zimmer_1128 full video, one template, all detections, our nandist scoring: 0.24-0.25 with wbfm's zimmer2leifer axes, 0.43 with x/y swapped; ~0.98 one frame from the template, decaying within 10-20 frames. fDNC is not scale-invariant: the Zimmer cloud is ~1.6x larger than fDNC's reference worms; per-frame checks give 0.63 at scale 0.7. Rescaled (GT-free, reference-RMS) full-video runs on zimmer_1128 and samuel 153 still running (r261029-4128796). The axis convention is picked with GT, so the best of four favours fDNC. Our network: ~0.97.

## 2026-10-29 19:54 — run

r261029-4128796 partial: fDNC on samuel 153 (NWB GT objects, GT-free size rescale 0.969), full video, single template: swap 0.809 (template 10) / 0.739 (266), swap_mirror 0.788, mirror 0.505 / 0.438, default 0.498 / 0.412; ours ~0.95. fDNC's own mean match probability (~0.79 for the swap conventions vs ~0.61) picks the best axis convention without GT. Rescaled zimmer_1128 and the 5-template votes still running. sbatch for all nine recordings now passes --rescale ref (not submitted).

## 2026-10-29 19:56 — run

r261029-4128796: fDNC zimmer_1128 rescaled (GT-free factor 0.624), template 10, full video, all detections: swap 0.615, swap_mirror 0.549, mirror 0.450, default 0.443 (ours ~0.97). Samuel 153 rescaled, single templates: best 0.809-0.835 (swap / swap_mirror). On both datasets fDNC's mean match confidence ranks the axis conventions as GT does, so the most-confident convention is a GT-free choice. 5-template votes still running.

## 2026-10-29 19:57 — decision

User (2026-10-29): BarlowTrack was co-submitted with ASCENT (Lu lab) because the two are so similar, so ASCENT is concurrent work, not missed prior work; the literature agents' novelty-gap framing is wrong. Brainstorm README corrected.

## 2026-10-29 19:59 — ask

ASCENT (Lu lab) is co-submitted with BarlowTrack. It has public code, trains per recording and reports 95.9% on our Leifer recording, so it could be run on the shared NWB benchmark with our metric, next to fDNC.

- [x] ask a261029-2 · on [[EVD - Published trackers run on the shared NWB benchmark]] · blocks [[EVD - Published trackers run on the shared NWB benchmark]] · Run ASCENT on our benchmark next to fDNC, or only cite it as concurrent work?

## 2026-10-29 20:00 — run

r261029-4117569 / r261029-4128796 (fDNC Samuel 153, GT-free rescale): a five-template vote does not help, 0.832 with swap / swap_mirror (the conventions fDNC is most confident in) and 0.48 with the other two; the best single template is 0.835 (swap_mirror, frame 532). Ours is ~0.95. The Zimmer 1128 multi-template runs are still going, writing to ~/.cache/barlow_scratch/fdnc/fdnc_results.jsonl.

## 2026-10-29 20:07 — run

fDNC Zimmer 1128 (r261029-4128796 et al.), more templates: with the GT-free rescale and swap / swap_mirror, templates 333 and 666 give 0.58-0.61; template 10 with swap stays best at 0.615. Without the rescale late templates are much worse (1250: 0.21-0.27, 195 objects, many extra detections; 1660: 0.14-0.16). Ours ~0.97.

## 2026-10-29 20:13 — answer

Not for now: reviewers did not ask for ASCENT. Cite it as concurrent work.

answered a261029-2

## 2026-10-29 20:13 — decision

Competitor scope (user, 2026-10-29): no ASCENT for now, since reviewers did not ask for it. ZephIR needs manual annotation of reference frames. Option the user raised: run the original wbfm tracker on the non-Zimmer recordings. The Flavell (5) and Samuel 153 projects already have step 1 (segmentation); steps 2-3 (SuperGlue tracklets, fDNC-template global matching, the project's snakemake) were never run there. Leifer and Samuel 137 / 184 also lack segmentation metadata (a261029-1).

## 2026-10-29 20:18 — decision

wbfm tracker baseline plan (user: build new projects from the NWB files, don't touch the existing ones). wbfm/scripts/0-create_new_project_from_nwb.py (sacred: nwb_file, project_dir) builds the project, unpacks the videos and masks to zarr, and recalculates metadata from the masks. Pipeline to combine_tracking_and_tracklets: 2a build_frame_objects, 2b match_frame_pairs (SuperGlue), 2c tracklets, 3a track_time_independent (SuperGlue against a template unless use_barlow_network), 3b combine. cluster_config asks for a GPU on 2a, 2b, 2c and 3a, but the code only uses CUDA if it is available (superglue.py:415) and runs on CPU otherwise, so a CPU-only cluster_config should work, only slower. Not yet tested.

## 2026-10-29 20:36 — run

r261029-4126504 (batch 22, visnet_z_scale), seed 43 epoch-14 proxy: Zimmer 1 / 1.55 / 3 / 4.6 = .762 / .778 / .823 / .865; Samuel 1 / 2.9 / 5 = .548 / .848 / .865; Flavell 1 / 2 = .785 / .775 (4 running). Undoing the z squash before ViSNet gains +0.10 on Zimmer and +0.32 on Samuel, far above seed noise (~0.04), still rising at the top of the range; flat on Flavell. Seed 44 still to come. Proxy only, so batch 23 checks the full video.

for [[QUE - Does adding neuron position information improve tracking]]

## 2026-10-29 20:36 — run

Batch 23 (batch23.sh, 3a60): larger visnet_z_scale (Zimmer 6 / 8, Samuel 8 / 12) and full-video accuracy at z_scale 1 vs the batch-22 best (Zimmer 4.6, Samuel 5), seed 43. Waits for batch 22.

run r261029-4158341 · pids 4158341 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch23.log · for [[QUE - Does adding neuron position information improve tracking]]

## 2026-10-29 20:52 — run

wbfm tracker baseline, Samuel 153 (e7c305d run_wbfm_tracker.py, local, CPU only): build a new project from 153.nwb under barlow_track_revisions/wbfm_tracker_baseline/samuel/ (user OK'd the location), run steps 2a-3b with default configs, score against the NWB GT. Also times the CPU run. The Samuel NWB masks hold only the ~97 GT neurons (98 GT), so no distractor objects, the same input our Samuel projects had.

run r261029-4167392 · pids 4167392 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/wbfm_tracker/samuel.log · for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-10-29 20:53 — decision

New sub-project (user, 2026-10-29): re-export the Leifer NeRVE worm1 recording from raw into a new NWB with a segmentation, so Leifer can join the per-recording round and the wbfm-tracker baseline (fixes i261023-2's missing metadata). Raw data found: leifer_data/Nguyen2017NeuronTracking/worm1/ (sCMOS_Frames_U16_1200x600.dat 73 GB, hiResData.mat, alignments.mat, outputFiles/PointsStats*.mat, manualSummary.mat); Leifer's MATLAB code is listed in files.txt inside the 352 GB tar. The existing Leifer_NeRVE_Worm1.nwb (someone else's export, video only) is the working example. A read-only exploration agent is mapping the format now; the real export waits for the next session.

## 2026-10-29 20:54 — run

Build the other seven wbfm-tracker baseline projects from their NWBs, create only (create_all_projects.sh): Samuel 137 / 184, Flavell x5, under barlow_track_revisions/wbfm_tracker_baseline/<label>/. The user runs the tracker on them with snakemake on the cluster. Note: samuel 184-004.nwb is only 0.15 GB for a 3002-frame video, so it may use external links; check its new project's data loads.

run r261029-4169997 · pids 4169997 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/wbfm_tracker/create_all.log · for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-10-29 20:58 — run

r261029-4167392 built the Samuel 153 project (wbfm_tracker_baseline/samuel/153/) but step 2a failed on import: after project creation, pytorch_lightning raises TypeError (pip's vendored typing_extensions got loaded); local env only. Fixed by importing pytorch_lightning first (14b581f). Rerun of the steps, CPU only, on the existing project.

run r261029-4174234 · pids 4174234 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/wbfm_tracker/samuel.log · for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-10-29 21:05 — run

r261029-4174234 failed in 2a after 4m50s: 96 of 120923 objects in the new Samuel 153 metadata are one-voxel, zero-brightness masks whose brightness-weighted centroid is (nan, nan, nan); wbfm's 2D encoder does int(z). Fix (67b1643): bbox-centre centroid for those objects, original kept as metadata_before_centroid_fix.pickle; applied by run_wbfm_tracker.py to every new project. Rerun, CPU only.

run r261029-4180779 · pids 4180779 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/wbfm_tracker/samuel.log · for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-10-29 21:07 — decision

Deferred to tomorrow (user): fix zero-brightness NaN centroids in wbfm itself. utils_metadata.py:76-78 (regionprops weighted_centroid, plain volume) and the props path at :92 (weighted_centroid via regionprops_table, used by recalculate_metadata_from_config) give (nan, nan, nan) when an object's red voxels are all 0; fall back to the geometric centroid. Then drop the barlow_track-side workaround (run_wbfm_tracker.fix_nonfinite_centroids, 67b1643) or keep it only for already-built projects.

- [ ] Fix NaN weighted centroids for zero-brightness objects in wbfm utils_metadata.py
- [ ] Check all eight wbfm_tracker_baseline projects have the centroid fix before telling the user the folder is ready

## 2026-10-29 21:09 — idea

Leifer re-export exploration done (b0c2763, barlow_track/scripts/scratch/leifer_export/README.md): NeRVE saved real masks for every detection (~117 per volume, CLstraight/pointStats*.mat baseImg + transform lookups back to raw), verified against raw intensities; GT is manualSummary.manualPoints, as in the old NWB; red is frame[:, :632].T of the .dat. Plan and risks in the README. Also: the old NWB is chunked 48 volumes deep, uncompressed, likely why Leifer evals take hours.

- [ ] Leifer export: choose the z rule (old 32-plane rule for comparability, or per-volume frames, nz 34)

## 2026-10-29 21:35 — run

r261029-4126504 (batch 22) done, seed 44 and the rest: b22_flavell_zs4.0_s43: 0.680;b22_zimmer_zs1.0_s44: 0.693;b22_zimmer_zs1.55_s44: 0.735;b22_zimmer_zs3.0_s44: 0.784;b22_zimmer_zs4.6_s44: 0.802;b22_samuel_zs1.0_s44: 0.611;b22_samuel_zs2.9_s44: 0.917;b22_samuel_zs5.0_s44: 0.918;b22_flavell_zs1.0_s44: 0.727;b22_flavell_zs2.0_s44: 0.739;b22_flavell_zs4.0_s44: 0.610;

## 2026-10-29 21:35 — run

Batch 22 conclusion (proxy, 2 seeds): visnet_z_scale gains replicate. Zimmer 1 -> 4.6: +0.10 / +0.11 (s43 / s44). Samuel 1 -> 5: +0.32 / +0.31, with 2.9 already almost as good. Flavell: 2 is flat (-0.01 / +0.01) and 4 hurts (-0.10 / -0.12); Flavell's voxels are probably near-isotropic already. Batch 23 (r261029-4158341, now running) checks the full video.

for [[QUE - Does adding neuron position information improve tracking]]

## 2026-10-29 22:01 — run

r261029-4180779 (wbfm tracker, Samuel 153, CPU only, local 56 cores): 2a 317 s, 2b SuperGlue frame pairs 1018 s, 2c 71 s; 3a SuperGlue against the template is running at ~3.4 frames/s (~7 min). So the wbfm tracker runs without a GPU at about 25 min per recording, and the cluster snakemake can use a CPU-only cluster_config. Still running: 3a, 3b, scoring.

for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-10-30 01:08 — decision

Leifer re-export started as a subagent (user, 2026-10-30). Decisions: z rule A first (the old file's 32-plane rule, so images stay pixel-identical and GT-only scores comparable), B later behind a flag; red only; NeRVE's own masks as CalciumSeriesSegmentationUntracked; GT from manualPoints, NeuronCentroids snapped to matched mask centroids (manual clicks kept separately) so eval's 0.5-voxel GT-to-mask mapping works; one volume per gzip chunk. Output: barlow_track_revisions/leifer_reexport/.

## 2026-10-30 01:08 — run

wbfm tracker, Samuel 153 (r261029-4180779), CPU only: accuracy 0.170 (gt_valid matching 0.173), 95 of 98 GT neurons assigned a column. Steps: 2a 317 s, 2b 1018 s, 2c 71 s, 3a 4402 s, 3b 4492 s (2.9 h in total; 3a's first progress bar was not the whole step). 0.17 is far below fDNC's 0.83 on the same recording, so the scoring may be wrong (ID spaces) before it says anything about the tracker; check first. Also: samuel_137 / 184 were built before the centroid fix; apply it.

## 2026-10-30 01:10 — run

Batch 23 so far (r261029-4158341, seed 43 proxy): Zimmer z_scale 6 0.874 (4.6 was 0.865), Samuel 8 0.794 and 12 0.551 (5 was 0.865), so the optimum is near the voxel aspect: Zimmer ~4.6-6, Samuel ~3-5. Zimmer z_scale 8 ran out of GPU memory beside batch 21 (rerun queued after batch 23). Full video, Zimmer z_scale 1: 0.922; the 4.6 eval is running.

## 2026-10-30 01:10 — run

Rerun of b23_zimmer_zs8.0_s43 (OOM), after batch 23 ends.

run r261030-102486 · pids 102486 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/posonly_fast/batch23_retry.log

## 2026-10-30 01:10 — run

Samuel 137 / 184 wbfm-tracker projects got the centroid fix afterwards (137: 600 objects, 184: 0). The Samuel NWB masks keep one label per GT neuron over time (99% of neurons), i.e. they are tracked masks; neither tracker reads mask labels, but keep it in mind for any label-based shortcut.

## 2026-10-30 01:10 — decision

wbfm_tracker_baseline folder ready for the user's snakemake: 8 projects (samuel, samuel_137, samuel_184, flavell x5) with the centroid fix; each snakemake/ has cluster_config_cpu_only.yaml next to the original (GPU gres replaced by the defaults). Samuel 153 already has tracks (0.170).

## 2026-10-30 02:56 — run

Leifer re-export done (subagent, 0314776): Leifer_NeRVE_Worm1_reexport_A.nwb / _B.nwb, validated (validate_A.log: 1496 / 1536 volumes pixel-identical, the rest explained; masks 117.5 per volume, in-mask red 1009 vs 106; 89.9% of GT matched; loads in ProjectData with raw_segmentation). Benchmark label leifer_reexport (b06d486). This answers the Leifer part of a261029-1 and works around i261023-2 (missing metadata). Building its wbfm project now (create only), for the per-recording round and the wbfm-tracker baseline.

run r261030-158650 · pids 158650 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/wbfm_tracker/leifer_reexport_create.log

## 2026-10-30 03:27 — run

r261030-158650: wbfm project from the Leifer re-export built in 31 min (wbfm_tracker_baseline/leifer_reexport/Leifer_NeRVE_Worm1_reexport_A/, metadata recalculated from the NeRVE masks, no NaN centroids), with cluster_config_cpu_only.yaml. The folder now holds 9 projects for the user's snakemake.

## 2026-10-30 03:28 — run

Batch 20 done (r261029-4011812 / r261029-4053272), Zimmer 100 frames, epoch-14 proxy, seeds 43 / 44: image-only lambd_obj 0.67 vs 0 = .617 / .444 vs .482 / .390 (+0.14 / +0.05); fused add 0.67 vs 0 = .687 / .626 vs .692 / .647 (-0.01 / -0.02). So the transpose term speeds up image-only training but not fused (position-dominated) training. Batch 21 (r261029-4059380), Samuel 153 image-only, 30 epochs, proxy at epoch 29: lambd_obj 0.67 vs 0 at 30 frames .559 / .499 vs .384 / .330 (+0.18 / +0.17), at 100 frames .556 / .549 vs .396 / .402 (+0.16 / +0.15). The gap did not close by epoch 29 and 30-frame 0.67 matches 100-frame 0.67: a data-efficiency and speed benefit for image networks, both seeds (goal d). Zimmer 40-epoch full-video cells still running.

for [[EVD - Loss-term ablation where accuracy is not saturated]]

## 2026-10-30 06:14 — decision

eval_accuracy --nwb_gt_mapping segmentation_id (opt-in, default centroid) maps GT to masks through NeuronSegmentationID, for the Leifer re-export v2 where GT positions stay the manual clicks (identical to the old NWB). Export v2 sent back to the agent: green channel registered from S2AHiRes, capped watershed pseudo masks for GT without a NeRVE mask (user: watershed OK, not too big), clicks as NeuronCentroids. User (2026-10-30): the old wbfm tracker is expected to do badly on new data; that is the point of a tracker that generalizes. Weekend round: also lambd_obj 1 (transpose term only).

## 2026-10-30 06:37 — decision

Weekend rounds staged (89508b5, README barlow_track/scripts/scratch/round_wk1_src/README.md): round_wk1_<label> on all 12 GT recordings, fusion {visual_only, add} x lambd_obj {0, 0.67, 1} at 100 frames, 30 epochs, jitter auto, visnet_z_scale = voxel aspect (Zimmer 4.615, Samuel 3.889, samuel_184 5.833 (xy 0.3 in its NWB; check), Leifer 4.65, Flavell 1); 2 seeds on zimmer_1128, samuel, flavell_2022033001. round_wk1f30_<label>: the same at 30 frames, 2 seeds, those 3 recordings. 126 GPU training jobs, evals in CPU jobs. All 12 objectives preflighted; CPU smoke test of the Samuel fused config with lambd_obj 1 trained and scored (z_scale 3.889 reached the model). Leifer trains on the re-export project, scored on the old NWB. Not launched (the user launches).

## 2026-10-30 06:46 — decision

Decision (user, 2026-10-30): train on every mask the data has, including the GT-seeded watershed masks in the Leifer re-export v2, as already happens on Samuel (all masks GT-seeded). So round_wk1_leifer must train on a wbfm project built from the v2 NWB (Leifer_NeRVE_Worm1_reexport_A2.nwb, being built), not the v1 NeRVE-only project; the Leifer round waits for it.

- [ ] When Leifer v2 is validated: point leifer_reexport (run_trials DATASETS) at A2, build its wbfm project, set TRAIN_PROJECTS['leifer'] to it, regenerate round_wk1_leifer, update the round README

## 2026-10-30 07:03 — decision

Decision (user, 2026-10-30): the Leifer re-export only adds segmentation and green; its GT must be the old NWB's exactly. Told the export agent: A2 keeps the old GT (drops the 208 extra clicks, stores volume 1535 z-flipped as the old file does, with its GT); only the red-line repair in 39 volumes remains as an image difference. v1 (A / B) snapped matched GT to mask centroids (median 0.49 um), so its centroids differ; A2 replaces it.

## 2026-10-30 07:11 — decision

Correction (user, 2026-10-30): keep the full GT and the correct data; the old NWB is a guide. A2 keeps the 208 manual clicks in volumes 78, 405, 945, 1516 (in manualSummary.manualPoints; the old export dropped them, the volumes where NeRVE blanked its detections) and stores volume 1535 un-flipped; GT positions are the raw clicks. Supersedes the 07:03 instruction.

## 2026-10-30 07:30 — decision

Sweep objective can now score NWB recordings on every detection (objective_nwb_objects: all, objective_nwb_gt_mapping; ba..., defaults unchanged). The Leifer weekend round will train on the v2 re-export project and score on the v2 NWB with all detections (~117 NeRVE masks per volume, ~40 of them non-GT, plus pseudo masks for missed GT) and segmentation_id mapping (user: use the non-GT detections for training and testing). Open: Flavell has untracked masks too (~180 per frame vs 163 GT) but the staged Flavell rounds score GT objects only.

## 2026-10-30 08:22 — run

Leifer re-export A2 done and validated (subagent, d4379ef; validate_A2.log checked): 26.5 GB, red + green (NeRVE's projective fit of S2AHiRes, corr 0.70 vs <= 0.19 for flips / 0.57 for a 1-frame offset; in-mask green 2.28x), NeRVE masks plus 12,019 capped watershed masks (median 94 voxels vs NeRVE 113), full GT as raw clicks (old points identical except volume 1535's pure z flip; +208 clicks), every GT point has a NeuronSegmentationID. 1671 unmatched clicks sit inside a NeRVE mask matched to another GT neuron (probably merged masks). leifer_reexport now points at A2 (f7c7151); building its wbfm project; v1 project moved to wbfm_tracker_baseline/leifer_reexport_v1_nerve_only.

run r261030-309374 · pids 309374 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/wbfm_tracker/leifer_reexport_A2_create.log

## 2026-10-30 08:32 — decision

Decision (user, 2026-10-30): test (and train) under the conditions of a new recording without GT, i.e. with every detection as distractors. Flavell training projects already hold every detection (152-177 objects per frame vs 136-173 GT neurons; NWB untracked masks 149-182); the staged Flavell objectives scored GT objects only, now nwb_objects all (make_per_recording_rounds OBJECTIVE_EXTRA). Scores on these cells are not comparable to earlier GT-object Flavell cells; distractors cannot be scored beyond their effect on GT neurons (misses / mismatches).

## 2026-10-30 08:56 — run

r261030-309374: Leifer A2 wbfm project built (metadata.pickle written, centroid fix applied; cluster_config_cpu_only.yaml added). round_wk1_leifer_reexport can launch (41bec69).

## 2026-10-30 09:40 — decision

User launched the weekend rounds (round_wk1_*, round_wk1f30_*) on 2026-10-30. Speed / robustness plan agreed for next week (the user): DataLoader workers only with tests of reproducibility; mixed precision deferred; fewer epochs, benchmarking intermediate checkpoints; label propagation uses a full CPU per seed, so try it on GPU (--track_device cuda exists; it OOMed next to the trained model on 12 GB) rather than parallel seeds; training resume from checkpoints (the wbfm snakemake steps already restart).

- [ ] Speed: DataLoader num_workers > 0 for crop training, with a test that augmented batches are identical to num_workers 0 for a fixed seed
- [ ] Speed: save per-epoch checkpoints and score several epochs of one trial, to pick shorter schedules
- [x] Speed: time full-video label propagation on GPU (--track_device cuda) in its own job vs CPU
- [ ] Robustness: resume training from the last checkpoint on restart / Slurm requeue, with a test

## 2026-10-30 09:55 — run

Timing matrix: fused (add) and keypoints-only training, GPU vs 16 CPU threads, 20 frames x 1 epoch on samuel, zimmer_1128, samuel_137, flavell_2022033001, leifer_reexport, plus each objective eval on 16 CPU threads; to decide which jobs go to CPU nodes

run r261030-356454 · pids 356454 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/timing_probe/matrix.log

## 2026-10-30 10:02 — run

Timing matrix rerun with the data phase split (8ae97da): fused and keypoints-only, GPU vs 16 CPU threads, 5 recordings, plus objective evals on 16 CPU threads. First result, Samuel fused GPU 16 steps: global augmentation 4.0 s of 6.1 s data, crop augment 1.4 s, volume read 0.4 s, forward+backward 3.4 s; the scipy full-volume warp outweighs the GPU compute

run r261030-589527 · pids 589527 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/timing_probe/matrix.log

## 2026-10-30 11:01 — result

r261030-589527 partial (CPU fused zimmer eval, samuel_137/flavell/leifer CPU fused still running; GPU fused OOM'd next to the other job except Samuel/137). Per step: posonly fwd+bwd CPU16 1.0-1.7 s vs GPU 0.13-0.17 s (~8-10x); 30 epochs x 100 frames posonly on CPU16 ~1-1.5 h. Fused CPU16 backward 29-32 s/step (forward 2 s) vs GPU fwd+bwd 0.2 s on Samuel (~150x); Zimmer fused CPU16 ~38 s/step. GPU fused Samuel step: data 0.45 s (global augment 0.30) vs fwd+bwd 0.20 s. Samuel fused CPU16 eval: embed 989 s, labelprop 1027 s, 34 min.

## 2026-10-30 11:35 — run

Scaling probe (2243786 + run_scaling.sh): fused GPU Samuel with 4 DataLoader workers vs 0, and position-only CPU at 4/8/32 threads on Zimmer and Leifer. Before it: worker reads fixed (zarr ZipStore reopened per worker) and checked identical on 5 recordings

run r261030-1366476 · pids 1366476 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/timing_probe/scaling.log

## 2026-10-30 11:55 — result

r261030-1366476 (scaling): position-only CPU per step fwd+bwd, Zimmer 4/8/16/28 threads 3.3/2.7/1.6/1.9 s, Leifer 2.3/1.3/1.2/1.0 s: saturates at 8-16 threads (28 shared cores with the GPU job's lanes). Fused GPU Samuel with 4 DataLoader workers: data/step 0.42 -> 0.06 s, epoch 22.4 -> 18.9 s at 20 frames (fixed per-epoch proxy 9 s dominates); ~1.9x per epoch projected at 100 frames.

## 2026-10-30 12:11 — result

Zimmer 1128 data loading, 32 items (one frame each, two views), cores 16-47: 0 workers 105-136 s (3.3-4.3 s/item), 2 workers 69 s, 8 workers 19 s, 16 workers 12 s, batches identical in every case. Workers parallelize across items, so they scale nearly linearly up to 16. Also corrected: Samuel 153 is the easy Samuel recording; 137 / 184 are the hardest (0.47 / 0.43 for the paper-ablation image-only network).

## 2026-10-30 13:21 — decision

Auto DataLoader workers for Slurm GPU training jobs (aab23eb), so the weekend jobs that start after the user pulls get them: num_workers 0 -> SLURM_CPUS_PER_TASK - 2 (6 on the weekend's 8 CPUs) for zarr-backed projects, probe batch with fallback to 0, global torch RNG restored. Samuel GPU: step-0 loss bit-identical to num_workers 0, val loss within GPU run-to-run noise (0.0025343 vs 0.0025359 / 0.0025351). Rounds become mixed in speed only, not in results. Template key train_gpu_job_cpus (default 8) for future rounds.

## 2026-10-30 13:54 — result

round_wk1 check: no training failures (all finished trials trained 30 epochs, 0.2-1.6 h), but every CPU objective eval on an NWB label without a LABS entry (samuel_137, samuel_184, five Flavell, leifer_reexport, wk1f30_flavell) died at start in eval_accuracy._check_args ('needs an explicit frame source'); ~12 trials so far marked FAILED in Ax. Samuel 153 rounds score (0.916 / 0.919 / f30 0.933). Fixed in 3e695bb, verified end to end on 30 frames of samuel_137, flavell_2022033001, leifer_reexport. Recovery: the user pulls (eval jobs import the code at start, so later evals succeed); after each round's dispatcher ends, rerun it with --skip_completed_grid_jobs to submit eval-only jobs for the trained-but-unscored trials. Not a second dispatcher while one runs (it would retrain unfinished trials).

## 2026-10-30 14:26 — decision

Staged round_snap1 (b6925cf, README round_snap1_src/README.md): epoch_snapshots [1, 3, 10, 20] + final 30, scored per snapshot in the CPU eval job (objective_snapshots), fusion {visual_only, add} x lambd_obj {0, 0.67} on zimmer_1128, samuel_137, flavell_2022033001, leifer_reexport, 16 GPU jobs, train_gpu_job_cpus 16. Asks: fewer-epoch training (speed goal b) and whether the transpose loss trains faster (goal d). Not launched: the user launches it.

## 2026-10-30 16:17 — run

Cluster epoch times with DataLoader workers (epoch_times_from_runs.py over round_wk1*/round_snap1_*, 53 finished 30-epoch x 100-frame jobs; s per epoch = training + val + proxy). 14 workers: Samuel 13-18 (A30), Zimmer 38-39 (A30) / 56 (L4), Leifer 35-45 (A30/L4) / 59-73 (T4), Flavell 62-70 (L4) / 93-104 (T4); a 30-epoch job takes 0.1-0.15 h (Samuel), 0.3-0.5 h (Zimmer, Leifer), 0.5-0.9 h (Flavell). Same GPU, 0 vs 6 vs 14 workers: Leifer L4 167-177 / 64 / 45, Samuel A30 27 / 16 / 13-18, Flavell L4 78-103 / 48-66 / 62-70. Epoch time now follows GPU type (A30 < L4 < T4), so jobs are compute-bound: serial augmentation per item (local, 0 workers) is Samuel 0.4 s, Flavell 1.0 s, Zimmer 3.4 s, i.e. 80 items / 14 workers = 2-19 s per epoch, under the GPU time. GPU-side warping is not worth it now (at most Zimmer's few seconds); the per-epoch matching proxy (local GPU 6-10 s on Samuel, roughly half its epoch) is the larger fixed cost. round_wk1_samuel_137's dispatcher stopped at 13:59 (3 eval failures in a row, the fixed NWB-label bug): needs the --skip_completed_grid_jobs rerun. round_wk1 Zimmer trials still pending in the queue after 6.5 h with the same Slurm request as round_snap1's running Zimmer jobs. First snapshot records (samuel_137 trial 0, visual_only lo 0): epoch 1 0.885, 3 0.883, 10 0.893, 20 0.877, 30 0.852.

## 2026-10-30 16:30 — run

Smoke check for round_bayes_wk2: the top of each planned augmentation range (scale_jitter 0.25, rotation 180, crop rotation 180, crop noise p 0.5, keypoint dropout 0.3, spurious detections 0.3; elastic_scale 0 vs 1.5 at std (1,4,4), length 40), fused, local GPU, 8 workers, 10 frames x 2 epochs, on samuel_137 / zimmer_1128 / leifer_reexport / flavell_2022033001: does it run, and what elastic costs per item

run r261030-2581777 · pids 2581777 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/augmax.log

## 2026-10-30 16:44 — run

round_bayes_wk2 launched by the user (2026-10-30 16:42, code 55cc4a3): one Ax search per recording (zimmer_1128, samuel_137, flavell_2022033001, leifer_reexport), --total_budget 120 each, 13 parameters (fusion, self_layers, lambd_obj, lr, epochs 2-30, num_frames 30-500, 7 GT-free augmentations). All four dispatchers submitted their first 10 trials. Weekend run that sets the space for the Tue-Thu 3-day run. Smoke check r261030-2581777: samuel_137 and zimmer_1128 run at the top of every augmentation range, with and without elastic; leifer and flavell pending.

## 2026-10-30 16:58 — run

r261030-2581777: all 8 cells ran (4 recordings x elastic 0 / 1.5, every other augmentation at the top of its round_bayes_wk2 range). Per epoch of 8 steps, 8 workers on 10 local cores, data wait elastic 0 -> 1.5: Zimmer 8.3 -> 19.4 s, Leifer 7.5 -> 18.8 s, Flavell 6.4 -> 7.3 s, Samuel 3.0 -> 3.5 s, against fwd+bwd 1-3.6 s. So strong elastic on the large volumes (Zimmer, Leifer) makes training data-bound again, maybe 2-3x slower trials; that is where GPU-side warping would pay off, if Ax favours elastic. The matching proxy is 8-28 s per epoch (Zimmer 28 s), the largest fixed cost.

## 2026-11-02 08:17 — run

round_bayes_wk2 status 2026-11-02: flavell/leifer/samuel dispatchers all stopped silently at 2026-11-01 04:42 (no traceback, simultaneous: host/session ended) after 106/90/108 trials; their in-flight training jobs finished but the CPU evals were never submitted (12/10/10 trained-unscored). zimmer dispatcher died 2026-10-30 20:35 on 'sbatch: Unable to contact slurm controller' after 18 trials (8 scored, 9 trained-unscored, 1 no ckpt). Only real failures: flavell trials 28 (all embeddings NaN) and 48 (1468 NaN rows). Best acc: flavell .985, leifer .885 (median .878, flat), samuel .931, zimmer .980 (Sobol only). Top-5 everywhere fusion=add, lambd_obj ~0.4-0.7.

- [ ] restart round_bayes_wk2 dispatchers (budgets flavell 14, leifer 30, samuel 12, zimmer 102) to score the trained-unscored trials
- [ ] dispatcher: retry sbatch on transient slurm controller errors instead of dying

## 2026-11-02 08:25 — run

Leifer deep dive (user, 2026-11-02: round_bayes_wk2 Leifer flat at .878-.885 is too low even with distractors): re-track round_bayes_wk2_leifer_reexport trial 57's cached embeddings on CPU (37cddd2 decompose_errors.py) with every detection / GT masks only / NeRVE masks only, and break the errors down by neuron, frame, mask type and wrong-label kind

run r261102-3049660 · pids 3049660, 3049661, 3049662 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/leifer_deep_dive/trial57/all.log, /home/charles/.cache/barlow_scratch/leifer_deep_dive/trial57/gt.log, /home/charles/.cache/barlow_scratch/leifer_deep_dive/trial57/nerve.log

## 2026-11-02 08:29 — run

Leifer deep dive: per-mask geometry of the A2 re-export (3cb590e mask_geometry.py), to test whether distractors near GT rows in embedding space are fragments or duplicates of GT neurons. trial 57 kNN (SVD 50, 6000 GT queries, 10 NN from other frames): same neuron 0.882, distractor 0.094, other GT 0.024; 37% of detections (70359 / 189378) are non-GT.

run r261102-3052801 · pids 3052801 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/leifer_deep_dive/masks.log

## 2026-11-02 12:36 — dropped

killed by the session crash at 2-4 of 25 seeds; relaunched

dropped r261102-3049660

## 2026-11-02 12:36 — dropped

killed by the session crash before output; relaunched

dropped r261102-3052801

## 2026-11-02 12:36 — run

Relaunch (setsid) of the Leifer deep dive after a session crash: mask_geometry.py and decompose_errors.py all/gt/nerve on trial 57, same commit 3cb590e

run r261102-3155203 · pids 3155203, 3155204, 3155205, 3155206 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/leifer_deep_dive/masks.log, /home/charles/.cache/barlow_scratch/leifer_deep_dive/trial57/all.log, /home/charles/.cache/barlow_scratch/leifer_deep_dive/trial57/gt.log, /home/charles/.cache/barlow_scratch/leifer_deep_dive/trial57/nerve.log

## 2026-11-02 12:38 — run

Leifer deep dive (user): are the 4 NeRVE-blank volumes with the 208 extra clicks corrupted? blank_volumes.py at 7475607

run r261102-3156626 · pids 3156626 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/leifer_deep_dive/blank.log

## 2026-11-02 13:34 — dropped

killed with the terminal scope by systemd-oomd at 12:40 (user-slice memory pressure from 3 parallel labelprops + 16-worker geometry)

dropped r261102-3155203

## 2026-11-02 13:34 — dropped

same oomd kill at 12:40

dropped r261102-3156626

## 2026-11-02 13:34 — dead-end

Both session crashes today (08:35, 12:40) were systemd-oomd killing the terminal scope: my parallel Leifer deep-dive runs pushed user-slice memory pressure past 50%; also took Firefox. Rerun sequentially with ulimit -v caps (run_all.sh, 89b284b), outputs in the session scratchpad (/tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive; ~/.cache is read-only in this session). Lesson: one labelprop at a time locally (each Leifer eval is budgeted 64 GB on the cluster).

run r261102-2490 · pids 2490 · host charles-Precision-7820-Tower · logs /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/driver.log

## 2026-11-02 13:34

r261102-2490 runs run_all.sh as of ec2c82a (89b284b plus the env-var fallback only)

## 2026-11-02 13:40 — todo

Real cause of today's OOMs: wbfm load_final_project_data_from_nwb does da.squeeze(h5py dataset) on CalciumSeriesSegmentation(Untracked), which reads the whole array (35 GB for Leifer A2) on every ProjectData load; 16 geometry workers each loaded it. Same 35 GB at the start of every cluster Leifer eval (64 GB jobs). Fix belongs in wbfm (wrap in da.from_array before squeeze; wbfm repo read-only this session); deep-dive scripts patch it in-process (bf5bfd3 lazy_nwb.py). Blank-volume and geometry steps rerun with it (40 GB cap) next to decompose_all.

- [ ] wbfm: wrap the NWB segmentation datasets in da.from_array before da.squeeze in load_final_project_data_from_nwb (finished_project_data.py ~943, ~951)

run r261102-3482 · pids 3482 · host charles-Precision-7820-Tower · logs /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/blank.log, /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/masks.log

- [ ] issue i261102-1

## 2026-11-02 13:41 — dropped

lazy_nwb tripped on wbfm's da.from_array of a dask array; fixed

dropped r261102-3482

## 2026-11-02 13:41 — run

Rerun blank-volume check and mask geometry with the fixed lazy loader (b907dc0; NWB loads in 8 s)

run r261102-4055 · pids 4055 · host charles-Precision-7820-Tower · logs /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/blank.log, /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/masks.log

## 2026-11-02 13:41

r261102-2490: decompose_nerve and decompose_gt steps will run decompose_errors.py at 9505316 (lazy NWB load only; tracking and scoring unchanged)

## 2026-11-02 13:44 — run

r261102-4055 (blank_volumes.py + mask_geometry.py at 75b… lazy loader): (1) The 4 NeRVE-blank volumes 78/405/945/1516 are corrupted: mean red ~2020 vs ~110 in neighbours, p99 at saturation (63-65k), one plane at mean 31-33k in 405/945, correlation with the previous volume ~0 (neighbours 0.3-0.7). Clicks there (39-75 of 79) sit on bright voxels, but these are bad frames, not a detector failure; candidates to drop. (2) Non-GT masks (70359, 46/frame) are mostly separate cells, not duplicates: nearest GT mask centroid median 3.74 um (GT-GT nearest 4.59), <2 um only 4.2% (GT-GT 2.4%), <3 um 29.5% (GT-GT 11.7%); smaller (median 78 vs 133 voxels) and deeper in z (median plane 18.8 vs 15.7): plausibly the unannotated neurons. 2863 non-GT NeRVE masks contain a GT click (the merged / offset cases). Pseudo masks: only 6.8% have a non-GT NeRVE mask within 3 um, so the GT-seeded masks rarely duplicate a NeRVE detection.

## 2026-11-02 14:14 — dropped

decompose_all killed at seed 22/25 in the 14:12 terminal crash (blank and masks steps had finished)

dropped r261102-2490

## 2026-11-02 14:14 — run

Relaunch decompose_errors.py all/nerve/gt (9505316) on trial 57 sequentially in its own systemd user unit leifer-decompose (MemoryMax 120G), so a memory kill stays inside the unit and a terminal crash does not take it

run r261102-3202160 · pids 3202160 · host charles-Precision-7820-Tower · logs /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/decompose_all.log, /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/decompose_nerve.log, /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/decompose_gt.log

## 2026-11-02 14:17 — dropped

restarted as three parallel units: one labelprop uses ~8 cores (serial main loop) and 5 GB, so the subsets can run side by side

dropped r261102-3202160

## 2026-11-02 14:17 — run

decompose_errors.py all/nerve/gt (9505316) on trial 57 in parallel, units leifer-decompose-{all,nerve,gt}, MemoryMax 40G each, 12 threads

run r261102-3204514 · pids 3204514, 3204522, 3204525 · host charles-Precision-7820-Tower · logs /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/decompose_all.log, /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/decompose_nerve.log, /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/decompose_gt.log

## 2026-11-02 14:25 — decision

round_bayes_wk2 per-parameter read (80-98 scored trials; Zimmer only 8): Leifer is insensitive to everything: Sobol-phase std 0.036 (Samuel 0.095, Flavell 0.166), visual_only mean 0.857 vs add 0.873, round_snap1 epoch-1 snapshot already 0.84; top-10 sit at the top of scale_jitter (0.25), crop_degrees_z (~160-180) and epochs (30), nothing else at an edge. Samuel 137 (the hardest Samuel recording; earlier best 0.662, image-only 0.468) reaches 0.931; its top 10 all self_layers 1, and epochs and num_frames at the upper bounds (30, 500). Everywhere: add fusion, lambd_obj ~0.4-0.5 at the top, low global rotation (max_degrees_z top-10 median 0.06-0.14 of range).

## 2026-11-02 14:33 — run

Generate round_bayes_wk3_<label> (round_bayes_wk2's search, same grid, on the 8 other GT recordings: zimmer_1123/1210, flavell (2025_07_01), flavell 0316-03/0324-02/0330-02, samuel (153), samuel_184) with --preflight, in unit wk3-preflight

run r261102-3214879 · pids 3214879 · host charles-Precision-7820-Tower · logs /tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/wk3_gen.log

## 2026-11-02 15:04 — run

r261102-3204514 (trial 57, cached embeddings re-tracked as the objective): every detection 0.880 (reproduces 0.885 within seed noise); GT masks only 0.9665 (0.976 on NeRVE-mask GT, 0.881 on pseudo-mask GT); NeRVE masks only 0.884. So the non-GT detections cost ~9 points: 10700 of 12113 mismatches go to a non-GT mask, only 1413 to another GT neuron. Errors concentrate: worst 10 neurons hold 41% (neuron_013/076/044/049/032/024 at 0.48-0.69 with all detections, 0.95-0.99 GT-only). The stolen-by masks sit next to the true one: predicted mask to true GT mask centroid median 2.84 um (59% < 3 um; GT-GT nearest 4.59), median 93 voxels, 11% hold a GT click; errors flicker (runs median 1-2 frames, not identity swaps). The 26 frames without NeRVE masks score 0.43 (frames 1524+ 0.05). Pseudo masks are not the problem (removing them changes nothing). Next: are the stealing masks splits of the same cell (z-adjacent) or touching neighbours.

## 2026-11-02 15:04

r261102-3214879: round_bayes_wk3 generated for 8 recordings, all preflights OK (c806e42)

## 2026-11-02 15:05

Leifer stealing masks (the 10700 mismatches onto non-GT masks, trial 57): offset mostly in z, |dz| median 2.18 um (1.45 planes) vs dxy 1.55 um; 61% more z than xy, 30% within 1 um in xy (stacked above/below the GT mask); baseline non-GT masks within 4 um of GT: 41% mostly-z, |dz| 1.56 / dxy 2.09. Points to NeRVE z-splits of single cells. In HANDOFF.md (leifer_deep_dive).

## 2026-11-02 15:12 — decision

Decision (user, 2026-11-02): Leifer needs work on the dataset (z-split NeRVE masks) before more sweeps; the Tue-Fri launch covers the other recordings only (launch_tue_fri.sh leaves leifer_reexport out).

## 2026-11-02 15:46 — decision

Decision (user, 2026-11-02): sweep round configs live in the cluster's config_files/<round>/, not in the repo. round_bayes_wk3_* and round_bayes_wk3_src (grid, base, README, launch_tue_fri.sh) copied to config_files/ (identical, checked with diff) and removed from git (555a125). Launch: bash $B/round_bayes_wk3_src/launch_tue_fri.sh after git pull. 45 older round_* folders still sit in barlow_track/scripts/scratch/.

## 2026-11-02 17:25 — run

wk3 launch (17:21): Flavell x4 and Samuel x2 submitted 6 trials each; wk2 zimmer_1128/samuel_137 score-only restarts submitted their CPU evals. round_bayes_wk3_zimmer_1123/1210 dispatchers (7781791/2) OOM-killed at 32G during the objective preflight (_load_project_gt -> ProjectData with NWB, likely i261102-1); no trials created, safe to resubmit with sbatch --mem=96G. First launch at 15:48 failed only because the launch shell lacked the conda env (user relaunched).

## 2026-11-02 18:01 — decision

best_of_table dropped every per-recording sweep record (lab samuel_137 / flavell_2022* / leifer_reexport -> 'unknown lab'), so Samuel 137 showed <0.9 instead of 0.931 (round_bayes_wk2_samuel_137 trial 63). Also, Flavell nwb_objects=all records had been filed under the '(GT objects)' rows. Fixed in 61efb24: lab matched by prefix, every-detection records in their own rows.

## 2026-11-02 18:04 — decision

best_of_table split (fe39543): main table = every-detection / project-segmentation scores; GT-object rows move to a second table when the recording has an every-detection score. Found while doing it: round_bayes_wk3_samuel / _samuel_184 and wk2_samuel_137 score GT objects only (no objective_nwb_objects: all), although thresholded_segmentation NWBs exist for 137/153/184/185 and the one every-detection Samuel 153 score is 0.898 (sc_bayes0_153_nwball) vs 0.971 GT-only. Asked the user whether to switch Samuel to every detection.

## 2026-11-02 18:05 — decision

Correction to 18:04: no Samuel scoring change needed. Every Samuel NWB segmentation (plain and thresholded) is GT-only (checked 2026-10-28, user agreed); the 0.898 nwb_objects=all on 153 drops 76% of GT. best_of_table excludes those records again (Samuel stays GT objects in the main table).

## 2026-11-02 18:16 — run

wbfm opt-in lazy NWB segmentation (lazy_segmentation arg / WBFM_NWB_LAZY_SEGMENTATION, uncommitted on wbfm dev at 20ed314e): load check identical sampled-volume hashes eager vs lazy, peak RSS Samuel 137 3.56->0.30 GB, Flavell 7.22->0.33 GB, Leifer A2 lazy 0.44 GB (eager ~35 GB). End-to-end check: eval_accuracy trained wk2 flavell_2022033001 trial 83, nwb_objects all, 150 frames, 5 seeds, lazy then eager; accuracies must match

run r261102-3394883 · pids 3394883 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/53a39840-2ccb-4fa4-8a78-2b0160b71611/scratchpad/lazy/e2e_driver.log

## 2026-11-02 18:18 — run

Leifer z-split test, stage 1 (zsplit_merge.py pairs at 3287a70, lazy wbfm NWB segmentation): every vertically touching mask pair per frame of A2 with contact fraction; no GT used

run r261102-3398854 · pids 3398854 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/53a39840-2ccb-4fa4-8a78-2b0160b71611/scratchpad/zsplit/pairs.log

## 2026-11-02 18:20 — run

Leifer z-split test, stage 2 (zsplit_merge.py track at 3287a70): trial 57 cached embeddings, merge vertically touching masks with contact_frac >= 0.3 / 0.5 (largest-mask row) and 0.3 (mean row), GT mapped through the merge, re-track 25 seeds and score as the objective (baseline 0.880, GT-only 0.9665). Pairs stage: 53396 touching pairs, contact_frac median 0.36

run r261102-3403676 · pids 3403676, 3403749, 3403774 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/53a39840-2ccb-4fa4-8a78-2b0160b71611/scratchpad/zsplit/track_0.3_largest.log, /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/53a39840-2ccb-4fa4-8a78-2b0160b71611/scratchpad/zsplit/track_0.5_largest.log, /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/53a39840-2ccb-4fa4-8a78-2b0160b71611/scratchpad/zsplit/track_0.3_mean.log

## 2026-11-02 18:24 — decision

Lazy NWB segmentation works (r261102-3394883): eval_accuracy trained wk2 flavell_2022033001 trial 83, nwb_objects all, 150 frames, 5 seeds gives identical results lazy vs eager (accuracy 0.947694, misses 384, mismatches 739); lazy 1:36 wall / 2.6 GB peak RSS, eager 2:32 / 16.3 GB. Load alone: sampled-volume hashes identical; Samuel 137 3.56->0.30 GB, Flavell 7.22->0.33 GB, Leifer A2 0.44 GB lazy (eager ~35 GB). The Dec 2024 napari segfault (wbfm 2f409136) came from lazy image data, which has stayed lazy since; the segmentation went eager only by accident in a4819064 (Aug 2025, da.squeeze on the h5py dataset). wbfm change (lazy_segmentation arg, env WBFM_NWB_LAZY_SEGMENTATION, default off) is uncommitted on wbfm dev; barlow_track c77ab22 sets the env default to 1 in eval_accuracy.py and train_barlow_clusterer.py. Fixes i261102-1 once wbfm is pushed and pulled on the cluster.

## 2026-11-02 18:28 — run

Samuel 137 one-change pilot (s137_pilot/run_pilot.py at 1b2c519) from round_bayes_wk2_samuel_137 trial 63 (0.931): base (same seed; step-0 loss matches the cluster run), base_seed2, lambd_obj 0 (no object term; the wk2 sweeps never sampled < 0.1), lambd_obj 1, reaugment_each_epoch true. Trained locally on the GPU in two streams, scored as the objective (nwb GT objects, CPU embedding and labelprop, 25 seeds). For [[QUE - Does the object-space loss term improve tracking in any regime]] and a261027-1. Existing-data read (sweep_param_effect.py at 62cd4c8): wk2 partial dependence of accuracy on lambd_obj over 0.1-1 is flat on Flavell 0330-01 (0.911-0.914), Leifer (0.868-0.871) and Samuel 137 (0.859-0.863); importance rank 5-8 of 13; top-10 trials sit at 0.35-0.8

run r261102-3413234 · pids 3413234, 4109000 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_a.log, /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_b.log

## 2026-11-02 18:29

r261029-4169997 (create_all_projects.sh): all seven wbfm-tracker baseline projects were created without errors (Samuel 137, 184-004, Flavell 0331-01, 0316-03, 0324-02, 0330-01, 0330-02 under barlow_track_revisions/wbfm_tracker_baseline/<label>/); the Flavell five needed 0 centroid fixes, and the two Samuel projects print no centroid-fix line (to check against the open to-do 'all eight have the centroid fix'). These are the projects the user runs the wbfm tracker on with snakemake; no tracker scores yet in this log.

## 2026-11-02 18:29

r261030-356454 (first timing matrix) was superseded by its rerun r261030-589527 with the data phase split (same log, matrix.log); results are in the 2026-10-30 11:01 entry. Its own failures (gpu leifer add, cpu samuel_137/leifer add+eval exit 1) were the GPU OOM next to the other job and the evals noted there.

## 2026-11-02 18:29

Correction to the 18:29 r261030-356454 entry: the log does not say why its runs failed (GPU fused train exit 1 on flavell/leifer/zimmer, CPU samuel_137/leifer evals exit 1; the tracebacks are only the subprocess wrappers). The OOM cause in the 11:01 entry is for the rerun r261030-589527; do not read it as this run's cause.

## 2026-11-02 18:30 — dropped

restarted after 10 min with num_workers 4 (data-bound at 0: 220 s/100 steps vs 50 s on the cluster) in three streams; partial dirs removed

dropped r261102-3413234

## 2026-11-02 18:30 — run

Samuel 137 pilot relaunched (run_pilot.py at 6e68a5e, num_workers 4 for every variant; the user freed the GPU for the coming days): streams a base+reaug, b lambd0+lambd1, c base_seed2. Same design as the dropped r261102-3413234

run r261102-365990 · pids 365990, 366025, 366044 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_a.log, /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_b.log, /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_c.log

## 2026-11-02 19:02 — result

Leifer z-split merge (r261102-3403676, zsplit_merge.py 00ac3c0; trial 57 cached embeddings, GT-free merge of vertically touching masks): on the cells both score, accuracy 0.894 -> 0.938 (contact_frac >= 0.3, largest piece's row), 0.942 (0.3, mean row), 0.890 -> 0.928 (0.5). Headline vs the 0.880 baseline overstates: merging drops GT cells where two GT neurons end in one group (0.3: 699 groups, 8404 cells, baseline 0.70 on them; 0.5: 247 groups, 4846 cells), so compare on the same cells. Reading: the gain (~4.5 pt) matches the share of baseline mismatches whose stealing mask gets merged into the GT mask's group (6453 of 119019 cells = 5.4 pt at 0.3), so it is mostly the score accepting either piece, not better tracking. If the stacked pieces are one cell, the segmentation (not the network) costs ~5 of the ~9 points; if they are touching neighbours, the merge is hiding errors. Needs a look at a few in napari (e.g. neuron_044) and a guard against merging two annotated neurons before any re-segmentation. GT-only remains 0.9665.

## 2026-11-02 20:13 — run

Sweep state 19:54 (user away until ~Fri): wk3 Flavell x4 and Samuel x2 running cleanly (no failed trials), zimmer_1123 dispatcher resubmitted at 96G (7781913) and submitting; zimmer_1210 resubmit and most wk1/wk1f30 grid restarts still queued (dispatcher jobs start one at a time). Monitoring stopped at the user's request.

- [x] On return: check zimmer_1210 and wk1 grid dispatchers started; tally wk3 per-recording bests with best_of_table.py
- [ ] Decide wrapper default memory (96G) or cheap preflight for Zimmer projects (i261102-1)

## 2026-11-02 21:07 — result

Samuel 137 pilot, first two scores (r261102-365990, CPU objective as the sweep): base (trial 63 config and seed) 0.920, base_seed2 0.884. Trial 63 scored 0.931 in the sweep. A 3.6 pt spread between two retrains of one config is larger than any lambd_obj effect the wk2 partial dependence shows (0.004), so single runs per variant cannot answer QUE object-space loss; and trial 63's 0.931 is the best of ~80 noisy draws, so expect regression to the mean in any retrain. Added seed-paired variants (0869829) and a 4th stream.

## 2026-11-02 21:07 — run

Samuel 137 pilot stream d: base_seed3, lambd0_seed2, reaug_seed2 (run_pilot.py 0869829; seed-2 variants pair with base_seed2)

run r261102-2656039 · pids 2656039 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_d.log

## 2026-11-02 21:08 — result

Samuel 137 pilot (r261102-365990): lambd0 (lambd_obj 0, trial 63's seed) 0.893 vs base at the same seed 0.920: -2.7 pt, inside the 3.6 pt seed spread (base_seed2 0.884). Not conclusive; lambd0_seed2 (r261102-2656039) pairs with base_seed2.

## 2026-11-02 21:20 — run

Labelprop seed noise on Samuel 137 (user asked whether more labelprop seeds would shrink the 3.6 pt base vs base_seed2 spread): re-track the cached embeddings of base and base_seed2 at labelprop seed 1 and 2 (25 seeds) and seed 0 with 100 seeds (s137_pilot/lp_seeds.py 32ce9cf, CPU only, CUDA hidden: the first launch OOM'd loading the checkpoint onto the full GPU)

run r261102-77194 · pids 77194, 79170 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/53a39840-2ccb-4fa4-8a78-2b0160b71611/scratchpad/lpseeds/base.log, /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/53a39840-2ccb-4fa4-8a78-2b0160b71611/scratchpad/lpseeds/base_seed2.log

## 2026-11-02 22:55 — result

Labelprop seed noise on Samuel 137 (r261102-77194, lp_seeds.py 32ce9cf, cached embeddings). base: lp seed 0/1/2 at 25 seeds 0.920/0.927/0.927, 100 seeds 0.928. base_seed2: 0.884/0.878/0.876, 100 seeds 0.884. Labelprop seed spread within a network ~0.8 pt; 100 seeds lands at or above the 25-seed mean (+0.3, +0.5 pt) for 4x the time (64 vs 16 min). The two training seeds differ by 4.4 pt at 100 seeds: the spread is training-seed variance, not labelprop. The local retrain of trial 63 at 100 seeds (0.928) reproduces its sweep score (0.931); base's original 0.920 was a low labelprop draw.

## 2026-11-03 00:02 — result

Samuel 137 pilot (r261102-2656039): base_seed3 0.872. Trial 63's config over three training seeds: 0.920 (its own seed; 0.928 at 100 labelprop seeds), 0.884, 0.872; mean 0.892, sd ~2.5 pt. Trial 63's 0.931 was largely a lucky training seed: the sweep's best trial overstates its config by ~3-4 pt. lambd0 (0.893, trial 63's seed) sits at the config's seed mean.

## 2026-11-03 00:16 — result

Samuel 137 pilot (r261102-365990) at trial 63's training seed: base 0.920, reaug 0.917, lambd1 (object term only) 0.899, lambd0 (feature term only) 0.893. The config's own training-seed spread is 0.872-0.920 (base, base_seed2, base_seed3), so no variant differs from base beyond seed noise. Launched seed-3 variants so every arm has 3 training seeds (2854f93).

## 2026-11-03 00:16 — run

Samuel 137 pilot streams e (lambd1_seed2, lambd0_seed3) and f (reaug_seed3, lambd1_seed3), run_pilot.py 2854f93; with stream d (lambd0_seed2, reaug_seed2) this completes 3 training seeds x {base, lambd0, lambd1, reaug}

run r261103-1931191 · pids 1931191, 2222987 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_e.log, /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_f.log

## 2026-11-03 00:52 — surprise

reaug_seed3 (r261103-1931191, stream f) crashed in its first validation epochs: _correlation_for_plot multiplied unpaired position views (2 vs 3 objects, < 2 shared ids) and the trial was discarded without a checkpoint. Fixed in bc3c1bf (skip such batches; regression test), pushed. Affects any position-pipeline run with object dropout where a val batch's views share < 2 ids, more likely with reaugment_each_epoch; cluster rounds crash the same way until they pull. Partial dir removed, reaug_seed3 queued after stream f (unit s137-pilot-g).

## 2026-11-03 00:52 — run

reaug_seed3 rerun with the bc3c1bf fix, queued after stream f

run r261103-732047 · pids 732047 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/s137_pilot/stream_g.log

## 2026-11-03 06:14 — result

Samuel 137 pilot complete (r261102-365990, r261102-2656039, r261103-1931191, r261103-732047; run_pilot.py 2854f93, train fix bc3c1bf for reaug_seed3; trial 63 config, labelprop seed 0 x 25 seeds, CPU objective). Training seeds trial63/2/3: base 0.920/0.884/0.872 (mean 0.892, sd 2.5 pt); lambd0 (no object term) 0.893/0.804/0.821 (mean 0.839), paired vs base -2.7/-8.0/-5.1 pt, mean -5.3, paired t p 0.07; lambd1 (object term only) 0.899/0.864/0.912 (mean 0.892), paired -2.1/-2.0/+4.0, mean 0.0; reaug 0.917/0.859/0.911 (mean 0.896), paired -0.2/-2.5/+3.9, mean +0.4. Reading: removing the object-space term costs ~5 pt on Samuel 137 (same sign at all 3 seeds); the feature-space term adds nothing beyond the object term; reaugment_each_epoch has no detectable effect. The wk2 sweep never sampled lambd_obj < 0.1, which is why its partial dependence looked flat. One dataset, 3 seeds.

## 2026-11-03 06:23 — run

Leifer replication of the Samuel 137 lambd0 result: round_bayes_wk2_leifer_reexport trial 57 (sweep 0.885, lambd_obj 0.41) retrained as base and lambd0 at training seeds trial57/2/3, scored as the sweep objective (NWB all objects, segmentation_id GT mapping, labelprop 25 seeds, CPU). run_pilot.py 7c849f8 --name leifer_pilot. Streams a (base, lambd0) and b (base_seed2, lambd0_seed2) on the GPU; c (base_seed3, lambd0_seed3) waits for a, since a third ~5 GB process OOM'd. ~3.7 h training per run locally (321 s/100 steps vs 172 on the cluster) plus ~1 h scoring; all done ~03:00 on 2026-11-04.

run r261103-716084 · pids 716084, 742568, 1663423 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_pilot/stream_a.log, /home/charles/.cache/barlow_scratch/train/leifer_pilot/stream_b.log, /home/charles/.cache/barlow_scratch/train/leifer_pilot/stream_c.log · for [[EVD - Loss-term ablation where accuracy is not saturated]]

## 2026-11-03 07:56 — run

User asked for object-space-only runs. Leifer pilot stream d: lambd1 (lambd_obj 1) at training seeds trial57/2/3, queued after stream b (GPU fits two Leifer runs). run_pilot.py 7c849f8. Expected done ~06:00 on 2026-11-04.

run r261103-946326 · pids 946326 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_pilot/stream_d.log · for [[EVD - Loss-term ablation where accuracy is not saturated]]

## 2026-11-03 15:50 — dead-end

Leifer resegment (leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist) segmentation job 7781649 failed: project_config pointed subfolder_configs.preprocessing at the SOURCE project's preprocessing_config.yaml, so the relative bounding_boxes_fname resolved to leifer_reexport/.../dat (no bbox there) while the alt_preprocessing rule had written the bbox into the resegment project's dat/. Fixed: preprocessing now points at the resegment project's own dat/preprocessing_config.yaml, with preprocessed_red/green_fname made absolute to the source zarrs. Backups *.bak_20261103. Not relaunched.

## 2026-11-03 15:55 — run

Leifer resegment (stardist) segmentation run locally via RUNME.sh -c -s segmentation, cluster queue full; after the preprocessing-config path fix

run r261103-1181386 · pids 1181386 · host charles-Precision-7820-Tower · logs /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline/leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist/snakemake/log_snakemake_local_20261103.out

## 2026-11-03 16:47 — result

Leifer pilot (r261103-716084): base 0.8775, base_seed2 0.8792 (trial 57's sweep score 0.885). Training-seed spread on Leifer is small so far (0.2 pt vs 3.6 on Samuel 137). lambd0 and lambd0_seed2 finishing training.

## 2026-11-03 16:47 — decision

Checked the wk3 cluster rounds after the user's wbfm pull (cluster wbfm f4f555d5, pulled 2026-11-03 15:55; cluster barlow_track still 555a125, so WBFM_NWB_LAZY_SEGMENTATION is never set and NWB segmentation stays eager): all 8 dispatchers alive (37-41 of 120 submitted per Flavell/Samuel round, 24-29 Zimmer), no Traceback in any job log since the pull, jobs started after the pull load NWBs and embed, records written after the pull have normal accuracies. Only failure today: round_bayes_wk3_samuel trial 18 at 05:56, before the pull, a NaN checkpoint refused by the eval and marked FAILED for Ax (NaN discipline working). Zimmer 'bigtiff.btf does not exist' errors are logged lines from 2026-11-02, not crashes.

## 2026-11-03 16:55 — decision

User pulled barlow_track on the cluster (bc3c1bf, 2026-11-03 16:54): new eval and train jobs now set WBFM_NWB_LAZY_SEGMENTATION=1 (lazy NWB segmentation, with wbfm f4f555d5) and carry the validation-plot fix. Running dispatchers keep their old in-memory code; jobs they submit import the new files (eval_accuracy changed by the env-var line only). Watching all wk3 job logs for tracebacks and new FAILED trials.

## 2026-11-03 17:36

r261103-1181386 done: Leifer stardist resegmentation (leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist) finished locally in ~1h40, exit 0. 1536 frames, median 129 objects/frame (max 145); metadata.pickle + masks.zarr (385 MB) in 1-segmentation/. Later pipeline steps left to the user on the cluster.

## 2026-11-03 17:51 — run

Mapping Leifer re-export GT (79 neurons, NWB clicks + NeRVE masks) onto the stardist masks of leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist, to build a final_3d_tracks_df so eval_accuracy --source project can score this project like the paper wbfm projects. Script: scratchpad map_gt.py (click-voxel label and max-overlap-with-NeRVE-mask label per (t, neuron)).

run r261103-3422453 · pids 3422453 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/map_gt.log

## 2026-11-03 17:53 — dropped

too slow (per-neuron full-volume compare); replaced by a joint-histogram version

dropped r261103-3422453

## 2026-11-03 17:53 — run

Rerun of the GT-to-stardist cross-check mapping (click-voxel label, max overlap with the GT neuron's NeRVE mask), joint-histogram version, 16 workers.

run r261103-437335 · pids 437335 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/map_gt.log

## 2026-11-03 17:53 — run

Leifer StarDist resegmentation (wbfm_tracker_baseline/leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist, user request; GT still being finalized): train trial 57's config on it as base and lambd1 (object term only) at training seeds trial57 and 2, train only (run_pilot.py f0e7f7d --project_path --train_only --name leifer_reseg), to score once the GT exists. Checked first: project loads, 1536 frames, masks are StarDist's (120-133 objects per frame, 94% foreground overlap with NeRVE, not identical), metadata matches mask labels per frame; segment_config's segmentation_fname still names the NeRVE masks.zarr but is not what loads. Queued behind the Leifer pilot streams c and d (GPU fits two Leifer runs); ~3 h each, done ~midday 2026-11-04.

run r261103-604205 · pids 604205, 604562 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/stream_e.log, /home/charles/.cache/barlow_scratch/train/leifer_reseg/stream_f.log

## 2026-11-03 17:54 — surprise

Leifer pilot (r261103-716084): lambd0 (no object term) 0.8872 and 0.8792 vs base 0.8775 and 0.8792 at the same training seeds (trial57, 2): paired +1.0 / 0.0 pt. The Samuel 137 lambd0 cost (-5.3 pt) does not show on Leifer so far; seed 3 and lambd1 still to come. Leifer may be bottlenecked elsewhere (z-split masks, HANDOFF finding 7), which would make it insensitive to the loss.

## 2026-11-03 17:57 — decision

Cross-check r261103-437335 result: 97.9% of the 119,019 Leifer GT clicks fall inside a stardist mask; that mask equals the max-overlap stardist mask of the GT neuron's NeRVE mask for 95.3%; 2.9% of clicks share a stardist mask with another GT click (likely stardist merges). Click->nearest-centroid distance median 1.1 vox, p95 4.1 vox (xy 0.3226 um, z 1.5 um). Decision (with user): build stardist GT with a tight distance cap, click inside the mask, objects claimed by 2+ GT neurons left unassigned; script barlow_track/scripts/project_reanalysis/gt_tracks_from_clicks.py, provenance JSON next to the output for the methods section.

## 2026-11-03 17:57 — run

Dry runs of gt_tracks_from_clicks.py on the stardist Leifer project at max_dist_um 0.75 / 1.0 / 1.5, to choose the cap.

run r261103-1354985 · pids 1354985, 1354986, 1354987 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/dry_0.75.log, /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/dry_1.0.log, /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/dry_1.5.log

## 2026-11-03 17:58 — dropped

user: stop the old-data Leifer runs and train on the StarDist resegmentation instead; lambd1 stopped mid-training, partial dir removed

dropped r261103-946326

## 2026-11-03 17:58 — dropped

superseded: user asked for a basic sweep on the resegmentation instead

dropped r261103-604205

## 2026-11-03 17:58 — decision

Leifer pilot (r261103-716084) stopped on the user's call after 4 runs (user: the NeRVE data is saturated): base_seed3 and lambd0_seed3 never finished, base_seed3 partial dir removed. Kept: base 0.878/0.879, lambd0 0.887/0.879 (seeds trial57/2).

## 2026-11-03 17:59 — dead-end

sweep_param_effect.py (62cd4c8) partial dependence is wrong for log_scale parameters: the forest is fit on log values but the grid is evaluated in raw units, so the line is flat (lr, num_frames, epochs all flat on Leifer wk2). lambd_obj (linear) unaffected. Bins are fine.

- [ ] fix sweep_param_effect.py partial dependence grid for log_scale params

## 2026-11-03 18:02 — run

Writing stardist Leifer GT tracks: max_dist_um 1.5 (primary, set as final_3d_tracks_df; one z-plane, clicks are z-quantized to 1.5 um planes) and 1.0 (stricter sensitivity variant). Dry runs r261103-1354985: assigned 63.8% / 75.1% / 86.6% of 119,019 clicks at 0.75 / 1.0 / 1.5 um; shared-object drops 419 / 687 / 1252; outside-mask drops 19 / 74 / 1020; 208 clicks in the 4 empty volumes. No global click-centroid bias (median signed offset < 0.06 um); worst neurons sit in large (p90-size) objects or are ~1 plane off in z.

run r261103-2423961 · pids 2423961, 2423962 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/write_1.5.log, /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/write_1.0.log

## 2026-11-03 18:04 — run

Rerun of the GT writes (r261103-2423961 failed: wbfm add_metadata_to_df_raw_ind needs a likelihood column; now likelihood 1 where assigned).

run r261103-3393482 · pids 3393482, 3393483 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/write_1.5.log, /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/write_1.0.log

## 2026-11-03 18:07 — run

Basic local sweep on the Leifer StarDist resegmentation (user: NeRVE data is saturated; GT still being finalized, so train only, score later): trial 57's config with lambd_obj {0, 0.41, 1} x num_frames {206, 400}, one training seed (Leifer seed spread 0.2 pt), run_pilot.py 5a30d94 --name leifer_reseg --project_path <stardist project> --train_only. Stream e: base, lambd0, lambd1, nf400; stream f: lambd0_nf400, lambd1_nf400. Crop z 8 kept: StarDist objects z extent median 6, p90 7, max 10 (NeRVE 3/4/7), 98% fit. ~4 h per 206-frame run, ~8 h per 400-frame run; done ~14:00 on 2026-11-04.

run r261103-1358870 · pids 1358870, 1445434 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/stream_e.log, /home/charles/.cache/barlow_scratch/train/leifer_reseg/stream_f.log

## 2026-11-03 18:07 — run

Smoke test: eval_accuracy --mode trained --source project on the stardist Leifer project with its new GT (1.5 um), baseline_leifer/trial_0, CPU, first 100 frames, 5 labelprop seeds. Only checks the pipeline end to end; the score is not a benchmark number.

run r261103-671865 · pids 671865 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/smoke.log

## 2026-11-03 18:17 — run

Smoke rerun (r261103-671865 OOM'd: load_barlow_model put the model on the full Titan V despite --device cpu); now CUDA_VISIBLE_DEVICES='' and 50 frames.

run r261103-2200226 · pids 2200226 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/smoke.log

## 2026-11-03 18:28 — todo

r261103-2200226: smoke eval of the stardist Leifer project with its new GT ran end to end (50 frames, baseline_leifer/trial_0, 0.860; pipeline check only). GT built in r261103-3393482 (1.5 um primary, 1.0 um variant); r261103-1354985 dry runs and r261103-437335 cross-check summarized in [[DATA - Leifer NeRVE recording re-segmented with stardist with ground truth assigned from manual clicks]]. Messaged BarlowTrack_local_runs that the GT is ready.

- [x] commit barlow_track/scripts/project_reanalysis/gt_tracks_from_clicks.py and cite the commit in the stardist Leifer DATA note

## 2026-11-03 18:38 — run

Leifer StarDist project GT is ready (from the BarlowTrack_cluster_runs session: 79 Leifer click neurons assigned to StarDist objects within 1.5 um, inside the mask, one GT neuron per object, 86.6% of clicks assigned; final_3d_tracks_df = 3-tracking/manual_annotation/gt_tracks_from_clicks_1.5um.h5; segmentation r261103-1181386; volumes 78, 405, 945, 1516 empty). Scoring on it (score_reseg.py 10939a3; --source project, project = GT, labelprop 25 seeds seed 0, CPU, CUDA hidden): references trial 57 (NeRVE-trained, cluster) and the local NeRVE base retrain, then each leifer_reseg sweep checkpoint as it lands. Note GT is 86.6% of the NeRVE GT's clicks, so scores are not directly comparable with the NeRVE 0.885; the references give the like-for-like baseline.

run r261103-3915910 · pids 3915910 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/score.log

## 2026-11-03 18:38

BarlowTrack_local_runs is scoring checkpoints on the stardist Leifer project against the 1.5 um GT ([[DATA - Leifer NeRVE recording re-segmented with stardist with ground truth assigned from manual clicks]]). Scorer: scratch/s137_pilot/score_reseg.py @10939a3, run r261103-3915910, output ~/.cache/barlow_scratch/train/leifer_reseg/scores.jsonl. It scores the NeRVE-trained references (wk2 Leifer trial 57 and its local retrain), then a 6-run stardist-trained sweep (r261103-1358870). It writes nothing to the project.

## 2026-11-03 19:55 — dead-end

r261103-3915910: first reference (ref_trial57_nerve) FAILED after embedding+tracking, at the record write: extra_record 'weights' collides with the eval's own field. Fixed in 6072f44 (drop the key); scorer restarted (same unit leifer-reseg-score), reusing the cached trial57 embedding.

run r261103-77928 · pids 77928 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/score.log

## 2026-11-03 20:41 — result

r261103-77928: ref_trial57_nerve (wk2 Leifer trial 57, trained on NeRVE) scored on the StarDist project vs its click GT (1.5 um): accuracy 0.9448 (misses 2873, mismatches 2818, of 103052 GT cells; 44 min CPU, labelprop 25 seeds seed 0). Same network on the NeRVE project scored ~0.885, but not comparable: StarDist GT covers 86.6% of clicks (the unassigned 13% may be the hard ones), objects differ. The fair comparison is the reseg-trained sweep runs vs this reference on the same GT.

## 2026-11-03 22:06 — result

r261103-77928: ref_local_nerve (local retrain of Leifer trial 57 on NeRVE, leifer_pilot/base) on the StarDist click GT: 0.9464 (misses 2847, mismatches 2679 of 103052). With ref_trial57_nerve 0.9448, the NeRVE-trained reference is ~0.945 on this GT; the reseg-trained runs compare against that. Training is slower than estimated: 206-frame runs ~5 h (11.7 min/epoch x 26), 400-frame ~9.5 h, so the grid ends ~19:00 2026-11-04. ~300 'centroid outside volume' crop skips per run, mostly StarDist centroids at z 31.5 rounding to plane 32 of a 32-plane volume; negligible against ~3600 steps of crops.

## 2026-11-04 00:16 — result

r261103-1358870 / r261103-77928: leifer_reseg base (trial 57 config, trained on the StarDist project) on the StarDist click GT: 0.9476 (misses 2971, mismatches 2430 of 103052), vs the NeRVE-trained references 0.9448 / 0.9464. +0.1 to +0.3 pt, within Leifer seed spread (~0.2): training on the StarDist segmentation buys nothing measurable over NeRVE training when scored on StarDist objects. Fewer mismatches (2430 vs ~2750) but more misses (2971 vs ~2860).

## 2026-11-04 01:08 — run

Why do 13.4% of the Leifer clicks fail to become StarDist GT (user asks: clean exclusion categories? StarDist misses small?). gt_click_exclusions.py (5db47c7) replays the 1.5 um rule and puts each click in one category. 40-volume check: assigned 86.1%, in a StarDist mask whose centroid is >1.5 um away (median 1.9 um; z voxel is 1.5 um, objects ~6 planes) 9.7% = recoverable by the rule, coincident clicks 1.3%, one mask holding two neurons' clicks 0.8%, background within 1 um of a mask 1.4%, StarDist missed an object NeRVE has 0.5%, background in both 0.1%. Full run on all 1536 volumes started.

run r261104-885770 · pids 885770 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/click_categories/run.log

## 2026-11-04 01:13 — result

r261104-885770 (all 1536 volumes, 119019 clicks; reproduces the GT's 103052 assigned exactly): assigned 86.6%; click inside a StarDist mask no other click is in but centroid >1.5 um (median 1.88, p90 2.67 um) 9.0%; coincident clicks (<1.5 um) 1.1%; one mask holding two neurons' clicks 1.1%; background <=1 um from a mask 1.3%; StarDist missed an object NeRVE has 0.6%; background in both 0.06%; empty volumes 0.2%. Also 939 assigned clicks (0.8%) sit in a mask holding another neuron's click (merge the rule kept, because the other click was beyond 1.5 um). StarDist genuinely missing a neuron is 0.6-2.0% (upper bound counts near-mask). A rule 'click inside a mask that holds no other neuron's click' gives 94.8% coverage (per-neuron p10 0.62 -> 0.90). neuron_070 is an outlier: 35% background, 11% coincident.

## 2026-11-04 01:22 — decision

User: new wbfm project (configs only) with a GT keeping clicks inside masks and near mask edges, then evaluate networks on it. Created Leifer_NeRVE_Worm1_reexport_A2_stardist_maskgt (leifer_resegment/, wbfm make_project_like keeping preprocessing/segmentation/training_data linked to the StarDist project; verified identical masks, centroids, red data at t 0 and 700). gt_tracks_from_clicks.py --rule mask (3e2fdc7, edge 1.0 um, coincide 1.5 um), set as its final_3d_tracks_df. Expected ~96% of clicks. Scoring will use project = StarDist project (cached embeddings) and gt = the maskgt project.

run r261104-3312389 · pids 3312389 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/maskgt.log

## 2026-11-04 01:27 — result

r261104-3312389 first mask GT build 94.65%: edge clicks made masks 'shared' and removed in-mask clicks (fixed 16a906d). r261104-568713 rebuilt (gt_tracks_from_clicks.py --rule mask, edge 1.0 um, coincide 1.5 um) into Leifer_NeRVE_Worm1_reexport_A2_stardist_maskgt, set as its final_3d_tracks_df: 113499 of 119019 clicks = 95.4% (vs 86.6%); per-neuron coverage p10 0.91, median 0.98. Excluded: coincident clicks 1541 (1.3%), mask holding 2+ neurons' clicks 3003 (2.5%; ~850 of them edge clicks whose nearest mask is another neuron's), no mask within 1 um 768 (0.65%), empty volumes 208. Edge assignments 661. Not >96%: near-mask clicks mostly sit next to another neuron's mask.

## 2026-11-04 01:27 — run

Scoring refs + leifer_reseg variants against the maskgt GT (project = StarDist project for cached embeddings, gt = _maskgt project; score_reseg.py d7b6519 --gt --results scores_maskgt.jsonl, same tags).

run r261104-2000932 · pids 2000932 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/score_maskgt.log

## 2026-11-04 01:28 — decision

User: finish the wbfm pipeline (traces) on the GT-tracked projects, dry runs first. Dry run without tracks planned the full wbfm tracking chain (frame objects, tracklets, tracking, combine), which would redo tracking; so each project's GT h5 was copied to 3-tracking/postprocessing/combined_3d_tracks.h5 (final_3d_tracks_df still names the GT file). Dry runs then plan only extract_full_traces + traces on both projects (StarDist 1.5 um GT, _maskgt mask GT). Running locally (no slurm here), sequential, 24 cores.

run r261104-2551987 · pids 2551987 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/traces.log

## 2026-11-04 01:42 — result

r261104-2551987 done: snakemake traces (extract_full_traces) finished on both GT-tracked projects, ~7 min each, no errors in the project logs. 4-traces/ has red/green traces (79 neurons), reindexed masks, paper traces, grid plots. Missing GT frames are zero intensity in the raw traces (fraction 0.065 maskgt, 0.151 StarDist 1.5 um = 1 - GT coverage exactly). paper_traces keep 76 neurons with the mask GT vs 62 with the 1.5 um GT (the paper-trace filter drops neurons with too much missing data).

## 2026-11-04 02:12 — result

r261104-2000932: ref_trial57_nerve on the mask GT (113499 cells): 0.9351 (misses 1258, mismatches 6107) vs 0.9448 on the 1.5 um GT (103052 cells; misses 2873, mismatches 2818). The clicks the mask rule adds are harder: back-of-envelope (sets not exactly nested) accuracy on the ~10.4k added cells ~0.84. So the 1.5 um GT flattered the score by ~1 pt; mismatches doubled, misses halved. Cached embedding reused (44 min = tracking only).

## 2026-11-04 02:57 — result

r261104-2000932: ref_local_nerve on the mask GT: 0.9372 (misses 1279, mismatches 5847) vs 0.9464 on the 1.5 um GT. Both NeRVE references lose ~1 pt on the mask GT (0.935/0.937). base (StarDist-trained) next.

## 2026-11-04 03:43 — result

r261104-2000932: leifer_reseg base on the mask GT 0.9385 (misses 1183, mismatches 5800) vs NeRVE refs 0.9351/0.9372: same +0.1-0.3 pt as on the 1.5 um GT, within seed spread. lambd0_nf400 finished training (r261103-1358870 stream f, now lambd1_nf400); lambd0 still training on stream e.

## 2026-11-04 04:55 — result

r261103-3915910 scorer (restarted as r261103-77928): leifer_reseg lambd0_nf400 (feature term only, 400 frames, StarDist) on the 1.5 um GT 0.9416 (misses 3672, mismatches 2343) vs base 0.9476: -0.6 pt, all of it misses (+700); single seed. lambd0 finished training (stream e now lambd1).

## 2026-11-04 05:28 — surprise

r261104-2000932: lambd0_nf400 on the mask GT 0.9405 (misses 1024, mismatches 5725) vs base 0.9385: +0.2 pt, while on the 1.5 um GT it was -0.6 (0.9416 vs 0.9476, misses 3672). The ranking flips between GTs, so the difference is noise-level; the 1.5 um GT's misses are fragile (same tracking, sparser GT columns).

## 2026-11-04 06:28 — result

lambd0 (lambd_obj 0, nf default) on the Leifer StarDist reseg, 1.5 um GT: 0.9366 (3629 miss / 2908 mismatch), 1.1 pt below base 0.9476; mask-GT score pending

## 2026-11-04 07:00 — result

lambd0 on the Leifer StarDist reseg, mask GT: 0.9351, 0.3 pt below base 0.9385 (1.5 um GT: -1.1 pt). Object term off costs on both GTs at default num_frames; at nf400 it does not (-0.6 / +0.2)

## 2026-11-04 08:08 — run

Rescore older networks on the Leifer StarDist reseg against both GTs (1.5 um and mask), stream 1 of 3: wk2 Leifer trials across the accuracy ranking (does the NeRVE ranking transfer?), wk1 t1, snap1 t0, zimmer_1128 t0 and samuel_137 t63 (cross-dataset). CPU, labelprop 25 seeds, c8da74e

run r261104-2447344 · pids 2447344 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/score_old1.log

## 2026-11-04 08:08 — run

Rescore older networks on the Leifer StarDist reseg against both GTs (1.5 um and mask), stream 2 of 3: wk2 Leifer trials across the accuracy ranking (does the NeRVE ranking transfer?), wk1 t1, snap1 t0, zimmer_1128 t0 and samuel_137 t63 (cross-dataset). CPU, labelprop 25 seeds, c8da74e

run r261104-2447732 · pids 2447732 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/score_old2.log

## 2026-11-04 08:08 — run

Rescore older networks on the Leifer StarDist reseg against both GTs (1.5 um and mask), stream 3 of 3: wk2 Leifer trials across the accuracy ranking (does the NeRVE ranking transfer?), wk1 t1, snap1 t0, zimmer_1128 t0 and samuel_137 t63 (cross-dataset). CPU, labelprop 25 seeds, c8da74e

run r261104-2447888 · pids 2447888 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/score_old3.log

## 2026-11-04 08:12 — dropped

user asked to use the benchmark script instead (run_trials_on_all_ground_truth.py with leifer_stardist datasets)

dropped r261104-2447344

## 2026-11-04 08:12 — dropped

user asked to use the benchmark script instead (run_trials_on_all_ground_truth.py with leifer_stardist datasets)

dropped r261104-2447732

## 2026-11-04 08:12 — dropped

user asked to use the benchmark script instead (run_trials_on_all_ground_truth.py with leifer_stardist datasets)

dropped r261104-2447888

## 2026-11-04 08:14 — run

Benchmark (run_trials_on_all_ground_truth.py, 9a1d725) on the new datasets leifer_stardist + leifer_stardist_maskgt, CPU, 2 workers: round_bayes_wk2_leifer_reexport trials 57 74 65 50 63 37 76 14 9 24 (spread over the NeRVE ranking, ranks 0-79). Results in wbfm_tracker_baseline/leifer_resegment/benchmark/<round>.jsonl

run r261104-4089020 · pids 4089020 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/bench_wk2.log

## 2026-11-04 08:14 — run

Benchmark on leifer_stardist + _maskgt, 1 worker, older and cross-dataset networks in turn: wk1_leifer t1, snap1_leifer t0, wk2_zimmer_1128 t0, wk2_samuel_137 t63

run r261104-4089023 · pids 4089023 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/bench_other.log

## 2026-11-04 08:39 — surprise

r261104-4089020: the trial 9 leifer_stardist cell died on OSError EIO reading red_data from the CIFS mount (/lisc is mounted soft, so a slow read surfaces as an I/O error). Transient: the other cells read on. Its chained maskgt cell embeds by itself and writes the shared cache, so a rerun of the benchmark at the end (skips recorded cells) fills the stardist cell from cache

## 2026-11-04 09:32 — run

Click misses vs the StarDist 180-object cap (user: are no-mask clicks dim objects the cap removed?): objects per volume, click-to-mask distance and brightness per category, all volumes, ac

run r261104-3278557 · pids 3278557 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/click_misses_vs_cap/run.log

## 2026-11-04 09:32

r261104-3278557 runs gt_click_misses_vs_cap.py at 6481f73 (the run line above was cut short)

## 2026-11-04 09:37 — result

r261104-3278557 (6481f73): the StarDist 180-object cap never fired on the Leifer reseg. StarDist 103-145 objects per volume (median 129, 1532 volumes), NeRVE 117-154 (median 125, sampled every 10th; min 37). Clicks with no mask are not dim removed objects: within 1 um of a mask they are brighter than assigned clicks (median 1.5x, red 3x5x5 box, relative to the volume's assigned clicks); the 700 with a NeRVE mask but no StarDist mask within 1 um are dimmer (median 0.51x, p10 0.18x, about the dimmest kept object 0.17x) and sit 1.6 um median from a StarDist mask (p90 7.8). Only ~290 clicks (0.25%) are >2 um from any StarDist object

## 2026-11-04 09:59 — run

NeRVE GT project Leifer_NeRVE_Worm1_reexport_A2_nervegt (make_project_like of the A2 NeRVE project, configs only; user asked for a full project with NeRVE segmentation and GT, traces to compare with the paper). GT by gt_tracks_from_clicks.py --rule segmentation_id --max_label 1000 (7799d00): the NWB's own NeRVE label per click, pseudo masks dropped: 107,000 of 119,019 clicks (89.9%); 10,603 on pseudo masks, 1,416 in the 26 empty volumes, none shared; per-neuron coverage p10 0.81, median 0.97, min 0.0007. GT copied to combined_3d_tracks.h5, dry run planned only extract_full_traces + traces; now running traces

run r261104-1389743 · pids 1389743 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/traces_nervegt.log

## 2026-11-04 10:19 — dead-end

r261104-1389743 wrote empty traces (red_traces.h5 0 columns, no grid plots): wbfm reindex_segmentation's lookup table has 1000 entries, the NeRVE project's pseudo masks are labels >= 1000, so every volume raised IndexError and its reindexed mask was zeroed. Fixed in wbfm (dlc_for_wbfm dev, utils_segmentation.reindex_segmentation pads the table with background for higher labels; uncommitted, synthetic test passes). Removed the empty outputs, dry run planned only extract_full_traces + traces, rerunning

## 2026-11-04 10:19 — run

Traces rerun on the NeRVE GT project with the wbfm lookup-table fix (uncommitted in dlc_for_wbfm)

run r261104-2149802 · pids 2149802 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/traces_nervegt2.log

## 2026-11-04 10:30 — result

r261104-2149802 done: NeRVE GT project traces with the wbfm lookup fix, 0 IndexErrors, 79 neurons in red/green traces (11.8% of frames zero or NaN), 69 in paper_traces, heatmap and grid plots written (sent to the user next to the two StarDist heatmaps)

## 2026-11-04 10:49 — decision

User: labels >= 1000 are a sign of upstream errors, so no wbfm workaround. Reverted the wbfm lookup padding; wbfm (dlc_for_wbfm dev, uncommitted) now raises a clear ValueError naming the volume and label when masks or tracks hold a label >= the 1000-entry table, instead of silently zeroing the volume. The NeRVE GT project got its own NeRVE-only segmentation (drop_mask_labels.py, 7a84d3f): the 12,019 pseudo masks (labels >= 1000, 1,162,620 voxels, in all 1536 volumes) set to background, every other voxel and the metadata unchanged (max metadata label 152; checked at volumes 0, 700, 1500), project_config segmentation -> its own 1-segmentation/segment_config.yaml. The original A2 project is untouched. GT copy refreshed in combined_3d_tracks.h5 (newer masks made the dry run plan the whole tracking chain; after the refresh only extract_full_traces + traces)

## 2026-11-04 10:49 — run

Traces on the NeRVE GT project with its NeRVE-only masks, stock wbfm plus the new label check

run r261104-1391957 · pids 1391957 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/traces_nervegt3.log

## 2026-11-04 10:59 — result

r261104-1391957 done: NeRVE GT project traces on its NeRVE-only masks with stock wbfm (plus the new label check), no IndexError or ValueError, red traces 79 neurons, paper traces 69, plots written: same shapes as the discarded workaround run

## 2026-11-04 11:04 — decision

User (2026-11-04): leave the original A2 NeRVE project's masks as they are; labels >= 1000 only matter for pipeline trace extraction, which is needed once, for the NeRVE GT, and is done in _nervegt

## 2026-11-04 11:13 — result

Trace cliffs in the StarDist GT heatmaps (user: tracking error, or the GT lost a dim neuron?), trace_cliff_check.py 7226561, output ~/.cache/barlow_scratch/train/leifer_reseg/trace_cliffs/. Not tracking: the two cliff neurons (038, 060; paper trace early-minus-late 0.58 and 0.35 in the mask GT, next largest 0.07) have stable GT objects (click to assigned centroid median 0.65 / 0.63 um, longest run > 2 um 4 / 2 frames; 100-frame median object volume within 0.77-1.51x), and over all 79 neurons the longest run > 2 um is 10 frames and no window has the object volume > 2x or < 0.5x. It is wbfm's per-channel exponential bleach correction: raw green/red rises (038: 0.50 early -> 0.63 late), dr_over_r_50 without bleach correction rises too (-0.12 -> 0.14), with it falls (0.32 -> -0.26); background subtraction (75 per pixel x volume) adds a little for 038. Both neurons' red and green fall about 3x over the first ~200 frames, which the single exponential fits badly. The dip at frames ~690-790 (040, 081, 050, 034) is in all three projects including the NeRVE one, so it is in the recording or clicks, not the segmentation. No GT-side filter would remove either; per the user, no filtering on the traces post hoc. 038 is in no NeRVE paper trace because NeRVE never detected it (1495 of 1524 clicks on pseudo masks)

## 2026-11-04 11:21 — result

neuron_038 in the StarDist heatmaps (user: high first half, low second, worse than a jump): the heatmap is not paper_traces but calc_default_traces(dr_over_r_20, interpolate, rolling mean), which puts r0 at the 20th percentile; there 038 averages 0.76 first half vs 0.04 second (mask GT; 0.88 centroid GT; colour scale tops at 0.72), 060 0.47, every other neuron under 0.11. Without bleach correction (bleach_correct=False) 038 rises slowly like every neuron (-0.22 first minus second half; red bleaches faster than green). Its red is 6.5x brighter in frames 0-50 than 300-600 and green 3.3x (median neuron 1.3x and 1.1x), which the per-channel exponential overcorrects. Red crops at its click (t = 5, 40, 150, 400, 800, 1300): one neuron, separate from its neighbour, the same StarDist mask on it every time; it is just very bright at the start (and again near t 800). Not a tracking or GT error. Quick looks (inline), not a recorded result

## 2026-11-04 12:33 — result

lambd1 (lambd_obj 1, object term only) on the Leifer StarDist reseg, mask GT: 0.9383 (920 miss / 6078 mismatch), level with base 0.9385; centroid GT pending

## 2026-11-04 12:37 — result

lambd1 on the Leifer StarDist reseg, centroid GT: 0.9394 (3698 / 2548), 0.8 pt below base 0.9476 (mask GT: level, 0.9383 vs 0.9385)

## 2026-11-04 14:38 — result

r261104-4089020 / r261104-4089023 partial: wk2 t37 leifer_stardist 0.9330, wk2 t24 0.8916; snap1 t0 0.9347 / maskgt 0.9358 (other rounds unit); no new failures

## 2026-11-04 14:40 — decision

Committed the wbfm clear-error change for mask labels >= 1000 (dlc_for_wbfm dev a1cd2225, not pushed), per the user

## 2026-11-04 17:10 — result

r261104-4089020 / r261104-4089023 partial: wk2 t37 maskgt 0.9253, t24 maskgt 0.8721, t57 leifer_stardist 0.9448 (matches the earlier score_reseg ref_trial57_nerve 0.9448), zimmer_1128 t0 on leifer_stardist 0.9256

## 2026-11-04 18:37 — result

r261103-77928 / r261104-2000932: lambd1_nf400 on the Leifer StarDist reseg: centroid GT 0.9414, mask GT 0.9384 (base 0.9476 / 0.9385)

## 2026-11-04 20:25 — result

r261104-4089020 / r261104-4089023 partial: wk2 t50 0.9485 / maskgt 0.9408 (best so far, level with local base), t57 maskgt 0.9351, t63 0.9416, samuel_137 t63 0.9388, zimmer_1128 t0 maskgt 0.9211

## 2026-11-04 20:32 — run

Re-track base (Leifer StarDist reseg) from its cached embedding with --save_tracks, centroid then mask GT, labelprop on GPU, for an error-pattern analysis (are errors in GT entries we were too generous to keep?)

run r261104-3677569 · pids 3677569 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/retrack.log

## 2026-11-04 20:35 — dropped

labelprop on the GPU ran out of memory next to the nf400 training; rerun on CPU

dropped r261104-3677569

## 2026-11-04 20:35 — run

Re-track base (Leifer StarDist reseg) on CPU from its cached embedding with --save_tracks (4ae0c07), centroid and mask GT in parallel, for the error-pattern analysis

run r261104-4092161 · pids 4092161, 4092164 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/retrack_c.log, /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/retrack_m.log

## 2026-11-04 21:50 — run

Overnight CPU queue for the Leifer reseg checks (user: rigorously test the resegmentation; 15 jobs, 4 parallel, all with --save_tracks, scripts at 2b0e31d): (1/7) base, t57, t50, ref_local_nerve embedded and scored on the NeRVE-masks GT project _nervegt; (10) t57 re-tracked on the NWB all-objects path to split errors on pseudo masks; (2) base on the 1.0 um GT (_gt1um copy); (4) merge persistence for StarDist and NeRVE masks; (5) labelprop seeds 1-4 base centroid GT, 1-2 base mask GT, 1-2 t57 centroid GT. Queue: /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/overnight_queue.txt

run r261104-3067468 · pids 3067468 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/queue_status.log

## 2026-11-04 21:50 — decision

User decisions for the Leifer reseg subproject: finish the local training, then test the resegmentation rigorously overnight; it gets a full supp figure. Report accuracy that does not assume the GT is complete (gt_valid matching, next to nandist). More labelprop seeds yes; training seeds no (the user runs GPU sweeps; leave the GPU after nf400). Traces with every segmentation and tracker, and compare them. Also check how the original NeRVE benchmark (0.885, all 119,019 clicks, pseudo masks tracked as objects) treats pseudo-mask clicks.

## 2026-11-04 22:17 — decision

User: trust the GT, drop clicks only for a very good reason; merges and coincident clicks are good reasons (merges get a figure panel); unattached clicks count as StarDist misses and are reported; score with gt_valid. Click jitter (click_jitter.py, 2b0e31d+; per-frame change of click-to-centroid offset, good-size frames): sigma per axis z/x/y 0.35/0.20/0.20 um on StarDist masks (3-d 0.44, p95 of 3-d jitter 1.7 um), 0.27/0.18/0.18 on NeRVE (0.37, p95 1.3). Mask GT with edge 1.5 um (c5a605a) in new copy _maskgt15: 113,689 assigned, 437 no mask within 1.5 um (misses), 208 in 4 empty volumes, 3,144 merges and 1,541 coincident excluded. Scoring base on it.

run r261104-3000639 · pids 3000639 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/base_maskgt15.log

## 2026-11-04 22:23 — run

Click-to-object Mahalanobis distance (object voxel covariance; surface at sqrt 5) for every click, StarDist and NeRVE masks (click_mahalanobis.py, user: centroid distance scaled by object size; the Euclidean mixtures show no clear clusters). Earlier, click_distance_gmm.py (3b45037): background-click edge distance with brightness splits into bright clicks just off a mask (1,690, 0.3-0.8 um, 1.4-1.5x) and dim clicks 1.7 um out (526, 0.41x; 40% within 1.5 um of a mask, so the 1.5 um edge rule would attach ~210 likely misses to a neighbour), plus neuron_038's 59 misclicks (jumps of 4-19 um for 1-4 frames to dim spots no segmentation has, frames 273-1285).

run r261104-3792074 · pids 3792074 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/click_mahalanobis/run.log

## 2026-11-04 22:40 — result

r261104-3792074 (click_mahalanobis.py a1b... see git log): StarDist, every click's Mahalanobis distance to its nearest object (object voxel covariance): one population that falls off sharply at the ellipsoid surface sqrt5 (median 0.50, p99 2.61; GMM BIC picks 1 component); 2,137 clicks (1.8%) beyond sqrt5, 95% of them outside the mask; NeRVE-segmented fraction and brightness fall with distance (2.24-3: 1,539 clicks, 1.5x bright, 51% NeRVE; beyond 4.47: 0.34x, 32%; beyond 10: 0%, 0.18x, mostly neuron_038). NeRVE masks: 4.5% component at d_M ~17 (0.2% NeRVE, 0.29x) = the ~10.6k clicks on pseudo masks, so the measure recovers NeRVE's own misses. gt_click_rules.py: click-to-nearest-click spacing has an excess below ~1 um over even NeRVE's object spacing (StarDist objects never closer than 2.3 um); misclick statistic (pairwise robust motion reference, ±5 frames) has a clean jitter null (p99.9 1.42 um) but 8,456 clicks beyond 2x that, concentrated in neurons 035/038/065/060 and at their own brightness, i.e. real non-rigid motion the null lacks; not yet a justified rule. Earlier mean-of-neighbours reference was biased by missing neighbours (dead end).

## 2026-11-04 23:05 — decision

User decisions on the Leifer GT for the supp figure: headline accuracy is StarDist strict (a click outside every StarDist mask counts as a missed segmentation, i.e. an error), gt_valid matching, over every kept click; merges and coincident clicks excluded, each with its own justification panel (coincident cutoff 1.0 um, from the click-spacing excess over object spacing). Supp: four accuracies, each with a panel on why it is not the headline: NeRVE masks only; NeRVE plus pseudo masks for clicks it missed (the original 0.885, needs the strongest justification); StarDist plus added masks for clicks outside it; StarDist with growing attachment shells (edge 0.5-3 um, and Mahalanobis sqrt5), noting the shell bias: the GT then names the nearest object, which the tracker is also drawn to. Implementation: gt_tracks_from_clicks.py writes per-click fates (_status.npz) for mask and segmentation_id rules (coincident exclusion now in both, Mahalanobis attach, status-only mode for the NWB label space); score_gt_variants.py scores one run's saved tracks against every variant offline (labelprop never sees the GT); add_click_masks.py builds the StarDist + click-masks project.

## 2026-11-04 23:11 — run

Build the Leifer GT variants (gt_tracks_from_clicks.py at the commit above, coincident 1.0 um everywhere): StarDist strict (in mask only, headline), edge shells 0.5/1/1.5/2/3 um, Mahalanobis sqrt5; NeRVE masks only (max_label 1000); NeRVE + pseudo masks in NWB label space (status only). Into each project's 3-tracking/manual_annotation/variants/.

run r261104-2025465 · pids 2025465 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/gt_variants/sd_strict.log

## 2026-11-04 23:27 — run

Device control: base seed 0 on the StarDist centroid GT re-tracked with labelprop on the GPU (CPU embedding cache), starts when nf400 training (pid 3460809) exits; compare with the CPU 0.9476 before moving more tracking to the GPU (user: GPU free through tomorrow morning). Embedding stays on CPU (i261027-2 GPU crop padding).

run r261104-237552 · pids 237552 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/base_centroidgt_gputrack.log

## 2026-11-04 23:47 — result

r261104-2025465 (GT variants, 891f275) + score_gt_variants.py on base seed 0 CPU tracks: StarDist strict (in mask only, coincident 1.0 um and merges excluded) 112,996 assigned of 115,507 kept, 2,511 segmentation misses; base 0.9404 on assigned, 0.9199 over all kept (headline), gt_valid = nandist at this coverage. Shells: Mahalanobis sqrt5 0.9400/0.9219; edge 0.5 0.9393/0.9254; 1.0 0.9385/0.9305; 1.5 0.9376/0.9324; 2.0 0.9371/0.9330; 3.0 0.9368/0.9336. Strict to 3 um attaches 1,004 clicks of which ~53% are tracked to the attached object (tracker errors +471): the shell-attached clicks are much harder than average. NeRVE masks only: 106,552 assigned, 11,663 missed of 118,215 kept (no merges); NeRVE+pseudo: all 118,215 assigned.

## 2026-11-05 00:10 — run

StarDist + click masks control (supp variant C; add_click_masks.py at 2a.. see git log): new project Leifer_NeRVE_Worm1_reexport_A2_stardist_clickmasks (make_project_like, own segmentation): an ellipsoid of median StarDist extent (semi-axes 2.68 z, 6.87 x, 6.71 y voxels) on background voxels around each of the strict GT's 2,511 missed clicks (median 366 voxels); then its strict GT (coincident 1.0 um) set as final tracks; then base embedded (CPU) and tracked (labelprop CPU) with --save_tracks.

run r261105-2217669 · pids 2217669 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/base_clickmasks.log

## 2026-11-05 00:10 — run

base on the original NeRVE scoring (NWB, every object incl. pseudo masks, segmentation_id GT mapping; supp variant B), CPU embedding and tracking, --save_tracks, to score against the nerve_pseudo variant next to trial 57.

run r261105-2247910 · pids 2247910 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/base_nwb_allobj.log

## 2026-11-05 00:16 — result

nf400 finished training (leifer-reseg-e done; all six sweep variants trained); the scorers pick it up on CPU (centroid and mask GT, old metrics). GPU free; device control r261104-237552 starts on it.

## 2026-11-05 02:37 — decision

r261104-237552: GPU label propagation on base seed 0 (Leifer StarDist, 1.5 um centroid GT) gives 0.9476, the same as CPU (misses 2984 vs 2971, mismatches 2416 vs 2430). Not bitwise identical, but equal accuracy, so further label-propagation runs may use the GPU; embedding stays on CPU (i261027-2).

## 2026-11-05 02:37 — run

Leifer NWB all-object re-tracks (t57, base) failed at argparse (--lab leifer_reexport); relaunched with --lab leifer and GPU label propagation, for the original-NeRVE panel of the Leifer supp figure

run r261105-2919358 · pids 2919358, 2919361 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/t57_nwb_allobj.log, /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/base_nwb_allobj.log · for [[QUE - How much do the segmentation and the ground truth conversion change Leifer tracking accuracy and traces]]

## 2026-11-05 02:37

r261104-3000639 (base on the 1.5 um mask GT, maskgt15 project) ended; tracks saved to base_maskgt15.h5, scored with score_gt_variants with the other runs. r261105-2247910 failed at argparse (bad --lab), relaunched above.

## 2026-11-05 02:38 — run

Rerun of the wk2 trial 9 leifer_stardist benchmark cell (first run hit OSError EIO reading masks from the mount; the maskgt cell with the same embedding succeeded, 0.9141)

run r261105-2920870 · pids 2920870 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/bench_wk2_t9.log

## 2026-11-05 02:41 — run

GPU label-propagation re-track (saved tracks, then score_gt_variants on sd_strict, Mahalanobis and edge shells) of every network benchmarked on the Leifer StarDist project: 14 benchmark cells, 6 local sweep variants, wk2 t50/t57 seeds 1-4; queue /home/charles/.cache/barlow_scratch/train/leifer_reseg/gpu_retrack/queue.txt (scripts at ab90a41)

run r261105-2924449 · pids 2924449 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/gpu_retrack/logs/queue_status.log · for [[QUE - How much do the segmentation and the ground truth conversion change Leifer tracking accuracy and traces]]

## 2026-11-05 02:42 — result

r261104-3067468 (overnight queue) nervegt part: networks tracked on the NeRVE-masks-only Leifer project and scored (score_gt_variants, ab90a41) against nerve_only (NeRVE masks, pseudo-mask clicks counted as segmentation misses: 11,663 of 118,215 kept). gt_valid accuracy on assigned / all kept: base 0.8925/0.8045, t57 0.8909/0.8030, t50 0.8865/0.7991, ref_local_nerve (trained on NeRVE masks) 0.8879/0.8003. About 10k mismatches vs about 2.4k on the StarDist project with the same networks, so the NeRVE segmentation is harder to track even on the clicks it segments, and a network trained on it does no better.

## 2026-11-05 02:42 — result

r261104-3067468 seeds and GT copies, scored on the StarDist variants (gt_valid, assigned/all kept): base seed0 strict 0.9404/0.920, seed1 0.9411/0.921, seed2 0.9381/0.918; edge 3 um shell 0.9343-0.9373/0.931-0.934. Seed 0 tracks from the 1.0 um, mask and maskgt15 project copies score identically to 4 digits (label propagation never sees the GT; tracks differ by a few clicks between CPU runs). GPU tracks (r261104-237552) match CPU to 4 digits on every variant.

## 2026-11-05 02:42 — result

Leifer benchmark on the StarDist project complete except wk2 trial 9 centroid cell (rerun r261105-2920870): wk2 t50 0.9485, t57 0.9448, t63 0.9416, t74 0.9408, t65 0.9402, t76 0.9402, t37 0.9330, t14 0.9266, t24 0.8916 (1.5 um centroid GT, nandist); mask GT 0.8721-0.9418. These cells saved no tracks; r261105-2924449 re-tracks them on the GPU to score them on the strict GT.

## 2026-11-05 02:44 — result

r261104-3067468 merges part (gt_merge_persistence.py 2b0e31d, mask label under each click voxel): StarDist 3,484 clicks (2.9%) in a mask that also holds another neuron's click, 127 pairs; NeRVE masks 4,073 (3.4%), 121 pairs. Merges are brief: no pair has a run of 50+ frames (longest 11, neuron_050/070, click distance median 1.4 um); 2 StarDist and 3 NeRVE pairs share masks in 100+ frames in total; most merged pairs' clicks are 1.5-5 um apart, i.e. two clicked neurons the segmentation fused for a few frames. For the merge panel: a fused mask cannot be right for both neurons, and the rate is the same on both segmentations, so excluding merges does not favour StarDist. NeRVE's own segmentation_id is one-to-one although 3.4% of its clicks sit on a voxel of a mask shared with another click: NeRVE's assignment is not mask membership (check for the NeRVE panel).

## 2026-11-05 02:44 — dropped

GPU OOM with two label propagations at once (wk2_t57_s0); restarted one at a time with OOM retry

dropped r261105-2924449

## 2026-11-05 02:44 — run

GPU re-track queue restarted one job at a time with OOM retry (same queue, same scripts ab90a41)

run r261105-2927825 · pids 2927825 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/gpu_retrack/logs/queue_status.log · for [[EVD - On the strict Leifer StarDist ground truth the base network tracks 92 percent of kept clicks]]

## 2026-11-05 02:55 — run

Traces for GT- vs tracker-tracked comparison (item 8): tracks_to_project.py (9f14b48) made three copies with final tracks = StarDist strict GT (_stardist_strictgt), base tracker tracks matched gt_valid on sd_strict (_stardist_track_base; reproduces 0.9404), base tracker tracks on NeRVE masks matched on nerve_only (_nervegt_track_base). Matched tracks are written at every volume they hold an object (6,275 volumes beyond the GT's assigned clicks for base on StarDist). Dry runs plan only extract_full_traces + traces; snakemake traces sequential, 16 cores.

run r261105-2936414 · pids 2936414 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/traces_cmp/traces.log · for [[QUE - How much do the segmentation and the ground truth conversion change Leifer tracking accuracy and traces]]

## 2026-11-05 02:57 — result

compare_traces.py (validated before commit; reran at the commit below for the record) on existing GT-tracked traces: StarDist 1.5 um centroid GT vs mask GT (same segmentation): raw ratio r = 1.000 on shared frames and same object 100%, paper traces median r 0.992 (only the gap interpolation differs). NeRVE masks vs StarDist mask GT, both GT-tracked: raw green/red ratio per-neuron r median 0.908 (p10 0.78, min 0.56, 33 of 78 below 0.9), paper traces median 0.841 (p10 0.60, 51 of 69 below 0.9). The segmentation alone moves the traces noticeably.

## 2026-11-05 05:10 — run

r261105-2919358 failed: both NWB all-object re-tracks hit CUDA OOM in label propagation (the GPU re-track queue held 6.65 GB). Rerun one after the other on the free GPU from the cached embeddings.

run r261105-3009009 · pids 3009009 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/t57_nwb_allobj.log, /home/charles/.cache/barlow_scratch/train/leifer_reseg/error_patterns/logs/base_nwb_allobj.log · for [[EVD - Networks tracked on NeRVE's own Leifer masks score 5 points lower than on StarDist masks]]

## 2026-11-05 05:12 — result

r261105-2927825 done (28/28): every network benchmarked on the Leifer StarDist project re-tracked on the GPU from its cached embedding (labelprop seed 0) and scored on the strict GT (gt_valid = nandist to 4 digits). Strict all-kept: wk2 t50 0.9223 (seeds 0-4: 0.9220-0.9239), t74 0.9231, lambd0_nf400 0.9215, t63 0.9213, nf400 0.9207, t65 0.9204, base 0.9199, lambd1 0.9200, lambd1_nf400 0.9199, t76 0.9198, ref_local_nerve 0.9198, snap1 t0 0.9176, t57 0.9167 (seeds 0-4: 0.9162-0.9186), lambd0 0.9160, wk1 t1 0.9117, t14 0.9094, samuel137 t63 0.9082, t37 0.9078, zimmer1128 t0 0.9029, t9 0.8964, t24 0.8562. Strict ranks t57 below t74/t63/t65 (it was second on the 1.5 um centroid GT); the top 12 lie within 0.4 pt, about two labelprop-seed spreads (t50 0.19 pt, t57 0.24 pt).

## 2026-11-05 05:12 — result

r261105-2920870 done: wk2 trial 9 leifer_stardist (1.5 um centroid GT) 0.9247, misses 3718, mismatches 4037; benchmark round complete (20/20). The partial rerun had rewritten round_bayes_wk2_leifer_reexport_unified.json with its single record (the driver writes the run's own records); rebuilt it from the jsonl, latest record per tag (20).

## 2026-11-05 05:12 — result

r261105-2217669 done (StarDist + click masks, control C): base embedded and tracked on the _stardist_clickmasks project (2,511 added masks), strict GT there assigns all 115,507 kept clicks: 0.9350 (gt_valid = nandist; tracker misses 968, mismatches 6,538) vs 0.9199 on all kept on plain StarDist strict. So about 1,700 of the 2,511 added-mask clicks (about 69%) are tracked correctly, if the rest is unchanged: a segmentation that misses nothing (built from the GT) gains 1.5 pt.

## 2026-11-05 05:12 — result

r261105-2936414 done: snakemake traces on _stardist_strictgt, _stardist_track_base, _nervegt_track_base (13 min each, exit 0). compare_traces.py (5c74d01), per-neuron Pearson r median (ratio of summed green/red; paper traces): tracker vs GT on StarDist 0.987 / 0.975 (8 of 79 neurons below 0.9 on ratio; same object median 98.7% of shared frames); tracker vs GT on NeRVE masks 0.943 / 0.938 (24 of 78 below 0.9); NeRVE vs StarDist both GT-tracked 0.908 / 0.842 (33 below 0.9); NeRVE GT vs StarDist tracker 0.881 / 0.816. On StarDist masks the tracker changes traces less than the segmentation choice does. Output ~/.cache/barlow_scratch/train/leifer_reseg/traces_cmp/compare/ (summary.json, per-pair csv, trace_agreement.png).

## 2026-11-05 05:23 — result

r261105-3009009 done (NWB all objects incl. pseudo masks, segmentation_id GT, GPU labelprop): t57 0.8859 (misses 1421, mismatches 12157 of 119,019), base 0.8849, reproducing the original NeRVE benchmark (0.885). score_gt_variants --split_label_at 1000 (45a04d2) on nerve_pseudo (gt_valid, coincident 1.0 um excluded, 118,215 kept): t57 0.8899 overall, 0.8949 on the 106,552 clicks NeRVE segmented, 0.8444 on the 11,663 pseudo-mask clicks; base 0.8887 / 0.8932 / 0.8478. Tracking the pseudo masks as objects barely changes accuracy on NeRVE's own clicks (t57 0.8909 without them), so the low NeRVE number is not caused by the pseudo masks: NeRVE-segmented clicks carry ~10.4k mismatches either way.

## 2026-11-05 07:31 — run

fDNC (published model) on the Leifer recording with the new segmentations: StarDist project and NeRVE-masks-only project, every detection, 4 axis conventions, templates 10 + 4 spread plus the GT-free vote, GT-free rescale (fdnc_track_and_score.py @579dea7, --save_tracks). Tracks scored offline with score_gt_variants.py against sd_strict / nerve_only / nerve_pseudo. Axis convention to be picked by fDNC's mean match probability (GT-free), as on Samuel. Paper reports fDNC 79.1%, NeRVE 82.9% on this recording (per-volume template-match accuracy, Yu et al. 2021).

run r261105-3083686 · pids 3083686, 3083837, 3083724, 3083840, 3083653, 3083660, 3083657, 3083663 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/fdnc/leifer/logs/stardist_default.log, /home/charles/.cache/barlow_scratch/fdnc/leifer/logs/stardist_swap.log, /home/charles/.cache/barlow_scratch/fdnc/leifer/logs/stardist_mirror.log, /home/charles/.cache/barlow_scratch/fdnc/leifer/logs/stardist_swap_mirror.log, /home/charles/.cache/barlow_scratch/fdnc/leifer/logs/nervegt_default.log, /home/charles/.cache/barlow_scratch/fdnc/leifer/logs/nervegt_swap.log, /home/charles/.cache/barlow_scratch/fdnc/leifer/logs/nervegt_mirror.log, /home/charles/.cache/barlow_scratch/fdnc/leifer/logs/nervegt_swap_mirror.log · for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-11-05 07:31 — decision

Misclick rule (user, 2026-11-05): clicks with no segmentation nearby cannot be settled without more manual annotation. They stay in the strict GT and count as segmentation misses; no drop rule.

## 2026-11-05 07:31 — decision

NeRVE's pointStatsNew has tracked points without masks: rows past pointIdx in straightPoints / trackIdx are 'error correction' positions (NeRVE tutorial). 158,778 tracked detections and 21,799 tracked extra points (12%); per track median 3%, 90th pct 39%. Detections' pointIdx = the re-export's NeRVE mask labels (centroids within ~0.35 px). Plan (agreed with user): score NeRVE's tracks only on NeRVE masks; headline = detections only; supplement = extra points mapped through NeRVE's own CLstraight transforms to the mask they land in. Never transfer NeRVE points onto StarDist masks (user: the points would differ with another segmentation).

## 2026-11-05 07:35 — run

NeRVE's own tracks to saved tracks (nerve_tracks.py @82c84fb): detections only, and detections + error-correction points mapped to the mask under them in the NeRVE-only project and in the NWB segmentation (with pseudo masks). 4-volume check: detections' straightPoints map back to their own mask label 100%, rawPoints within 0.28 px; 53 of 67 extra points land inside a NeRVE mask (being counted: tracked vs untracked detection).

run r261105-3087916 · pids 3087916 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/nerve_tracker/nerve_tracks.log · for [[EVD - Published trackers run on the shared NWB benchmark]]

## 2026-11-05 07:38 — result

Worst Leifer trace neurons (inspect_trace_neurons.py @f8f20c1, out ~/.cache/barlow_scratch/train/leifer_reseg/traces_cmp/inspect/). StarDist, GT-tracked vs base-tracked: the 4 neurons with r_paper < 0.5 (023, 015, 068, 024) and 058 are the same low-activity neurons that also disagree GT vs GT across segmentations (SNR 2.0-3.8 vs median 4.5 for r >= 0.9; Spearman r_paper vs SNR 0.43). Differences are hundreds of short runs (longest 4-27 volumes), almost never a swap with another GT neuron (diff_to_other_gt <= 0.05): one side holds an unannotated object 3-4 um away. Two kinds: ambiguous clicks (023, 058: click about equidistant, within 1 um, from both objects on 77-88% of differing frames; 023's click sits 1.19 um from its object even on agreeing frames vs 0.53 typical) and clear tracker flicker (015, 024, 068: click 0.6-1.0 um from the GT object, tracker's object 2.8-3.5 um away). Spikiness (median |x_t - mean of neighbours|) of the ratio trace: the tracker's trace is smoother than the GT-tracked one in 73 of 79 neurons and all 7 with r_paper < 0.8 (023 0.030 vs 0.059, 024 0.022 vs 0.058): single-volume ratio spikes in the GT traces coincide with differing frames, i.e. the clicks hop between two objects. On NeRVE masks the tracker is smoother in only 58%. Caveat: smoother is not proof of correct; a tracker consistently on the wrong neighbour is smooth too.

## 2026-11-05 07:39 — idea

Nguyen et al. 2017 (PLoS CB, NeRVE) on this recording: GT 'previously hand annotated and published' (Nguyen 2016), 'nine researchers collectively annotated 70 neurons from each of the 1519 volumes' with a MATLAB GUI; not said whether annotators saw NeRVE output (likely independent, it predates NeRVE). NeRVE fills missing neurons by consensus + TPS interpolation and also REPLACES a detected position more than 3 s.d. from the consensus (the pointStatsNew extra points). Their accuracy pairs automated and manual neurons by closest match, only 'neurons consistently found by both', error = matched to a manual neuron other than its most frequent one. fDNC paper (Yu 2021) on the same recording: fDNC 79.1%, NeRVE 82.9%, single-template NeRVE 73.1% (per-volume match accuracy vs a template).

## 2026-11-05 07:48 — result

r261105-3083686 (fDNC) and r261105-3087916 (NeRVE tracks) recorded in [[EVD - On the new Leifer ground truths fDNC tracks 82 percent and NeRVE 89 percent of assigned clicks]]: fDNC 0.821 on StarDist strict (base 0.940), NeRVE detections-only 0.886 on its own masks (base 0.893). NeRVE's 12% extra points mostly sit on detections another NeRVE track holds (17,830 of 21,799) and recover only 4.2% of the clicks NeRVE did not segment, so they do not explain the gap on all-kept clicks. Leifer cell of [[EVD - Published trackers run on the shared NWB benchmark]].

## 2026-11-05 07:58 — result

Coincident clicks in the Samuel GTs (NWB clicks, grid spacing from the NWB; quick inline check): other-neuron click within 1.0 um: 153 46 of 120,923 clicks (0.04%, one pair neuron_003/014, 23 frames in 9 runs, median distance 2.62 um), 137 none, 184 366 of 210,336 (0.17%, 10 pairs). Leifer for reference: 804 of 119,019 (0.68%, 74 pairs). Within 1.5 um: 153 0.26%, 137 0.04%, 184 7.87%, Leifer 1.30%. 184 is denser overall (nearest other click median 2.20 um vs 3.2-3.4 in 153/137, 4.5 Leifer; 36% of clicks have a neighbour within 2 um) and has persistent close pairs: neuron_054/080 median 1.28 um over 2,310 shared frames (p10 1.07), 004/022 median 1.58, 018/055 1.71: either tightly packed neighbours or one neuron annotated twice; not checkable without the images (and 184's 0.3 um xy spacing, unlike 0.45 for 153/137, is unverified). Samuel evals give the tracker the clicks as objects, so 1 um coincidences can cost at most ~0.2 pt.

## 2026-11-05 08:03 — result

gt_click_coincidence.py @43203f6 on Samuel 153/137/184, Flavell, Leifer re-export: recorded in [[EVD - Coincident GT clicks are rare in the Samuel and Flavell ground truths and in Leifer are brief jumps onto a neighbour]]. Leifer's coincident pairs are normally 2.6-5 um apart and touch in short runs (one click jumping to the neighbour), supporting their exclusion panel.

## 2026-11-05 08:05 — result

r: neuron_removal_scores.py @62921a6 recorded in [[EVD - Half of the base network's Leifer StarDist errors are clicks where every tracker agrees on another object]]; consensus_error_clicks.csv (3,618 clicks) prepared for the user's manual check of suspect Leifer neurons.

## 2026-11-05 08:26 — run

Labelprop seeds 1-4 for wk2 t74, wk2 t63 and lambd0_nf400 on the Leifer StarDist project (GPU queue, one job at a time, saved tracks scored on sd_strict and shell variants), for seed spread in the strict-GT network table; base, t50 and t57 already have 5 seeds.

run r261105-3124433 · pids 3124433 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/gpu_retrack/logs/queue_status.log · for [[EVD - On the strict Leifer StarDist ground truth the base network tracks 92 percent of kept clicks]]

## 2026-11-05 08:26 — done

committed at 891f275 and cited in the StarDist DATA notes

Done: commit barlow_track/scripts/project_reanalysis/gt_tracks_from_clicks.py and cite the commit in the stardist Leifer DATA note

## 2026-11-05 08:55 — run

Click snapshots for the user's manual check (click_snapshots.py @c85a7fc), 16 consensus-error clicks for each of the 10 suspect Leifer neurons, out /home/charles/.cache/barlow_scratch/train/leifer_reseg/neuron_removal/snapshots/. First look, neuron_023: a pair of touching nuclei with the click on their boundary in all 8 sampled volumes; the GT object and the trackers' object are the two halves, so its 36% consensus errors are an ambiguous annotation, not tracking.

run r261105-3145457 · pids 3145457 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/neuron_removal/snapshots.log

## 2026-11-05 08:57 — run

NeRVE vs StarDist per-click comparison for base seed 0 (nerve_vs_stardist_clicks.py), all volumes: why NeRVE masks are harder. 30-volume test: clicks failing only on NeRVE have NeRVE splitting the StarDist object into 2+ masks in 68% (12% for clicks correct on both), nearest NeRVE neighbour 2.57 vs ~3.6 um; NeRVE masks are ~4x smaller overall (median ~120 vs ~520 voxels).

run r261105-3147438 · pids 3147438 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/nerve_vs_stardist/run.log · for [[EVD - Networks tracked on NeRVE's own Leifer masks score 5 points lower than on StarDist masks]]

## 2026-11-05 08:59 — issue

Samuel 184-004's NWB grid_spacing (0.3, 0.3, 1.75 um) is probably wrong: wbfm convert_harvard_data.py writes 0.45 for Harvard data, and in pixels 184's head matches 153/137 (click-cloud s.d. major/minor 43.3/11.5 px vs 46.9/12.0 and 42.8/12.9; nearest xy click 3.7 vs 3.8/4.3 px). At 0.45 um, 184 has no coincident clicks within 1 um (vs 366 at 0.3) and its 'persistent close pairs' vanish (nearest other click median 3.01 um, like 153/137). Consequence: make_per_recording_rounds.voxel_aspect read the NWB, so round_wk1_samuel_184 and round_bayes_wk3_samuel_184 trained with visnet_z_scale 5.833 instead of 3.889 (z stretched 1.5x in the ViSNet position input). Only 184 is affected; crop sizes are in voxels. Needs confirmation from the Samuel lab or the raw metadata.

- [ ] issue i261105-1 · affects [[DATA - Samuel lab recordings 137 and 184-004 with ground truth]], [[EVD - Coincident GT clicks are rare in the Samuel and Flavell ground truths and in Leifer are brief jumps onto a neighbour]]

## 2026-11-05 08:59

i261105-1: [[EVD - Coincident GT clicks are rare in the Samuel and Flavell ground truths and in Leifer are brief jumps onto a neighbour]] revised in place with the 0.45 um row for 184; its conclusion no longer depends on 184's spacing.

## 2026-11-05 09:00 — result

r261105-3147438 done: NeRVE's deficit is its over-splitting of nuclei (recorded in [[EVD - Networks tracked on NeRVE's own Leifer masks score 5 points lower than on StarDist masks]]). r261105-3145457 done: snapshots for 10 neurons. Quick look: 023 and 044 are touching-nucleus pairs with the click on or near the shared boundary (ambiguous GT); 072's clicks sit clearly in the GT object while every tracker takes the nucleus below (real tracking errors, keep).

## 2026-11-05 09:07 — decision

Sweep accuracy objective refactored (f3fadf8): the eval now runs inside the trial's GPU job with label propagation on the GPU (objective_track_device defaults to cuda, objective_crop_window defaults to slide like training per i261027-2, CUDA OOM redoes all seeds on CPU). objective_cpu_job is opt-in only. Reason: Leifer labelprop takes ~2 min on a GPU vs 822 s on 16 cluster CPUs (job 7770675), and the CPU job also embeds on CPU (827 s). Past round_* templates keep objective_cpu_job: true as the record; new rounds should drop it and cut objective_minutes. AGENTS.md updated (git-ignored).

- [ ] new round templates: drop objective_cpu_job, lower objective_minutes

## 2026-11-05 09:13 — decision

Eval bottleneck found and fixed (43e7a0a): wbfm's get_all_neuron_metadata_for_single_time evaluates volumes_with_no_neurons (scans every frame) on each call, 0.2-0.3 s a frame on Leifer/Zimmer, ~5-8 min per full-video eval on GPU and CPU alike (training already bypassed it in volume_data). data_loading._single_time_metadata gives identical output on 4 projects incl. empty frames at ~0.0005 s a frame. Scores unchanged by construction. Unrelated: test_barlow_position projector-norm tests (2) fail on HEAD before this change.

## 2026-11-05 09:14

r261105-3124433: wk2 t74, t63 and lambd0_nf400 seeds 1-4 done; seed s.d. 0.08-0.17 points on strict all-kept for all five 5-seed networks, means t50 .923 > t74 .922 > lambd0_nf400 .921 > t63 .920 > t57 .917. Added to [[EVD - On the strict Leifer StarDist ground truth the base network tracks 92 percent of kept clicks]].

## 2026-11-05 09:15

r261102-3398854 (z-split stage 1, pairs): 53,396 vertically touching mask pairs in 1,536 frames, contact_frac p10/50/90 0.05/0.36/0.91; used as the input of stage 2 (r261102-3403676), whose result is recorded there.

## 2026-11-05 09:17 — decision

CPU label-propagation phase profile (labelprop_phases.py at 6bec9b4, Leifer wk2 t57 embedding, 196,626 objects, 3 seeds, local Xeon): propagation is ~97% of CPU labelprop time, 517 s/seed on 1 thread, 102 s/seed on 16 (8.3 cores busy, so torch_sparse spmm scales sub-linearly); NNDescent 34 -> 11 s, spectral 7 s, SVD 5 s, Python edge loop 1.6 s (vectorised in the commit after 43e7a0a, identical edge_index). No other single-threaded step of note in labelprop; the one in the eval was the per-frame metadata scan (43e7a0a). GPU labelprop ~2.3 s/seed. Local 16-thread seed time is 3x the cluster job 7770675's 30 s/seed (other CPU, not checked).

## 2026-11-05 09:17 — run

Cluster state 2026-11-05 09:15, read from the round folders (no squeue here).

wk3 Bayesian rounds: all 8 dispatchers alive, 52-73 of 120 trials scored per recording. No traceback in any job log since 2026-11-03 17:00.

Best (trial) against the earlier best under the same scoring:
- flavell 0.982 (t57) vs 0.949 (wk1 t4)
- 0316-03 0.989 (t49) vs 0.974
- 0324-02 0.980 (t47) vs 0.956
- 0330-02 0.970 (t61) vs 0.947
- samuel 153 0.985 (t49) vs 0.984
- samuel_184 0.957 (t27) vs 0.933
- zimmer_1123 0.987 (t45) vs 0.979 (visnet_full t27)
- zimmer_1210 0.942 (t28) vs 0.920

On every recording, fusion add beats visual_only (best add vs best visual_only, 1-22 pt; 0330-02 0.970 vs 0.745, samuel 0.985 vs 0.926). visual_only has only 12-17 trials per round, mostly before Ax converged, so its bests are undersampled.

The winning lambd_obj differs by recording: about 0-0.05 on flavell 2025-07-01 and zimmer_1210 (top 2), 0.35-0.8 elsewhere. self_layers 0, 1 or 2 by recording.

wk1 grid restarts:
- zimmer_1128 nf100: 10/12 done, all 0.958-0.976 (saturated).
- wk1f30 nf30: 10/12 done. lambd_obj 0 costs visual_only 4-5 pt (0.920/0.887 vs 0.959/0.941 at 0.67) but not add (0.954/0.961 vs 0.962).
- samuel_137: 3/9 done. visual_only lambd_obj 1.0 at seed 43 scored 0.587, against 0.853/0.829 at 0 and 0.67.

All scores are the sweep objective (each trial scored on its own training recording), best of about 60 trials.

## 2026-11-05 09:17 — done

done 2026-11-05: wk3 bests above in the notebook; zimmer_1210 and wk1 grid dispatchers running

Done: On return: check zimmer_1210 and wk1 grid dispatchers started; tally wk3 per-recording bests with best_of_table.py

## 2026-11-05 10:16 — done

2026-11-05: GPU ~2.3 s/seed vs 102 s/seed on 16 local threads (labelprop_phases.py, 6bec9b4) and 30 s/seed in cluster job 7770675; sweeps now track on the trial GPU by default (f3fadf8)

Done: Speed: time full-video label propagation on GPU (--track_device cuda) in its own job vs CPU

## 2026-11-05 10:17 — run

End-to-end check of the new sweep default (f3fadf8 + 43e7a0a + 167a169) on an idle local GPU: full Leifer StarDist eval of wk2 trial 57 as a trial job now runs it (fresh embedding, --device cuda --crop_window slide, labelprop on cuda, 25 seeds, project GT). Wall time sets objective_minutes; accuracy should match the GPU re-track of the cached CPU embedding (r261105-2927825).

run r261105-3209644 · pids 3209644 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/train/leifer_reseg/gpu_inline_eval/eval.log

## 2026-11-05 10:24 — surprise

i261105-1: Susoy et al. 2021 (Cell 184:5122, doi:10.1016/j.cell.2021.08.024, zotero:975IU5CK), STAR Methods p. e3: the Samuel-lab spinning-disc system has a pixel size of 0.45 um and ~1.75 um between optical sections (20 per volume, 256x512 px area). This matches wbfm convert_harvard_data.py (0.45, 0.45, 1.75) and the pixel-space match of 184 with 153/137, so 184's NWB value of 0.3 um xy is very likely wrong. Not proof: 184-004 has 16 planes, not 20, so it may be a different acquisition; the issue stays open until the lab or the raw metadata confirms.

## 2026-11-05 10:25 — result

r261105-3209644 done: full Leifer StarDist eval of wk2 t57 as a sweep trial now runs it (fresh GPU embedding, slide crops, GPU labelprop x25, metadata fix): 7 min 13 s wall (embed 239 s for 1,536 frames / 196,626 objects, track 118 s incl. 25 seeds at 2.3-3.0 s and spectral sync), peak RSS 5.6 GB. Accuracy 0.94479 vs 0.94475 for the GPU re-track of the cached CPU embedding (misses 2,846 vs 2,871). Embeddings agree to p99 relative row difference 1.9e-6; 4 of 196,626 rows differ by > 1e-3 (max 6%). Suggest objective_minutes ~60 for Leifer-size videos on the cluster (local GPU, network-mounted data).

## 2026-11-05 10:32 — decision

User decisions (2026-11-05) for the next sweep:
1. Hold recordings out of hyperparameter tuning; the frozen base config is chosen on a tuning subset only.
2. The objective is a lower quantile across the tuning recordings (about the lower quartile), not the mean, which saturated datasets dominate, and not the min, which is too noisy.
3. The base settings are frozen after this sweep. Hyperparameters may then change per recording only through GT-free information (e.g. crop size from physical voxel size).
4. Resegmenting Samuel with stardist is a subproject; the user expects problems, because the data look different.

## 2026-11-05 10:32 — surprise

GT-free proxy check (scripts/scratch/gt_free_proxy/proxy_corr.py, proxy_partial.py, uncommitted). Method: 11 wk2/wk3 per-recording rounds (Zimmer 1128 left out, n=9), image-plus-position trials only (37-94 per round); last logged epoch of each trial's stats.json; Spearman correlation with final accuracy.

Raw correlations:
- The summed val_loss has none (|rho| < 0.4, except Flavell 0316-03 at -0.62).
- The per-batch val loss and the train loss correlate at -0.46 to -0.83 on most recordings, but only -0.07/-0.47 on Samuel 137 and -0.18/-0.50 on Samuel 184, the least saturated.

Partial correlations, controlling for num_frames, epochs, lr, lambd_obj, self_layers and the three augmentation settings: these shrink to -0.15 to -0.49 (val loss) and +0.05 to -0.59 (train loss). The descriptor mean fraction is the best single one on Samuel 137 (partial -0.64).

Reading: most of the loss-accuracy correlation is training budget. Within a budget, lower loss is weakly better, which is not enough to tune on.

- [x] test tracking self-consistency (agreement between label-propagation seeds, track purity) as a GT-free proxy on the cached embeddings of the wk2/wk3 rounds (emb_cache holds every trial)

## 2026-11-05 10:57 — decision

User approved (2026-11-05) the held-out split for the shared-config sweep.
- Tuning set: Zimmer 1210 and 1123, Flavell 03-30-02 and 03-24-02, Samuel 137 and 184.
- Held out: Zimmer 1128, Flavell 2025-07-01, 03-16-03 and 03-30-01, Samuel 153, Leifer stardist.
- Objective: 25th percentile across the tuning recordings of (accuracy minus that recording's best wk2/wk3 score).
Also approved: the Samuel stardist resegmentation (start with 153) and testing tracking self-consistency as a GT-free proxy.

## 2026-11-05 11:13 — run

Timing test of the GT-free consistency proxy (consistency_proxy.py) on wk3 zimmer_1210 trial 28. The first attempt (2 trials plus a second-seed run, under timeout 900) never finished a trial, CPU-bound at about 11 cores; this rerun prints per-stage times.

run r261105-3249979 · pids 3249979 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy_test.log

## 2026-11-05 11:14 — answer

Default to true (user, 2026-11-05): frozen view pairs are non-standard and came from a bug, not a design choice. At least one more sweep will run with it on, so problems would show there. Done in 2085014.

answered a261027-1

## 2026-11-05 11:14 — decision

a261027-1 acted on (2085014): reaugment_each_epoch now defaults to true via train_barlow_clusterer.reaugment_enabled (unit test), the project template writes it explicitly, and s137_pilot/run_pilot.py pins false for its recorded variants. Compute is unchanged: the augmentation was already recomputed every epoch from the same seed. The LR is constant, so the Flavell reaug run's stop at epoch 32 has no schedule confound. Saved configs of position runs from 10-05 to 11-05 lack the key and need false to reproduce (AGENTS.md). Cluster rounds change only after a pull, and a killed trial of a running round retrained after the pull would re-augment.

- [ ] weekend round: sweep augmentation strength with reaug on, plus a frozen-vs-reaug control (best config, 2-3 training seeds, Flavell + one other dataset)

## 2026-11-05 11:18 — dropped

GPU label propagation OOM after 9 seeds (fragmentation), CPU fallback ~1 h per trial; restarting with expandable_segments

dropped r261105-3249979

## 2026-11-05 11:18 — run

Rerun of the consistency-proxy timing test with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run r261105-3408664 · pids 3408664 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy_test.log

## 2026-11-05 11:23 — run

Label propagation memory: torch_sparse.spmm gathers an (edges x labels) buffer per layer, about 3.6 GB on Zimmer 1210 (260k objects, k=20, ~175 labels), freed and reallocated 100 times per seed at a seed-dependent size. That fragments the CUDA cache, and multi-seed GPU propagation OOMed at seed 10 (r261105-3249979). Not a leak: each seed's result goes to the CPU. Fix (uncommitted): utils_label_propagation.sparse_adjacency builds the normalized adjacency once as CSR, and each layer is adj @ H. Tests: 16/16 pass, including 2 new ones (CSR matches spmm to 1e-5 with identical labels on a kNN graph; a shared adjacency equals one built per call). Real-data check: wk3 zimmer_1210 trial 28 rescored from its cached embedding with GPU label propagation, against the recorded 0.9418 (3007 misses, 6394 mismatches); peak GPU memory logged.

run r261105-3413687 · pids 3413687 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/rescore_t28.log

## 2026-11-05 11:42 — run

Samuel 153 stardist resegmentation (subproject): wbfm_tracker_baseline/samuel_resegment/153_stardist, a copy of samuel/153 (whose masks hold only GT neurons) reading the source's preprocessed volumes (1331 x 20 x 320 x 192). Model students_and_lukas_3d_zarr (Zimmer-trained) at native resolution. Run locally with RUNME.sh -c -s segmentation (alt_preprocessing bbox, then segmentation).

run r261105-3423621 · pids 3423621 · host charles-Precision-7820-Tower · logs /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline/samuel_resegment/153_stardist/snakemake/log_snakemake_local_20261105.out

## 2026-11-05 11:45 — run

CSR vs spmm label comparison on Zimmer 1210 trial 28 (4 seed volumes, 100 layers, GPU and CPU): fraction of objects whose top label differs, and max probability difference

run r261105-3441300 · pids 3441300 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/csr_vs_spmm.log

## 2026-11-05 11:47 — surprise

Other session (BarlowTrack_cluster_runs) replaced torch_sparse.spmm with a CSR adjacency in label propagation (uncommitted, utils_label_propagation.py): fixes GPU OOM at seed 10 on Zimmer 1210 (spmm's edges x labels buffer, ~3.6 GB/layer). CPU check here on the Leifer t57 graph (196,626 objects): spmm 1064 vs CSR 93 ms/layer at 16 threads (8.3 vs 14.7 cores), 571 vs 61 s/seed on 1 thread, max diff 3e-7. So CPU labelprop drops from ~45 to ~4 min per 25-seed eval once committed; the 'propagation is 97% of CPU time' entry above describes spmm.

## 2026-11-05 11:47 — run

Smoke test of the new shared-config dispatcher optimize_shared_config.py (2 trials, tuning samuel_184 + zimmer_1210, 12 frames, 1 epoch, local submitit). Plumbing only.

run r261105-3452335 · pids 3452335 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/round_shared_smoke/smoke.log

## 2026-11-05 11:47

r261105-3441300 (GPU part): CSR adjacency vs torch_sparse spmm on zimmer_1210 t28, 4 seeds x 100 layers: identical top labels, max |prob diff| 3e-7. So r261105-3413687's 0.94166 vs recorded 0.94181 is GPU-vs-CPU float order, not CSR. r261105-3413687: GPU full eval with CSR, 7.3 min, peak 1.05 GB (spmm OOMed at seed ~10). r261105-3408664: first proxy test, agree_mean 0.0096 was the perm-indexing bug (fixed).

## 2026-11-05 11:48 — run

GT-free tracking self-consistency proxy (consistency_proxy.py, CSR propagation on GPU) on 12 accuracy-stratified trials each of wk2_samuel_137, wk3_samuel_184, wk3_zimmer_1210 (the non-saturated recordings), to see whether agreement/confidence/dup rank trials like accuracy does. ~3 min/trial.

run r261105-3461106 · pids 3461106 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/run_all.log

## 2026-11-05 11:49 — run

GT-free voxel-size consistency check from segmented-object spread (voxel_size_from_masks.py, 6 frames/recording), calibrated on Zimmer (0.325/1.5 um) and Leifer StarDist (0.3226/1.5 um): checks the suspect Samuel 184 xy (0.3 vs 0.45) and Flavell z (0.3 isotropic?) before a um-based crop rule. zimmer_1128 sigma_xy 3.02 vox, sigma_z 1.54; samuel_137 1.80 / 1.05 (implied xy 0.55 um vs stated 0.45; GT-seeded masks, different segmenter).

run r261105-3468094 · pids 3468094 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/voxel_sizes.log

## 2026-11-05 11:53 — decision

Staged (not launched) config_files/round_bayes_wk4_leifer_stardist_strict: wk3's grid on the Leifer StarDist project, scored against the strict click GT under a new benchmark label leifer_stardist_strict (DATASETS entry added; the old leifer_stardist label scores the 1.5 um centroid GT, so reusing it would have mis-skipped cells). num_frames to 1000, eval in the GPU job, train_hours 72. Preflight OK (79 neurons, 1536 frames); template attaches in Ax. Needs the uncommitted DATASETS/make_per_recording_rounds/CSR changes on the cluster.

## 2026-11-05 11:53 — surprise

test_staged_round_templates_create_an_experiment_and_resume failed on a clean tree (KeyError 'values': a staged scratch round has range params); test now takes a range's bounds. 72/72 pass.

## 2026-11-05 11:54 — surprise

r261105-3468094 (voxel_size_from_masks.py, uncommitted; median per-object std of mask voxel coords, 6 frames each), calibrated on Zimmer StarDist (sigma_xy 3.0 vox = 0.98 um). Leifer StarDist comes out at xy 0.322 um vs 0.3226 stated: the xy estimate is right across labs with the same segmenter. z is not (Leifer 1.9 vs 1.5 um): axial blur differs, ignore z. Segmenter matters ~1.45x (Leifer NeRVE masks give 0.47). (1) Samuel 184 sigma_xy 1.64 vox vs 137 1.80 and 153 1.71, all GT-seeded watershed: 184's voxels match 137/153 (stated 0.45 um), so its NWB grid_spacing xy 0.3 is very likely wrong (0.3 would make its nuclei ~1.5x larger in voxels than the others). (2) Flavell (5 recordings, consistent): sigma_xy 1.2 vox, sigma_z 1.75 vox: under the stated 0.3 um isotropic voxels nuclei would be 0.36 um sigma, ~2.7x smaller than Zimmer StarDist's, beyond the segmenter spread; implied xy ~0.8 um/vox. If so, the 64-vox xy crop spans ~50 um on Flavell (Zimmer: 21 um), i.e. half the head, and visnet_z_scale 1.0 is wrong too. Needs Flavell's real voxel size (Atanas 2023 methods; not found by a quick search).

- [ ] Find Flavell's true voxel size (Atanas et al. 2023 Cell methods / NWB provenance) and Samuel 184's; then the um crop rule
- [x] Commit voxel_size_from_masks.py before citing r261105-3468094 in a node

## 2026-11-05 11:55 — run

Check that eval_accuracy on the _strictgt project (the wk4 Leifer StarDist objective) reproduces BarlowTrack_local_runs' score_gt_variants number: wk2 leifer t57 seed 0, expected accuracy_assigned 0.9370.

run r261105-3497624 · pids 3497624 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/strict_check/eval.log

## 2026-11-05 11:55 — dropped

GPU half done (identical labels, 3e-7); CPU half redundant with BarlowTrack_local_runs' CPU benchmark (3e-7) and its cached 8.2 GB starved the smoke test

Stopped PID 3441300.

dropped r261105-3441300

## 2026-11-05 12:04 — run

Resume test of optimize_shared_config.py on the smoke round: r261105-3452335's two trials each lost one cell to GPU OOM from my leftover r261105-3441300 (8.2 GB cached); the rest of the plumbing worked (per-label accuracy, regret, failed cell fails the trial). Expect: the missing cells are retrained, then the trials complete.

run r261105-3528506 · pids 3528506 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/round_shared_smoke/smoke_resume.log

## 2026-11-05 12:07 — result

r261105-3461106 partial: on 12 accuracy-stratified trials each, mean agreement of the 25 labelprop runs with their spectral-sync consensus tracks accuracy at Spearman 0.99 (all) / 0.94 (top half, n=6) on BOTH wk2_samuel_137 and wk3_samuel_184; mean consensus confidence 0.99/0.89 and 0.98/0.89. Val loss reached at most -0.5 partial. Next: every trial of samuel_137 to control for the settings (budget confound).

## 2026-11-05 12:07 — run

Consistency proxy on all ~109 wk2_samuel_137 trials (seeded with the 12 done), for a partial correlation controlling for num_frames/epochs/lr/lambd_obj/self_layers/augmentation, i.e. whether agreement says more than the training budget.

run r261105-3531457 · pids 3531457 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/all_samuel137.log

## 2026-11-05 12:08 — run

Samuel 153 StarDist re-segmentation (r261105-3423621) finished: median 51 objects/volume (28-94 over every 60th volume) vs ~76 GT-seeded masks/volume in the NWB. Now assigning the GT centroids to the StarDist masks with Leifer's strict rule (mask at the click voxel, edge 0, coincide 1 um, merges unassigned; voxel 1.75/0.45/0.45 um from the NWB) to measure GT coverage.

run r261105-3532825 · pids 3532825 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/samuel_gt.log

## 2026-11-05 12:10 — run

Samuel 153 StarDist GT coverage (r261105-3532825, strict rule): 30% of 120,923 GT clicks assigned; 36% outside every mask, 34% in masks shared by 2+ GT neurons, median neuron coverage 10%. Unusable as is (the user expected trouble). Likely cause: Samuel nuclei are ~1.7 vox xy spread vs ~3.0 on the Zimmer data the StarDist model was trained on. Testing resampling before StarDist (stardist_rescale_test.py, 8 volumes, zooms z,xy = 1,1 / 1,1.4 / 1,1.75 / 1.17,1.75).

run r261105-3535238 · pids 3535238 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/stardist_rescale.log

## 2026-11-05 12:10 — run

r261105-3497624 died on CUDA OOM loading the model (11:55, while r261105-3441300 held 8.2 GB); relaunched the same strict-GT check (expect 0.9370).

run r261105-3536566 · pids 3536566 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/strict_check/eval.log

## 2026-11-05 12:31 — result

r261105-3535238 (Samuel 153, 8 volumes, 665 GT clicks, strict rule): resampling before StarDist helps a lot. zoom z,xy 1,1: 32% assigned (67% in a mask, 34% in masks shared by 2+ GT neurons, 56 objects); 1,1.4: 58% (84/26, 98); 1,1.75: 70% (88/18, 134); 1.17,1.75 (Zimmer object size): 74% (91/17, 155 objects). Leifer strict for comparison: 95% of clicks. Trying stronger zooms for the remaining merges.

## 2026-11-05 12:32 — run

Samuel StarDist zoom scan, round 2: z,xy 1.17,2.1 / 1.5,1.75 / 1.5,2.1 / 2,2.1 / 1.17,2.5.

run r261105-3554388 · pids 3554388 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/stardist_rescale2.log

## 2026-11-05 12:32 — result

r261105-3528506: optimize_shared_config.py resume works: both smoke trials reattached, only the 2 OOM'd cells retrained, then both completed with objectives (-0.40, -0.29: lower-quartile regret of 2 labels, 12-frame 1-epoch trainings). shared_results.jsonl keeps the earlier failed line per trial; the last line per trial is current. r261105-3461106 (12 trials, wk3_zimmer_1210): agreement separates broken from working networks (rho 0.62 all, agree_p10 0.73) but not within the top half (0.936-0.943; agree_mean rho -0.77, plateau ~0.83). So the consistency proxy works on the hard Samuel recordings, not near saturation.

## 2026-11-05 12:33 — issue

Samuel 184-004's NWB grid_spacing xy (0.3 um) is very likely wrong: its segmented nuclei are the same size in voxels as Samuel 137/153's (0.45 um) within 10% ([[EVD - Segmented nucleus sizes contradict the stated voxel sizes of the Flavell and Samuel 184 recordings]]). Every samuel_184 run with --z_scale_from_voxels (round_bayes_wk3_samuel_184) used visnet_z_scale 1.75/0.3 = 5.83 instead of ~3.9. Accuracies stay valid as measured; the tuned config may be off and the per-recording z-scale rule is wrong for 184.

- [x] issue i261105-2 · affects [[DATA - Samuel lab recordings 137 and 184-004 with ground truth]]

## 2026-11-05 12:33 — issue

Flavell NWBs state 0.3 um isotropic voxels, but their segmented nuclei are ~2.7x smaller in voxels than Zimmer's in xy and taller in z than in xy, implying ~0.6-0.8 um xy pixels ([[EVD - Segmented nucleus sizes contradict the stated voxel sizes of the Flavell and Samuel 184 recordings]]). Flavell runs with --z_scale_from_voxels used visnet_z_scale 1.0, and the 64-voxel xy crop may span ~40-50 um there (Zimmer: 21 um). Need the true voxel size (Atanas et al. 2023 methods / acquisition metadata).

- [ ] issue i261105-3 · affects [[DATA - Flavell lab benchmark recordings]]

## 2026-11-05 12:34 — decision

Staged (not launched) config_files/round_shared_v1: shared-config Ax sweep (optimize_shared_config.py), 6 tuning recordings, lower-quartile regret vs wk2/wk3 bests (unchanged as of 12:35), wk3 grid minus fusion (add fixed), num_frames to 1000, 60 trials x 6 GPU jobs, 4 trials at a time. Template loads and builds all 6 cells with wk3's eval settings. sbatch_dispatcher.sbatch now takes BARLOW_DISPATCHER to run it as a Slurm job. Open: samuel_184 and Flavell z-scales from the suspect grid_spacing (i261105-2/3), kept as in wk3.

## 2026-11-05 12:41 — result

r261105-3554388 (Samuel 153 zoom scan 2, 8 volumes, 665 clicks; assigned / in mask / shared / objects): 1.17,2.1: 79/94/15/180; 1.5,1.75: 76/93/17/176; 1.5,2.1: 83/93/10/198; 2,2.1: 88/94/6.6/236; 1.17,2.5: 83/91/8/194. Larger z zoom cuts merges most. Object counts (236 vs ~83 clicks/volume, ~98 GT neurons) may mean over-splitting: scan 3 adds median object volume (um^3) to compare with Zimmer StarDist (615 vox x 0.158 um^3 = ~97 um^3).

## 2026-11-05 12:41 — run

Samuel zoom scan 3 with object volumes: 1,1 / 1.17,1.75 / 2,2.1 / 2.5,2.1 / 2,2.5 / 3,2.5.

run r261105-3564506 · pids 3564506 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/stardist_rescale3.log

## 2026-11-05 12:58 — result

r261105-3536566: eval_accuracy --mode trained on the _strictgt project (the wk4 Leifer StarDist objective, nandist) reproduces BarlowTrack_local_runs' score_gt_variants number for wk2 leifer t57 seed 0: 0.93707 (1101 misses, 6010 mismatches over 112,996 assigned clicks) vs 0.9370 expected. The staged round_bayes_wk4_leifer_stardist_strict objective is correct. 47 min locally (CPU crops, 25 seeds on GPU).

## 2026-11-05 12:58 — result

Strict-GT objective checked (other session BarlowTrack_cluster_runs): eval_accuracy --mode trained on Leifer_NeRVE_Worm1_reexport_A2_stardist_strictgt (nandist, 25 seeds, GPU labelprop with CSR) gives wk2 t57 seed 0 = 0.93707 over 112,996 GT entries, matching score_gt_variants' sd_strict accuracy_assigned 0.9370. The staged round_bayes_wk4_leifer_stardist_strict objective (label leifer_stardist_strict) is therefore accuracy over assigned clicks; headline all-kept = 0.97826 x it on these masks (README of that round).

## 2026-11-05 12:58 — result

Interim r261105-3531457 (96 wk2_samuel_137 trials): agree_mean rho 0.95 all, 0.72 top half (n=48); partial over 78 fusion=add trials controlling for 8 settings 0.88 (conf_mean 0.90); val loss partial was at most ~0.5. The proxy's top pick is the true best trial (t63, 0.9311); top 3 average 0.924. Replicating on every wk3_samuel_184 trial.

## 2026-11-05 12:58 — run

Consistency proxy on every wk3_samuel_184 trial (replication of the samuel_137 partial correlation).

run r261105-3581004 · pids 3581004 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/all_samuel184.log

## 2026-11-05 13:05 — result

r261105-3564506 (zoom scan 3, object volumes): raw StarDist objects on Samuel are 117 um^3 (merges; Zimmer ~97); z,xy 1.17,1.75: 45 um^3; 2,2.1: 27; 2.5,2.1: 23; 3,2.5: 14 um^3. The best click-assignment zooms split nuclei. Recorded in [[EVD - Upsampling Samuel volumes before StarDist assigns most ground truth clicks but shrinks objects below nucleus size]]. r261105-3531457 done (109 samuel_137 trials): partial rho 0.89 over 91 add trials; recorded in [[EVD - Agreement of label propagation runs with their consensus ranks Samuel networks by accuracy without ground truth]].

- [ ] Commit the session's scripts (consistency_proxy/corr/partial, proxy_corr/partial, voxel_size_from_masks, stardist_rescale_test, optimize_shared_config) and CSR propagation, then cite commits in the in-progress Evidence and mark them done
- [ ] Samuel StarDist: re-segment the full video at zoom 1.17,1.75 (and 2,2.1) and score a tracker on each, to see whether fragments hurt tracking

## 2026-11-05 13:05 — done

Done on wk2_samuel_137 (all 109), wk3_samuel_184 and wk3_zimmer_1210 (12 each): agree_mean rho 0.94, partial 0.89 on Samuel 137, picks the true best; Zimmer 1210 only separates broken networks. All samuel_184 trials running (r261105-3581004). See the EVD on label-propagation agreement.

Done: test tracking self-consistency (agreement between label-propagation seeds, track purity) as a GT-free proxy on the cached embeddings of the wk2/wk3 rounds (emb_cache holds every trial)

## 2026-11-05 13:05 — done

Folded into the broader commit to-do of 13:05

Done: Commit voxel_size_from_masks.py before citing r261105-3468094 in a node

## 2026-11-05 13:08 — surprise

Fast test suite with all of today's uncommitted changes: 303 passed, 10 skipped, 1 failed: test_step0_embed_and_augment.py::test_embed_single_volume_subset (CPU input vs CUDA weights), which fails identically on a clean tree, so it predates today's changes; not fixed (project-dependent test, unclear which side should move).

## 2026-11-05 13:55 — result

r261105-3581004 (all 76 wk3_samuel_184 trials) replicates samuel_137: agree_mean rho 0.93, partial 0.87 over 61 add trials (conf_mean 0.95 / 0.89); top half 0.62; the proxy's top pick is the true best (t27, 0.957). Recorded in [[EVD - Agreement of label propagation runs with their consensus ranks Samuel networks by accuracy without ground truth]].

## 2026-11-05 14:19 — decision

The user (2026-11-05): the Samuel recordings should all have the same voxel size. Re-read Susoy et al. 2021 (zotero:975IU5CK) STAR Methods p. e3: one spinning-disc system, 0.45 um pixels, 20 sections ~1.75 um apart, 256x512 px. Note the recordings are of the male TAIL (tail neurons, not head), which may explain why mask-based sizes put Samuel ~20% above 0.45 relative to Zimmer head nuclei. make_per_recording_rounds.voxel_aspect now overrides samuel_184 to (1.75, 0.45) um: visnet_z_scale 3.89 (wk3 used 5.83); round_shared_v1 README updated.

## 2026-11-05 14:19

i261105-2 duplicates i261105-1 (same Samuel 184 grid_spacing problem, logged this morning from convert_harvard_data.py and the paper); track it under i261105-1.

## 2026-11-05 14:19 — resolved

Duplicate of i261105-1

resolved i261105-2

## 2026-11-05 14:27 — decision

Pre-weekend checks (2026-11-05 14:30): 1000-frame training fits (wk2/wk3 median 1.1 s, p90 1.8 s per frame-epoch -> 9-15 h at 1000x30, train_hours 72); volumes are read per item from zarr, so RAM does not grow with num_frames; Leifer StarDist red_data is zarr with one volume per chunk (DataLoader workers on). Shared dispatcher's Slurm path uses the same train_job_params/walltime/failure handling as optimize_hyperparameters. round_bayes_wk3_flavell_2022031603 has recorded nothing since 06:03: its dispatcher is alive and waits on 6 GPU training jobs (e.g. 7898108, 7898221, 7898715, 7898919, submitted 01:54-04:45) that never started (no logs); no dependency, so they are pending in Slurm for an unknown reason (needs squeue). wk3 rounds are at 57-82 of 120 scored.

## 2026-11-05 14:32 — run

Consistency proxy (consistency_proxy.py @2511d60) on every trial of wk3_zimmer_1210, zimmer_1123, flavell_2022033002, flavell_2022032402 (two GPU streams, overnight): (1) is the Zimmer top-half failure just saturation (all 57+ zimmer_1210 trials, partial correlation)? (2) GT-free per-recording references for a proxy-objective shared sweep (the user, 2026-11-05: search on the consistency proxy).

run r261105-3650602 · pids 3650602, 3650603 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/run_zimmer.log, /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/run_flavell.log

## 2026-11-05 14:40 — surprise

Spectral sync rounds each run's singular-vector block to a permutation independently (round_block_to_partial_perm), which is not rotation-invariant: on synthetic equal-sized blobs (tied singular values) some runs are misaligned and agreement is only ~0.72. On real data it is near-optimal: Samuel 137 t63 per-run agreement through the perms 0.643 vs 0.664 with the best Hungarian matching of each run to the consensus, no run labels mapped to unused consensus columns. So the consistency proxy measures real run-to-run disagreement. (Diagnostic script: scratchpad perm_diag.py, quick look.)

## 2026-11-05 14:40 — run

Local smoke of optimize_shared_config.py with objective_metric agree_mean (consistency recorded in every eval record via utils_consistency / tracker.labelprop_runs; 2 trials, samuel_184 + zimmer_1210, 12 frames 1 epoch).

run r261105-3662154 · pids 3662154 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/round_shared_smoke_proxy/smoke.log

## 2026-11-05 14:42 — run

The Flavell half of r261105-3650602 matched no trials (Flavell embeddings are emb_<label>_trained_allobj_<tag>.npz; consistency_proxy.py's tag regex did not strip allobj_) and exited at once; fixed the regex, relaunched the Flavell stream.

run r261105-3662972 · pids 3662972 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/run_flavell.log

## 2026-11-05 14:47 — run

Spectral-sync deep dive (the user, 2026-11-05). Findings from the code: (1) round_block_to_partial_perm rounds each run's block of right singular vectors on its own, which depends on the arbitrary basis of the singular subspace; column weighting (invsqrt-count) gives every consensus label about the same singular value, so that basis is close to arbitrary; (2) the input is top-1 indicators (probabilities discarded: topk=1 and row-normalised); (3) aggregate_consensus loops in Python over every labelled object (most of the 30-44 s). Added opt-in options (defaults unchanged): rounding='cpqr' (rotate onto pivoted-QR anchor rows, Damle et al. 2019), refine_iters (alternating Hungarian re-alignment of each run to the others' consensus), vectorised aggregate (identical output). Synthetic blobs: refine raises agreement 0.72->0.88 and purity 0.72->0.76 (noise 4); cpqr alone no change. Now scoring tracking accuracy of independent / refine / cpqr_refine on 7 cached embeddings (Samuel 137 t63/t50, 184 t27/t32, Zimmer 1210 t52/t36, Flavell 0330-02 t61).

run r261105-3668466 · pids 3668466 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/sync_cmp/run.log

## 2026-11-05 16:35 — run

r261105-3668466 (compare_sync, 711ba82:barlow_track/scripts/scratch/spectral_sync/compare_sync.py): same propagation, only the sync differs, 7 networks. Accuracy independent -> refine -> cpqr_refine: S137 t63 .9313/.9339/.9338, S137 t50 .8765/.8856/.8880, S184 t27 .9571/.9575/.9575, S184 t32 .9254/.9244/.9241, Z1210 t52 .9432/.9407/.9406, Z1210 t36 .9294/.9296/.9316, F033002 t61 .9688/.9713/.9714. Mean +0.16 pt (refine) / +0.22 pt (cpqr_refine), 5/7 not worse; largest gain on the weakest network (+0.9/+1.2 pt), losses up to -0.26 pt (Zimmer t52). Agreement rises in all 7 (+0.015 to +0.035), partly by construction since refinement maximises agreement with the consensus. Sync time ~10 s.

## 2026-11-05 16:35 — decision

Keep the spectral-sync refinement opt-in (default independent rounding) for the weekend shared sweep: the mean gain (~0.2 pt) is below the spread between networks, and switching would shift accuracies and agreement against the wk2/wk3 records the proxy references come from. Revisit after the sweep, e.g. as a final-tracking option.

## 2026-11-05 16:36 — run

r261105-3662154 (local DEBUG smoke of optimize_shared_config with objective_metric agree_mean on samuel_184 + zimmer_1210, 2 Ax trials): end to end OK; shared_results records metric, per-recording value, accuracy and regret; Ax completed both trials. DEBUG-sized training so values are low (agree .34-.41).

## 2026-11-05 16:37 — decision

Switched staged round_shared_v1 to objective_metric agree_mean (cluster config_files/round_shared_v1/shared_template.yaml; accuracy version kept as .accuracy_objective.yaml.bak): references from proxy_references.py, zimmer_1210 .8509, zimmer_1123 .8603, flavell_2022033002 .7906, samuel_137 .6426, samuel_184 .7036; flavell_2022032402 pending r261105-3662972 (dispatcher refuses to start without it).

- [x] add reference_agree_mean for flavell_2022032402 (and refresh zimmer_1123 when its scoring ends) in round_shared_v1/shared_template.yaml

## 2026-11-05 17:15 — decision

Spectral sync now defaults to CPQR rotation + Hungarian refinement (a5cb2f1, the user: turn on the improvement; pushed with 5695edc/711ba82). Accuracies and agreement recorded before a5cb2f1 used independent rounding; consistency records now carry 'sync'. The running proxy scorings r261105-3650602 / r261105-3662972 started before the change and stay on the old sync.

## 2026-11-05 17:15 — run

Re-score the top 5 trials by old agree_mean of samuel_137, samuel_184, flavell_2022033002, zimmer_1210 with the new default sync, to re-derive round_shared_v1's reference_agree_mean (the old references were under independent rounding).

run r261105-3772374 · pids 3772374 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/run_refs_newsync.log

## 2026-11-05 17:46 — run

r261105-3772374: new-sync (cpqr_refine10) agree_mean of the top-5 old-sync trials -> references samuel_137 .6654 (t63), samuel_184 .7210 (t27), flavell_2022033002 .8169 (t71; old-sync max was t61), zimmer_1210 .8648 (t33). Written into round_shared_v1/shared_template.yaml; old-sync zimmer_1123 value removed so the dispatcher refuses to launch until it is re-derived. (accuracy in these rows is the round's recorded old-sync accuracy.)

## 2026-11-05 17:46 — run

Re-score the top 5 trials of zimmer_1123 (now) and flavell_2022032402 (after r261105-3662972 ends) with the new default sync for round_shared_v1's last two references.

- [x] add zimmer_1123 and flavell_2022032402 new-sync reference_agree_mean (proxy_references.py on scratchpad/proxy/newsync_*.jsonl) to round_shared_v1/shared_template.yaml

run r261105-3795686 · pids 3795686 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/run_refs_newsync_2.log

## 2026-11-05 17:55 — run

r261105-3795686 (part 1): zimmer_1123 new-sync reference_agree_mean .8730 (t56, recorded acc .9857; round best .9868), added to round_shared_v1. flavell_2022032402 remains (waits for r261105-3662972).

## 2026-11-05 18:54 — run

r261105-3662972 ended: flavell_2022032402 all 75 trials, agree_mean rho all .86 / better half .37 / partial .77 (63 add trials; settings explain .45); old-sync max-agree trial t47 = round best .9798. r261105-3795686 ended: new-sync top-5 agreement 47 .8517, 57 .8564, 73 .8517, 45 .8515, 61 .8483 -> reference .8564; round_shared_v1 now has all 6 references (zimmer_1210 .8648, zimmer_1123 .8730, flavell_2022033002 .8169, flavell_2022032402 .8564, samuel_137 .6654, samuel_184 .7210). A killed wrapper left a second identical chain running, so the newsync files hold duplicate (identical) rows.

## 2026-11-05 18:54 — surprise

Under the new sync the top-5 agreement values bunch within ~0.005-0.008 and the max-agreement trial moves off the true best on two Flavell rounds (033002: t61 .9695 -> t71 .9675; 032402: t47 .9798 -> t57 .9725, recorded old-sync accuracies). Refinement pulls each run towards the consensus, which may blur the proxy among the best networks; 5 trials per round is too few to say.

## 2026-11-05 18:54 — decision

Follow-up to the surprise entry above: no blurring. Among each round's top 5 the new sync's agreement spread is equal or wider (e.g. samuel_184 .022 -> .028, zimmer_1210 .005 -> .008), and its Spearman with recorded accuracy is noise-level either way (n=5; mean over 6 rounds old +0.05, new +0.15). The pick moving on two Flavell rounds is within that noise.

## 2026-11-05 18:54 — done

all 6 new-sync references in round_shared_v1

Done: add reference_agree_mean for flavell_2022032402 (and refresh zimmer_1123 when its scoring ends) in round_shared_v1/shared_template.yaml

## 2026-11-05 18:54 — done

done, r261105-3795686

Done: add zimmer_1123 and flavell_2022032402 new-sync reference_agree_mean (proxy_references.py on scratchpad/proxy/newsync_*.jsonl) to round_shared_v1/shared_template.yaml

## 2026-11-05 19:50 — run

25 vs 100 label-propagation seeds for the agreement proxy (user asked: the tracker default and the paper use 100; eval/sweep use 25). consistency_proxy.py --num_seeds (d782242), default CPQR+refinement sync, same trials at both counts: samuel_137 20 trials evenly spaced in accuracy rank, flavell_2022032402 15 trials from its better half (where the 25-seed ranking was weak, rho .37). Compares Spearman with recorded accuracy and time per trial.

run r261105-3868317 · pids 3868317 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/run_seeds100.log · for [[EVD - Agreement of label propagation runs with their consensus ranks Samuel networks by accuracy without ground truth]]

## 2026-11-05 20:14 — dead-end

r261105-3868317 failed: every cell died at once because the /lisc CIFS mount was down (OSError 112 Host is down, ~20:10); nothing computed. ygg drop itself also fails while /lisc is down (it stats run logs there).

## 2026-11-05 20:14 — run

Rerun of r261105-3868317 (25 vs 100 seeds), waiting up to 10 h for the /lisc mount to answer, then the same script.

run r261105-3881276 · pids 3881276 · host charles-Precision-7820-Tower · logs /tmp/claude-1000/-home-charles-Current-work-repos-barlow-track/9f349deb-c45d-4543-9ef0-50ad477ae9d4/scratchpad/proxy/run_seeds100.log · for [[EVD - Agreement of label propagation runs with their consensus ranks Samuel networks by accuracy without ground truth]]

## 2026-11-05 22:34 — decision

r261105-3881276 (25 vs 100 seeds, new sync, same trials): samuel_137 (20 trials, acc .488-.931) agree_mean rho 25 .97 / 100 .98, better half .82 / .88, same pick (t63 = best), rho(25,100) 1.00. flavell_2022032402 better half (15 trials, acc .973-.980) rho 25 .47 / 100 .29, upper half .57 / .57, pick t47 (= best .9798) at 25 vs t45 (.9757) at 100, rho(25,100) .57, so within that 0.7 pt range the ranking is mostly propagation noise at either count. 100 seeds cost 2.7x (Samuel 23 -> 62 s/trial) to 3.2x (Flavell 85 -> 276 s). Decision: keep 25 seeds for the proxy and the sweep; 100 buys nothing measurable.

## 2026-11-06 08:49 — surprise

wk3 per-recording rounds (dispatchers 7781793-7781798, 7781913, 7781914) found cancelled on 2026-11-06 morning; their logs stop 19:28-19:38 on 11-05, around when the /lisc CIFS mount dropped here (~20:10). No cluster notice. Stopped at 63-87 of 120 trials, mostly plateaued (best within the last 1-52 trials; new bests at most +0.1 pt). Not resumed: the shared sweep supersedes them.

## 2026-11-06 08:49 — run

round_shared_v1 launched by the user (dispatcher job 7910480, 2026-11-06 08:45, commit d782242): Ax on the 0.25-quantile of (agree_mean - reference) over the 6 tuning recordings, 60 trials, 4 in flight = 24 GPU jobs, all started at once (cluster quiet). First check: no tracebacks; 23 of 24 cells training normally; trial_2/flavell_2022033002 (job 7910495) logged loss 0 with zero embedding std at step 0, never seen in wk3's 159 jobs on that recording, being watched.

run r261106-j7910480 · jobs 7910480 · host charles-Precision-7820-Tower · logs /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_shared_v1/dispatcher_7910480.log

## 2026-11-06 10:15 — decision

Storage cleanup by the user on the cluster (2026-11-06, /lisc scratch 97% -> 92% full, 390 GB -> 1007 GB free): (1) every trial_*/checkpoints/checkpoint.pth outside round_shared_v1 (1456 resume checkpoints, 317 GB; finished trials keep resnet50.pth); (2) all .pth/.npz/.pt/.pickle in the 14 rounds trained before 2026-10-20 (new_image_embedding, initial_position_embedding[_samuel], v2-v5_position_embedding, round_2x2, round_center, round_bayes_samuel, visnet_test, round_bayes_zimmer, round_flavell, round_visnet_full; ~214 GB). Their exp_results, configs and logs are kept, so the numbers in the notes citing them stay traceable, but those networks and embeddings can no longer be re-scored. Not done: deleting the weaker trials' cached embeddings (list of 901 files, 407 GB, on the local scratchpad).

## 2026-11-06 12:33 — issue

Eval crops of project-source recordings truncate fractional crop centres (evaluate_trained_checkpoint default round_crop_centers=False -> _centroid_boxes int(z)) while training (volume_data.extract_crops) rounds them: objects with a fractional coordinate >= .5 are cropped one voxel off at eval, a whole z-plane in z. Systematic, so every wk2/wk3 and round_shared_v1 accuracy shares it and comparisons within them hold; absolute accuracies on project-source recordings may be low. NWB source rounds. Found by a read-only code review subagent 2026-11-06, not yet measured.

- [ ] issue i261106-1 · affects [[@BarlowTrack revision accuracy evaluation]]

## 2026-11-06 12:33 — decision

Code review subagent (read-only, 2026-11-06): labelprop/spectral-sync core verified (CSR = spmm to 1.8e-7, index alignment, vectorised aggregate, cpqr/refine; synthetic 40% label noise: independent .962, cpqr 1.000, cpqr+refine 1.000). Other findings: smoke runs in a round dir would be resumed as real trials (checked: round_shared_v1's 99 records are all full-length, 25 seeds, cpqr_refine10, so not affected); proxy_references.py does not filter on sync/num_seeds (the shared round's references are 25 seeds + new sync, so not affected); consensus K capped at max seed label + 10 with one null column per run (plausible accuracy loss on every-detection data, untested); refine_perms keeps stale mappings for zero-count labels (tiny effect on agreement); dup_frac accepted as objective but is constant 0; CUDA fallback only catches OutOfMemoryError; minor edge crashes.

- [ ] measure the eval crop-centre rounding issue: re-score a few networks with round_crop_centers=True (needs re-embedding)
- [ ] test the spectral-sync K cap (max label + 10) on an every-detection recording

## 2026-11-06 12:33 — decision

Held-out scoring mode for optimize_shared_config.py (--score_heldout, --top/--trials, --replicates, --heldout_labels; heldout_results.jsonl) drafted and smoke-tested locally on held-out samuel (2 replicates, 12-frame smoke: .761 / .812). Not in the repo yet: this resumed session's sandbox makes the repo read-only except research/tree; draft in the session scratchpad heldout/optimize_shared_config.py.

## 2026-11-06 14:10 — decision

Code-review fixes prepared as a patch (repo read-only in this session; applies cleanly to d782242, 117 tests pass on a patched copy): shared sweep resumes only records scored at the run's eval length and refuses to mix smoke and full trials (params.yaml now records smoke); dup_frac rejected as an objective; restart after a crash between mkdir and params.yaml no longer dies; proxy_references.py filters on sync and num_seeds; label propagation falls back to CPU on cuSPARSE/cuBLAS/CUDA resource errors too, re-raising others; eval_accuracy maps '<name>_dupNN' to its base object instead of crashing. Plus the held-out scoring mode and the labelprop_opt plan/harness (subagent, harness reproduces the tracker exactly; batch1 = 33 variants x 4 cases x 2 seeds, ~3 h CPU). Deliberately not changed mid-sweep because they shift scores: eval crop-centre rounding (i261106-1), the consensus K cap, stale perms of zero-count labels in refine_perms.

## 2026-11-06 14:32 — decision

labelprop_opt PLAN.md updated for the local GPU: speed batch first (GPU reproduction check), batch 1 as two GPU streams (~2.5 h), added seeds100. Fragment errors now split into err_frag_split / err_frag_distractor / err_frag_other in sweep_labelprop records, because seed count only targets one of three fragmentation causes (neuron absent from seed volumes); the S137 seed-count null says nothing about fragments (Samuel has none).

## 2026-11-06 14:36 — run

Label propagation tracker screen on the local GPU (labelprop_opt, uncommitted edits on ec7d6ed): chain script /home/charles/.cache/barlow_scratch/labelprop_opt/chain_speed_batch1.sh runs the speed batch (GPU reproduction check), then batch 1 (34 variants + base, 4 dev cases, seeds 0/1) as two GPU streams.

run r261106-361651 · pids 361651 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/speed_dev.log, /home/charles/.cache/barlow_scratch/labelprop_opt/batch1_dev_s0.log, /home/charles/.cache/barlow_scratch/labelprop_opt/batch1_dev_s1.log · for [[EVD - One-change screen of label propagation variants on four development recordings]]

## 2026-11-06 14:39 — run

r261106-361651 speed batch, S137 t50 seed 0: base 0.8879 reproduces the recorded 0.8880; local and batch5 equal base exactly on GPU; exact kNN 0.8881 with graph 1.5 s vs 24.6 s. layers30 0.8960 (+0.81 pt) and layers50 0.8945, one seed only: batch 1 tests it on 8 pairs.

## 2026-11-06 14:46 — decision

r261106-361651: chain stopped before batch 1 (killed the chain shell only); the speed batch continues as pid 361654. Speed so far: base reproduces S137 t50 (0.8879 vs 0.8880) and S184 t27 (0.9577 vs 0.9575); local and batch5 are identical to base; exact is +0.02 pt with a 5x faster graph. layers30 is +0.82 on S137 t50 and +0.03 on S184 t27. Machine was underused: each process is serial CPU (Python sync, 4.5 min Zimmer load) with seconds of GPU work, so batch 1 now runs as 8 streams.

## 2026-11-06 14:46 — decision

spread99 replaced by spread90/spread97 (9a32928): its rationale (mislabelled seeds) cannot apply, because seed labels are the seed volume's own objects; alpha=0.99 is about base; lower alpha gives a smooth hop discount comparable to fewer layers (question from the user).

## 2026-11-06 14:46 — run

Batch 1 of the label propagation screen, 8 streams (4 dev cases x seeds 0/1, 7 threads each, shared GPU), at barlow_track@9a32928; starts when the speed batch (pid 361654) ends.

run r261106-369804 · pids 369804 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch1.log · for [[EVD - One-change screen of label propagation variants on four development recordings]]

## 2026-11-06 15:02 — decision

Zimmer project load profiled (Z1210, 246 s for two opens): load_final_project_data_from_nwb spends ~95 s per open in pynwb read (hdmf constructs ~17k containers incl. 3334 per-volume segmentation objects; arg checks + type-map deepcopies) and ~23 s in convert_nwb_to_trace_dataframe; volumes themselves are lazy. run_eval and the harness opened the same project twice; fixed in 3fb6198 (reuse when GT path == project path), ~2 min saved per Zimmer eval. Further speedup would need a cache of seg metadata + GT tracks or a lazier wbfm NWB loader.

## 2026-11-06 15:02 — decision

Library check for the user: the kNN graph has no same-volume cannot-link (build_knn_graph is plain NNDescent; one-label-per-volume is enforced only after propagation, by the greedy per-run pass and the final Hungarian). svd_center is close to a no-op in U*S mode (Euclidean distances are translation invariant; only the 50-d subspace changes); it matters only with whitening.

## 2026-11-06 15:08 — dropped

replaced before it started: batch 1 relaunched as 12 streams (seed 2 added, CPU/GPU split) to use the idle machine

dropped r261106-369804

## 2026-11-06 15:08 — run

Batch 1 relaunched as 12 streams (4 dev cases x seeds 0/1/2; Z1210 and F0330 seeds 0/1 propagate on GPU, the rest on CPU) at barlow_track@3fb6198. Speed batch (r261106-361651): Z1210 base 0.9315 vs 0.9316 and F0330 base 0.9717 vs 0.9714 reproduce; layers30 -0.15 pt on F0330 (S137 +0.82, S184 +0.03).

run r261106-384883 · pids 384883 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch1.log · for [[EVD - One-change screen of label propagation variants on four development recordings]]

## 2026-11-06 15:16 — run

Per-run depth/hop diagnostic (diag_layers_hops.py at da3555b) on the 4 dev cases, 5 seed runs each, clamp + spread 0.9/0.97. Smoke on S137 t50 (3 runs): per-run accuracy on neurons present in the seed volume peaks at L=30 (0.652) and falls at 100 (0.629) and 300 (0.577) while coverage is already 1.0 by L=20 (max hop 11): fewer layers helps by less bleeding, not by abstaining. spread0.9 plateaus at 0.649. Objects whose neuron is absent from the seed volume (19%) are always labelled, necessarily wrongly; they are 95% of objects at >=10 hops. No dependence on |t - t_seed|.

run r261106-398256 · pids 398256 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/diag_samuel_137.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_samuel_184.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_zimmer_1210.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_flavell_2022033002.log

## 2026-11-06 15:18 — dropped

8 of 12 streams hit CUDA OOM: CPU streams opened 368 MB CUDA contexts each and 4 GPU streams peak ~3.5 GB; stopped, 32 records kept, resumed as r-next

dropped r261106-384883

## 2026-11-06 15:18 — run

Batch 1 resumed (32 records kept) as 12 streams with only Z1210 s0 and F0330 s0 on the GPU and CUDA_VISIBLE_DEVICES='' for the CPU streams, at barlow_track@da3555b (harness unchanged since 3fb6198).

run r261106-400753 · pids 400753 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch1.log · for [[EVD - One-change screen of label propagation variants on four development recordings]]

## 2026-11-06 15:34 — dropped

restarted with confidence metrics (mass, top-1 prob, hops as predictors of correctness)

dropped r261106-398256

## 2026-11-06 15:34 — surprise

User recalls (from building the tracker): low-mass nodes did not predict correctness, and dropping them hurt final accuracy. Diag smoke S137 t50 (2 runs): mass is NOT ~1 after 100 clamped layers (median 0.008; walks rarely hit a seed volume in 100 steps), so it decays with distance; it predicts a correct per-run label weakly (AUC 0.69) but flags absent-neuron objects better (0.83). Top-1 probability predicts correctness best (AUC 0.85; dropping its lowest 20% removes 38% of wrong, 5% of correct votes). Whether dropping helps FINAL accuracy is what thr03 in batch 1 tests.

## 2026-11-06 15:34 — run

Depth/hop/confidence diagnostic on the 4 dev cases, 5 runs each (diag_layers_hops.py at 5094974).

run r261106-416617 · pids 416617 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/diag_samuel_137.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_samuel_184.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_zimmer_1210.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_flavell_2022033002.log · for [[EVD - Per-run label accuracy falls with propagation depth beyond about 30 layers]]

## 2026-11-06 16:14 — decision

NWB load speed (subagent, wbfm branch nwb-load-speed, 7 commits on a1cd2225, worktree /home/charles/Current_work/repos/dlc_for_wbfm_nwbspeed): per-read type-map copy (hdmf 3.14 deep-copied it per attribute; fixed upstream in hdmf 4.2), vectorised convert_nwb_to_trace_dataframe, pynwb skips Seg_tpoint_1..N (voxel masks via h5py), hybrid loading keeps NWB segmentation lazy and skips NWB traces the project has, no h5py path searches on dereferenced objects. Verified here: 6 new tests pass; Zimmer 1210 project load + finished-neuron GT (lazy segmentation, as the harness) 287.6 s on dev vs 20.6 s on the branch under load ~50, with df_gt, final_tracks and segmentation metadata at t=0/1/833/1666 exactly equal. Not merged: waiting on the user.

## 2026-11-06 16:57 — run

Batch 1: Zimmer and Flavell at seeds 1/2 ran on CPU at ~30 min propagation per variant under load (prop_s 1750-1870 s, vs ~23 s on the GPU); stopped them (records kept) and queued them on the GPU after the Zimmer s0 stream ends: two queues, Z1210 s1->s2 and F0330 s1->s2.

run r261106-473331 · pids 473331 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/gpu_queues.log · for [[EVD - One-change screen of label propagation variants on four development recordings]]

## 2026-11-06 16:57 — surprise

Depth diagnostic r261106-416617 done (4 dev cases, 5 runs each). Per-run accuracy on present neurons vs layers (L=30 / 100 / 300): S137 .654/.632/.585, S184 .839/.821/.789, Z1210 .859/.858/.857, F0330 .837/.837/.832. Bleeding with depth is a Samuel effect; the distractor recordings are flat from L=20. Z1210: 6% of GT objects are unreachable from a seed volume (disconnected graph pieces; Flavell ~0.3%). Correctness AUC, mass / top-1 prob / -hops: S137 .64/.81/.60, S184 .64/.90/.61, Z1210 .61/.75/.59, F0330 .64/.82/.61: mass is a weak predictor everywhere (agrees with the user's memory), prob is better. Yet in batch 1 (partial) thr03 (drop prob<0.3 before sync) is -0.32 pt and probw -0.28 pt: the per-run signal does not carry into the final accuracy, again as the user found.

## 2026-11-06 16:57 — decision

NWB: does a newer stack make the branch's table skipping (fix 2) unnecessary? Plain full pynwb read of the Zimmer no-image NWB (1667 per-volume tables), scratch Python 3.11 env: pynwb 4.2.0 / hdmf 6.2.0 / HDF5 2.0.0 39.9 s vs wbfm env pynwb 2.8.3 / hdmf 3.14.6 / HDF5 1.14.2 116.2 s (load ~30). docval calls 0.51M vs 2.73M (type-map regression fixed upstream). The branch's fast read is 6.7 s on Python 3.8, so per-object construction (~20 ms/table) still dominates: the skipping is needed with any version. HDF5 itself is not the bottleneck (h5py attrs/name ~5 s).

## 2026-11-06 17:56 — run

Ax round 1 over label propagation settings (ax_labelprop.py at 5e40a5b): driver + Samuel CPU workers now, Zimmer/Flavell GPU+CPU workers after the batch-1 GPU queues end.

run r261106-506551 · pids 506551 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/ax/round1/driver.log · for [[EVD - Bayesian search over label propagation settings on four development recordings]]

## 2026-11-06 18:00 — run

Ax round 1: started early CPU workers for Zimmer and Flavell (the launcher's GPU workers wait for the batch-1 GPU queues, and no Ax trial completes without all 4 cases).

run r261106-510144 · pids 510144, 510145 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/ax/round1/worker_z_cpu0.log, /home/charles/.cache/barlow_scratch/labelprop_opt/ax/round1/worker_f_cpu0.log · for [[EVD - Bayesian search over label propagation settings on four development recordings]]

## 2026-11-06 20:46 — run

Batch 2 (confirmation) started at d8a1335: 16 variants (base, the six batch-1 promotions, svd150, temporal2, temporal10, and combinations c_seeds, c_all = svd100+seeds50+stratified+temporal5+tsmooth05, c_all_s100, c_all_notemp, c_all_gh, c_all_l30) on CONFIRM_CASES x seeds 0-2. Leifer t57 then Z1210 t52 in one GPU queue; S137 t63, S184 t32, F0324 t28 as 9 CPU streams. The user asked to make sure Leifer is tested.

run r261106-595573 · pids 595573 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch2.log, /home/charles/.cache/barlow_scratch/labelprop_opt/batch2_confirm_leifer_reexport_s0.log · for [[EVD - Confirmation of screened label propagation variants on held-out recordings]]

## 2026-11-06 20:46 — decision

Ax round 1 stopped after its 8 attached trials (driver killed before any Sobol trial; the launcher's STOP file was removed so the workers finish the queue). Reason: the search space lacked temporal_w, and temporal5 is the largest Zimmer gain in batch 1 (+1.38 pt). Round 2 (d8a1335) adds temporal_w in {0,1,2,5,10}, imports round 1's complete trials as completed Ax trials, and attaches temporal5, seed_strat, tsmooth05, c_all, c_all_gh; 50 trials, 4 Sobol.

## 2026-11-06 20:46 — run

Ax round 2 started (driver waits for round 1's queue to empty).

run r261106-595574 · pids 595574 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/ax/round2/launch.log, /home/charles/.cache/barlow_scratch/labelprop_opt/ax/round2/driver.log · for [[EVD - Bayesian search over label propagation settings on four development recordings]]

## 2026-11-06 20:48 — run

r261106-595573: the 3 F0324 streams failed at once (parallel streams copied the embedding to the same .part file; fixed in afe2dce), and the GPU queue ran Z1210 before Leifer (CONFIRM_CASES order); stopped it. Relaunched both with Leifer first.

run r261106-598057 · pids 598057 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch2_fix.log, /home/charles/.cache/barlow_scratch/labelprop_opt/batch2_confirm_leifer_reexport_s0.log · for [[EVD - Confirmation of screened label propagation variants on held-out recordings]]

## 2026-11-06 20:53 — run

Fragment-merging diagnostic on Z1210 t36 and F0330 t61, seed 0, CPU (S137 smoke: no candidate merges, as expected without distractors).

run r261106-602653 · pids 602653 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/diag_merge_zimmer_1210.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_merge_flavell_2022033002.log · for [[EVD - Fragment merging of final label propagation tracks on recordings with distractors]]

## 2026-11-06 20:58 — run

The user: the StarDist Leifer re-segmentation (strict click GT) is probably the paper's Leifer, so test it too. Added it to the harness (EXTRA_LABELS; embedding of wk2 Leifer t57 on the StarDist objects from the benchmark cache) and to CONFIRM_CASES (3756753). Batch 2 on it, seeds 0-2, CPU. Its base at seed 0 should reproduce 0.9371 (r261105-3536566).

run r261106-606174 · pids 606174 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch2_stardist.log, /home/charles/.cache/barlow_scratch/labelprop_opt/batch2_confirm_leifer_stardist_strict_s0.log · for [[EVD - Confirmation of screened label propagation variants on held-out recordings]]

## 2026-11-06 20:58 — run

Fragment-merging diagnostic queued on Leifer StarDist t57 (after r261106-602653).

run r261106-606521 · pids 606521 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/diag_merge_leifer_stardist_strict.log · for [[EVD - Fragment merging of final label propagation tracks on recordings with distractors]]

## 2026-11-06 22:01 — run

r261106-595574: its wait on round 1 never ended (`ls jobs claimed` prints the directory headers even when both are empty); killed the launcher (CPU workers kept), set round 1's STOP, started the Zimmer GPU worker and the driver by hand. Round 2 imported round 1's 8 trials (best ax4 seeds100 .9418, Z1210 .9414; ax1 svd100 .9429) and attached trials 8-12.

run r261106-633070 · pids 633070 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/ax/round2/launch_part2.log, /home/charles/.cache/barlow_scratch/labelprop_opt/ax/round2/driver.log · for [[EVD - Bayesian search over label propagation settings on four development recordings]]

## 2026-11-06 22:01 — surprise

Fragment merging on Z1210 t36 s0 (r261106-602653): of 85 true fragment pairs (same majority GT neuron) among 1067 candidates, the median shares 98% of the smaller label's volumes (10th percentile 76%), so the fragments are not time-disjoint pieces of one track but concurrent tracks, mostly with low GT share (distractor tracks), that hold the neuron in some volumes. A time-disjoint merge rule finds almost none (2 merges, +0.25 pt at ratio 3, overlap 0.2). AUCs: centroid ratio .78, overlap .68, consensus affinity .84. The merge bound (+2.5 pt here) needs per-volume reassignment between a neuron track and its distractor partner, not label merging.

## 2026-11-06 22:02 — run

Fragment-pair diagnostic on Z1210 t36 s0: are the partner tracks of fragmented neurons pieces of the same nucleus (over-segmentation) or neighbouring nuclei? Flavell's merge diagnostic showed the same concurrency as Zimmer (36 true pairs, median overlap 0.99; consensus affinity AUC .95).

run r261106-640685 · pids 640685 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/diag_fragment_pairs_zimmer_1210.log · for [[EVD - Fragment merging of final label propagation tracks on recordings with distractors]]

## 2026-11-06 22:12 — run

Fragment errors are long swaps with a neighbour, so try voting along temporal chains (adjacent-volume mutual-NN links) before the final Hungarian: lp_core chain_w. Smoke on S137 t50 s0 (cached consensus): chain_w 1 +0.61 pt, chain_w 5 -0.07. Grid chain_w x tsmooth on S137, Z1210, F0330, then Leifer StarDist.

run r261106-647605 · pids 647605 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/diag_chain_zimmer_1210.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_chain_flavell_2022033002.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_chain_samuel_137.log, /home/charles/.cache/barlow_scratch/labelprop_opt/diag_chain_leifer_stardist_strict.log · for [[QUE - Can the label propagation tracker be made more accurate on fixed embeddings]]

## 2026-11-06 23:03 — result

Chain voting (chain_w, 7b4f1df), r261106-647605 on cached base consensus at seed 0: Z1210 t36 +1.66 pt at chain_w 1 (+1.61 at 0.5), almost all from unlabelled objects (1.60 -> 0.14%; fragments unchanged, 3.14 -> 3.20%); Zimmer chains are long (median object in a chain of 159 volumes) and 99.8% of links join the same GT neuron. S137 t50 +0.61 at chain_w 1 (+0.52 at 0.5; chains median 7, links 90% right). F0330 t61 +0.09 at 0.25-1 (chains median 15, links 97.6%). chain_w >= 5 hurts everywhere (-0.6 to -1.2 pt on F0330, S137). With tsmooth 0.5 the two overlap: on S137 they cancel. Added chain05, chain1, c_all_chain, c_all_gh_chain (tsmooth 0 in the combos) to batch 2 (b5e20d2).

## 2026-11-06 23:03 — run

Batch 2 add-on: the chain variants for streams started before b5e20d2.

run r261106-673582 · pids 673582 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch2_addon.log · for [[EVD - Confirmation of screened label propagation variants on held-out recordings]]

## 2026-11-06 23:04 — run

Leifer StarDist t57 s0 (r261106-606521, r261106-647605): base .9371 reproduces the strict-GT objective (.93707, r261105-3536566). Fragments are 5.6% of GT points (merge bound .9878); 98 true fragment pairs, median overlap 0.985, no merge passes anywhere in the grid; chain voting +0.17 pt at most (chain_w 1). Fragment-pair diagnostic on it now.

run r261106-674571 · pids 674571 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/diag_fragment_pairs_leifer_stardist_strict.log · for [[EVD - Fragmented neurons on Zimmer swap with a neighbouring nucleus for long stretches]]

## 2026-11-06 23:06 — run

Batch 2 rescheduled: the Leifer StarDist CPU streams (r261106-606174, 6 records in 2 h) and the add-on launcher (r261106-673582; its S137 reruns keep going) stopped. One GPU queue now runs StarDist s0-2 first, after the running Z1210 t52 s0, then Z1210 s1, s2, s0 and Leifer re-export s0-2 for the chain variants; S184 and F0324 CPU streams rerun for the chain variants when theirs end.

run r261106-677223 · pids 677223 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch2_v3.log, /home/charles/.cache/barlow_scratch/labelprop_opt/batch2_confirm_leifer_stardist_strict_s0.log · for [[EVD - Confirmation of screened label propagation variants on held-out recordings]]

## 2026-11-06 23:06 — dropped

moved to the GPU queue of r261106-677223 (CPU too slow); its 6 records are kept

dropped r261106-606174

## 2026-11-06 23:06 — dropped

replaced by r261106-677223; its S137 reruns continue

dropped r261106-673582

## 2026-11-07 01:08 — decision

Ax round 2 driver restarted (PID 742683) at 73022a2: GPEI re-suggested trial 12 (c_all_gh) as trial 17 on the same seeds; repeats of an arm now run on new seed blocks. Killed duplicate F0324 seed 2 stream 600696 (596195 runs the same). Dev-set objective so far: c_all_gh 0.9530, c_all 0.9500, base about 0.936.

## 2026-11-07 01:09 — dropped

superseded by v4: F0324 moved off the CPU (1.5 h per variant there, 20 s propagation on GPU)

dropped r261106-677223

## 2026-11-07 01:09 — dropped

F0324 CPU streams stopped; remaining variants in v4 on the GPU

dropped r261106-595573

## 2026-11-07 01:09 — dropped

F0324 CPU streams stopped; remaining variants in v4 on the GPU

dropped r261106-598057

## 2026-11-07 01:09 — run

Batch 2 scheduler v4: two GPU lanes over one locked list (StarDist s2 first, then F0324 s0-2, Z1210 s1/s2/s0, Leifer re-export s0-2, S184, StarDist s1 fill-in); lane 1 after StarDist s1 (740207).

run r261107-743860 · pids 743860 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/launch_batch2_v4.log · for [[EVD - Confirmation of screened label propagation variants on held-out recordings]]

## 2026-11-07 01:10 — decision

Ax round 2 workers: stopped the Zimmer and Flavell CPU workers (30-50 min per job against 1-3 min on the GPU; they held trials back) and requeued their 3 open jobs; added a Flavell GPU worker (745196) and second S184 / S137 CPU workers.

## 2026-11-07 01:33 — run

Ax round 2 stopping at 21 trials: GPEI re-suggested c_all_gh (dev 0.9530, deterministic: trial 17 = trial 12) for trials 17-20; 18-20 run it on seeds 2-7 as a seed-spread estimate. Round 3 (2111174) starts when round 2's driver (759602) exits: imports rounds 1-2, adds chain_w {0, 0.25, 0.5, 1}, starts c_all_gh_chain / c_all_gh_ts_chain025 / c_all_gh_svd150, replaces a third suggestion of one arm by a one-parameter neighbour; 45 trials.

run r261107-759603 · pids 759603 · host charles-Precision-7820-Tower · logs /home/charles/.cache/barlow_scratch/labelprop_opt/ax/round3/launch.log · for [[EVD - Bayesian search over label propagation settings on four development recordings]]

## 2026-11-07 02:24 — result

diag_switch_jumps (9218f4a) on the base tracker, seed 0, local frame of 8 nearest GT neurons: Leifer StarDist 6663 label switches over 78 neurons, at 68% only the GT object jumps (median 3.4 um vs typical 0.9), at 10% only the track; with the NWB clicks, 51% of switches are the GT object jumping while the click stays. In error volumes the click is 1.41 um from its GT mask centre (0.52 in correct volumes), 2.8 um from the tracked object, which is 3.9 um away. Zimmer 1210: 1144 switches, only GT 23%, only track 30%, both 46%.
