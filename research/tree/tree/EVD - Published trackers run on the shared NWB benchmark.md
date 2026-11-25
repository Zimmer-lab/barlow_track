---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-7dd4-728a-8921-7eba42772be7
status: open
priority: high
dateCreated: 2026-10-23T00:05:06+02:00
dateModified: 2026-11-24T19:00:00+01:00
confidence: high
projects:
  - "[[barlowtrack]]"
---

## Purpose

A same-input comparison with other methods (R1.M2, R3.W2): same detections, frames, reference identities and accuracy function for BarlowTrack and the competitor.

## Method

Scope (user, 2026-10-29 / 2026-11-23): new comparator runs are limited to the published fDNC model; ASCENT is cited as concurrent work, ZephIR needs manual reference annotation.

**Recordings and BarlowTrack side.** The 12 recordings of the revision benchmark, round_shared_v1 trial 59 (`config_files/round_shared_v1/exp_results_v2.jsonl`, labelprop v2, 50 seeds, nandist matching): 5 Flavell, 3 Zimmer, 3 Samuel, 1 Leifer. The paired number is the trial-59 network (tag `round_shared_v1_trial59_<lab>`). Six recordings also have four independent retrainings of the same recipe (`trial59r1`-`r4`); for those the table adds their mean and range over r0-r4. Two exact duplicate rows in the jsonl (flavell_2022033001, zimmer_1210) were deduplicated.

**fDNC side** (`barlow_track/scripts/scratch/fdnc_baseline/fdnc_track_and_score.py`, run r261124 below; protocol unchanged from the Zimmer 1128 / Samuel 153 / Leifer runs):
- Published model `fDNC/model/model.bin` (nh128, nl6), CPU, fDNC's own Hungarian assignment.
- Templates: frame 10 (the wbfm pipeline template) plus 4 evenly spread (`np.linspace(0, T-1, 6)[1:-1]`), and fDNC's 5-template vote (wbfm `calc_bipartite_from_candidates`, templates mapped to template 10 by fDNC's own match, no GT).
- One GT-free rescale per recording to the median RMS radius of fDNC's 11 NeuroPAL reference worms (factor from the template-10 cloud).
- Four axis conventions (default / swap / mirror / swap_mirror). **The reported convention is the one whose vote has the highest mean fDNC match probability** (GT-free). No GT accuracy enters templates, scale or convention.
- New rule this round, needed once: a spread template frame with < 2 detections moves to the nearest frame with at least the median detection count (counts only). It fired only on Samuel 137 (frame 1060 has no GT objects, moved to 1061); no other recording's templates change.
- Voxel sizes as the revision uses them: Zimmer project, Leifer 0.3226 / 1.5 µm, Flavell NWB 0.3 µm isotropic, Samuel 0.45 / 1.75 µm (184-004's NWB says 0.3 µm xy; overridden as in `make_per_recording_rounds._NWB_VOXEL_OVERRIDE_UM`).

**Detections and references, per lab (identical for both methods):**
- *Flavell:* every detection of the NWB's raw_segmentation (`--nwb_objects all`); reference = the NWB `final_tracks` of each `flavell_data.nwb` (the Flavell lab's own tracker output, i.e. the version the preprint's 96.7% is against, not the 2025_07_01 project GT; see [[EVD - Two Flavell ground-truth versions disagree on about a tenth of neuron identities]]), full final_tracks (no finished-neuron filter exists), neuron fields read by name (i261026-1 fix), GT mapped to the mask whose centroid is within 0.5 voxel (centroid mapping; 0 GT positions unmapped on all five).
- *Zimmer:* every segmented object of the paper GT project; reference = the project's finished-neuron final_tracks, matched on raw_segmentation_id.
- *Samuel:* **oracle detections: the NWB's GT objects themselves** (no segmentation of all nuclei exists for these Susoy et al. 2021 recordings); reference = NWB final_tracks.
- *Leifer worm 1:* every object of the StarDist re-segmentation; reference = the strict click GT (`_strictgt` project = sd_strict). Scored on assigned clicks (112,996, the recorded metric) and on all kept clicks: correct / 115,507 (= assigned + 2,305 clicks with no mask + 206 in volumes without objects; 2,708 merged and 804 coincident clicks excluded, decision 2026-11).

**Pairing checks** (`fdnc_paired_table.py`), all 12 pass:
1. Frames: same n_frames.
2. Detections: fDNC's per-frame detection ids equal the ids BarlowTrack embedded (trial-59 embedding cache `lin_to_t_seg`), frame by frame (e.g. Flavell 2022-03-31-01: 286,727 detections in 1,600 frames on both sides). On Samuel 137 / 184, 4 frames with a single GT object are skipped by BarlowTrack and unmatched by fDNC: misses for both.
3. Reference identities and denominators: the GT matrix fDNC is scored against equals the `gt` matrix saved with the BarlowTrack tracks (`tracks_v2/round_shared_v1_trial59_<lab>.h5`) cell by cell; totals equal. (Samuel 153: fDNC's matrix has 5 trailing frames BarlowTrack's lacks; they hold no GT entry.) Leifer: the scored GT cells are exactly the 112,996 assigned clicks of `sd_strict_status.npz`.
4. Scoring: BarlowTrack's recorded accuracy recomputes exactly from its saved gt / pred_matched with `calculate_accuracy`; fDNC is scored with the same steps as `eval_accuracy.run_eval` (align, pad, `match_columns_to_gt` nandist, `calculate_accuracy`).

**Run r261124** (2026-11-24, local CPU, charles-Precision-7820-Tower): barlow_track @ bcbcf5e with uncommitted `fdnc_track_and_score.py` (sha1 `47691d65eca5`; Samuel 137 rerun at `7cb2caa2b118` after adding the template rule) and new `fdnc_paired_table.py` (sha1 `020be9313bbc`). Commands: `~/.cache/barlow_scratch/fdnc/r1m2_trial59/run_all.sh`, then `python barlow_track/scripts/scratch/fdnc_baseline/fdnc_paired_table.py`. Output: `~/.cache/barlow_scratch/fdnc/r1m2_trial59/` (`fdnc_results.jsonl`, `paired_table.csv`, `paired_table_details.json`, per-recording `_detections.npz` / `_scored_gt.h5`, logs).

Reused vs new: Zimmer 1128 and Samuel 153 (cached 2026-10-29 tracks, `~/.cache/barlow_scratch/fdnc/`) and Leifer (cached 2026-11-05 StarDist tracks, `.../fdnc/leifer/`) were **rescored** with the current scoring code after checking every cached match against today's detections (Zimmer 1128 and Samuel 153 reproduce their cached scores exactly; Leifer is now scored against the strict-GT project exactly as trial 59 is, instead of offline through `score_gt_variants.py`). The other 9 recordings are **new runs**.

## Results

Accuracy (nandist matching). BarlowTrack: trial-59 network; "r0-r4" = mean (range) over the five trainings where they exist.

| Recording | Detections | GT neurons / entries | fDNC (convention) | BarlowTrack | BarlowTrack r0-r4 | fDNC row |
|---|---|---|---|---|---|---|
| Flavell 2022-03-31-01 | all NWB | 163 / 231,268 | 0.793 (default) | 0.976 | 0.972 (0.968-0.976) | new |
| Flavell 2022-03-16-03 | all NWB | 173 / 258,078 | 0.881 (swap) | 0.981 | 0.987 (0.981-0.990) | new |
| Flavell 2022-03-24-02 | all NWB | 146 / 206,024 | 0.773 (default) | 0.977 | – | new |
| Flavell 2022-03-30-01 | all NWB | 160 / 231,279 | 0.732 (swap) | 0.979 | 0.980 (0.979-0.981) | new |
| Flavell 2022-03-30-02 | all NWB | 136 / 187,290 | 0.813 (default) | 0.973 | – | new |
| Zimmer 1123 | all segmented | 61 / 98,642 | 0.646 (swap) | 0.980 | – | new |
| Zimmer 1128 | all segmented | 131 / 206,701 | 0.627 (swap) | 0.986 | 0.987 (0.985-0.989) | rescored |
| Zimmer 1210 | all segmented | 103 / 161,548 | 0.727 (default) | 0.951 | – | new |
| Samuel 153 | GT objects (oracle) | 98 / 120,923 | 0.832 (swap) | 0.993 | 0.991 (0.988-0.994) | rescored |
| Samuel 137 | GT objects (oracle) | 92 / 85,353 | 0.540 (swap_mirror) | 0.938 | – | new |
| Samuel 184 | GT objects (oracle) | 85 / 210,336 | 0.636 (swap) | 0.977 | – | new |
| Leifer worm 1, assigned clicks | all StarDist | 79 / 112,996 | 0.821 (swap_mirror) | 0.941 | 0.937 (0.923-0.942) | rescored |
| Leifer worm 1, all kept clicks | all StarDist | 79 / 115,507 | 0.803 | 0.920 | 0.917 | rescored |

BarlowTrack scores higher on **12 of 12** recordings, fDNC on 0, no ties (a tie = equal to three decimals). The gap is 0.10-0.40 (median 0.21). The count is the same with BarlowTrack's r0-r4 mean or its worst training. fDNC's errors are mostly mismatches, e.g. Zimmer 1128: 15,695 misses and 61,476 mismatches (BarlowTrack 899 / 1,999).

GT-free convention choice vs GT: the highest-confidence vote is also the GT-best convention on 11 of 12; on Samuel 137 it picks swap_mirror (0.540) over swap (0.543), costing 0.003. Swap or swap_mirror wins on Samuel, Leifer, Zimmer 1123 / 1128 and two Flavell recordings; default or mirror wins on three Flavell recordings and Zimmer 1210, so no single convention fits all recordings. The x/y swap is what matters: the wrong swap family scores 0.20-0.56, while the mirror partner of the chosen convention is within 0.05 of it everywhere.

## Conclusion

On identical detections, frames, reference identities and scoring, the published fDNC model with a GT-free 5-template protocol tracks 0.54-0.88 of GT entries across the 12 benchmark recordings, against 0.94-0.99 for BarlowTrack trial 59, which is higher on every recording.

## Caveats

- **One competitor.** fDNC only; other published trackers were not run (scope decision).
- **Samuel uses oracle detections** for both methods (GT objects only), which likely flatters both; Flavell's reference is algorithmic (the Flavell lab's tracker), not manual.
- **fDNC was used as a whole-video tracker** (every frame matched to templates, one identity per template neuron for the video), which is our metric, not fDNC's per-volume template-matching accuracy (Yu et al. 2021). It was trained on NeuroPAL point clouds of another microscope; position-only, no colour.
- **Protocol choices are ours** (templates 10 + 4 spread, reference-RMS rescale, four conventions, vote), all GT-free; a better template choice could raise fDNC (on Leifer the best single template, picked with GT, gave 0.856 vs the vote's 0.821; see [[EVD - On the new Leifer ground truths fDNC tracks 82 percent and NeRVE 89 percent of assigned clicks]]).
- **BarlowTrack replicates:** six recordings have one trial-59 network only. Where five trainings exist, the trial-59 network is within 0.006 of their mean and the spread is ≤ 0.02 (Leifer r4 is the low outlier at 0.923), far below the 0.10-0.40 gap; fDNC is deterministic.
- Code not yet committed (scratch scripts, sha1 above).
