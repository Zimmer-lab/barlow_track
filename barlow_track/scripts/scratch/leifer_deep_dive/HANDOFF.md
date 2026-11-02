# Handoff: Leifer re-export and its low tracking accuracy (2026-11-02)

For a fresh session working on the Leifer NeRVE worm1 re-export. Read this first, then
`../leifer_export/README.md` (how the export was built) and the lab notebook
(`research/tree/NOTEBOOK.md`, search "Leifer"). Load the `ygg:tree` skill before touching the tree.

## The problem

Every network scored on the re-export lands at 0.84-0.885 accuracy, whatever its settings,
while the same kind of network scored 0.972 on the old Leifer NWB (GT objects only, crops at
the clicks). The user (2026-11-02): some loss is expected from distractors, but not this much.

- round_bayes_wk2_leifer_reexport: 80 scored Ax trials, median 0.878, max 0.885 (trial 57);
  the 26 random Sobol trials have std 0.036 (Samuel 137: 0.095, Flavell: 0.166). Image-only
  networks average 0.857 vs 0.873 fused; round_snap1's epoch-1 snapshot already scores 0.84.
  So ~12% of cells are wrong nearly independent of the network: look at data and scoring.
- Best trial 57: misses 2003, mismatches 11978 of 119019 GT cells.
- Zimmer is also scored with many non-GT detections and reaches 0.96-0.98, so distractors
  alone are not obviously the explanation.

## Data

| What | Path (cluster, mounted locally under /lisc) |
|---|---|
| Re-export used now (A2) | `/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/leifer_reexport/Leifer_NeRVE_Worm1_reexport_A2.nwb` (26.5 GB) |
| Its wbfm project (training) | `.../barlow_track_revisions/wbfm_tracker_baseline/leifer_reexport/Leifer_NeRVE_Worm1_reexport_A2/project_config.yaml` |
| Old NWB (video + GT only) | `.../barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb` |
| Raw NeRVE release | `.../barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1/` |
| Weekend round | `.../barlow_track_revisions/config_files/round_bayes_wk2_leifer_reexport/` (trials, `exp_results.jsonl`, `emb_cache/` with every trial's embeddings) |

A2 contents: 1536 volumes, (632, 600, 32) zxy-ish as (T, Z, X, Y) = (1536, 32, 632, 600) in
ProjectData; red (channel 0) and green registered to red; NeRVE masks (labels < 1000) plus
12,019 GT-seeded watershed pseudo masks (label 1000 + GT neuron index) for GT clicks with no
NeRVE mask within ~2.6 um; GT = raw manual clicks of 79 neurons, `NeuronSegmentationID` finite
for every GT point. Voxel size z 1.5 um, xy 0.3226 um.

Write permission (user): only `barlow_track_revisions/leifer_reexport/` and
`barlow_track_revisions/wbfm_tracker_baseline/`. Never write under `barlow_track_paper/`.

## How it is scored

`eval_accuracy.py --source nwb --nwb_objects all --nwb_gt_mapping segmentation_id` (template
keys `objective_nwb_objects: all`, `objective_nwb_gt_mapping: segmentation_id`): every mask in
the NWB is cropped at its regionprops centroid and embedded; GT neuron j at frame t is the mask
whose label equals its `NeuronSegmentationID` (two GT on one label: both dropped; none are).
Label propagation, 25 seeds, SVD 50, nandist column matching. All 119,019 GT positions map to a
mask. Code: `barlow_track/scripts/eval_accuracy.py` `_nwb_all_objects`, `run_eval`.

## Findings so far (2026-11-02)

1. Not a channel mix-up: in-mask / background red 1277 / 112 vs green 245 / 98; the training
   project's red equals the NWB's channel 0.
2. The 4 volumes NeRVE left without detections (78, 405, 945, 1516; they hold the 208 clicks
   the old NWB dropped) are corrupted frames: mean red ~2020 vs ~110 in neighbours, p99 at
   saturation, one plane at mean 31-33k in 405 / 945, correlation with the previous volume ~0
   (neighbours 0.3-0.7). Candidates to drop from training and scoring (0.2% of GT; cannot
   explain the gap). Output: `blank.log` (below).
3. 26 volumes have no NeRVE masks at all (12, 78, 188, 405, 535, 624, 635, 678, 762, 945,
   1434, 1469, 1516, 1523-1535): only pseudo masks, i.e. only GT objects, there.
4. Non-GT masks (70,359, ~46 per volume, 37% of detections) are mostly separate cells, not
   duplicates of GT: nearest GT-mask centroid median 3.74 um (GT-to-GT nearest 4.59 um), < 2 um
   only 4.2%; smaller (median 78 vs 133 voxels) and deeper in z (median plane 18.8 vs 15.7).
   Plausibly the unannotated neurons. 2,863 non-GT NeRVE masks contain a GT click (merged or
   offset cases). Only 6.8% of pseudo masks have a non-GT NeRVE mask within 3 um.
5. Trial 57 embeddings, kNN from other frames (6000 GT queries, 10 NN): same neuron 0.882,
   non-GT detection 0.094, another GT neuron 0.024. Confusions are mostly with non-GT masks.
6. **The non-GT detections cost the ~9 points** (decompose_errors.py on trial 57's cached
   embeddings, run r261102-3204514): every detection 0.880, GT masks only **0.9665** (0.976 on
   NeRVE-mask GT, 0.881 on pseudo-mask GT), NeRVE masks only 0.884. So images, crops at mask
   centroids and the GT mapping are fine; the pseudo masks are not the problem.
   - 10,700 of 12,113 mismatches go to a non-GT mask, 1,413 to another GT neuron.
   - Concentrated: the worst 10 neurons hold 41% of errors. neuron_013/076/044/049/032/024
     score 0.48-0.69 with every detection but 0.95-0.99 GT-only.
   - The stealing mask sits right next to the true one: centroid distance median 2.84 um
     (59% < 3 um; GT-to-GT nearest 4.59 um), median 93 voxels, 11% contain a GT click.
   - Errors flicker (runs of median 1-2 frames, longest ~17), not identity swaps: frame by
     frame the tracker picks the neighbouring mask instead of the GT one.
   - The 26 frames without NeRVE masks score 0.43 (frames 1524+ 0.05).
   - The offset is mostly in z: |dz| median 2.18 um (1.45 planes) vs dxy median 1.55 um; 61%
     of stealing masks are offset more in z than xy and 30% sit within 1 um in xy, i.e.
     stacked right above or below the GT mask. Baseline (all non-GT masks within 4 um of a GT
     mask): 41% mostly-z, |dz| 1.56 / dxy 2.09 um. This points to NeRVE splitting cells in z.
   **Main open question:** are the stealing masks NeRVE splits of the same cell (e.g. z-adjacent
   pieces; then the GT is arguably fine and the segmentation should merge them, or the scoring
   should accept either piece) or genuine touching neighbours (then the tracker has a real
   discrimination problem). Look at a few in napari (neuron_044: 82% within 3 um).

7. **Merging z-stacked masks (GT-free rule) would recover ~4.5 points** (zsplit_merge.py, run
   r261102-3403676). Masks in vertical voxel contact (contact / smaller mask's largest plane
   >= 0.3) merged per frame, re-tracked from trial 57's cached embeddings: 0.894 -> 0.942 on the
   same GT cells (mean row; 0.938 largest-piece row; threshold 0.5: 0.890 -> 0.928). 60% of the
   stealing masks are in such contact with the GT mask (48% at 0.5). The gain is about the size of
   the baseline mismatches that merging turns into hits (5.4 pt), i.e. it measures how much of the
   loss is "picked the stacked piece", not a better tracker. The rule also merges two annotated
   neurons in 699 groups (0.3) / 247 (0.5): needs a guard (z-extent, size) before it becomes a
   segmentation fix. Still open: are the pieces one cell? Look in napari.

## Open ideas, not yet checked

- Do the same GT cells fail in every trial (data / GT problem) or different ones (model)?
  `decompose_errors.py` writes `<subset>_cells.npz` (G = GT label, P = matched prediction, frame
  x neuron); run it on a second trial's cache and compare the error masks.
- Crop centre: A2 crops at mask centroids, the old score at clicks; NeRVE masks of 113 voxels
  median vs click offsets up to 2.6 um.
- Are the non-GT cells in z planes the old 32-plane rule clips (deeper planes, partial cells)?
- Score the old-NWB-best network (round_visnet_full trial 27, 0.972 on the old NWB) on A2 with
  all objects, and A2 with `--nwb_objects gt`, to separate data from scoring.
- Drop the 4 corrupted volumes (and maybe the 26 NeRVE-blank ones) from training.

## Tools in this folder (all committed)

- `decompose_errors.py --emb <npz> --subset all|gt|nerve --out <dir>`: re-tracks one trial's
  cached embeddings exactly as the objective does, prints the breakdown (per neuron, per frame,
  mask type, wrong-label kind, time blocks). ~35-60 min, ~5 GB, ~8 cores.
- `mask_geometry.py --out masks.npz`: per mask centroid, voxels, nearest GT click and distance,
  clicks inside, is-GT. ~5 min with 4 workers.
- `blank_volumes.py`: image checks of the 4 corrupted volumes and their neighbours.
- `lazy_nwb.py`: `load_nwb_project(path)` loads the NWB without reading the segmentation
  into memory (8 s). **Always use it locally.**
- `run_all.sh`: sequential driver with memory caps.

Outputs of today's runs: `/tmp/claude/claude-1000/-home-charles-Current-work-repos-barlow-track/ba25cb4a-3496-406f-97c9-e1aff6a1e52c/scratchpad/leifer_deep_dive/`
(`blank.log`, `masks.npz`, `decompose_{all,nerve,gt}.log` and `{all,nerve,gt}_cells.npz`).
Temporary: copy what you need.

## Hard-won lessons (do not re-learn)

- **Memory.** wbfm's `load_final_project_data_from_nwb` does `da.squeeze` on the h5py
  segmentation and reads all 35 GB on every load (issue i261102-1; fix belongs in wbfm,
  `finished_project_data.py` ~943/951, not yet made). Parallel loads tripped systemd-oomd
  twice on 2026-11-02 and killed the user's terminal and Firefox. Use `lazy_nwb.py`, and run
  long local jobs in their own unit: `systemd-run --user --unit=NAME -p MemoryMax=40G ...`
  (survives terminal crashes; an OOM stays inside the unit).
- One label propagation uses ~8 cores (serial main loop) and ~5 GB; three can run at once.
- The local GPU: the user freed it for this project again on 2026-11-02 (for the coming days).
- Python: `/home/charles/anaconda3/envs/wbfm/bin/python`; log runs with
  `ygg log "..." --pid N --log PATH`; commit as you go with the session trailer.
- `copy_nwb_file=False` in wbfm's `build_project_structure_from_nwb_file` deletes the source
  NWB: always pass True.
- No augmentation or filtering fitted to GT.
