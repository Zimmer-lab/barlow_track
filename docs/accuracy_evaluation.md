# Handoff: position-encoding accuracy evaluation (Sept 2026)

Goal: compare tracking accuracy of image-only vs position-aware Barlow embeddings,
untrained and trained, on four ground-truth datasets. All work below was done on a
**CPU-only** machine (no NVIDIA driver) with 56 cores. Nothing here modifies the
`/lisc` source projects; working copies live in `/tmp/claude`.

## 1. Environment

```bash
PY=/home/charles/anaconda3/envs/wbfm/bin/python
export NUMBA_CACHE_DIR=/tmp/claude/numba_cache MPLCONFIGDIR=/tmp/claude/mpl
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   # a third-party pytest plugin imports napari -> numba crash
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16  # scale to concurrency
```

* `~/.config` is **read-only** in this sandbox; numba/matplotlib need the redirects above or
  every `umap`/`hdbscan` import dies with `RuntimeError: cannot cache function ... no locator`.
  The repo test suite sets these automatically in `barlow_track/tests/conftest.py`.
* pytest must run with `--assert=plain` (same numba issue) and slow tests need `--runslow`.
* PyTorch has no CUDA here: everything forces `torch.device('cpu')`. Throughput reference
  (paper arch, batched backbone, ~10 cores): ~100-150 neurons/s embedding; labelprop-25 on
  120k points ≈ 25-35 min; on 283k points ≈ 2 h.

## 2. Data: projects, copies, ground truth

Analyzed projects contain only tracking data; raw volumes are referenced remotely, so
lightweight copies are cheap. Copies were made with wbfm's helper (drops analysis products):

```python
from wbfm.utils.projects.project_config_classes import make_project_like
make_project_like(src_config, '/tmp/claude/eval_projects',
                  steps_to_keep=['segmentation', 'preprocessing'],  # absolute remote refs; everything else fresh
                  new_project_name='zimmer')  # -> zimmer | flavell | leifer | samuel
```

Existing copies (reuse, ~200 KB each): `/tmp/claude/eval_projects/{zimmer,flavell,leifer,samuel}/project_config.yaml`.
**Zimmer fix:** its config lacks `exposure_time`, which crashes `ProjectData.load_final_project_data`
(`IncompleteConfigFileError` via `physical_units.py`). The copy was patched with
`exposure_time: 12` (same placeholder the flavell/leifer/samuel configs already carry).

| lab | working copy | GT source | GT content |
|---|---|---|---|
| zimmer | `eval_projects/zimmer` (1667 fr) | `.../wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format/project_config.yaml` | 131 finished neurons |
| flavell | `eval_projects/flavell` (1600 fr) | `.../barlow_track_paper/flavell_data/images_for_charlie/flavell_data.nwb` | 163 named neurons |
| samuel | `eval_projects/samuel` (1331 fr) | `.../barlow_track_paper/samuel_data/153.nwb` | 98 named neurons |
| leifer | NWB directly (1536 fr, `--source nwb`) | `.../barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb` | 79 named neurons, xyz + red volume in-file |

Leifer special case: its source `1-segmentation/metadata.pickle` **no longer exists on disk**, so the
analyzed leifer project cannot be (re-)embedded through the standard path. The NWB carries both the
volume `(1536, 32, 632, 600)` and GT xyz, so `eval_accuracy.py --source nwb` crops around GT positions
instead (this absorbed the old standalone `eval_leifer_accuracy.py`, deleted Oct 2026 after a
bit-identical validation run). NOTE: the old leifer script's `attention` mode used add-fusion +
attention while every other lab used concat-fusion + attention; the merged script standardizes
`attention` = concat + layernorm for all labs. Pre-merge leifer attention records used the old
add-fusion variant AND the old (ungated, bare-ReLU) architecture — do not compare them directly
with post-merge attention numbers.

## 3. Networks / weights

| weights | path | arch |
|---|---|---|
| untrained (paper) | `/lisc/.../wbfm/TrainedBarlow/untrained_{zimmer,flavell,leifer,samuel}/trial_0/resnet50.pth` | emb 64, target (8,64,64), random init |
| trained image-only | `/lisc/.../fieseler/barlow_track_revisions/config_files/initial_position_embedding/trial_4/resnet50.pth` | emb 1024, target (4,64,64) |
| trained position (in training) | `.../initial_position_embedding/trial_*/checkpoints/checkpoint.pth` | emb 512, target (8,32,32), concat+self-attn |

**Format check for position checkpoints** (all old trials are image-only: 45 keys, no `kenc`):

```bash
$PY -u -c "
import torch
sd = torch.load('<trial>/checkpoints/checkpoint.pth', map_location='cpu')
sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
keys = list(sd.keys())
print(len(keys), 'kenc:', any('kenc' in k for k in keys),
      'self_gnn:', any('self_gnn' in k for k in keys),
      'fuse_mlp:', any('fuse_mlp' in k for k in keys))
"
```

Real position nets show ~77 keys with all three True. Note: old `args.pickle` files lack `model_type`,
so `load_barlow_model` falls back to plain BarlowTwins and fails on the extra keys — construct
`BarlowVolumeAttention` explicitly (as the eval scripts do via `hasattr` dispatch) or ensure new runs
save `model_type` (current `train_barlow_clusterer.py` stamps it for fresh and resume runs).

## 4. Scripts (vendored in `barlow_track/scripts/`)

* `eval_accuracy.py` — main A/B: `--lab {zimmer,flavell,samuel,leifer} --mode {image,position,posonly,attention,trained}`.
  `--source {project,nwb}` selects the frame source (nwb = crop around GT xyz from the GT NWB itself;
  required for leifer). `--weights` + `--mode trained` evaluates any checkpoint (descriptor stage via
  `--descriptor_stage {auto,backbone,fused,contextual,projected}`; `--center_per_volume` / `--l2_per_volume`
  apply eval-time per-frame normalization). `--cluster {global,labelprop}`
  (default `labelprop`), `--num_seeds` (default 25; paper used 100). `--skip_embed` re-tracks saved
  embeddings from `/tmp/claude/emb_<lab>_<mode>.npz` without re-embedding. Appends JSON lines to
  `/tmp/claude/exp_results.jsonl` (records carry a `source` field).
* `stored_acc.py` — zero-compute baseline: accuracy of the paper's saved `df_barlow_tracks.h5` vs GT.
* `var_test.py` — re-tracks saved samuel-attention embeddings with UMAP seeds 0,1,2 (variance check).
* `audit_format.py` — verifies identical loading + 100% seg-id overlap pred-vs-GT per lab.

`eval_accuracy.py` is also importable, which is how the hyperparameter sweep uses it as an
objective (`optimize_hyperparameters.py`, `objective: accuracy`):

```python
from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint
record = evaluate_trained_checkpoint(weights=ckpt, project=project_path,  # gt defaults to project
                                     tag=f'{sweep}_trial{n}_{dataset}', results_jsonl=..., emb_dir=...)
```

Three knobs exist for that caller: `--gt` (score a checkpoint against a chosen project instead of
the lab's default GT — the sweep always passes its own training project), `--track_device`
(`cpu` keeps the label-propagation graph off the GPU: a full-video graph is 100k-200k nodes with
dense per-step (N x classes) allocations, which OOMs a 12 GB GPU next to the trained model), and
the embedding cache, which is written atomically and validated (frame count) on reuse so a killed
sweep resumes without re-embedding.

Typical full run (labelprop-25, paper mode):

```bash
$PY -u barlow_track/scripts/eval_accuracy.py --lab zimmer --mode attention \
  --cluster labelprop --num_seeds 25
$PY -u barlow_track/scripts/eval_accuracy.py --lab leifer --source nwb --mode trained \
  --weights <trial>/resnet50.pth --cluster labelprop --num_seeds 25
```

Accuracy recipe everywhere (paper's): `rename_columns_using_matching(df_gt, df_pred,
column='raw_segmentation_id')` after `pad_with_nan_rows`, then `calculate_accuracy`
(`barlow_track/utils/utils_ground_truth.py`). Leifer uses `raw_neuron_ind_in_list` (its GT lacks
seg ids at match time; metadata join is skipped, raw ids come from the NWB directly).

## 5. Results so far

Untrained, identical backbone weights per lab (labelprop-25 unless noted):

| dataset | image | pos-only | +position (add) | +attention (concat) | paper stored |
|---|---|---|---|---|---|
| samuel | 0.769 | 0.001 / 0.566 | 0.568 | 0.622 | n/a |
| zimmer | 0.653 | 0.022 | 0.022 | 0.034 | 0.618 |
| flavell | 0.696 | 0.118 | 0.119 | 0.130 | 0.696 |
| leifer | 0.892 | 0.427 | 0.534 | 0.803 | 0.872 |

Trained image-only (trial_4) + labelprop-25: samuel **0.827**, zimmer **0.887**, flavell **0.791**, leifer **0.943**.
Global-clustering round (history): image 0.76/0.68/0.42/0.86; add-fusion collapses on zimmer/flavell (~0.00-0.02).

Read: untrained image reproduces the paper (flavell exact to 4 decimals); concat-attention is the best
position variant everywhere and near image-level on leifer/samuel; zimmer/flavell need *trained* fusion.
Labelprop on weak embeddings is seed-fragile (samuel posonly: 0.001 vs 0.566 across identical runs —
unseeded `random.shuffle` of seed times; UMAP-seed spread on fixed embeddings is only ±0.01).

## 6. Gotchas (all bitten before)

* torchio needs **4D** `(N,Z,X,Y)` tensors (N as channels, as legacy code does) — 5D raises.
* `KeypointEncoder`'s InstanceNorm needs N>1: `encode_position` returns zeros for N<2 (graceful
  visual-only fallback, in `barlow_superglue.py`); experiment scripts additionally skip <2-detection frames.
* Frames with **zero** detections crash `np.stack` — same `<2` skip covers it.
* `BarlowSuperGlue` (cross-volume matching) is a documented **stub**: no inference path, `forward()` raises.
* Model default is now `fusion: concat`; `fusion: position_only` benchmarks pure geometry.
* `load_barlow_model(fname, expected_args=config)` validates architecture strictly — including a
  latent-bug fix where training silently ignored custom `backbone_kwargs` (always used defaults).
* Leifer analyzed projects are unusable for fresh embedding (segmentation gone); use `--source nwb`.
* `exp_results.jsonl` entries without `cluster` are global-mode; `n_frames: 2/3` entries are sanity runs.
* Never `pkill -f` a pattern that appears in your own command line (it kills your shell); kill by PID.
* When editing the experiment loops, keep the per-frame body indented under `for t` — two separate
  runs were lost to dedent bugs that silently embedded one frame.
* **Full-video tracking does not fit on one GPU** next to the trained model (12 GB TITAN V OOMed in
  `clamped_label_propagation` after embedding 1667 frames); use `--track_device cpu`.
* Label propagation is seeded end to end (`--seed`), but two labelings of identical embeddings can
  still disagree on a small margin; treat sub-0.01 accuracy differences as noise.

## 7. Suggested next steps

1. Evaluate finished position trials on all four sets (`--mode trained --weights <resnet50.pth>`).
2. For publication numbers use labelprop-100 + fixed `random_state` (seed fragility above).
3. Train the `position_only` ablation (`use_position: true, fusion: position_only`) — smoke-tested.
4. Consider seeding `random.shuffle` in `track_using_label_propagation_clusterer` for reproducibility.
