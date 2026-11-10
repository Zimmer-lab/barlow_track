# round_sens_*: sensitivity tests around the frozen round_shared_v1 recipe (staged 2026-11-09, NOT launched)

The recipe of round_shared_v1 trial 59 (`base_train_config.yaml`: its base config plus the trial's parameters and
seed), trained anew on each recording and varied one factor at a time, scored by full-video tracking accuracy
(label propagation, in the training GPU job). Standard machinery only: `make_per_recording_rounds.py` writes
`round_sens_<round>_<label>/`, `optimize_hyperparameters.py` runs it (`--direct_parameter_sweep`,
`--one_at_a_time_sweep`, `--repetitions`), `sensitivity_report.py` reads the records. `launch_sensitivity.sh` does the
staging and the `sbatch` calls (usage in its header).

| round | sweep | repetitions | what |
|---|---|---|---|
| loss_volume | direct, 2 x 3 x 3 | 3 | fusion {visual_only, add} x lambd_obj {0, 0.347, 1} x num_frames {16, 64, 371}; (add, 0.347, 371) is the control |
| oat | one at a time, 31 values | 1 | num_frames {128, 256, 1000}; p_global_affine, max_degrees_z, elastic_scale, p_noise, crop_degrees_z, dropout_p, spurious_p each 0; lr x0.3 / x3; embedding_dim {64, 128, 1024}; projector_final {128, 2048}; target_sz_xy {32, 48}; epochs {0 (untrained), 10, 60}; fusion {position_only, concat}; position_encoder keypoint_mlp; use_attention true; backbone f_maps {2, 8}; visnet_hidden 128; visnet_neighbors 8; reaugment_each_epoch false |
| attention2 | direct, 1 | 1 | use_attention true with self_layers 2 (1 layer is an oat value) |
| crop_depth | direct, 2 | 1 | target_sz_z {8, 16} on every recording (Flavell's hand-set 16 vs the default 8; the recording's own value repeats the control) |
| matched_steps | direct, 1 | 1 | 64 frames, 174 epochs (the control's optimizer steps) |
| only_affine / only_elastic / only_noise / aug_none | direct, 1 | 1 | the other image augmentations (and for aug_none the position dropout and spurious objects) off; keypoint jitter stays |

Every config shares the recipe seed, so a variant is compared with the control cell of the same seed
(`sensitivity_report.py` pairs on label and seed); repetitions of loss_volume get `replicate_seed`s.
6 tuning recordings: 534 core jobs (loss_volume 324, oat 186, matched_steps 6, attention2 6, crop_depth 12) and 24 augmentation jobs.

Notes: fusion visual_only is the image-only network inside the position pipeline (use_position false would switch to
the legacy per-crop pipeline and ignore global_augment). Crop depth is not swept (Flavell uses 16, issue i261105-3 on its
voxel size). Not covered: HDBSCAN vs label propagation and cross-dataset transfer (base numbers on good networks only).
Read: `python barlow_track/scripts/sensitivity_report.py --out sensitivity_summary.tsv`.
