# round_flavell_legacy_crop_aug

Every position-pipeline network trails the preprint's legacy crop pipeline on Flavell by
about 15 points (0.78-0.82 against 0.95-0.96, NWB GT). Fusion, rotation, per-frame mean
drift, edge crops, frozen view pairs and GT-only objects are each ruled out locally. A
local run that removes the position pipeline's geometric augmentation from the crops looks
like the legacy pipeline from epoch 0: val loss 0.0043 and falling, proxy past every
earlier position run by epoch 4 (r261028-838680). That run has image features only, the
preprint per-crop view transforms and no global volume affine.

This round asks the two follow-up questions with training-seed replicates. Position
networks' Flavell accuracy spans about 12 points across seeds, so one run per cell is not
enough.

1. Does the global augmentation cost accuracy because it reaches the image crops?
2. With image views made the preprint way, does position fusion add to the image features?

## Grid

Baseline: the preprint Flavell config through the position pipeline. It uses
`crop_augment: legacy` (barlow.Transform's view 1 and view 2, probabilities from the `p_*`
keys), `reaugment_each_epoch: true`, and the revision's global augmentation: rotation 180°,
scale 0.1, translation (2, 8, 8), flip 0.25, jitter 1 voxel, dropout 0.2. 2 × 2 cells,
3 seeds each, 12 jobs:

| | `apply_to_volume: false` (keypoints only) | `apply_to_volume: true` (volume too) |
|---|---|---|
| `fusion: visual_only` | crops untouched by geometry (replicates the local legacy-matched run) | global affine and jitter reach the crops |
| `fusion: add` | **candidate design**: legacy-style image views plus augmented positions | revision pipeline plus legacy crop transforms |

With `apply_to_volume: false`, the affine, jitter and dropout move only the keypoints. Crops
come from the untransformed volume at the original centroids, so the image views differ only
by the per-crop transforms. With `fusion: visual_only`, the descriptor is the backbone output
(the position branch is built but unused), so the `auto` stage equals `backbone`.

Code: barlow_track@8ae7898 (flags, default off) and d02573b (dotted sweep names). Training
should take about 25 h per job; the legacy per-crop elastic transform on CPU dominates
(7 s/step locally). That is inside the 36 h training budget.

## Launch (cluster)

```bash
R=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_flavell_legacy_crop_aug
mkdir -p $R && cp train_config.yaml hyperparameter_search_template.yaml README.md $R/
python barlow_track/scripts/optimize_hyperparameters.py -p $R/hyperparameter_search_template.yaml \
    --direct_parameter_sweep --repetitions 3
```

## Rescore against the NWB GT

The objective scores against the project GT, which reads about 3.4 points low. Rescore each
trial's cached embedding against the NWB GT. Also embed at rounded crop centres, which is how
the position pipeline trains. On the comparator this is worth +1.9 points (0.795 → 0.814),
and the default truncation costs it that much.

```bash
P=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data
for N in $(seq 0 11); do
  T=round_flavell_legacy_crop_aug_trial${N}_flavell
  python barlow_track/scripts/eval_accuracy.py --lab flavell --source project \
      --project $P/projects/2025_07_01/project_config.yaml --gt $P/images_for_charlie/flavell_data.nwb \
      --mode trained --weights $R/trial_$N/resnet50.pth --skip_embed --emb_dir $R/emb_cache --tag $T \
      --track_device cpu --results_jsonl $R/rescore_nwb_gt.jsonl
  python barlow_track/scripts/eval_accuracy.py --lab flavell --source project \
      --project $P/projects/2025_07_01/project_config.yaml --gt $P/images_for_charlie/flavell_data.nwb \
      --mode trained --weights $R/trial_$N/resnet50.pth --round_crop_centers --emb_dir $R/emb_cache_round \
      --tag ${T}_round --track_device cpu --results_jsonl $R/rescore_nwb_gt.jsonl
done
```

## Reading it

- **visual_only, keypoints only near 0.95:** the position data path is fine once geometry stays
  off the crops. A lower **visual_only, volume** row says the global augmentation is the cost.
- **add, keypoints only** above its visual_only neighbour: position fusion helps when image
  views are made the preprint way. This answers the paper revision's question, on Flavell.
