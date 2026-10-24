# round_flavell_paper_ablation

Why do the revision networks trail the preprint on flavell? Measured on the same
eval path (`eval_accuracy --source project`, NWB GT, 25 labelprop seeds), the
preprint checkpoint scores 0.952 and the best revision net
(round_flavell_more_epochs trial_27) scores 0.924, or 0.915 at the backbone stage.
trial_27 also trained longer than the preprint net (17,900 vs 11,900 steps).

This round starts from the preprint config and toggles each design change.
The grid has 8 configs, each run with 2 training seeds:

| key | preprint | revision |
|---|---|---|
| `use_position` | false (legacy crop pipeline) | true (global augmentation + ViSNet, add fusion) |
| `embedding_dim` | 64 | 512 |
| `projector_norm` | batchnorm | layernorm |

`use_position` also switches the data pipeline: `train_barlow_clusterer.py` uses
the legacy `NeuronCropImageDataModule` unless position is on. Separating the two
would need a visual-only fusion mode, which doesn't exist yet. If `use_position`
turns out to matter, that split is the next round.

All 8 models were checked to build with the current code, and the objective
preflight passes (160 neurons, 1599 frames).

## Launch (cluster)

```bash
R=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_flavell_paper_ablation
python barlow_track/scripts/optimize_hyperparameters.py -p $R/hyperparameter_search_template.yaml \
    --direct_parameter_sweep --repetitions 2
```

## Rescore against the NWB GT (the GT behind the preprint's 96.7%)

The sweep objective scores against the project GT, which reads about 4 pt low:
preprint stored tracks score 0.927 there vs 0.968 against the NWB GT. The
rescore reuses each trial's cached embedding:

```bash
P=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data
for N in $(seq 0 15); do
  T=round_flavell_paper_ablation_trial${N}_flavell
  python barlow_track/scripts/eval_accuracy.py --lab flavell --source project \
      --project $P/projects/2025_07_01/project_config.yaml --gt $P/images_for_charlie/flavell_data.nwb \
      --mode trained --weights $R/trial_$N/resnet50.pth --skip_embed --emb_dir $R/emb_cache --tag $T \
      --track_device cpu --results_jsonl $R/rescore_nwb_gt.jsonl
done
```

## Local companion

`../retrain_paper_baseline_flavell/` retrains the preprint config with the
current code on the local machine. That corresponds to the (false, 64,
batchnorm) cell here, with seed 0 and evaluated directly against the NWB GT.
