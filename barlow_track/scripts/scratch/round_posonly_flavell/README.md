# round_posonly_{zimmer,flavell,samuel}

Can a position-only network learn neuron identity, and what stops the ViSNet position
branch from learning it? On Flavell the trained position branch of every fused network
matches about 10% of neurons between its own two training views and 2-3% between real
frames, no better than at random initialization, and spans about 2 effective dimensions.

One cause is in the encoder. pyg's `ExpNormalSmearing` places its RBF means for a cutoff
of 5 (its Angstrom default). At our cutoff of 0.35, every neighbour distance above about
0.14 x cutoff gets RBF features of about 0, so the encoder is blind to distances and an
untrained one gives every node the same output. Every earlier ViSNet run has this (issue
i261028-2). `visnet_rbf_rescale` (barlow_track@480b76c) builds ViSNet at cutoff 5 and
rescales the keypoints instead. On a real Flavell frame it lifts the untrained effective
rank from 4.0 to 14.7.

No position-only network has been trained before, so this is also the first such baseline.

## Grid (per dataset, 16 jobs, one seed)

Baseline: `fusion: position_only`, ViSNet (embedding 512, layernorm projector), global
augmentation as in the revision (rotation 180°, scale 0.1, translation (2, 8, 8)) plus
jitter 1 and dropout 0.2, all on the keypoints only (`apply_to_volume: false`, crops
untouched and unused), 50 epochs, frame-pair proxy each epoch (`val_match_frames: 30`).

| factor | values |
|---|---|
| `visnet_rbf_rescale` | false (every earlier run) / true |
| `visnet_cutoff_nn` | 0 = absolute radius 0.35 (23 / 10 / 15 NN on Zimmer / Flavell / Samuel), 3, 6, 12 NN (rescaled per frame, scale-invariant) |
| `global_augment.elastic_scale` | 0 / 1: elastic keypoint field matched to real lag-100 motion |

The matched fields (std z, x, y voxels / length) are Zimmer (0.2, 2.9, 2.9) / 19, Flavell
(1.1, 4.1, 4.1) / 41 and Samuel (0.2, 2.3, 2.3) / 10, from
[[EVD - The position augmentation deforms mid-range neighbour distances less than real motion does]].
Neighbourhoods are capped at the 32 nearest in every cell. The `rbf_rescale: false` cells
with `cutoff_nn > 0` stay mostly blind (the RBFs still span only 0.14 x the radius); they
are the factorial's control, not candidates.

Code: barlow_track@7641f83 or later (`visnet_rbf_rescale`, `visnet_cutoff_nn`,
`elastic_scale`; all default off).

## Launch (cluster)

```bash
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
for D in zimmer flavell samuel; do
  R=$B/round_posonly_$D
  mkdir -p $R && cp barlow_track/scripts/scratch/round_posonly_$D/*.yaml $R/
  # The dispatcher stays in the foreground until its sweep ends: background each
  # one, or the labs run one after another.
  nohup python barlow_track/scripts/optimize_hyperparameters.py -p $R/hyperparameter_search_template.yaml \
      --direct_parameter_sweep --repetitions 1 > $R/dispatcher.log 2>&1 &
done
```

Samuel trains on the loss objective: the sweep's accuracy objective scores against the training
project's own GT, and the samuel project has none (its GT is only in `samuel_data/153.nwb`; the
first launch stopped at the preflight). Once its trials have checkpoints, score them as every
earlier samuel number was scored:

```bash
python barlow_track/scripts/multiproject_scripts/run_trials_on_all_ground_truth.py \
    --trial_parent_dir $B/round_posonly_samuel --all --labs samuel --device cuda --slurm --no_fail_fast
```

## Reading it

- **rbf_rescale true well above false:** the RBF scale was what kept the position branch
  from learning. Fused networks need rerunning with the fix.
- **Nothing learns, even with the fix:** the position-only Barlow task itself is the
  problem (loss, views or encoder depth), not the RBFs.
- **Elastic on above off:** realistic keypoint deformation helps position generalize
  across the recording.
- **cutoff_nn:** which neighbourhood size carries identity, and whether it is the same
  on all three datasets (a single setting for one pipeline).

Position-only accuracy is scored against each project's own GT (Flavell project GT
reads ~3.4 pt low; rescore against the NWB GT if a cell is close to anything).
