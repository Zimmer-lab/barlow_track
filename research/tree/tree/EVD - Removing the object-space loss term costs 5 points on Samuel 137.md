---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a19fd4-dbde-729f-bfc3-f58885f4583a
status: done
priority: normal
dateCreated: 2026-11-03T06:14:49+01:00
dateModified: 2026-11-03T06:15:18+01:00
confidence: medium
outcome: succeeded
projects:
  - "[[barlowtrack]]"
completedDate: 2026-11-03
---

## Purpose

Test the object-space term below the accuracy ceiling, with training-seed replicates: the wk2 sweeps never sampled lambd_obj < 0.1, so their flat partial dependence (Samuel 137 0.859-0.863 over 0.1-1, sweep_param_effect.py 62cd4c8) says nothing about turning a term off.

## Method

Samuel 137 one-change pilot (see the Source; runs r261102-365990, r261102-2656039, r261103-1931191, r261103-732047). Trial 63's config (ViSNet position pipeline, lambd_obj 0.40) retrained with lambd_obj 0.40 (base), 0 (lambd0), 1 (lambd1), and with reaugment_each_epoch (reaug), at three training seeds each, scored as the sweep objective (NWB GT objects, label propagation 25 seeds, seed 0). Differences are paired by training seed.

## Results

| variant | seed trial 63 | seed 2 | seed 3 | mean | paired vs base |
|---|---|---|---|---|---|
| base (lambd_obj 0.40) | 0.920 | 0.884 | 0.872 | 0.892 | |
| lambd0 (feature term only) | 0.893 | 0.804 | 0.821 | 0.839 | -2.7 / -8.0 / -5.1, mean -5.3 pt, paired t p 0.07 |
| lambd1 (object term only) | 0.899 | 0.864 | 0.912 | 0.892 | -2.1 / -2.0 / +4.0, mean 0.0 |
| reaug | 0.917 | 0.859 | 0.911 | 0.896 | -0.2 / -2.5 / +3.9, mean +0.4 |

Noise: training seed sd 2.5 pt for base. Label propagation seed spread within one network ~0.8 pt (base 0.920/0.927/0.927 at seeds 0/1/2; base_seed2 0.884/0.878/0.876); 100 seeds gives 0.928 and 0.884, so the 4.4 pt gap between training seeds is the network, not the tracker (r261102-77194). Trial 63's own seed is a good draw: its sweep score 0.931 overstates the config's mean by ~4 pt.

## Conclusion

On Samuel 137 with the ViSNet position pipeline, training on the feature-space (redundancy-reduction) term alone loses 5.3 points of tracking accuracy (lower at all 3 training seeds), while the object-space term alone matches the 0.40 mixture (0.892 vs 0.892). So below the ceiling the object-space term carries the accuracy and the feature-space term adds nothing beyond it; the balance is not irrelevant, contrary to the preprint's near-ceiling result. reaugment_each_epoch has no detectable effect here (+0.4 ± ~3 pt).

## Caveats

One recording (Samuel 137), one config (trial 63), three training seeds; p 0.07 for lambd0. Training seed noise (sd 2.5 pt) is larger than the lambd1 and reaug differences, so those are "no effect detected", not "no effect". Scored on NWB GT objects (no distractor detections). reaug_seed2, reaug_seed3 and lambd0_seed3 trained after the bc3c1bf validation-plot fix, the others before; the fix touches only the wandb plot, not the loss. Needs a second dataset (Zimmer or Flavell) before it generalizes.
