---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1c054-977d-72fe-9a69-0c8c51c6e323
status: done
priority: high
dateCreated: 2026-11-08
dateModified: 2026-11-09T13:42:58+01:00
confidence: medium
outcome: succeeded
projects:
  - "[[barlowtrack]]"
completedDate: 2026-11-09
---

## Method

round_shared_v1 (see the Source): 60 Ax trials, each one config trained and scored on the 6 tuning recordings, objective the lower quartile of agree_mean relative to each recording's best per-recording (wk2/wk3) agree_mean, so 0 means "as consistent as the best per-recording network". Accuracy (labelprop-25, as eval_accuracy --mode trained) recorded alongside and compared with reference_accuracy, the best per-recording trial of wk2/wk3. Read from shared_results.jsonl (60 records, none failed; 360 slurm jobs without a traceback, OOM or cancellation).

## Result

The best trial by objective (trial 59, -0.0085) scores, against the per-recording best: samuel_184 0.967 (+1.0 pt), flavell_2022033002 0.971 (+0.1), zimmer_1123 0.987 (0.0), zimmer_1210 0.938 (-0.5), flavell_2022032402 0.972 (-0.8), samuel_137 0.909 (-2.2). Mean 0.957 vs 0.961 for the per-recording bests. Trial 50 has the same mean (0.957; samuel_137 -2.0, the rest within 0.7 pt). The best objective per block of 10 trials was -0.040, -0.064, -0.029, -0.018, -0.011, -0.009; mean accuracy over the first 10 trials was 0.842, over the last 10 0.955.

Where the ten best trials by objective agree: lambd_obj 0.30-0.46, epochs 14-30, num_frames 440-725, scale_jitter 0.14-0.19, elastic_scale 0.29-0.49, crop_degrees_z 31-58, p_noise 0.35-0.48, dropout_p 0.06-0.21, spurious_p 0.16-0.23, lr about 2e-5. Global z rotation is near 0 in 9 of the 10 (only trial 59 uses 41 deg). Samuel 137 is the only recording consistently 1.6-5 pt below its per-recording best among them.

## Conclusion

A single recipe, chosen with no ground truth, comes within 0.4 pt on average (0.0-0.8 pt on five of six recordings) of the best network tuned per recording on these six recordings; only Samuel 137 stays 2 pt below. The top ten trials converge on a narrow region (moderate augmentation, 400-700 frames, 14-30 epochs, little global z rotation), so the recipe is well defined enough to freeze.

## Caveats

Tuning recordings only: the six held-out recordings (zimmer_1128, flavell 2025-07-01, flavell_2022031603, flavell_2022033001, samuel 153, leifer_stardist_strict) have not been scored, so generalisation of the recipe is not shown. The per-recording references are maxima over many wk2/wk3 trials (optimistically biased), so "matches" means "within the selection noise of the baseline"; they had more freedom (fusion) and an older sync. Selection was on agree_mean with 25 seeds, and the top trials differ by 0.1-0.5 pt of objective, below its noise. "Little global rotation" is observed, not tested. The trial configs carry no reaugment_each_epoch key, so they ran with the default (true since 2085014, before the launch). Samuel 184 uses voxel size 0.45 x 0.45 x 1.75 um (z_scale 3.89), not the 5.83 of its reference. The sweep was scored with the label propagation recipe of the time (labelprop-25), not recipe v2. self_layers (and attn_gate_init) had no effect in this sweep: use_attention was false in every trial, so the network was BarlowWithPosition, which has no self-attention layers (found 2026-11-10); the sweep therefore searched one inert dimension, and attention within a volume is untested for the shared recipe.
