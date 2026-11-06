---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1ab7b-729d-741f-948c-0548c3f018f9
status: in-progress
priority: normal
dateCreated: 2026-11-05T12:32:36+01:00
dateModified: 2026-11-05T12:32:36+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Method

All scored fusion=add trials of the GT-tuned per-recording Bayesian rounds round_bayes_wk2_* and round_bayes_wk3_* (config_files/<round>/exp_results.jsonl), against the last logged value of each GT-free training metric in trial_N/log/stats.json: summed val loss, val_mean_batch_loss, train_loss, fused-embedding off-diagonal correlation, mean fraction, per-dimension std median (`val_val_match_accuracy_auto` left out: it uses GT names). Spearman raw, and partial: rank residuals after linear regression on the swept settings (log num_frames, log epochs, log lr, lambd_obj, self_layers, max_degrees_z, scale_jitter, elastic_scale). Scripts barlow_track/scripts/scratch/gt_free_proxy/proxy_corr.py and proxy_partial.py (uncommitted as of 2026-11-05).

## Result

The summed val loss does not correlate with accuracy. Per-batch val loss and train loss correlate at Spearman -0.46 to -0.83 raw, but only -0.15 to -0.5 once the settings are regressed out, weakest on Samuel 137 and 184 (the hard recordings). The embedding statistics are weaker still.

## Conclusion

Within a sweep, a lower loss mostly marks a longer or larger training run (more frames, epochs, a better lr), which also tracks better; beyond the settings it says little about accuracy. It cannot select a config without GT, least of all on the hard recordings where it matters.

## Caveats

Last-epoch values only; trials differ in epochs, so "last" is not one training stage. Partial correlations assume rank-linear effects of the settings. fusion=visual_only trials left out. Scripts not committed yet.
