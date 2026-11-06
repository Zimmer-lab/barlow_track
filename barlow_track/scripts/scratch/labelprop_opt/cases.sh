# Case lists for sweep_labelprop.py (round:trial:label; embeddings in <round>/emb_cache). Sourced by
# run_local.sh and labelprop_opt.sbatch. Recorded sweep accuracy (25 seeds, old sync) in comments.

# Screen (batch 1): one network per lab, strong and weak mixed
DEV_CASES=(
  round_bayes_wk2_samuel_137:50:samuel_137                   # .877 weak; Samuel = GT objects only
  round_bayes_wk3_samuel_184:27:samuel_184                   # .957 strong
  round_bayes_wk3_zimmer_1210:36:zimmer_1210                 # .929 weaker; project segmentation, distractors
  round_bayes_wk3_flavell_2022033002:61:flavell_2022033002   # .969 strong; NWB every detection
)

# Confirmation (batch 1 winners, batch 2): other networks of the same recordings + a held-out lab
# (Leifer, not looked at during the screen) + a held-out Flavell recording
CONFIRM_CASES=(
  round_bayes_wk2_samuel_137:63:samuel_137                   # .931 best S137
  round_bayes_wk3_samuel_184:32:samuel_184                   # .925
  round_bayes_wk3_zimmer_1210:52:zimmer_1210                 # .943 best Z1210
  round_bayes_wk2_leifer_reexport:57:leifer_reexport         # .885 held-out lab; NWB every detection
  round_bayes_wk3_flavell_2022032402:28:flavell_2022032402   # .973 held-out recording (median trial)
  round_bayes_wk2_leifer_reexport:57:leifer_stardist_strict  # .937 Leifer on StarDist masks, strict click GT
)                                                            # (the paper's Leifer; added 2026-11-06)
