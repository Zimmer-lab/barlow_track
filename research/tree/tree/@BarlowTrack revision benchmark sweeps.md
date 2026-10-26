---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a165a3-b2d7-7a28-8c25-dc4b95720a5d
status: done
priority: normal
dateCreated: 2026-10-19
dateModified: 2026-10-19
completedDate: 2026-10-19
run: barlow_track@bc9ff96:barlow_track/scripts/multiproject_scripts/run_trials_on_all_ground_truth.py
data:
  - /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/
projects:
  - "[[barlowtrack]]"
---

## Description

Benchmark driver for the revision hyperparameter rounds (Ax Bayesian or direct grids, launched with optimize_hyperparameters.py): scores every trial checkpoint of a round on the ground-truth sets with `eval_accuracy.py --mode trained`, writing `exp_results.jsonl` and `exp_results_unified.json` per round under config_files/<round>/. Tags `<round>_trial<N>_<lab>`, lab in zimmer_1128, flavell, leifer, samuel.

The results used in this tree are the summary under "Empirical state (Oct 2026 bayes rounds)" in the repository's AGENTS.md, which is git-ignored (not committed), so the summary is not versioned; the underlying records are on the cluster. Not rerun or checked here.
