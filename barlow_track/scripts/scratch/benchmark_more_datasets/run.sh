#!/bin/bash
# Score the best existing checkpoints of each kind (docs/best_of_table.md) on every further GT
# recording, CPU only (embedding and tracking), one Slurm job per (checkpoint, dataset) cell:
#   Zimmer 2022-11-23 worm 11 and 2022-12-10; Flavell 2022-03-31-01 (GT objects) and the four
#   forCharlie_2 recordings; Samuel 137 and 184-004.
# Checkpoints (all on /lisc):
#   fused:         round_visnet_full trial 27 (Zimmer), round_bayes_samuel trial 0 (Samuel)
#                  - both stages: auto (fused) and backbone (image features alone)
#   image-only:    round_flavell_paper_ablation trial 0 (Flavell), round_2x2 trial 6 (Zimmer)
#   position-only: round_posonly_zimmer trial 8 (RBF fix)
# Records go to each round's exp_results.jsonl (auto) or exp_results_backbone.jsonl (backbone,
# with its own embedding cache), tags <round>_trial<N>_<dataset> as usual. Run from the
# barlow_track checkout on the cluster after pulling; then rerun barlow_track/scripts/best_of_table.py.
set -u
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
D="zimmer_1123 zimmer_1210 flavell flavell_2022031603 flavell_2022032402 flavell_2022033001 flavell_2022033002 samuel_137 samuel_184"
DRV="python barlow_track/scripts/multiproject_scripts/run_trials_on_all_ground_truth.py --labs $D \
    --device cpu --track_device cpu --slurm --slurm_time 12:00:00 --slurm_cpus 16 --slurm_mem 48G --no_fail_fast"

$DRV --trial_parent_dir $B/round_visnet_full --trials 27 &
$DRV --trial_parent_dir $B/round_bayes_samuel --trials 0 &
$DRV --trial_parent_dir $B/round_flavell_paper_ablation --trials 0 &
$DRV --trial_parent_dir $B/round_2x2 --trials 6 &
$DRV --trial_parent_dir $B/round_posonly_zimmer --trials 8 &
for R in round_visnet_full:27 round_bayes_samuel:0; do
  $DRV --trial_parent_dir $B/${R%%:*} --trials ${R##*:} --descriptor_stage backbone \
      --results_jsonl $B/${R%%:*}/exp_results_backbone.jsonl --emb_dir $B/${R%%:*}/emb_cache_backbone &
done
wait
echo "benchmark_more_datasets done"
