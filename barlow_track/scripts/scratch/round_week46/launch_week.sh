#!/usr/bin/env bash
# Cluster runs for the week of 2026-11-10 (after the held-out scoring of round_shared_v1). Run on the login node:
#
#   cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
#   W=barlow_track/scripts/scratch/round_week46/launch_week.sh
#   bash $W rescore     # 1. label propagation v2 on every round_shared_v1 cell from emb_cache (8 GPU jobs,
#                       #    ~3 h each), tracks saved; then the cross-network absolute proxy (CPU, runs after)
#   bash $W heldout     # 2. held-out scoring of the remaining top-8 proxy trials (42 57 52 45, 2 replicates) and
#                       #    replicates 2-4 of trial 59: 48 + 18 GPU jobs (train ~4-6 h + score)
#   bash $W sens        # 3. sensitivity rounds around trial 59 on the 6 tuning recordings (staged 2026-11-10):
#                       #    534 core + 24 ablation GPU jobs, NUM_PARALLEL (default 2) per dispatcher, 54 dispatchers
#   bash $W hdbscan     # 4. HDBSCAN (global clustering) on trial 59's networks, all 12 recordings (CPU)
#   bash $W all         # 1-4 in that order
#   bash $W sens_rescore  # LATER, once `sens` has finished: v2 + saved tracks for every sensitivity cell, then the
#                         # cross-network proxy over all of a recording's networks (image-only, position-only, fused, ...):
#                         # rerun it when the per-round jobs are done to get cross_network_proxy_sens_v2.jsonl
#   bash $W status      # what has been recorded so far
#
# Everything resumes: rerun the same command (sens: SKIP=1 bash $W sens). Nothing here touches exp_results.jsonl
# of round_shared_v1 except `heldout`, which appends new cells as the earlier held-out scoring did.
set -euo pipefail
B=${B:-/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files}
export CODE=${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}
R=$B/round_shared_v1
P=barlow_track/scripts/scratch/gt_free_proxy
cd "$CODE"
mkdir -p $R/week46_logs

# Slurm interaction is the sweeps' own: every stage is a dispatcher run by sbatch_dispatcher.sbatch that submits its
# GPU / CPU jobs with submitit and optimize_hyperparameters' job parameters.
dispatch() {  # dispatcher script, job name, then its arguments
  local script=$1 name=$2; shift 2
  BARLOW_DISPATCHER=$script sbatch --job-name=$name --output=$R/week46_logs/${name}_%j.log \
      barlow_track/scripts/sbatch_dispatcher.sbatch "$@"
}

rescore() {
  dispatch $P/rescore_round.py rescore_v2 --round_dir $R --labelprop v2 --slurm 8 --then_proxy
}

heldout() {
  dispatch barlow_track/scripts/optimize_shared_config.py heldout_top8 -p $R --score_heldout --trials 42 57 52 45 --replicates 2
  dispatch barlow_track/scripts/optimize_shared_config.py heldout_t59_seeds -p $R --score_heldout --trials 59 --replicates 5
}

sens() {
  NUM_PARALLEL=${NUM_PARALLEL:-2} bash barlow_track/scripts/scratch/round_sens_src/launch_sensitivity.sh launch core
  NUM_PARALLEL=${NUM_PARALLEL:-2} bash barlow_track/scripts/scratch/round_sens_src/launch_sensitivity.sh launch ablations
}

hdbscan() {
  dispatch $P/rescore_round.py hdbscan_t59 --round_dir $R --cluster global --trials 59 --track_device cpu --slurm 1 --minutes 2880
}

sens_rescore() {
  for D in $B/round_sens_*/; do
    D=${D%/}; [ -f $D/exp_results.jsonl ] || continue
    n=$(wc -l < $D/exp_results.jsonl); m=0; [ -f $D/exp_results_v2.jsonl ] && m=$(wc -l < $D/exp_results_v2.jsonl)
    [ "$m" -ge "$n" ] && continue
    BARLOW_DISPATCHER=$P/rescore_round.py sbatch --job-name=rescore_$(basename $D) --output=$D/rescore_v2_%j.log \
        barlow_track/scripts/sbatch_dispatcher.sbatch --round_dir $D --labelprop v2 --slurm 1
  done
  python $P/cross_network_proxy.py --tracks_dir "$B/round_sens_*/tracks_v2" "$R/tracks_v2" \
      --results "$B/round_sens_*/exp_results_v2.jsonl" "$R/exp_results_v2.jsonl" \
      --out $B/cross_network_proxy_sens_v2.jsonl | tail -40 || true
}

status() {
  for f in exp_results_v2.jsonl exp_results_global.jsonl heldout_results.jsonl cross_network_proxy_v2.jsonl; do
    [ -f $R/$f ] && echo "$f: $(wc -l < $R/$f) lines" || echo "$f: not yet"
  done
  echo "sensitivity records: $(cat $B/round_sens_*/exp_results.jsonl 2>/dev/null | wc -l) of 558"
  squeue -u "$USER" -h -o '%j %T' | awk '{print $2}' | sort | uniq -c
}

case ${1:?rescore | heldout | sens | hdbscan | all | sens_rescore | status} in
  rescore|heldout|sens|hdbscan|sens_rescore|status) $1 ;;
  all) rescore; heldout; sens; hdbscan ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
