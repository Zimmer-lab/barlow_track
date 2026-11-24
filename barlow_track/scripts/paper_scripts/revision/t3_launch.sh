#!/usr/bin/env bash
# TEST_PLAN T3: per-time exclusivity ablation and a fair HDBSCAN, on trial 59's cached embeddings (no training,
# no embedding: every cell re-tracks round_shared_v1/emb_cache). Run on the login node:
#
#   cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
#   X=barlow_track/scripts/paper_scripts/revision/t3_launch.sh
#   bash $X smoke           # samuel 153 rep 0: the 3 recipes on this node's GPU, the 4 HDBSCAN cells on its CPU
#   bash $X labelprop_here  # v2_noE2, v2_noE3, v2_noexcl on all 36 trial-59 cells on this node's GPUs (NGPU=2
#                           # nohup processes, ~9 GPU-hours in all, so ~half a day); no Slurm jobs
#   bash $X labelprop       # the same as Slurm GPU jobs instead (SHARDS per variant), if the login GPUs are busy
#   bash $X hdbscan         # CPU Slurm jobs: the HDBSCAN 2 x 2 at the defaults on all 36 (global, global_soft,
#                           # global_excl_hard, global_excl) and the other 8 grid settings of plain global on the
#                           # 12 rep-0 cells
#   bash $X excl_tuned      # after hdbscan (CPU jobs): the other three 2 x 2 cells at each recording's tuned setting
#   bash $X status          # cells recorded per variant
#   bash $X report          # t3_report.py -> $R/t3_table.tsv
#   DRY=1 bash $X <stage>   # print the commands instead of submitting
#
# Every Slurm stage is a rescore_round.py dispatcher (sbatch_dispatcher.sbatch, submitit shard jobs with the
# sweeps' job parameters). Label propagation runs on a GPU (~5 min a cell). HDBSCAN is CPU work (UMAP, HDBSCAN,
# soft membership), so its cells run as CPU jobs (cpu_eval_job_params: 16 CPUs, 64G) with --track_device cpu;
# they read the GPU-made embedding cache (eval_accuracy reuses a "_slide" cache on CPU), so nothing is re-embedded.
# Finished cells skip on a rerun, so any stage can be relaunched after a failure.
set -euo pipefail
B=${B:-/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files}
CODE=${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}
R=$B/round_shared_v1
P=barlow_track/scripts/scratch/gt_free_proxy/rescore_round.py
SHARDS=${SHARDS:-4}          # GPU shards per label propagation variant (labelprop stage)
CPU_SHARDS=${CPU_SHARDS:-12}  # CPU shards per 36-cell HDBSCAN variant; grid settings (12 cells) get half
cd "$CODE"
mkdir -p $R/t3_logs

run() { if [ -n "${DRY:-}" ]; then echo "DRY: $*"; else "$@"; fi; }
dispatch() {  # $1 = job name, rest = rescore_round.py arguments
  local name=$1; shift
  run env BARLOW_DISPATCHER=$P sbatch --job-name=t3_$name --output=$R/t3_logs/${name}_%j.log \
      barlow_track/scripts/sbatch_dispatcher.sbatch --round_dir $R --trials 59 "$@"
}
MCS=(0.25 0.5 0.75)
MS=(0.005 0.02 0.05)

case ${1:-} in
  smoke)
    for v in v2_noE2 v2_noE3 v2_noexcl; do
      run python $P --round_dir $R --trials 59 --labs samuel --no_replicates --labelprop $v 2>&1 | tail -3
    done
    for c in global global_soft global_excl_hard global_excl; do
      run env CUDA_VISIBLE_DEVICES= python $P --round_dir $R --trials 59 --labs samuel --no_replicates --cluster $c \
          --track_device cpu 2>&1 | tail -3
    done
    ;;
  labelprop_here)
    # one nohup process per login-node GPU, each taking shard g of NGPU of every variant in turn (as
    # round_week47/launch_week.sh's *_here stages); logs in $R/t3_logs/labelprop_here_gpu<g>.log
    if pgrep -u "$USER" -f 'rescore_round\.py.*--shard' > /dev/null; then
      echo "login-node rescores already running (one stage at a time):"; pgrep -u "$USER" -af 'rescore_round\.py.*--shard'
      exit 1
    fi
    ngpu=${NGPU:-2}; th=${THREADS:-8}
    for ((g = 0; g < ngpu; g++)); do
      cmd="for v in v2_noE2 v2_noE3 v2_noexcl; do python -u $P --round_dir $R --trials 59 --labelprop \$v --shard $g --num_shards $ngpu; done"
      if [ -n "${DRY:-}" ]; then echo "DRY: CUDA_VISIBLE_DEVICES=$g nohup bash -c '$cmd'"; continue; fi
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th NUMBA_NUM_THREADS=$th nohup bash -c "$cmd" \
          > $R/t3_logs/labelprop_here_gpu$g.log 2>&1 < /dev/null &
      echo "GPU $g: pid $!, log $R/t3_logs/labelprop_here_gpu$g.log"
    done
    ;;
  labelprop)
    for v in v2_noE2 v2_noE3 v2_noexcl; do
      dispatch $v --labelprop $v --slurm $SHARDS
    done
    ;;
  hdbscan)
    # outputs are named <cluster>_mcs<a>_ms<b> (rescore_round.variant_name), defaults spelled out
    for c in global_soft global_excl_hard global_excl; do
      dispatch $c --cluster $c --track_device cpu --slurm $CPU_SHARDS
    done
    for a in "${MCS[@]}"; do for b in "${MS[@]}"; do
      if [ "$a" = 0.5 ] && [ "$b" = 0.02 ]; then reps=(--slurm $CPU_SHARDS)
      else reps=(--no_replicates --slurm $((CPU_SHARDS / 2))); fi
      dispatch global_mcs${a}_ms${b} --cluster global --track_device cpu "${reps[@]}" \
          --hdbscan_min_cluster_size $a --hdbscan_min_samples $b
    done; done
    ;;
  excl_tuned)
    # the rest of the 2 x 2 at each recording's tuned setting; one dispatcher per (cell, setting) for the recordings
    # that chose it (the default setting already ran in the hdbscan stage)
    python barlow_track/scripts/paper_scripts/revision/t3_report.py --out $R/t3_table.tsv > /dev/null
    python - "$R/t3_table.tsv" <<'EOF' | while read -r a b labs; do
import sys, re, collections
import pandas as pd
t = pd.read_csv(sys.argv[1], sep='\t', index_col=0)
by = collections.defaultdict(list)
for lab, s in t.get('hdbscan_tuned', pd.Series(dtype=str)).dropna().items():
    m = re.match(r'_mcs([\d.]+)_ms([\d.]+)$', s)
    if m and m.groups() != ('0.5', '0.02'):
        by[m.groups()].append(lab)
for (a, b), labs in by.items():
    print(a, b, ' '.join(labs))
EOF
      for c in global_soft global_excl_hard global_excl; do
        dispatch ${c}_mcs${a}_ms${b} --cluster $c --track_device cpu --no_replicates --slurm 2 \
            --hdbscan_min_cluster_size $a --hdbscan_min_samples $b --labs $labs
      done
    done
    ;;
  status)
    for f in $R/exp_results_v2.jsonl $R/exp_results_v2_no*.jsonl $R/exp_results_global*.jsonl; do
      [ -f $f ] || continue
      printf '%-55s %3s trial-59 cells\n' $(basename $f) $({ grep -o '"tag": "[^"]*trial59[^"]*"' $f || true; } | sort -u | wc -l)
    done
    squeue -u "$USER" -h -o '%i %j %T' | grep ' t3_' || echo "no t3 dispatchers in the queue"
    ;;
  report)
    python barlow_track/scripts/paper_scripts/revision/t3_report.py --out $R/t3_table.tsv
    ;;
  *)
    echo "usage: $0 smoke|labelprop_here|labelprop|hdbscan|excl_tuned|status|report"; exit 1 ;;
esac
