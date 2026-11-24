#!/usr/bin/env bash
# TEST_PLAN T3: per-time exclusivity ablation and a fair HDBSCAN, on trial 59's cached embeddings (no training,
# no embedding: every cell re-tracks round_shared_v1/emb_cache). Run on the login node:
#
#   cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
#   X=barlow_track/scripts/paper_scripts/revision/t3_launch.sh
#   bash $X smoke         # one cell (samuel 153, trial 59 rep 0) of each new variant, in this process; ~20-30 min
#   bash $X labelprop     # v2_noE2, v2_noE3, v2_noexcl on all 36 trial-59 cells (12 recordings + 24 replicates)
#   bash $X hdbscan       # the HDBSCAN 2 x 2 at the defaults on all 36 (global, global_soft, global_excl_hard,
#                         # global_excl), and the other 8 grid settings of plain global on the 12 rep-0 cells
#   bash $X excl_tuned    # after hdbscan: the other three cells of the 2 x 2 at each recording's GT-free tuned setting
#   bash $X status        # cells recorded per variant
#   bash $X report        # t3_report.py -> $R/t3_table.tsv
#   DRY=1 bash $X <stage> # print the commands instead of submitting
#
# Every stage is a rescore_round.py dispatcher (sbatch_dispatcher.sbatch, submitit shard jobs with the sweeps'
# GPU job parameters). GPU jobs even for HDBSCAN: the cached embeddings are keyed by the GPU crop window, and a
# CPU job misses them and re-embeds every cell on CPU (happened with the first HDBSCAN run, 2026-11-11).
# Finished cells skip on a rerun, so any stage can be relaunched after a failure.
set -euo pipefail
B=${B:-/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files}
CODE=${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}
R=$B/round_shared_v1
P=barlow_track/scripts/scratch/gt_free_proxy/rescore_round.py
SHARDS=${SHARDS:-4}
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
      run python $P --round_dir $R --trials 59 --labs samuel --no_replicates --cluster $c 2>&1 | tail -3
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
      dispatch $c --cluster $c --slurm $SHARDS
    done
    for a in "${MCS[@]}"; do for b in "${MS[@]}"; do
      if [ "$a" = 0.5 ] && [ "$b" = 0.02 ]; then reps=(--slurm $SHARDS); else reps=(--no_replicates --slurm 2); fi
      dispatch global_mcs${a}_ms${b} --cluster global "${reps[@]}" \
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
        dispatch ${c}_mcs${a}_ms${b} --cluster $c --no_replicates --slurm 1 \
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
    echo "usage: $0 smoke|labelprop|hdbscan|excl_tuned|status|report"; exit 1 ;;
esac
