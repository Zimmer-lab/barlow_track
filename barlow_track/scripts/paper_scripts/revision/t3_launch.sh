#!/usr/bin/env bash
# TEST_PLAN T3: per-time exclusivity ablation and a fair HDBSCAN, on trial 59's cached embeddings (no training,
# no embedding: every cell re-tracks round_shared_v1/emb_cache). Run on the login node:
#
#   cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
#   X=barlow_track/scripts/paper_scripts/revision/t3_launch.sh
#   bash $X smoke         # one cell (samuel 153, trial 59 rep 0) of each new variant, in this process; ~10-20 min
#   bash $X labelprop     # v2_noE2, v2_noE3, v2_noexcl on all 36 trial-59 cells (12 recordings + 24 replicates)
#   bash $X hdbscan       # HDBSCAN at the defaults on all 36 (as global_mcs0.5_ms0.02: new records with duplicate
#                         # statistics; the older exp_results_global.jsonl has none), global_excl on all 36, and the
#                         # other 8 grid settings on the 12 rep-0 cells
#   bash $X excl_tuned    # after hdbscan + report: global_excl at each recording's GT-free tuned setting
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
    run python $P --round_dir $R --trials 59 --labs samuel --no_replicates --cluster global_excl 2>&1 | tail -3
    ;;
  labelprop)
    for v in v2_noE2 v2_noE3 v2_noexcl; do
      dispatch $v --labelprop $v --slurm $SHARDS
    done
    ;;
  hdbscan)
    dispatch global_excl --cluster global_excl --slurm $SHARDS
    for a in "${MCS[@]}"; do for b in "${MS[@]}"; do
      if [ "$a" = 0.5 ] && [ "$b" = 0.02 ]; then reps=(--slurm $SHARDS); else reps=(--no_replicates --slurm 2); fi
      dispatch global_mcs${a}_ms${b} --cluster global "${reps[@]}" \
          --hdbscan_min_cluster_size $a --hdbscan_min_samples $b
    done; done
    ;;
  excl_tuned)
    # one dispatcher per distinct tuned setting, restricted to the recordings that chose it
    python barlow_track/scripts/paper_scripts/revision/t3_report.py --out $R/t3_table.tsv > /dev/null
    python - "$R/t3_table.tsv" <<'EOF' | while read -r a b labs; do
import sys, re, collections
import pandas as pd
t = pd.read_csv(sys.argv[1], sep='\t', index_col=0)
by = collections.defaultdict(list)
for lab, s in t['hdbscan_tuned_setting'].dropna().items():
    m = re.match(r'global_mcs([\d.]+)_ms([\d.]+)$', s)
    if m and m.groups() != ('0.5', '0.02'):  # the default setting's global_excl runs in the hdbscan stage
        by[m.groups()].append(lab)
for (a, b), labs in by.items():
    print(a, b, ' '.join(labs))
EOF
      dispatch excl_mcs${a}_ms${b} --cluster global_excl --no_replicates --slurm 1 \
          --hdbscan_min_cluster_size $a --hdbscan_min_samples $b --labs $labs
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
