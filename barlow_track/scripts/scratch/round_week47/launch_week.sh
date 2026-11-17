#!/usr/bin/env bash
# Cluster runs for the week of 2026-11-16. Run on the login node:
#
#   cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
#   W=barlow_track/scripts/scratch/round_week47/launch_week.sh
#   bash $W status         # per sensitivity round: recorded / expected, dispatcher alive, trial jobs in the queue
#   bash $W resume         # relaunch the dispatcher of every unfinished round whose dispatcher is gone
#   bash $W sens_rescore   # v2 + saved tracks for every sensitivity round with unrescored cells (one GPU job each)
#   bash $W sens_rescore_here  # the same on this machine's GPUs (login node; NGPU=2), no Slurm jobs
#   bash $W sens_proxy     # cross-network proxy pooled over round_sens_* and round_shared_v1 (CPU job)
#   bash $W transfer       # trial 59's network of each recording scored on the 11 others (132 cells, 8 GPU jobs)
#   bash $W transfer_report
#   DRY=1 bash $W <stage>  # print the sbatch commands instead of submitting
#
# resume: a dispatcher dies from the 7-day limit of sbatch_dispatcher.sbatch (the first ones launched
# 2026-11-10 ~15:35, so ~2026-11-17 15:35; Zimmer ones ~2026-11-19) or from a Slurm hiccup (oat_zimmer_1123,
# 2026-11-13). Its trial jobs keep running and write their own records. The relaunch passes
# --skip_completed_grid_jobs, which leaves out every grid job whose training finished (a trial trained but not
# scored is only re-scored), but a trial still TRAINING counts as not done and would be trained again, so a round
# with trial jobs still in the queue is left alone until they end (FORCE=1 relaunches anyway). Rerun resume
# whenever: it only acts on rounds that need it. Dispatchers that are alive are never touched.
#
# sens_rescore can run while rounds are still going: each rescore covers the cells recorded when it starts, and a
# rerun later fills the rest (finished cells skip). A round whose rescore job is still queued or running is skipped.
set -euo pipefail
B=${B:-/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files}
export CODE=${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}
R=$B/round_shared_v1
P=barlow_track/scripts/scratch/gt_free_proxy
S=barlow_track/scripts/scratch/round_sens_src
T=barlow_track/scripts/scratch/transfer/transfer_matrix.py
cd "$CODE"
mkdir -p $R/week47_logs

# jobs per (round, recording), as in launch_sensitivity.sh
declare -A EXPECTED=([loss_volume]=54 [oat]=31 [matched_steps]=1 [attention2]=1 [crop_depth]=2
                     [only_affine]=1 [only_elastic]=1 [only_noise]=1 [aug_none]=1)

run() { if [ -n "${DRY:-}" ]; then echo "DRY: $*"; else "$@"; fi; }
queue() { squeue -u "$USER" -h -o '%i %j'; }
ntags() { if [ -f "$1" ]; then { grep -o '"tag": "[^"]*"' "$1" || true; } | sort -u | wc -l; else echo 0; fi; }
recorded() { ntags $1/exp_results.jsonl; }
rescored() { ntags $1/exp_results_v2.jsonl; }
# job name present in the queue listing $2 (no pipe into grep -q: under pipefail its early exit reads as failure)
in_queue() { local names; names=$(awk '{print $2}' <<< "$2"); grep -qx "$1" <<< "$names"; }
# trial jobs this round's dispatchers submitted (submitit leaves <jobid>_submitted.pkl) that are still in the queue
inflight() { comm -12 <(ls $1 | sed -n 's/^\([0-9]*\)_submitted\.pkl$/\1/p' | sort -u) <(awk '{print $1}' <<< "$2" | sort -u) | wc -l; }
split_round() { local rest=${1#round_sens_}; L=$(echo $rest | awk -F_ '{print $(NF-1)"_"$NF}'); RN=${rest%_$L}; }

status() {
  local q; q=$(queue)
  printf '%-48s %9s %10s %7s %8s\n' round recorded dispatcher trials rescored
  for D in $B/round_sens_*/; do
    D=${D%/}; local name; name=$(basename $D); split_round $name
    local alive=no; in_queue "sens_${RN}_$L" "$q" && alive=yes
    printf '%-48s %4s/%-4s %10s %7s %8s\n' $name $(recorded $D) ${EXPECTED[$RN]} $alive $(inflight $D "$q") $(rescored $D)
  done
  for f in exp_results_transfer_v2.jsonl; do
    [ -f $R/$f ] && echo "$f: $(grep -o '"tag": "[^"]*"' $R/$f | sort -u | wc -l) of 132" || echo "$f: not yet"
  done
}

resume() {
  local q; q=$(queue)
  for D in $B/round_sens_*/; do
    D=${D%/}; local name; name=$(basename $D); split_round $name
    local n; n=$(recorded $D)
    [ "$n" -ge "${EXPECTED[$RN]}" ] && continue
    in_queue "sens_${RN}_$L" "$q" && continue  # dispatcher alive
    local k; k=$(inflight $D "$q")
    if [ "$k" -gt 0 ] && [ -z "${FORCE:-}" ]; then
      echo "$name: dispatcher gone, $n/${EXPECTED[$RN]} recorded, $k trial job(s) still queued or running: wait (FORCE=1 relaunches now and retrains those)"
      continue
    fi
    echo "$name: dispatcher gone, $n/${EXPECTED[$RN]} recorded: relaunching with --skip_completed_grid_jobs"
    run env LABELS=$L SKIP=1 NUM_PARALLEL=${NUM_PARALLEL:-2} bash $S/launch_sensitivity.sh launch $RN
  done
}

sens_rescore() {
  local q; q=$(queue)
  for D in $B/round_sens_*/; do
    D=${D%/}; local name; name=$(basename $D)
    [ "$(recorded $D)" -gt "$(rescored $D)" ] || continue
    in_queue "rescore_$name" "$q" && { echo "$name: rescore still queued or running"; continue; }
    run env BARLOW_DISPATCHER=$P/rescore_round.py sbatch --job-name=rescore_$name --output=$D/rescore_v2_%j.log \
        barlow_track/scripts/sbatch_dispatcher.sbatch --round_dir $D --labelprop v2 --slurm 1
  done
}

# The same rescore without Slurm, on this machine's GPUs (the login node's two): NGPU background processes, each
# taking shard g of NGPU of every round with unrescored cells, round after round. nohup, so logging out does not
# stop them; logs in $R/week47_logs/rescore_here_gpu<g>.log. Rounds with a Slurm rescore queued are left out.
sens_rescore_here() {
  local q; q=$(queue) ; local ngpu=${NGPU:-2}
  if pgrep -u "$USER" -f 'rescore_round.py.*--shard' > /dev/null; then
    echo "rescore_round.py shards already running here:"; pgrep -u "$USER" -af 'rescore_round.py.*--shard'; return 1
  fi
  local todo=()
  for D in $B/round_sens_*/; do
    D=${D%/}; local name; name=$(basename $D)
    [ "$(recorded $D)" -gt "$(rescored $D)" ] || continue
    in_queue "rescore_$name" "$q" && { echo "$name: Slurm rescore queued or running, left out"; continue; }
    todo+=("$D")
  done
  echo "${#todo[@]} rounds with unrescored cells, $ngpu GPU process(es)"
  [ ${#todo[@]} -gt 0 ] || return 0
  for ((g = 0; g < ngpu; g++)); do
    local cmd="for D in ${todo[*]}; do python -u $P/rescore_round.py --round_dir \$D --labelprop v2 --shard $g --num_shards $ngpu; done"
    if [ -n "${DRY:-}" ]; then echo "DRY: CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 nohup bash -c '$cmd'"; continue; fi
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup bash -c "$cmd" \
        > $R/week47_logs/rescore_here_gpu$g.log 2>&1 < /dev/null &
    echo "GPU $g: pid $!, log $R/week47_logs/rescore_here_gpu$g.log"
  done
}

sens_proxy() {
  run env BARLOW_DISPATCHER=$P/cross_network_proxy.py sbatch --job-name=proxy_sens_v2 --mem=64G \
      --output=$R/week47_logs/proxy_sens_v2_%j.log barlow_track/scripts/sbatch_dispatcher.sbatch \
      --tracks_dir "$B/round_sens_*/tracks_v2" "$R/tracks_v2" \
      --results "$B/round_sens_*/exp_results_v2.jsonl" "$R/exp_results_v2.jsonl" \
      --out $B/cross_network_proxy_sens_v2.jsonl
}

transfer() {
  run env BARLOW_DISPATCHER=$T sbatch --job-name=transfer_t59 --output=$R/week47_logs/transfer_t59_%j.log \
      barlow_track/scripts/sbatch_dispatcher.sbatch --round_dir $R --trial 59 --labelprop v2 --slurm 8
}

transfer_report() {
  python $T --round_dir $R --trial 59 --labelprop v2 --report
}

case ${1:?status | resume | sens_rescore | sens_rescore_here | sens_proxy | transfer | transfer_report} in
  status|resume|sens_rescore|sens_rescore_here|sens_proxy|transfer|transfer_report) $1 ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
