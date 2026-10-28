#!/bin/bash
# Does the position branch add anything off Flavell? Rescore the best existing fused ViSNet
# networks at descriptor_stage backbone (image features only) and auto (fused), same code,
# same GT, so the pair is like for like. Networks: round_visnet_full trial27 (best Zimmer
# 0.970 and Leifer 0.972) on zimmer_1128, samuel, leifer; round_bayes_samuel trial0 (best
# Samuel 0.950) on samuel. All have the distance-blind ViSNet (i261028-2), so fused ~
# backbone would say the trained position branch carries little.
# Local results only: record tags carry no stage, so each stage gets its own jsonl and
# embedding cache, and nothing is written next to the cluster rounds.
set -u
PY=/home/charles/anaconda3/envs/wbfm/bin/python
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
OUT=${OUT:-$HOME/.cache/barlow_scratch/runs/backbone_rescore_off_flavell}
mkdir -p "$OUT"
export NUMBA_CACHE_DIR=$HOME/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=$HOME/.cache/barlow_scratch/mpl

run() {  # round trial stage labs...
    local round=$1 trial=$2 stage=$3; shift 3
    $PY -u barlow_track/scripts/multiproject_scripts/run_trials_on_all_ground_truth.py \
        --trial_parent_dir "$B/$round" --trials "$trial" --labs "$@" \
        --descriptor_stage "$stage" --device cuda --sequential --no_fail_fast \
        --results_jsonl "$OUT/${round}_${stage}.jsonl" --emb_dir "$OUT/emb_${round}_${stage}"
}

# Fast cells first, Leifer (hours) last
for stage in backbone auto; do
    run round_bayes_samuel 0 "$stage" samuel
    run round_visnet_full 27 "$stage" zimmer_1128 samuel
done
for stage in backbone auto; do
    run round_visnet_full 27 "$stage" leifer
done
echo "done"
