#!/bin/bash
# The zimmer_1128 cells of run.sh OOM'd in GPU label propagation next to a training job
# (r261028-3159902); same cells, tracking on the CPU. Same jsonl and caches as run.sh.
set -u
PY=/home/charles/anaconda3/envs/wbfm/bin/python
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
OUT=${OUT:-$HOME/.cache/barlow_scratch/runs/backbone_rescore_off_flavell}
export NUMBA_CACHE_DIR=$HOME/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=$HOME/.cache/barlow_scratch/mpl
for stage in backbone auto; do
    $PY -u barlow_track/scripts/multiproject_scripts/run_trials_on_all_ground_truth.py \
        --trial_parent_dir "$B/round_visnet_full" --trials 27 --labs zimmer_1128 \
        --descriptor_stage "$stage" --device cuda --track_device cpu --sequential --no_fail_fast \
        --results_jsonl "$OUT/round_visnet_full_${stage}.jsonl" --emb_dir "$OUT/emb_round_visnet_full_${stage}"
done
echo "done"
