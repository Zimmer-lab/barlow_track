#!/bin/bash
# Build the new wbfm projects for the wbfm-tracker baseline (no tracker steps), one per NWB
# recording with masks; the user runs the tracker on them with snakemake on the cluster.
# Samuel 153 is built by its own local run (which also runs the steps on CPU as a test).
# Leifer is left out: its NWB has no segmentation (re-export sub-project).
cd /home/charles/Current_work/repos/barlow_track
export CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=/tmp/claude-1000/numba MPLCONFIGDIR=/tmp/claude-1000/mpl
for ds in samuel_137 samuel_184 flavell flavell_2022031603 flavell_2022032402 flavell_2022033001 flavell_2022033002; do
  echo "== $ds $(date)"
  /home/charles/anaconda3/envs/wbfm/bin/python -u barlow_track/scripts/scratch/wbfm_tracker_baseline/run_wbfm_tracker.py \
    --dataset $ds --create_only || echo "$ds FAILED"
done
echo "create_all done $(date)"
