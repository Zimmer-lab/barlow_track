#!/bin/bash
# Batches 15 and 16 (batch15.sh, batch16.sh: same cells) as one queue on 4 parallel GPU
# workers, all seed-43 cells first, so both batches report within ~3 h instead of ~4.5 h
# back to back. Each fast cell needs ~2 GB of GPU memory; starts once batch 14 is done.
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
until grep -q "batch14 done" $D/batch14.log 2>/dev/null; do sleep 30; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=300 visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
cells() {
  for s in 43 44; do
    echo "b15_jitauto_s$s position_augment.jitter_std=auto"
    echo "b16_pair0.5_s$s position_augment.pair_p=0.5"
    echo "b15_axscale0.1_s$s position_augment.axis_scale_jitter=0.1"
    echo "b16_pair1.0_s$s position_augment.pair_p=1.0"
    echo "b15_bend0.3_s$s position_augment.bend_max=0.3"
    echo "b15_cut0.5_s$s position_augment.cut_p=0.5 position_augment.cut_max=0.15"
    echo "b16_pair0.5_lag3_s$s position_augment.pair_p=0.5 position_augment.pair_lag=3"
    echo "b15_jit0_s$s position_augment.jitter_std=0.0"
    echo "b16_pair0.5_r1.5_s$s position_augment.pair_p=0.5 position_augment.pair_resid=1.5"
    echo "b15_gscale0_s$s global_augment.scale_jitter=0.0"
    echo "b15_gscale0.2_s$s global_augment.scale_jitter=0.2"
  done
}
run_cell() {
  name=$1; shift
  seed=${name##*_s}
  $P $name $B "$@" seed=$seed > $D/$name.cell.log 2>&1
  echo "$name exit $? $(grep -o 'epoch  14  proxy [0-9.]*' $D/$name.cell.log | tail -1)"
}
export -f run_cell; export P B D
cells | xargs -P 4 -L 1 bash -c 'run_cell "$@"' _
echo "queue15_16 done"
