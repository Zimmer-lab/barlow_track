#!/bin/bash
# Batch 7: seeds 44 and 45 for the three configs batch 5 ranked, so each has three seeds
# (b6_seed44 showed ~0.04 seed noise in the epoch-14 proxy, larger than the gaps): no vecnorm
# at cutoff 0.35 (b4_frames300_h128), vecnorm at 0.35 (b5_vecnorm), vecnorm at 0.5
# (b5_vecnorm_cutoff0.5, b6_seed44). Two streams in parallel; waits for batch 6 ($1).
cd /home/charles/Current_work/repos/barlow_track
while kill -0 "$1" 2>/dev/null; do sleep 60; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 global_augment.elastic_scale=1.0 num_frames=300 visnet_hidden=128"
VN="visnet_vecnorm=max_min"
( $P b7_novn_seed44 $B seed=44; $P b7_novn_seed45 $B seed=45; $P b7_vn_c0.5_seed45 $B $VN visnet_cutoff=0.5 seed=45 ) &
( $P b7_vn_seed44 $B $VN seed=44; $P b7_vn_seed45 $B $VN seed=45 ) &
wait
echo "batch7 done"
