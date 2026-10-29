#!/bin/bash
# long10_drop0 (0.932 full video) at hidden 512 (batch 12: +0.1 proxy over 128 at epoch 14).
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
$P long14_h512_drop0 visnet_cutoff_nn=0 num_frames=600 visnet_hidden=512 global_augment.elastic_scale=0.0 \
    position_augment.dropout_p=0.0 epochs=60 --eval
echo "long14 done"
