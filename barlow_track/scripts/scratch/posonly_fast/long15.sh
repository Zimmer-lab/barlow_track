#!/bin/bash
# visnet_cutoff_nn 0 vs 6 on the full-video measure. The epoch-14 proxy puts 6 ~0.05 below
# the fixed cutoff on both seeds (batch 17), round_posonly_zimmer's full video 0.04 above
# it (0.868 vs 0.825, its older base). Here at round_posonly2's base with its likely winner
# settings (width 512, 600 frames, 60 epochs, elastic 0, dropout 0.02, jitter auto) and the
# full-video zimmer_1128 eval, so the answer comes a day before the cluster round's.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="num_frames=600 visnet_hidden=512 global_augment.elastic_scale=0.0 epochs=60 position_augment.dropout_p=0.02 position_augment.jitter_std=auto"
$P --eval long15_cnn0 $B visnet_cutoff_nn=0 &
$P --eval long15_cnn6 $B visnet_cutoff_nn=6 &
wait
echo "long15 done"
