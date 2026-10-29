#!/bin/bash
# Long run of batch 5's best with batch 4's frames 600, 60 epochs, then full-video tracking on
# zimmer_1128. Compare with b2_nn0_el1.0_ep60 (100 frames, no vecnorm, cutoff 0.35): 0.843.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
$P long6_f600_h128_vn_c0.5_ep60 visnet_cutoff_nn=0 global_augment.elastic_scale=1.0 num_frames=600 \
    visnet_hidden=128 visnet_vecnorm=max_min visnet_cutoff=0.5 epochs=60 --eval
echo "long6 done"
