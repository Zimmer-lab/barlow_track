#!/bin/bash
# Batch 23: follow-up to batch 22 (visnet_z_scale). Seed 43 proxies rose monotonically with
# z_scale on Zimmer (1 / 1.55 / 3 / 4.6: .762 / .778 / .823 / .865) and Samuel (1 / 2.9 / 5:
# .548 / .848 / .865), and were flat on Flavell (1 / 2: .785 / .775). Still rising at the top
# of the range, so:
#  1. Larger values: Zimmer 6 / 8, Samuel 8 / 12, seed 43, epoch-14 proxy.
#  2. Full-video accuracy (--eval) at z_scale 1 vs the batch-22 best (Zimmer 4.6, Samuel 5),
#     seed 43: the proxy misjudged cutoff_nn on Samuel, so check the real metric.
# Same base as batch 22. Starts after batch 22; one worker (batch 20 / 21 share the GPU).
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
until grep -q "batch22 done" $D/batch22.log 2>/dev/null; do sleep 60; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="num_frames=300 visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0 position_augment.jitter_std=auto"
cells() {
  for zs in 6.0 8.0; do echo "zimmer - b23_zimmer_zs${zs}_s43 visnet_z_scale=$zs"; done
  for zs in 8.0 12.0; do echo "samuel - b23_samuel_zs${zs}_s43 visnet_z_scale=$zs"; done
  for zs in 1.0 4.6; do echo "zimmer --eval b23_zimmer_zs${zs}_eval_s43 visnet_z_scale=$zs"; done
  for zs in 1.0 5.0; do echo "samuel --eval b23_samuel_zs${zs}_eval_s43 visnet_z_scale=$zs"; done
}
run_cell() {
  lab=$1; flag=$2; name=$3; shift 3  # flag: --eval or - (argparse needs options before the overrides)
  [ -f $D/$name/resnet50.pth ] && { echo "$name already done"; return; }
  [ "$flag" = "-" ] && flag=""
  seed=${name##*_s}
  $P --lab $lab $flag $name $B "$@" seed=$seed > $D/$name.cell.log 2>&1
  echo "$name exit $? $(grep -o 'epoch  14  proxy [0-9.]*' $D/$name.cell.log | tail -1) $(grep -o '"accuracy": [0-9.]*' $D/$name.cell.log | tail -1)"
}
export -f run_cell; export P B D
cells | xargs -P ${NWORKERS:-1} -L 1 bash -c 'run_cell "$@"' _
echo "batch23 done"
