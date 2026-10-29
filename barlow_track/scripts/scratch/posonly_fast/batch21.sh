#!/bin/bash
# Batch 21: lambd_obj in image networks, training speed vs final accuracy and data efficiency.
# Batch 20 (Zimmer, image-only, 100 frames, seed 43): lambd_obj 0.67 0.617 vs 0 0.482 at
# epoch 14, reached in ~4x fewer epochs. The user (2026-10-29): in the preprint either loss
# trained well (final accuracy), dynamics and data efficiency unchecked; try Samuel too.
#  1. Samuel 153, image-only, lambd_obj 0 / 0.67 x 30 / 100 frames, 30 epochs (does 0 catch up?),
#     seeds 43 / 44.
#  2. Zimmer, image-only, lambd_obj 0 / 0.67, 100 frames, 40 epochs, seed 43, full-video eval.
# Same recipe as batch 20 (round_fused_fix_zimmer config, fusion visual_only). Starts after the
# batch 20 retry; one worker (crop training uses most of the 12 GB GPU).
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
until grep -q "batch20 done" $D/batch20_retry.log 2>/dev/null; do sleep 60; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
BASE=barlow_track/scripts/scratch/round_fused_fix_zimmer/train_config.yaml
SAMUEL=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data/projects/153/project_config.yaml
B="fusion=visual_only wandb_name=null"
cells() {
  for s in 43 44; do
    for f in 30 100; do
      for l in 0.0 0.67; do
        echo "b21_samuel_f${f}_lobj${l}_s$s --lab=samuel project_path=$SAMUEL num_frames=$f epochs=30 lambd_obj=$l"
      done
    done
  done
  for l in 0.0 0.67; do
    echo "b21_zimmer_e40_lobj${l}_s43 --eval num_frames=100 epochs=40 lambd_obj=$l"
  done
}
run_cell() {
  name=$1; flag=$2; shift 2  # flag: --lab=... or --eval (argparse needs options before the overrides)
  [ -f $D/$name/resnet50.pth ] && { echo "$name already done"; return; }
  seed=${name##*_s}
  $P --base $BASE $flag $name $B "$@" seed=$seed > $D/$name.cell.log 2>&1
  echo "$name exit $? $(grep -o 'epoch  [0-9]*  proxy [0-9.]*' $D/$name.cell.log | tail -1)"
}
export -f run_cell; export P B D BASE
cells | xargs -P ${NWORKERS:-1} -L 1 bash -c 'run_cell "$@"' _
echo "batch21 done"
