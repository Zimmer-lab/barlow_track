#!/bin/bash
# round_posonly_flavell trials 8 and 9 (RBF fix on, cutoff_nn 0, elastic 0 / 1) scored 0.870 / 0.881
# against the project GT, which is the Flavell lab's tracker output (issue i261028-3). Rescore
# the same checkpoints on the project segmentation against the NWB GT, CPU only (embedding and
# tracking), records local only (~/.cache/barlow_scratch/rescore_posonly_flavell/), nothing
# written to /lisc.
cd /home/charles/Current_work/repos/barlow_track
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_posonly_flavell
P=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data
O=$HOME/.cache/barlow_scratch/rescore_posonly_flavell
mkdir -p $O
for n in 8 9; do
  /home/charles/anaconda3/envs/wbfm/bin/python -u barlow_track/scripts/eval_accuracy.py --lab flavell \
      --source project --project $P/projects/2025_07_01/project_config.yaml \
      --gt $P/images_for_charlie/flavell_data.nwb --mode trained --weights $B/trial_$n/resnet50.pth \
      --device cpu --track_device cpu --emb_dir $O/emb_trial$n \
      --tag round_posonly_flavell_trial${n}_flavell_nwbgt --results_jsonl $O/exp_results.jsonl &
done
wait
echo "rescore_posonly_flavell done"
