"""Score checkpoints on the Leifer StarDist resegmentation against its click-derived GT.

eval_accuracy --mode trained --source project, project = GT = the StarDist project (its
final_3d_tracks_df is 3-tracking/manual_annotation/gt_tracks_from_clicks_1.5um.h5, built by the
BarlowTrack_cluster_runs session, r261103-1181386), label propagation 25 seeds at seed 0, CPU
embedding and tracking (as the sweep's CPU objective jobs). Scores, in order:

  ref_trial57_nerve   round_bayes_wk2_leifer_reexport trial 57 (trained on the NeRVE project)
  ref_local_nerve     the local Leifer pilot's base retrain of trial 57 (NeRVE project)
  <variant>           each leifer_reseg sweep run, waiting for its resnet50.pth

Already recorded tags are skipped, so it can be restarted. Run with CUDA_VISIBLE_DEVICES=''
while the GPU trains (load_barlow_model otherwise puts the checkpoint on the full GPU).

    CUDA_VISIBLE_DEVICES= python .../score_reseg.py --variants base lambd0 lambd1 nf400 lambd0_nf400 lambd1_nf400
"""
import argparse
import json
import os
import time

os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')

PROJECT = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline/'
           'leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist/project_config.yaml')
TRAIN = os.path.expanduser('~/.cache/barlow_scratch/train')
OUT = os.path.join(TRAIN, 'leifer_reseg')
REFS = {
    'ref_trial57_nerve': '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/'
                         'round_bayes_wk2_leifer_reexport/trial_57/resnet50.pth',
    'ref_local_nerve': os.path.join(TRAIN, 'leifer_pilot', 'base', 'resnet50.pth'),
}


def done_tags(path):
    if not os.path.exists(path):
        return set()
    tags = set()
    for line in open(path):
        try:
            tags.add(json.loads(line)['tag'])
        except (json.JSONDecodeError, KeyError):
            pass
    return tags


def score(name, weights, results):
    from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint
    tag = f'leifer_reseg_{name}'
    if tag in done_tags(results):
        print(f"[{name}] already scored", flush=True)
        return
    emb_dir = os.path.join(OUT, 'emb', name)
    os.makedirs(emb_dir, exist_ok=True)
    rec = evaluate_trained_checkpoint(weights=weights, project=PROJECT, gt=PROJECT, tag=tag, results_jsonl=results,
                                      emb_dir=emb_dir, source='project', device='cpu', cluster='labelprop',
                                      num_seeds=25, seed=0, extra_record={'variant': name})
    print(f"[{name}] accuracy {rec['accuracy']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='*', default=[])
    ap.add_argument('--no_refs', action='store_true')
    ap.add_argument('--poll', type=int, default=300)
    ap.add_argument('--training_units', nargs='*', default=[],
                    help='systemd user units training the variants: stop waiting once none is active')
    args = ap.parse_args()
    results = os.path.join(OUT, 'scores.jsonl')
    jobs = [] if args.no_refs else list(REFS.items())
    jobs += [(v, os.path.join(OUT, v, 'resnet50.pth')) for v in args.variants]
    pending = list(jobs)
    while pending:
        ready = [(n, w) for n, w in pending if os.path.exists(w)]
        if not ready:
            if args.training_units and not any(
                    os.system(f'systemctl --user is-active -q {u}') == 0 for u in args.training_units):
                print(f"training ended; never trained: {[n for n, _ in pending]}", flush=True)
                return
            time.sleep(args.poll)
            continue
        name, w = ready[0]
        pending.remove(ready[0])
        time.sleep(30)  # let the trainer finish writing
        try:
            score(name, w, results)
        except Exception as e:  # one bad checkpoint should not cost the rest
            print(f"[{name}] FAILED: {e!r}", flush=True)


if __name__ == '__main__':
    main()
