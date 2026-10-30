"""Time one short training (and optionally its objective eval) of a round_wk1 config.

Takes round_wk1_<label>/train_config.yaml, shortens it (epochs, num_frames), turns on
timing_log, and trains on the chosen device with a fixed number of CPU threads (pinned
with taskset, as a Slurm allocation would be). Per-epoch phase times land in the
training log as {"timing_epoch": ...} lines; --eval then scores the checkpoint with the
round's objective settings (embedding on --eval_device, label propagation on the CPU) and
the eval log has the embed / track times. For deciding which jobs can go to CPU nodes.

    python barlow_track/scripts/scratch/timing_probe/timing_probe.py --label samuel \
        --variant add --device cpu --threads 16 [--frames 20 --epochs 1] [--eval]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROUNDS = REPO / 'barlow_track/scripts/scratch'
OUT = Path(os.environ.get('TIMING_PROBE_OUT', Path.home() / '.cache/barlow_scratch/train/timing_probe'))
PY = sys.executable

# Model variants: the round_wk1 grid's fusions, and a keypoints-only position network
VARIANTS = {
    'add': dict(fusion='add'),
    'visual_only': dict(fusion='visual_only'),
    'posonly': dict(fusion='position_only', keypoints_only=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--variant', default='add', choices=list(VARIANTS))
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--first_core', type=int, default=0, help='taskset cores first_core..+threads-1')
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--eval', action='store_true', help="score the checkpoint as the round's objective")
    ap.add_argument('--eval_device', default='cpu', choices=['cpu', 'cuda'])
    args = ap.parse_args()

    name = f"{args.label}_{args.variant}_{args.device}{args.threads}_f{args.frames}e{args.epochs}"
    run_dir = OUT / name
    (run_dir / 'log').mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    rnd = ROUNDS / f'round_wk1_{args.label}'
    cfg = yaml.safe_load((rnd / 'train_config.yaml').read_text())
    cfg.update(VARIANTS[args.variant], epochs=args.epochs, num_frames=args.frames, timing_log=True,
               project_dir=str(run_dir), wandb_name=None)
    (run_dir / 'train_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))

    threads = str(args.threads)
    env = dict(os.environ, WANDB_MODE='offline', OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               NUMBA_NUM_THREADS=threads,
               NUMBA_CACHE_DIR=str(Path.home() / '.cache/barlow_scratch/numba_cache'),
               MPLCONFIGDIR=str(Path.home() / '.cache/barlow_scratch/mpl'))
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    cores = f"{args.first_core}-{args.first_core + args.threads - 1}"
    pin = ['taskset', '-c', cores]
    with open(run_dir / 'train.log', 'w') as log:
        subprocess.run(pin + [PY, '-u', 'barlow_track/scripts/train_barlow_clusterer.py', '-p',
                              str(run_dir / 'train_config.yaml')],
                       cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not args.eval:
        return
    eval_env = dict(env)
    if args.eval_device == 'cuda':
        eval_env.pop('CUDA_VISIBLE_DEVICES', None)
    code = ("import sys, yaml; "
            "from barlow_track.scripts.optimize_hyperparameters import resolve_objective; "
            "from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint; "
            "tpl = yaml.safe_load(open(sys.argv[1])); cfg = yaml.safe_load(open(sys.argv[2])); "
            "_, label, kw = resolve_objective(tpl); gt = kw.pop('gt') or cfg['project_path']; "
            "rec = evaluate_trained_checkpoint(weights=sys.argv[3] + '/resnet50.pth', "
            "project=cfg['project_path'], gt=gt, tag='timing_probe_' + label, "
            "results_jsonl=sys.argv[3] + '/eval.jsonl', emb_dir=sys.argv[3] + '/emb', "
            "device=sys.argv[4], **kw); "
            "print('EVAL accuracy', rec['accuracy'], 'minutes', rec['minutes'], flush=True)")
    with open(run_dir / 'eval.log', 'w') as log:
        subprocess.run(pin + [PY, '-u', '-c', code, str(rnd / 'hyperparameter_search_template.yaml'),
                              str(run_dir / 'train_config.yaml'), str(run_dir), args.eval_device],
                       cwd=REPO, env=eval_env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
