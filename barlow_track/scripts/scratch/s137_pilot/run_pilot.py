"""One-change-at-a-time pilot on Samuel 137 from round_bayes_wk2's best trial (63, 0.931).

Each variant copies trial 63's train_config.yaml, changes one thing, trains locally and is
scored exactly as the in-sweep objective scores it (optimize_hyperparameters.resolve_objective
with the round's template: source nwb, GT objects, 137.nwb, labelprop 25 seeds, track on CPU).

  base         trial 63 as is (same seed): does a local retrain reproduce 0.931?
  base_seed2   other seed: seed noise for everything below
  lambd0       lambd_obj 0: feature-space term only, no object term (the sweep never sampled < 0.1)
  lambd1       lambd_obj 1: object term only
  reaug        reaugment_each_epoch true (i261027-1, a261027-1)
  *_seed2/3    the same at another seed (seed 2 variants pair with base_seed2)

    python barlow_track/scripts/scratch/s137_pilot/run_pilot.py --variants base lambd0 reaug

Other rounds: --round round_bayes_wk2_leifer_reexport --trial 57 --name leifer_pilot (outputs and
record tags then use that name instead of s137_pilot). --project_path trains on another project
(e.g. a resegmentation of the same recording); --train_only skips scoring (no GT yet).
"""
import argparse
import copy
import os
import subprocess
import sys

import yaml

ROUNDS = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
ROUND = os.path.join(ROUNDS, 'round_bayes_wk2_samuel_137')
TRIAL = os.path.join(ROUND, 'trial_63')
NAME = 's137_pilot'
OUT = os.path.expanduser(f'~/.cache/barlow_scratch/train/{NAME}')
VARIANTS = {
    'base': {},
    'base_seed2': {'seed': 2},
    'lambd0': {'lambd_obj': 0.0},
    'lambd1': {'lambd_obj': 1.0},
    'reaug': {'reaugment_each_epoch': True},
    # Seed noise turned out large (base 0.920 vs base_seed2 0.884), so more seeds, paired by seed
    'base_seed3': {'seed': 3},
    'lambd0_seed2': {'seed': 2, 'lambd_obj': 0.0},
    'lambd1_seed2': {'seed': 2, 'lambd_obj': 1.0},
    'reaug_seed2': {'seed': 2, 'reaugment_each_epoch': True},
    'lambd0_seed3': {'seed': 3, 'lambd_obj': 0.0},
    'lambd1_seed3': {'seed': 3, 'lambd_obj': 1.0},
    'reaug_seed3': {'seed': 3, 'reaugment_each_epoch': True},
    # Leifer StarDist resegmentation grid: lambd_obj x num_frames (Leifer wk2's most important parameter)
    'nf400': {'num_frames': 400},
    'lambd0_nf400': {'lambd_obj': 0.0, 'num_frames': 400},
    'lambd1_nf400': {'lambd_obj': 1.0, 'num_frames': 400},
}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def run_variant(name, project_path=None, train_only=False):
    d = os.path.join(OUT, name)
    os.makedirs(os.path.join(d, 'log'), exist_ok=True)
    os.makedirs(os.path.join(d, 'checkpoints'), exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(TRIAL, 'train_config.yaml')))
    cfg = copy.deepcopy(cfg)
    # Local runs: 4 DataLoader workers (data-bound at 0; step-0 loss bit-identical and val loss
    # within run-to-run noise, notebook 2026-10-30, aab23eb). Every variant gets it, so they compare.
    cfg['num_workers'] = 4
    # Recorded runs (2026-11-02 to 11-05) trained before reaugment_each_epoch defaulted to true;
    # pin the old default so every non-reaug variant reproduces
    cfg.setdefault('reaugment_each_epoch', False)
    cfg.update(VARIANTS[name])
    if project_path:
        cfg['project_path'] = project_path
    cfg['project_dir'] = d
    cfg['wandb_name'] = f'{NAME}_{name}'
    cfg_path = os.path.join(d, 'train_config.yaml')
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    weights = os.path.join(d, 'resnet50.pth')
    if not os.path.exists(weights):
        print(f"[{name}] training", flush=True)
        subprocess.run([sys.executable, '-u', 'barlow_track/scripts/train_barlow_clusterer.py', '-p', cfg_path],
                       cwd=REPO, check=True)
    if train_only:
        print(f"[{name}] trained (train_only: not scored)", flush=True)
        return
    from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint
    from barlow_track.scripts.optimize_hyperparameters import resolve_objective
    tmpl = yaml.safe_load(open(os.path.join(ROUND, 'hyperparameter_search_template.yaml')))
    objective, label, eval_kwargs = resolve_objective(tmpl)
    kwargs = {k: v for k, v in eval_kwargs.items() if k not in ('gt', 'snapshots')}
    # The round scored in CPU-only jobs (objective_cpu_job), so embedding ran on the CPU with
    # its crop window (record crop_window 'cpu'); the GPU's default 'pad' crops differ at edges
    kwargs['device'] = 'cpu'
    print(f"[{name}] scoring ({objective}, {label}): {kwargs}", flush=True)
    rec = evaluate_trained_checkpoint(weights=weights, project=cfg['project_path'],
                                      gt=eval_kwargs.get('gt') or cfg['project_path'],
                                      tag=f'{NAME}_{name}', results_jsonl=os.path.join(OUT, 'results.jsonl'),
                                      emb_dir=os.path.join(d, 'emb'), extra_record={'variant': name}, **kwargs)
    print(f"[{name}] accuracy {rec['accuracy']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--round', default=None, help='round dir name under config_files (default Samuel 137 wk2)')
    ap.add_argument('--trial', type=int, default=None)
    ap.add_argument('--name', default=None, help='output dir and tag prefix (default s137_pilot)')
    ap.add_argument('--project_path', default=None, help="train on this project instead of the trial's")
    ap.add_argument('--train_only', action='store_true')
    args = ap.parse_args()
    global ROUND, TRIAL, NAME, OUT
    if args.round:
        ROUND = os.path.join(ROUNDS, args.round)
        TRIAL = os.path.join(ROUND, f'trial_{args.trial}')
        assert args.name, '--name is required with --round, so outputs do not mix with the Samuel pilot'
    if args.name:
        NAME = args.name
        OUT = os.path.expanduser(f'~/.cache/barlow_scratch/train/{NAME}')
    for v in args.variants:
        try:
            run_variant(v, args.project_path, args.train_only)
        except Exception as e:  # keep going: one failed variant should not cost the rest
            print(f"[{v}] FAILED: {e!r}", flush=True)


if __name__ == '__main__':
    main()
