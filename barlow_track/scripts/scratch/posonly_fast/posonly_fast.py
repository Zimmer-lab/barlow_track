"""Fast local position-only training: one cell from a base config plus key=value overrides.

Trains with keypoints_only (no volumes, no crops), so an epoch takes seconds, and prints the
per-epoch frame-pair matching proxy (val_match_frames held-out frames). For direction
finding only: promising cells go to a full eval (--eval) or the cluster.

Usage (repo root, wbfm env):
  python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py NAME [key=value ...] [--eval]
  e.g. ... rbf_off visnet_rbf_rescale=false
       ... elastic global_augment.elastic_scale=1.0 epochs=30
Values are parsed as YAML (true, 0.5, [1, 2]); dotted keys set nested entries.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ruamel.yaml import YAML

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = Path(os.environ.get('POSONLY_FAST_OUT', Path.home() / '.cache/barlow_scratch/train/posonly_fast'))
PY = sys.executable
EVAL_PROJECT = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/'
                'paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format')


def set_dotted(cfg, key, value):
    *parents, leaf = key.split('.')
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    if leaf not in node and not parents and leaf not in cfg:
        print(f"note: {key} is not in the base config", flush=True)
    node[leaf] = value


def proxy_curve(run_dir):
    rows = []
    stats = run_dir / 'log' / 'stats.json'
    if not stats.exists():
        return rows
    for d in json.loads(stats.read_text()):  # one JSON list of records
        if 'val_loss' in d and 'val_val_match_accuracy_auto' in d:
            rows.append((d['epoch'], d['val_val_match_accuracy_auto'], d.get('val_val_fused_eff_rank'),
                         d['val_loss'], d.get('time')))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('overrides', nargs='*')
    ap.add_argument('--base', default=str(HERE / 'base_zimmer.yaml'))
    ap.add_argument('--eval', action='store_true', help='full-video tracking accuracy on zimmer_1128')
    args = ap.parse_args()

    yaml = YAML()
    cfg = yaml.load(Path(args.base).read_text())
    for ov in args.overrides:
        k, v = ov.split('=', 1)
        set_dotted(cfg, k, YAML(typ='safe').load(v))
    run_dir = OUT / args.name
    (run_dir / 'log').mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    with open(run_dir / 'train_config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    env = dict(os.environ, WANDB_MODE='offline',
               NUMBA_CACHE_DIR=str(Path.home() / '.cache/barlow_scratch/numba_cache'),
               MPLCONFIGDIR=str(Path.home() / '.cache/barlow_scratch/mpl'))
    r = subprocess.run([PY, '-u', 'barlow_track/scripts/train_barlow_clusterer.py', '-p',
                        str(run_dir / 'train_config.yaml')], cwd=REPO, env=env)
    rows = proxy_curve(run_dir)
    print(f"\n== {args.name} {' '.join(args.overrides)}")
    for ep, acc, rank, vl, t in rows:
        print(f"  epoch {ep:3d}  proxy {acc:.3f}  rank {rank if rank is None else round(rank, 1)}  "
              f"val_loss {vl:.4f}  t {t}s")
    if r.returncode != 0:
        sys.exit(r.returncode)
    if args.eval:
        subprocess.run([PY, '-u', 'barlow_track/scripts/eval_accuracy.py', '--lab', 'zimmer',
                        '--source', 'project', '--project', EVAL_PROJECT, '--gt', EVAL_PROJECT,
                        '--mode', 'trained', '--weights', str(run_dir / 'resnet50.pth'),
                        '--device', 'cuda', '--track_device', 'cpu', '--emb_dir', str(run_dir / 'emb'),
                        '--tag', f'posonly_fast_{args.name}_zimmer_1128',
                        '--results_jsonl', str(run_dir / 'eval.jsonl')], cwd=REPO, env=env, check=True)


if __name__ == '__main__':
    main()
