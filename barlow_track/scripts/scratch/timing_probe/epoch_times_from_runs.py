"""Per-epoch wall times of finished cluster training jobs, from their logs and stats.json.

For every submitit training log in the given rounds: the node (hence GPU type), the
DataLoader workers, fusion and lambd_obj, then from the trial's log/stats.json the
seconds between consecutive validation records (one epoch of training plus its
validation) and the training time to the last one.

    python barlow_track/scripts/scratch/timing_probe/epoch_times_from_runs.py round_snap1_* round_wk1_*
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

CONFIGS = Path('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files')
GPU = {'c': 'T4', 'd': 'T4', 'g01': 'L40s', 'g02': 'L40s', 'g03': 'L40s', 'g04': 'L40s',
       'g05': 'L40s', 'g06': 'RTX6000', 'g07': 'A30', 'g08': 'A30', 'g09': 'H200', 'h': 'L4'}


def gpu_of(node):
    short = node.replace('node-', '')
    return GPU.get(short) or GPU.get(short[0], '?')


def runs(round_dir):
    for out in sorted(round_dir.glob('*_0_log.out')):
        text = out.read_text(errors='replace')
        m = re.search(r"Starting training with args: namespace\((.*)\)\. Stats in folder: (\S+)", text)
        if not m:
            continue
        a, trial = m.group(1), Path(m.group(2))
        node = re.search(r'GPU info: IDs \S+ on (\S+)', text)
        workers = re.search(r'num_workers (\d+) \(SLURM', text)
        stats = trial / 'log' / 'stats.json'
        if not stats.exists():
            continue
        val = [r['time'] for r in json.loads(stats.read_text()) if 'val_loss' in r]
        if len(val) < 3:
            continue
        ep = np.diff(val)
        yield dict(round=round_dir.name, trial=trial.name, job=out.name.split('_')[0],
                   gpu=gpu_of(node.group(1)) if node else '?', node=node.group(1) if node else '?',
                   workers=int(workers.group(1)) if workers else 0,
                   fusion=re.search(r"fusion='(\w+)'", a).group(1),
                   lambd_obj=float(re.search(r'lambd_obj=([\d.e-]+)', a).group(1)),
                   frames=int(re.search(r'num_frames=(\d+)', a).group(1)),
                   epochs=len(val), ep_med=float(np.median(ep)), ep_first=float(val[0]),
                   total_h=val[-1] / 3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rounds', nargs='+')
    args = ap.parse_args()
    dirs = sorted({d for pat in args.rounds for d in CONFIGS.glob(pat) if d.is_dir()})
    print(f"{'round':34s} {'trial':8s} {'gpu':7s} {'wk':>3s} {'fusion':12s} {'lo':>4s} {'fr':>4s} "
          f"{'ep':>3s} {'s/ep':>6s} {'ep0 s':>6s} {'h':>5s}")
    for d in dirs:
        for r in runs(d):
            print(f"{r['round']:34s} {r['trial']:8s} {r['gpu']:7s} {r['workers']:3d} {r['fusion']:12s} "
                  f"{r['lambd_obj']:4.2f} {r['frames']:4d} {r['epochs']:3d} {r['ep_med']:6.1f} "
                  f"{r['ep_first']:6.0f} {r['total_h']:5.2f}")


if __name__ == '__main__':
    main()
