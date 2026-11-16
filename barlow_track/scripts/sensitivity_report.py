"""Read the round_sens_* sensitivity rounds: accuracy per variant and recording, paired against the control.

A trial's variant is how its train_config.yaml differs from the frozen recipe (scratch/round_sens_src/
base_train_config.yaml; per-recording keys ignored), e.g. "lambd_obj=0.0, num_frames=16"; the control is the
empty difference. Trials that share a seed are paired. Records come from each round's exp_results.jsonl
(latest per tag), as in sweep_param_effect.py.

    python barlow_track/scripts/sensitivity_report.py [--rounds 'round_sens_*'] [--out summary.tsv]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import yaml

ROUNDS = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch', 'round_sens_src', 'base_train_config.yaml')
PER_RUN_KEYS = {'project_dir', 'project_path', 'wandb_name', 'target_sz_z', 'visnet_z_scale', 'seed', 'DEBUG', 'dryrun',
                'model_fname', 'epoch_snapshots'}


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def _close(x, y):
    """Equal as numbers; '1e-06' (a string in YAML 1.1) equals 1e-06."""
    try:
        return bool(np.isclose(float(x), float(y), rtol=1e-9)) and not isinstance(x, bool)
    except (TypeError, ValueError):
        return False


def recording_target_sz_z(label, base):
    """The crop depth a recording's rounds use by default (make_per_recording_rounds: Flavell 16, else the base's)."""
    from barlow_track.scripts.scratch.make_per_recording_rounds import TARGET_SZ_Z
    return TARGET_SZ_Z.get(str(label).split('_')[0], base.get('target_sz_z', 8))


def variant_name(cfg, base, label=None):
    """'k=v, k=v' over the keys where cfg differs from base (per-recording keys excluded); '' for the control.

    target_sz_z is per recording (Flavell 16), so with label it is compared against that recording's
    default instead of being ignored: the crop_depth round's other depth is a variant, not the control.
    """
    a, b = flatten(cfg), flatten(base)
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k.split('.')[0] in PER_RUN_KEYS or k in PER_RUN_KEYS:
            continue
        x, y = a.get(k), b.get(k)
        same = (x == y) or _close(x, y)
        if not same:
            diffs.append(f'{k}={x:.4g}' if isinstance(x, float) else f'{k}={x}')
    if label is not None and 'target_sz_z' in cfg and int(cfg['target_sz_z']) != int(recording_target_sz_z(label, base)):
        diffs.append(f"target_sz_z={int(cfg['target_sz_z'])}")
    return ', '.join(sorted(diffs))


def collect(rounds_dir, pattern, base):
    """[(label, variant, seed, accuracy)] over all records of the matching rounds."""
    rows = []
    for d in sorted(glob.glob(os.path.join(rounds_dir, pattern))):
        path = os.path.join(d, 'exp_results.jsonl')
        if not os.path.exists(path):
            continue
        latest = {}
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('tag'):
                latest[r['tag']] = r
        for tag, r in latest.items():
            m = re.search(r'_trial(\d+)_(.+)$', tag)
            cfg_path = os.path.join(d, f'trial_{m.group(1)}', 'train_config.yaml') if m else None
            if not m or r.get('accuracy') is None or not os.path.exists(cfg_path):
                continue
            cfg = yaml.safe_load(open(cfg_path))
            rows.append((m.group(2), variant_name(cfg, base, m.group(2)), int(cfg['seed']), float(r['accuracy'])))
    return rows


def summarize(rows):
    """Per variant: mean accuracy per label, mean paired difference to the control (same label and seed).

    Several trials of one variant can share (label, seed) (the control runs in loss_volume and again in
    crop_depth); each is kept, and a variant is paired with the mean of the control's trials of that seed.
    """
    acc = defaultdict(lambda: defaultdict(list))  # variant -> (label, seed) -> [accuracy, ...]
    for label, var, seed, a in rows:
        acc[var][(label, seed)].append(a)
    ctrl = {k: float(np.mean(v)) for k, v in acc.get('', {}).items()}
    labels = sorted({l for l, *_ in rows})
    out = []
    for var, cells in acc.items():
        per_label = {l: [a for (ll, s), v in cells.items() if ll == l for a in v] for l in labels}
        diffs = [a - ctrl[k] for k, v in cells.items() if k in ctrl and var != '' for a in v]
        out.append(dict(variant=var or 'control', n=sum(len(v) for v in cells.values()),
                        accuracy={l: float(np.mean(v)) if v else None for l, v in per_label.items()},
                        paired_diff=float(np.mean(diffs)) if diffs else None,
                        paired_sd=float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None, n_pairs=len(diffs)))
    out.sort(key=lambda r: (r['variant'] != 'control', r['variant']))
    return labels, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rounds_dir', default=ROUNDS)
    ap.add_argument('--rounds', default='round_sens_*')
    ap.add_argument('--base', default=BASE)
    ap.add_argument('--out', default=None, help='also write the table here (tsv)')
    a = ap.parse_args()
    base = yaml.safe_load(open(a.base))
    labels, rows = summarize(collect(a.rounds_dir, a.rounds, base))
    fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
    lines = ['\t'.join(['variant', 'n'] + labels + ['paired_diff', 'paired_sd', 'n_pairs'])]
    for r in rows:
        lines.append('\t'.join([r['variant'], str(r['n'])] + [fmt(r['accuracy'][l]) for l in labels]
                               + [fmt(r['paired_diff']), fmt(r['paired_sd']), str(r['n_pairs'])]))
    print('\n'.join(lines))
    if a.out:
        with open(a.out, 'w') as f:
            f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
