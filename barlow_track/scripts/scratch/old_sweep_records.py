"""Tabulate the scored trials of every revision sweep round by pipeline arm and lab.

Each record in <round>/exp_results.jsonl carries the trial's full train_config. Records are
deduplicated by tag (latest wins, as the benchmark driver does). Arms, from the training
code (train_barlow_clusterer.py): use_position or use_attention selects the volume pipeline
(global affine before crop, photometric-only per-crop transform) and a model that fuses
position (BarlowVolumeAttention subclasses BarlowWithPosition); both false selects the
legacy crop pipeline (per-crop affine + elastic) with an image-only model.

Caveats printed with the table: Flavell scored with --source nwb has the neuron-order bug
(i261026-1, accuracy capped at 0.908); Flavell --source project reads ~4 pt below the NWB
GT; rounds scored before the 2026-10-05 fixes (labelprop, crops, indices) are not
comparable; ViSNet trials before 60e4068 (2026-10-08 10:10) can NaN on duplicate points.

Usage (wbfm env):
  python old_sweep_records.py [--rounds round_2x2 ...] [--csv out.csv]
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'


def arm(cfg):
    if not (cfg.get('use_position') or cfg.get('use_attention')):
        return 'legacy crop, image only'
    enc = cfg.get('position_encoder', 'keypoint_mlp')
    att = f"attn{cfg.get('self_layers', 2)}" if cfg.get('use_attention') else 'no attn'
    return f"volume, {enc}, {cfg.get('fusion', 'add')}, {att}"


def load(rounds):
    rows = []
    for f in sorted(glob.glob(f'{ROOT}/*/exp_results.jsonl')):
        rnd = os.path.basename(os.path.dirname(f))
        if rounds and rnd not in rounds:
            continue
        by_tag = collections.OrderedDict()
        for line in open(f):
            if line.strip():
                x = json.loads(line)
                by_tag[x['tag']] = x
        for x in by_tag.values():
            c = x.get('train_config', {})
            steps = int(c.get('epochs', 0)) * int(c.get('num_frames', 0)) * float(c.get('train_fraction', 0.8))
            rows.append(dict(
                round=rnd, trial=x['tag'].split('_trial')[-1].split('_')[0], lab=x['lab'],
                source=x.get('source'), accuracy=x.get('accuracy', np.nan), arm=arm(c),
                train_on=os.path.basename(os.path.dirname(c.get('project_path', '')))[:28],
                steps=steps, epochs=c.get('epochs'), num_frames=c.get('num_frames'),
                emb=c.get('embedding_dim'), proj_norm=c.get('projector_norm'), lr=float(c.get('lr', np.nan)),
                crop=f"{c.get('target_sz_z')}x{c.get('target_sz_xy')}", seed=c.get('seed')))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', nargs='*')
    ap.add_argument('--csv')
    args = ap.parse_args()
    df = load(args.rounds)
    df['cell'] = df.lab + '/' + df.source
    pd.set_option('display.width', 250, 'display.max_rows', 500, 'display.max_colwidth', 40)
    for rnd, d in df.groupby('round', sort=False):
        print(f'\n=== {rnd}: {d.trial.nunique()} trials; trained on {sorted(d.train_on.unique())}; '
              f'steps {sorted(d.steps.unique())}; crop {sorted(d.crop.unique())}')
        t = d.groupby(['arm', 'cell']).accuracy.agg(['count', 'mean', 'max']).unstack('cell')
        print(t.round(3).to_string())
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == '__main__':
    main()
