"""Partial Spearman between the GT-free self-consistency metrics (consistency_proxy.py) and accuracy,
controlling for the swept settings: rank residuals after linear regression on num_frames, epochs, lr
(log), lambd_obj, self_layers and the global augmentation strengths, the same controls as
proxy_partial.py (val loss), so the two proxies compare directly. A proxy that only reflects the training
budget loses its correlation here; one that carries information beyond the settings keeps it.

Only fusion=add trials (the architecture every recording converged to). The settings come from each
trial's record in <round>/exp_results.jsonl.

    python barlow_track/scripts/scratch/gt_free_proxy/consistency_partial.py consistency.jsonl
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata, spearmanr

C = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
METRICS = ['agree_mean', 'agree_p10', 'conf_mean', 'conf_med', 'assigned_frac', 'complete_tracks_50']
CTRL = [('num_frames', True), ('epochs', True), ('lr', True), ('lambd_obj', False), ('self_layers', False),
        ('max_degrees_z', False), ('scale_jitter', False), ('elastic_scale', False)]


def resid(y, X):
    y = rankdata(y)
    X = np.column_stack([np.ones(len(y))] + [rankdata(c) for c in X.T])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def settings(round_name):
    out = {}
    for line in open(os.path.join(C, round_name, 'exp_results.jsonl')):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if x.get('accuracy') is not None:
            out[x['tag']] = x.get('train_config') or {}
    return out


def main(path):
    by_round = defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by_round[r['round']].append(r)
    for rnd, rows in sorted(by_round.items()):
        tc_by_tag = settings(rnd)
        acc, met, ctl = [], [], []
        for r in rows:
            tc = tc_by_tag.get(r['tag'], {})
            if tc.get('fusion') != 'add':
                continue
            ga = tc.get('global_augment') or {}
            try:
                c = [float(tc.get(p, ga.get(p))) for p, _ in CTRL]
            except TypeError:
                continue
            acc.append(r['accuracy'])
            met.append([r[m] for m in METRICS])
            ctl.append([np.log(v) if lg else v for v, (_, lg) in zip(c, CTRL)])
        n = len(acc)
        if n < len(CTRL) + 5:
            print(f"{rnd}: only {n} fusion=add trials, too few for {len(CTRL)} controls")
            continue
        acc, met, ctl = np.array(acc), np.array(met, dtype=float), np.array(ctl)
        ra = resid(acc, ctl)
        settings_only = 1 - np.var(ra) / np.var(rankdata(acc))
        print(f"\n{rnd}: n={n} fusion=add trials; the settings explain {settings_only:.2f} of the "
              f"accuracy rank variance")
        print(f"  {'metric':20s} {'raw':>6s} {'partial':>8s}")
        for j, m in enumerate(METRICS):
            raw = spearmanr(met[:, j], acc).correlation
            part = np.corrcoef(resid(met[:, j], ctl), ra)[0, 1]
            print(f"  {m:20s} {raw:6.2f} {part:8.2f}")


if __name__ == '__main__':
    main(sys.argv[1])
