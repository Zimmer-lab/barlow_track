"""Partial Spearman (rank residuals after linear regression on the swept settings) between GT-free metrics and accuracy."""
import json, glob, os, numpy as np
from scipy.stats import rankdata, spearmanr
C = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
M = ['val_mean_batch_loss', 'train_loss', 'val_val_fused_offdiag_corr', 'val_val_fused_mean_frac', 'val_val_val_std_median']
CTRL = [('num_frames', True), ('epochs', True), ('lr', True), ('lambd_obj', False), ('self_layers', False),
        ('max_degrees_z', False), ('scale_jitter', False), ('elastic_scale', False)]
def resid(y, X):
    y = rankdata(y); X = np.column_stack([np.ones(len(y))] + [rankdata(c) for c in X.T])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ b
print(f"{'round':28s} {'n':>3s} " + ' '.join(f'{m[-12:]:>13s}' for m in M) + '   (raw | partial)')
for r in sorted(glob.glob(C + '/round_bayes_wk[23]_*/exp_results.jsonl')):
    name = r.split('/')[-2].replace('round_bayes_', '')
    if 'zimmer_1128' in name: continue
    recs = {}
    for l in open(r):
        try: x = json.loads(l)
        except Exception: continue
        if x.get('accuracy') is not None: recs[x['tag']] = x
    acc, met, ctl = [], [], []
    for t, x in recs.items():
        tc = x.get('train_config') or {}
        if tc.get('fusion') != 'add': continue
        f = f"{C}/{r.split('/')[-2]}/trial_{t.split('_trial')[1].split('_')[0]}/log/stats.json"
        if not os.path.exists(f): continue
        try: st = json.load(open(f))
        except Exception: continue
        last = {}
        for e in st:
            for k, v in e.items():
                if k in M and isinstance(v, (int, float)): last[k] = v
        if not all(np.isfinite(last.get(m, np.nan)) for m in M): continue
        ga = tc.get('global_augment') or {}
        c = [float(tc.get(p, ga.get(p))) for p, _ in CTRL]
        acc.append(x['accuracy']); met.append([last[m] for m in M]); ctl.append([np.log(v) if lg else v for v, (_, lg) in zip(c, CTRL)])
    acc, met, ctl = np.array(acc), np.array(met), np.array(ctl)
    ra = resid(acc, ctl)
    cells = []
    for j in range(len(M)):
        raw = spearmanr(met[:, j], acc).correlation
        part = np.corrcoef(resid(met[:, j], ctl), ra)[0, 1]
        cells.append(f'{raw:5.2f} | {part:5.2f}')
    print(f"{name:28s} {len(acc):3d} " + ' '.join(f'{c:>13s}' for c in cells))
