"""Spearman correlation between GT-free training metrics (last logged epoch) and final tracking accuracy, per sweep round."""
import json, glob, os, numpy as np
from scipy.stats import spearmanr
C = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
M = ['val_loss', 'val_mean_batch_loss', 'test_loss', 'val_loss_original', 'val_loss_transpose', 'train_loss',
     'val_val_fused_eff_rank', 'val_val_fused_offdiag_corr', 'val_val_fused_mean_frac', 'val_val_val_std_median',
     'val_val_match_accuracy_auto']
rows = {}
for r in sorted(glob.glob(C + '/round_bayes_wk[23]_*/exp_results.jsonl')):
    name = r.split('/')[-2].replace('round_bayes_', '')
    recs = {}
    for l in open(r):
        try: x = json.loads(l)
        except Exception: continue
        if x.get('accuracy') is not None: recs[x['tag']] = x
    acc, met = [], {m: [] for m in M}
    for t, x in recs.items():
        if (x.get('train_config') or {}).get('fusion') != 'add': continue
        trial = t.split('_trial')[1].split('_')[0]
        f = f"{C}/{r.split('/')[-2]}/trial_{trial}/log/stats.json"
        if not os.path.exists(f): continue
        try: st = json.load(open(f))
        except Exception: continue
        last = {}
        for e in st:
            for k, v in e.items():
                if k in M and isinstance(v, (int, float)): last[k] = v
        acc.append(x['accuracy'])
        for m in M: met[m].append(last.get(m, np.nan))
    acc = np.array(acc)
    out = {}
    for m in M:
        v = np.array(met[m], float); ok = np.isfinite(v)
        out[m] = spearmanr(v[ok], acc[ok]).correlation if ok.sum() > 8 else np.nan
    rows[name] = (len(acc), np.median(acc), np.percentile(acc, 90) - np.percentile(acc, 10), out)
short = {'val_loss': 'val', 'val_mean_batch_loss': 'valmean', 'test_loss': 'test', 'val_loss_original': 'val_feat',
         'val_loss_transpose': 'val_obj', 'train_loss': 'train', 'val_val_fused_eff_rank': 'effrank',
         'val_val_fused_offdiag_corr': 'offdiag', 'val_val_fused_mean_frac': 'meanfrac', 'val_val_val_std_median': 'std',
         'val_val_match_accuracy_auto': 'match(GT)'}
print(f"{'round':28s} {'n':>3s} {'med':>5s} {'p90-10':>6s} " + ' '.join(f'{short[m]:>9s}' for m in M))
for n, (k, med, spread, out) in rows.items():
    print(f"{n:28s} {k:3d} {med:.3f} {spread:6.3f} " + ' '.join(f'{out[m]:9.2f}' for m in M))
