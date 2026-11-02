"""How much does one swept hyperparameter move tracking accuracy? Read from finished sweep rounds.

For each round (config_files/<round>/), joins every accuracy record in exp_results.jsonl
(latest per tag) with its trial's train_config.yaml and reports, for --param:
  - Spearman correlation with accuracy, over all trials and over the first --n_sobol trials
    (Ax's random Sobol phase: the only unbiased sample; later trials are chosen by the model);
  - mean / max accuracy per bin of the parameter;
  - a random-forest fit of accuracy on every swept parameter: permutation importance rank of
    --param among them, and the partial dependence of accuracy on --param (other parameters at
    their observed values), which controls for parameters that co-vary in the Bayesian phase.

    python barlow_track/scripts/sweep_param_effect.py --param lambd_obj \
        --rounds 'round_bayes_wk2_*' 'round_bayes_wk3_*'
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml

ROUNDS = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'


def _get(cfg, dotted):
    for k in dotted.split('.'):
        if not isinstance(cfg, dict) or k not in cfg:
            return None
        cfg = cfg[k]
    return cfg


def load_round(rdir):
    tmpl = yaml.safe_load(open(os.path.join(rdir, 'hyperparameter_search_template.yaml')))
    hps = tmpl.get('hyperparameters', [])
    recs = {}
    for f in sorted(glob.glob(os.path.join(rdir, 'exp_results*.jsonl'))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r.get('accuracy'), (int, float)) and np.isfinite(r['accuracy']) and r.get('trial_dir'):
                recs[r['tag']] = r  # latest wins, as the benchmark driver dedups
    rows = []
    for r in recs.values():
        cfg_path = os.path.join(r['trial_dir'], 'train_config.yaml')
        if not os.path.exists(cfg_path):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        trial = int(os.path.basename(r['trial_dir'].rstrip('/')).split('_')[-1])
        vals = {}
        for hp in hps:
            v = _get(cfg, hp['name'])
            if hp['type'] == 'choice' and not hp.get('is_ordered'):
                v = hp['values'].index(v) if v in hp['values'] else None
            vals[hp['name']] = v
        rows.append(dict(trial=trial, acc=r['accuracy'], tag=r['tag'], **vals))
    return hps, sorted(rows, key=lambda x: x['trial'])


def analyze(name, hps, rows, param, n_sobol, bins, seed):
    from scipy.stats import spearmanr
    rows = [r for r in rows if r.get(param) is not None]
    if len(rows) < 8:
        print(f"{name}: {len(rows)} usable trials, skipped")
        return
    acc = np.array([r['acc'] for r in rows])
    p = np.array([float(r[param]) for r in rows])
    tri = np.array([r['trial'] for r in rows])
    sob = tri < n_sobol
    rho, pv = spearmanr(p, acc)
    line = f"{name}: n {len(rows)} (Sobol {sob.sum()}), acc median {np.median(acc):.3f} max {acc.max():.3f}; " \
           f"spearman all {rho:+.2f} (p {pv:.2g})"
    if sob.sum() >= 6:
        rs, ps = spearmanr(p[sob], acc[sob])
        line += f", Sobol {rs:+.2f} (p {ps:.2g})"
    print(line)
    edges = np.asarray(bins)
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < edges[-1] else (p >= lo) & (p <= hi)
        cells.append(f"[{lo:g},{hi:g}] n {m.sum():2d} mean {acc[m].mean():.3f} max {acc[m].max():.3f}"
                     if m.any() else f"[{lo:g},{hi:g}] n  0")
    print("   bins: " + " | ".join(cells))
    top = np.argsort(-acc)[:10]
    print(f"   top-10 {param}: {np.round(np.sort(p[top]), 2).tolist()}")

    names = [hp['name'] for hp in hps if all(r.get(hp['name']) is not None for r in rows)]
    if param not in names or len(rows) < 20:
        return
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    X = np.array([[float(r[n]) for n in names] for r in rows])
    for j, hp in enumerate(hps):
        if hp['name'] in names and hp.get('log_scale'):
            X[:, names.index(hp['name'])] = np.log(np.maximum(X[:, names.index(hp['name'])], 1e-12))
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=seed, oob_score=True).fit(X, acc)
    imp = permutation_importance(rf, X, acc, n_repeats=20, random_state=seed).importances_mean
    order = np.argsort(-imp)
    rank = int(np.flatnonzero(names[i] == param for i in order)[0]) if False else \
        [names[i] for i in order].index(param) + 1
    print(f"   RF (OOB R2 {rf.oob_score_:.2f}): {param} importance rank {rank}/{len(names)}; top 4: " +
          ", ".join(f"{names[i]} {imp[i]:.4f}" for i in order[:4]))
    j = names.index(param)
    grid = np.linspace(edges[0], edges[-1], 6)
    pd_vals = []
    for g in grid:
        Xg = X.copy()
        Xg[:, j] = g
        pd_vals.append(rf.predict(Xg).mean())
    print("   partial dependence: " + " ".join(f"{g:g}:{v:.3f}" for g, v in zip(grid, pd_vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--param', default='lambd_obj')
    ap.add_argument('--rounds', nargs='+', default=['round_bayes_wk2_*'])
    ap.add_argument('--n_sobol', type=int, default=26)
    ap.add_argument('--bins', type=float, nargs='+', default=[0, 0.1, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dirs = sorted({d for pat in args.rounds for d in glob.glob(os.path.join(ROUNDS, pat))})
    for d in dirs:
        if not os.path.exists(os.path.join(d, 'hyperparameter_search_template.yaml')):
            continue
        hps, rows = load_round(d)
        if not any(hp['name'] == args.param for hp in hps):
            continue
        analyze(os.path.basename(d), hps, rows, args.param, args.n_sobol, args.bins, args.seed)


if __name__ == '__main__':
    main()
