"""How often do two GT neurons' clicks coincide? A segmentation-free check for any click-only GT (NWB).

Per recording: every click's distance (um, NWB imaging-plane grid_spacing) to the nearest other neuron's click in
the same volume; clicks with a partner within each --radii; per neuron pair, how often they are within the
smallest radius, how long those stretches last, and the pair's median distance over all shared volumes (a pair
always ~1 um apart is one neuron clicked twice or a tight neighbour; brief touches are neighbours passing).

Writes <out>/coincidence_summary.json, <out>/<name>_pairs.csv and <out>/nearest_click_distance.png.

    python .../gt_click_coincidence.py --nwb samuel_153=.../153.nwb leifer=.../reexport_A2.nwb --out DIR
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def grid_spacing(path):
    import h5py
    with h5py.File(path, 'r') as h:
        opt = h['general/optophysiology']
        gs = [np.asarray(opt[k]['grid_spacing'][()], float) for k in opt if 'grid_spacing' in opt[k]]
    if not gs:
        raise SystemExit(f'{path}: no grid_spacing')
    return gs[0]  # (x, y, z) like the image axes


def clicks_um(path, spacing=None):
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    gs = grid_spacing(path) if spacing is None else np.asarray(spacing, float)
    a = _load_nwb_arrays(ProjectData.load_final_project_data(path, allow_hybrid_loading=True, verbose=0))
    um = np.array([gs[2], gs[0], gs[1]])
    return np.stack([a['gz'], a['gx'], a['gy']], 2) * um, [str(n) for n in a['neurons']], gs


def _runs(idx):
    if not len(idx):
        return []
    return [len(r) for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)]


def analyse(C, names, radii):
    T, N, _ = C.shape
    nn_all, n_clicks = [], 0
    within = {r: 0 for r in radii}
    exact = 0
    r0 = min(radii)
    for t in range(T):
        js = np.flatnonzero(np.isfinite(C[t]).all(1))
        n_clicks += len(js)
        if len(js) < 2:
            continue
        D = cdist(C[t, js], C[t, js])
        np.fill_diagonal(D, np.inf)
        m = D.min(1)
        nn_all.append(m)
        for r in radii:
            within[r] += int((m < r).sum())
        exact += int((m < 1e-6).sum())
    nn = np.concatenate(nn_all) if nn_all else np.zeros(0)
    # pairs: distance time series of every pair that is ever within r0
    rows = []
    for i in range(N):
        d = np.linalg.norm(C[:, i + 1:] - C[:, i:i + 1], axis=2)  # (T, N - i - 1)
        close = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=0) < r0
        for k in np.flatnonzero(close):
            j = i + 1 + k
            dij = d[:, k]
            idx = np.flatnonzero(dij < r0)
            runs = _runs(idx)
            rows.append(dict(neuron_a=names[i], neuron_b=names[j], shared_volumes=int(np.isfinite(dij).sum()),
                             volumes_within=int(len(idx)), runs=len(runs), longest_run=max(runs),
                             median_distance_um=float(np.nanmedian(dij)),
                             p10_distance_um=float(np.nanpercentile(dij, 10))))
    pairs = pd.DataFrame(rows).sort_values('volumes_within', ascending=False) if rows else pd.DataFrame()
    s = dict(volumes=T, neurons=N, clicks=n_clicks, exact_duplicates=exact,
             nearest_other_click_um=dict(median=round(float(np.median(nn)), 3), p1=round(float(np.percentile(nn, 1)), 3),
                                         p10=round(float(np.percentile(nn, 10)), 3)),
             clicks_with_partner_within={f'{r}um': dict(n=within[r], frac=round(within[r] / max(n_clicks, 1), 5))
                                         for r in radii},
             pairs_within_min_radius=int(len(pairs)),
             persistent_pairs=int((pairs.median_distance_um < 2 * r0).sum()) if len(pairs) else 0)
    return s, pairs, nn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nwb', nargs='+', required=True, help='NAME=path.nwb')
    ap.add_argument('--radii', nargs='+', type=float, default=[1.0, 1.5, 2.0])
    ap.add_argument('--spacing', nargs='*', default=[],
                    help='NAME=x,y,z um: override the NWB grid_spacing (e.g. one suspected wrong)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = {}
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    override = {k: [float(x) for x in v.split(',')] for k, v in (s.split('=', 1) for s in args.spacing)}
    for spec in args.nwb:
        name, path = spec.split('=', 1)
        C, names, gs = clicks_um(path, override.get(name))
        s, pairs, nn = analyse(C, names, args.radii)
        s.update(nwb=os.path.abspath(path), grid_spacing_xyz_um=gs.tolist(), spacing_overridden=name in override)
        summary[name] = s
        pairs.to_csv(os.path.join(args.out, f'{name}_pairs.csv'), index=False)
        print(name, json.dumps({k: v for k, v in s.items() if k != 'nwb'}), flush=True)
        if len(pairs):
            print(pairs.head(8).round(2).to_string(index=False), flush=True)
        v = np.sort(nn)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=name)
    for r in args.radii:
        ax.axvline(r, color='0.7', lw=0.6, ls='--')
    ax.set_xscale('log')
    ax.set_xlim(0.3, 20)
    ax.set_xlabel('distance to nearest other neuron\'s click (um)')
    ax.set_ylabel('fraction of clicks')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'nearest_click_distance.png'), dpi=150)
    with open(os.path.join(args.out, 'coincidence_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
