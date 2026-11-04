"""Evidence for two click-level GT exclusion rules that need no segmentation: coincident clicks and misclicks.

Coincident clicks. Two GT neurons' clicks closer than any two segmented cells ever are cannot both be on
their own cell. Compares, per volume, the distance between every pair of clicks of different neurons with
the distance between every segmented object and its nearest neighbour (centroids from each project's
1-segmentation/metadata.pickle), in um. Reports both distributions' low percentiles and the count of click
pairs below each object-spacing percentile and below fixed distances, and how persistent each close pair is.

Misclicks. A click that leaves the position its neighbouring frames give it and comes back. Worm motion is
removed pairwise: for neuron j and each of its K nearest neurons k (nearest by median distance over the
recording), the vector D_jk(t) = click_j(t) - click_k(t) minus its median over frames t-w..t+w (t excluded)
is e_jk(t); the misclick statistic is |median over k of e_jk(t)| (median per axis, over the neighbours with a
click at t and in the window, at least --min_neighbours). A click that jumps shifts every e_jk alike; a
neighbour that jumps, or is missing, moves only its own term, which the median ignores. (A mean over the
neighbours' clicks is not used: a missing neighbour shifts it by several um.) A null for pure annotation jitter: the same
statistic under independent Gaussian jitter of the measured s.d. per axis (--sigma_um, from click_jitter.py),
simulated through the same K-neighbour and median steps. Reports the observed and null percentiles, counts
beyond multiples of the null's 99.9th percentile, run lengths of flagged frames, and the neurons involved.
With --categories (gt_click_exclusions.py) and --brightness (gt_click_misses_vs_cap.py), an independent check
for each threshold: of the flagged and the unflagged clicks, the fraction on a StarDist mask, on a NeRVE
mask (the NWB's raw_segmentation_id < 1000), on neither, and their median brightness relative to the
volume's assigned clicks. Misclicks should sit off both segmentations and be dim.

    python .../gt_click_rules.py --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb \
        --projects stardist=.../project_config.yaml nerve=... --sigma_um 0.35 0.20 0.20 --out <dir>
"""
import argparse
import json
import os
import pickle

import numpy as np


def neighbours(C, K):
    """K nearest neurons of each neuron by median click distance over the recording, (N, K)."""
    med = np.nanmedian(C, axis=0)
    D = np.linalg.norm(med[:, None] - med[None], axis=2)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :K]


def pairwise_deviation(C, nb, w, min_neighbours):
    """|median_k (D_jk(t) - median_{s in t-w..t+w, s != t} D_jk(s))| per (t, neuron), um (module docstring)."""
    T, N, _ = C.shape
    K = nb.shape[1]
    out = np.full((T, N), np.nan)
    for j in range(N):
        D = C[:, j][:, None, :] - C[:, nb[j]]                        # T, K, 3
        E = np.full_like(D, np.nan)
        for t in range(T):
            lo, hi = max(0, t - w), min(T, t + w + 1)
            win = np.concatenate([D[lo:t], D[t + 1:hi]])
            if len(win):
                E[t] = D[t] - np.nanmedian(win, axis=0)
        ok = np.isfinite(E).all(2)
        enough = ok.sum(1) >= min_neighbours
        m = np.nanmedian(np.where(ok[..., None], E, np.nan), axis=1)
        out[:, j] = np.where(enough, np.linalg.norm(m, axis=1), np.nan)
    return out


def runs(mask):
    out, r = [], 0
    for m in mask:
        if m:
            r += 1
        elif r:
            out.append(r)
            r = 0
    if r:
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--projects', nargs='*', default=[], help='NAME=project_config.yaml (object spacing)')
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.5, 0.3226, 0.3226])
    ap.add_argument('--sigma_um', type=float, nargs=3, default=[0.35, 0.20, 0.20],
                    help='annotation jitter s.d. per axis z x y (click_jitter.py)')
    ap.add_argument('--K', type=int, default=6, help='neighbours for the motion reference')
    ap.add_argument('--w', type=int, default=5, help='half-window (frames) for the temporal median')
    ap.add_argument('--min_neighbours', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', default=None, help="gt_click_exclusions.py's click_categories.npz")
    ap.add_argument('--brightness', default=None, help="gt_click_misses_vs_cap.py's npz")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    scale = np.asarray(args.voxel_um)
    nwb = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(nwb)
    names = list(a['neurons'])
    C = np.stack([a['gz'], a['gx'], a['gy']], 2) * scale  # um
    T, N, _ = C.shape
    out = dict(args=vars(args), volumes=T, neurons=N, clicks=int(np.isfinite(C).all(2).sum()))

    # ---- Coincident clicks vs object spacing
    pair_d, pair_key, pair_t = [], [], []
    for t in range(T):
        ok = np.flatnonzero(np.isfinite(C[t]).all(1))
        if len(ok) < 2:
            continue
        p = C[t, ok]
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        iu = np.triu_indices(len(ok), 1)
        pair_d.append(d[iu])
        pair_key.append(np.stack([ok[iu[0]], ok[iu[1]]], 1))
        pair_t.append(np.full(len(iu[0]), t))
    pair_d, pair_key, pair_t = np.concatenate(pair_d), np.concatenate(pair_key), np.concatenate(pair_t)
    nn_click = np.full((T, N), np.nan)
    for t in range(T):
        ok = np.flatnonzero(np.isfinite(C[t]).all(1))
        if len(ok) > 1:
            d = np.linalg.norm(C[t, ok][:, None] - C[t, ok][None], axis=2)
            np.fill_diagonal(d, np.inf)
            nn_click[t, ok] = d.min(1)
    spacing = {}
    for name, cfg in (p.split('=', 1) for p in args.projects):
        with open(os.path.join(os.path.dirname(os.path.abspath(cfg)), '1-segmentation', 'metadata.pickle'), 'rb') as f:
            md = pickle.load(f)
        nn = []
        for t in range(T):
            df = md.get(t)
            if df is None or len(df) < 2:
                continue
            c = np.stack(df['centroids'].values).astype(float) * scale
            d = np.linalg.norm(c[:, None] - c[None], axis=2)
            np.fill_diagonal(d, np.inf)
            nn.append(d.min(1))
        nn = np.concatenate(nn)
        spacing[name] = nn
        out.setdefault('object_nn_spacing_um', {})[name] = dict(
            n=int(len(nn)), pct_0_1_1_5_50=np.percentile(nn, [0.1, 1, 5, 50]).round(2).tolist(), min=round(float(nn.min()), 2))
    out['click_nn_spacing_um_pct_0_1_1_5_50'] = np.nanpercentile(nn_click, [0.1, 1, 5, 50]).round(2).tolist()
    thr = {f'{k}um': float(k) for k in (1.0, 1.5, 2.0, 2.5)}
    for name, nn in spacing.items():
        thr[f'{name}_object_spacing_p0.1'] = float(np.percentile(nn, 0.1))
        thr[f'{name}_object_spacing_p1'] = float(np.percentile(nn, 1))
    close = {}
    for k, v in thr.items():
        m = pair_d < v
        keys = {tuple(x) for x in pair_key[m]}
        close[k] = dict(threshold_um=round(v, 2), click_pairs=int(m.sum()),
                        clicks_involved=int(len({(t, j) for t, (j1, j2) in zip(pair_t[m], pair_key[m]) for j in (j1, j2)})),
                        neuron_pairs=len(keys))
    out['coincident_click_pairs'] = close
    m15 = pair_d < 1.5
    per = {}
    for (j1, j2), t in zip(pair_key[m15], pair_t[m15]):
        per.setdefault((names[j1], names[j2]), []).append(int(t))
    top = sorted(per.items(), key=lambda kv: -len(kv[1]))[:12]
    out['coincident_1_5um_top_neuron_pairs'] = [dict(pair=list(k), frames=len(v), longest_run=max(runs(np.isin(np.arange(T), v))))
                                                for k, v in top]

    # ---- Misclicks: deviation from the temporal median of the motion-corrected position
    nb = neighbours(C, args.K)
    dev = pairwise_deviation(C, nb, args.w, args.min_neighbours)
    rng = np.random.default_rng(args.seed)
    sig = np.asarray(args.sigma_um)
    # Null: the same pipeline on pure jitter around a fixed relative position (each neuron and its K
    # neighbours jittered independently; worm motion cancels exactly by construction)
    Cn = np.where(np.isfinite(C), rng.normal(0, 1, C.shape) * sig, np.nan)
    dev_null = pairwise_deviation(Cn + np.nanmedian(C, axis=0, keepdims=True), nb, args.w, args.min_neighbours)
    obs, nul = dev[np.isfinite(dev)], dev_null[np.isfinite(dev_null)]
    q999 = float(np.percentile(nul, 99.9))
    out['misclick'] = dict(
        observed_dev_um_pct_50_90_99_99_9=np.percentile(obs, [50, 90, 99, 99.9]).round(2).tolist(),
        null_dev_um_pct_50_90_99_99_9=np.percentile(nul, [50, 90, 99, 99.9]).round(2).tolist(),
        null_p99_9_um=round(q999, 2))
    on_sd = on_nerve = rel = None
    if args.categories:
        cz = np.load(args.categories)
        if list(cz['neurons']) != names:
            raise SystemExit('neuron order differs between the NWB and click_categories.npz')
        cc = list(cz['cats'])
        on_sd = np.isin(cz['cat'], [cc.index(k) for k in ('assigned', 'in_mask_far_centroid', 'merged_object')])
        gs = a['gs']
        on_nerve = np.isfinite(gs) & (gs > 0) & (gs < 1000)
    if args.brightness:
        bz = np.load(args.brightness)
        rel = bz['inten'] / bz['ref_level'][:, None]

    def check(mask):
        r = {}
        if on_sd is not None:
            r.update(on_stardist_mask=round(float(on_sd[mask].mean()), 3), on_nerve_mask=round(float(on_nerve[mask].mean()), 3),
                     on_neither=round(float((~on_sd[mask] & ~on_nerve[mask]).mean()), 3))
        if rel is not None:
            r['brightness_rel_median'] = round(float(np.nanmedian(rel[mask])), 3)
        return r

    valid_dev = np.isfinite(dev)
    if on_sd is not None or rel is not None:
        out['misclick_check_unflagged_below_null_p99_9'] = check(valid_dev & (dev <= q999))
    for f in (1, 2, 3, 5):
        flag = dev > f * q999
        rl = [r for j in range(N) for r in runs(flag[:, j])]
        by = {names[j]: int(flag[:, j].sum()) for j in range(N) if flag[:, j].sum()}
        out['misclick'][f'beyond_{f}x_null_p99_9'] = dict(
            threshold_um=round(f * q999, 2), clicks=int(flag.sum()), frac=round(float(flag.sum() / max(len(obs), 1)), 5),
            null_clicks_expected=round(float((nul > f * q999).mean() * len(obs)), 1),
            run_lengths_1_2_3_4_5plus=[int(sum(r == k for r in rl)) for k in (1, 2, 3, 4)] + [int(sum(r >= 5 for r in rl))],
            top_neurons=dict(sorted(by.items(), key=lambda kv: -kv[1])[:8]), **check(flag))
    np.savez(os.path.join(args.out, 'click_rules.npz'), dev=dev, dev_null=dev_null, nn_click=nn_click,
             pair_d=pair_d, pair_key=pair_key, pair_t=pair_t, names=np.array(names), **{f'objnn_{k}': v for k, v in spacing.items()})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.linspace(0, 8, 161)
    ax[0].hist(nn_click[np.isfinite(nn_click)], bins=bins, density=True, histtype='step', color='k',
               label='click to nearest other neuron\'s click')
    for i, (name, nn) in enumerate(spacing.items()):
        ax[0].hist(nn, bins=bins, density=True, histtype='step', color=f'C{i}', label=f'{name} object to nearest object')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('um')
    ax[0].set_title('Coincident clicks vs cell spacing')
    ax[0].legend(fontsize=7)
    lb = np.linspace(-1.5, 1.6, 140)
    ax[1].hist(np.log10(obs[obs > 0]), bins=lb, density=True, histtype='step', color='k', label='observed')
    ax[1].hist(np.log10(nul[nul > 0]), bins=lb, density=True, histtype='step', color='C1', label='jitter null')
    for f in (1, 3):
        ax[1].axvline(np.log10(f * q999), ls='--', lw=0.8, color='0.4')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('log10 um: click minus its temporal median (motion-corrected)')
    ax[1].set_title('Misclicks')
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'click_rules.png'), dpi=90)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
