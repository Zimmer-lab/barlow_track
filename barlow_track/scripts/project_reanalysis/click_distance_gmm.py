"""Do the click-to-segmentation distances split into "annotation noise" and "object the segmentation missed"?

Fits Gaussian mixtures (sklearn, 1-4 components, BIC picks) to two distance histograms from
gt_click_exclusions.py's click_categories.npz (StarDist project, Leifer manual clicks):

  edge     clicks on background (no StarDist mask at the click voxel): distance to the nearest mask voxel
           (um), alone and jointly with the click's red intensity relative to the volume's assigned clicks
           (gt_click_misses_vs_cap.py's npz); log10 of both
  centroid every click with a finite click-to-nearest-centroid distance: log10 of that distance

and reports each component (weight, mean, s.d. back in um), the distance where the far component's
posterior crosses 0.5, and, as an independent check, the fraction of clicks in each component that NeRVE
segmented (the NWB's raw_segmentation_id at the click is a NeRVE label, < --pseudo_min_label). A component
of real misses should be NeRVE-segmented much more often than one of clicks jittered off a mask.

    python .../click_distance_gmm.py --categories .../click_categories.npz --brightness .../click_misses_vs_cap.npz \
        --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb --out <dir>
"""
import argparse
import json
import os

import numpy as np


def fit(X, kmax, seed):
    from sklearn.mixture import GaussianMixture
    fits = [GaussianMixture(k, covariance_type='full', n_init=5, random_state=seed).fit(X) for k in range(1, kmax + 1)]
    bic = [float(g.bic(X)) for g in fits]
    return fits, bic


def describe(g, X, nerve, names, edges=(1.0, 1.5)):
    order = np.argsort(g.means_[:, 0])
    lab = g.predict(X)
    comps = []
    for rank, c in enumerate(order):
        m = lab == c
        comps.append(dict(component=rank, weight=round(float(g.weights_[c]), 4), n=int(m.sum()),
                          **{f'{nm}_geo_mean': round(float(10 ** g.means_[c, i]), 3) for i, nm in enumerate(names)},
                          **{f'{nm}_log10_sd': round(float(np.sqrt(g.covariances_[c][i, i])), 3)
                             for i, nm in enumerate(names)},
                          frac_nerve_segmented=round(float(nerve[m].mean()), 3) if m.any() else None,
                          **{f'frac_first_dim_within_{e}um': round(float((10 ** X[m, 0] <= e + 1e-6).mean()), 3)
                             for e in edges if m.any()}))
    return comps, order


def crossing(g, order, lo, hi, other=None):
    """Distance (um) where the posterior of the farthest component first exceeds 0.5 (others fixed at median)."""
    d = np.logspace(np.log10(lo), np.log10(hi), 2000)
    X = np.log10(d)[:, None]
    if other is not None:
        X = np.column_stack([X, np.full(len(d), other)])
    post = g.predict_proba(X)[:, order[-1]]
    hit = np.flatnonzero(post > 0.5)
    return round(float(d[hit[0]]), 3) if len(hit) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--categories', required=True)
    ap.add_argument('--brightness', required=True)
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--pseudo_min_label', type=int, default=1000)
    ap.add_argument('--kmax', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    d = np.load(args.categories)
    cat, cats = d['cat'], list(d['cats'])
    b = np.load(args.brightness)
    rel = b['inten'] / b['ref_level'][:, None]
    nwb = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(nwb)
    if list(a['neurons']) != list(d['neurons']):
        raise SystemExit('neuron order differs between the NWB and click_categories.npz')
    gs = a['gs'][:cat.shape[0]]
    nerve = np.isfinite(gs) & (gs > 0) & (gs < args.pseudo_min_label)
    out = dict(args=vars(args))

    # Edge distances of background clicks
    bg = np.isin(cat, [cats.index(c) for c in ('background_near_mask', 'missed_found_by_ref', 'background_both')])
    e = d['edge_dist']
    m1 = bg & np.isfinite(e) & (e > 0)
    x = np.log10(e[m1])[:, None]
    fits, bic = fit(x, args.kmax, args.seed)
    k = int(np.argmin(bic))
    comps, order = describe(fits[k], x, nerve[m1], ['edge_um'])
    out['edge_1d'] = dict(n=int(m1.sum()), bic=[round(v, 1) for v in bic], best_k=k + 1, components=comps,
                          far_posterior_0_5_at_um=crossing(fits[k], order, 0.3, 30) if k else None,
                          frac_nerve_segmented_all=round(float(nerve[m1].mean()), 3))
    # Two components regardless of BIC, for a single noise / miss split
    if k != 1:
        c2, o2 = describe(fits[1], x, nerve[m1], ['edge_um'])
        out['edge_1d']['two_components'] = dict(components=c2, far_posterior_0_5_at_um=crossing(fits[1], o2, 0.3, 30))
    # Jointly with brightness
    m2 = m1 & np.isfinite(rel) & (rel > 0)
    X = np.column_stack([np.log10(e[m2]), np.log10(rel[m2])])
    fits2, bic2 = fit(X, args.kmax, args.seed)
    k2 = int(np.argmin(bic2))
    comps2, order2 = describe(fits2[k2], X, nerve[m2], ['edge_um', 'brightness_rel'])
    out['edge_and_brightness_2d'] = dict(n=int(m2.sum()), bic=[round(v, 1) for v in bic2], best_k=k2 + 1,
                                         components=comps2)

    # Click to nearest centroid, every click
    cd = d['centroid_dist']
    m3 = np.isfinite(cd) & (cd > 0) & (cat >= 0)
    x3 = np.log10(cd[m3])[:, None]
    fits3, bic3 = fit(x3, args.kmax, args.seed)
    k3 = int(np.argmin(bic3))
    comps3, order3 = describe(fits3[k3], x3, nerve[m3], ['centroid_um'])
    out['centroid_1d'] = dict(n=int(m3.sum()), bic=[round(v, 1) for v in bic3], best_k=k3 + 1, components=comps3,
                              far_posterior_0_5_at_um=crossing(fits3[k3], order3, 0.05, 30) if k3 else None)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for axi, (xx, g, o, lab) in zip(ax, [(x[:, 0], fits[k], order, 'background click to nearest StarDist mask'),
                                         (x3[:, 0], fits3[k3], order3, 'click to nearest StarDist centroid')]):
        axi.hist(xx, bins=80, density=True, color='0.7')
        grid = np.linspace(xx.min(), xx.max(), 400)[:, None]
        resp = g.predict_proba(grid) * np.exp(g.score_samples(grid))[:, None]
        for r, c in enumerate(o):
            axi.plot(grid[:, 0], resp[:, c], label=f'component {r}')
        axi.set_xlabel(f'log10 um: {lab}')
        axi.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'gmm.png'), dpi=90)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
