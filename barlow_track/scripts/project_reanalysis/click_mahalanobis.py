"""Click-to-object distance scaled by each object's own size and shape (Mahalanobis), for every GT click.

Euclidean click-to-centroid distances mix object size with annotation error: a click 2 um from the centre
of a large object can be on it, the same offset from a small one is off it. Here each segmented object's
voxel coordinates (um, z x y) give a covariance S (plus voxel**2 / 12 per axis, the spread of one voxel), and
a click p is d_M = sqrt((p - mu)^T S^-1 (p - mu)) from it. For a solid ellipsoid the surface is at d_M =
sqrt(5) = 2.24 in every direction, so clicks on an object sit below about 2.24 whatever its size, and a
click on an object the segmentation missed is far from every object. Per click: the smallest d_M over the
volume's objects, its object, and whether the click voxel is inside that object's mask.

Reports the d_M histogram (log bins), fractions beyond sqrt(5) and 2 sqrt(5), Gaussian mixtures on log10
d_M (1-4 components, BIC), and per component the fraction NeRVE segmented (the NWB's raw_segmentation_id at
the click, < --pseudo_min_label) and the click's red intensity relative to the volume's assigned clicks.

    python .../click_mahalanobis.py --projects stardist=.../project_config.yaml nerve=... \
        --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb --brightness .../click_misses_vs_cap.npz --out <dir>
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_FIVE = np.sqrt(5.0)


def objects_in_volume(seg, scale, max_label=None):
    """Labels, means (M, 3) and inverse covariances (M, 3, 3) of every mask, in um."""
    from scipy import ndimage
    labels = np.unique(seg)
    labels = labels[labels > 0]
    if max_label is not None:
        labels = labels[labels < max_label]
    if not len(labels):
        return labels, np.zeros((0, 3)), np.zeros((0, 3, 3))
    idx = np.nonzero(np.isin(seg, labels))
    lab = seg[idx]
    P = np.stack(idx, 1) * scale
    order = np.argsort(lab, kind='stable')
    lab, P = lab[order], P[order]
    bounds = np.searchsorted(lab, labels)
    ends = np.append(bounds[1:], len(lab))
    mu = np.zeros((len(labels), 3))
    inv = np.zeros((len(labels), 3, 3))
    reg = np.diag(scale ** 2 / 12.0)
    for k, (b, e) in enumerate(zip(bounds, ends)):
        q = P[b:e]
        mu[k] = q.mean(0)
        S = np.cov(q.T, bias=True) if len(q) > 1 else np.zeros((3, 3))
        inv[k] = np.linalg.inv(S + reg)
    return labels, mu, inv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--projects', nargs='+', required=True, help='NAME=project_config.yaml')
    ap.add_argument('--gt_nwb', required=True)
    ap.add_argument('--brightness', default=None, help="gt_click_misses_vs_cap.py's npz (inten, ref_level)")
    ap.add_argument('--max_label', type=int, default=1000, help='ignore mask labels >= this (pseudo masks)')
    ap.add_argument('--pseudo_min_label', type=int, default=1000)
    ap.add_argument('--voxel_um', type=float, nargs=3, default=[1.5, 0.3226, 0.3226])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--frames', type=int, default=None)
    ap.add_argument('--kmax', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from sklearn.mixture import GaussianMixture
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    scale = np.asarray(args.voxel_um)
    nwb = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(nwb)
    C = np.stack([a['gz'], a['gx'], a['gy']], 2)
    gs = a['gs']
    nerve_all = np.isfinite(gs) & (gs > 0) & (gs < args.pseudo_min_label)
    rel = None
    if args.brightness:
        b = np.load(args.brightness)
        rel = b['inten'] / b['ref_level'][:, None]
    out = dict(args=vars(args))
    for name, cfg in (p.split('=', 1) for p in args.projects):
        prj = ProjectData.load_final_project_data(cfg, allow_hybrid_loading=True, verbose=0)
        seg_all = prj.raw_segmentation
        T = min(seg_all.shape[0], C.shape[0], args.frames or 10 ** 9)
        N = C.shape[1]
        shape = np.array(seg_all.shape[1:])

        def one(t):
            seg = np.asarray(seg_all[t])
            labels, mu, inv = objects_in_volume(seg, scale, args.max_label)
            dm = np.full(N, np.nan)
            de = np.full(N, np.nan)
            inside = np.zeros(N, bool)
            ok = np.isfinite(C[t]).all(1)
            if not len(labels) or not ok.any():
                return t, dm, de, inside, len(labels)
            p = C[t][ok] * scale
            diff = p[:, None, :] - mu[None]                                  # n, M, 3
            d2 = np.einsum('nmi,mij,nmj->nm', diff, inv, diff)
            k = d2.argmin(1)
            dm[ok] = np.sqrt(d2[np.arange(len(p)), k])
            de[ok] = np.linalg.norm(diff[np.arange(len(p)), k], axis=1)
            vox = np.clip(np.round(C[t][ok]).astype(int), 0, shape - 1)
            inside[ok] = seg[vox[:, 0], vox[:, 1], vox[:, 2]] == labels[k]
            return t, dm, de, inside, len(labels)

        DM, DE = np.full((T, N), np.nan), np.full((T, N), np.nan)
        INS = np.zeros((T, N), bool)
        n_obj = np.zeros(T, int)
        with ThreadPoolExecutor(args.workers) as ex:
            for t, dm, de, ins, n in ex.map(one, range(T)):
                DM[t], DE[t], INS[t], n_obj[t] = dm, de, ins, n
        np.savez(os.path.join(args.out, f'mahalanobis_{name}.npz'), dm=DM, de=DE, inside=INS, n_obj=n_obj)
        m = np.isfinite(DM) & (DM > 0)
        x = np.log10(DM[m])
        nerve = nerve_all[:T][m]
        edges = [0, 0.5, 1, 1.5, 2.24, 3, 4.47, 7, 10, 20, np.inf]
        hist = [dict(bin=f'{lo}-{hi}', n=int(((DM[m] >= lo) & (DM[m] < hi)).sum()),
                     frac_nerve_segmented=round(float(nerve[(DM[m] >= lo) & (DM[m] < hi)].mean()), 3)
                     if ((DM[m] >= lo) & (DM[m] < hi)).any() else None,
                     frac_click_inside_that_mask=round(float(INS[:T][m][(DM[m] >= lo) & (DM[m] < hi)].mean()), 3)
                     if ((DM[m] >= lo) & (DM[m] < hi)).any() else None,
                     **({'brightness_rel_median': round(float(np.nanmedian(rel[:T][m][(DM[m] >= lo) & (DM[m] < hi)])), 3)}
                        if rel is not None and ((DM[m] >= lo) & (DM[m] < hi)).any() else {}))
                for lo, hi in zip(edges[:-1], edges[1:])]
        X = x[:, None]
        fits = [GaussianMixture(k, n_init=3, random_state=0).fit(X) for k in range(1, args.kmax + 1)]
        bic = [round(float(g.bic(X)), 1) for g in fits]
        g = fits[int(np.argmin(bic))]
        lab = g.predict(X)
        comps = []
        for r, c in enumerate(np.argsort(g.means_[:, 0])):
            mm = lab == c
            comps.append(dict(component=r, weight=round(float(g.weights_[c]), 4), n=int(mm.sum()),
                              dm_geo_mean=round(float(10 ** g.means_[c, 0]), 3),
                              log10_sd=round(float(np.sqrt(g.covariances_[c].ravel()[0])), 3),
                              frac_nerve_segmented=round(float(nerve[mm].mean()), 3),
                              **({'brightness_rel_median': round(float(np.nanmedian(rel[:T][m][mm])), 3)}
                                 if rel is not None else {})))
        out[name] = dict(clicks=int(m.sum()), dm_pct_50_90_99=np.percentile(DM[m], [50, 90, 99]).round(2).tolist(),
                         frac_beyond_sqrt5=round(float((DM[m] > _FIVE).mean()), 4),
                         n_beyond_sqrt5=int((DM[m] > _FIVE).sum()),
                         n_beyond_2sqrt5=int((DM[m] > 2 * _FIVE).sum()),
                         frac_inside_mask_among_beyond_sqrt5=round(float(INS[:T][m][DM[m] > _FIVE].mean()), 3),
                         histogram=hist, gmm_bic=bic, gmm_best_k=int(np.argmin(bic)) + 1, gmm_components=comps)
        print(name, json.dumps({k: v for k, v in out[name].items() if k not in ('histogram',)}), flush=True)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(-1.5, 2, 120)
        ax.hist(x, bins=bins, color='0.75', label='all clicks')
        ax.hist(x[nerve], bins=bins, histtype='step', color='C0', label='NeRVE segmented the click')
        ax.hist(x[~nerve], bins=bins, histtype='step', color='C3', label='NeRVE did not')
        ax.axvline(np.log10(_FIVE), color='k', ls='--', lw=0.8, label='ellipsoid surface, sqrt(5)')
        ax.set_yscale('log')
        ax.set_xlabel('log10 Mahalanobis distance, click to nearest object')
        ax.set_title(name)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f'mahalanobis_{name}.png'), dpi=90)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
