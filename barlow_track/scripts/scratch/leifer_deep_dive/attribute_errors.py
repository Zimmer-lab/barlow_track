"""Price the GT / segmentation problems of the Leifer re-export A2 in accuracy points.

Reads the outputs of decompose_errors.py (<subset>_cells.npz: G = GT mask label, P = matched
prediction, frame x neuron) and mask_geometry.py (masks.npz), plus the GT clicks from the NWB.
Each GT cell gets one category, first match wins:
  blank     volume without NeRVE masks (only pseudo masks there)
  dup       another GT click within --dup_um (two GT neurons clicked on one cell)
  merged2   its NeRVE mask holds 2+ GT clicks
  sliver    pseudo mask under --sliver_vox voxels (seed fell inside a NeRVE mask)
  big       NeRVE mask over --big_factor x the median NeRVE mask (merged with an unannotated cell)
  pseudo    any other pseudo mask (NeRVE missed the neuron)
  clean     the rest
and prints, per run, cells / errors / accuracy per category, the accuracy with each category
dropped, and what the wrong labels in clean cells were (non-GT detection, other GT mask, miss).
Nothing is refitted and the GT is not changed: this only splits the existing score.

    python barlow_track/scripts/scratch/leifer_deep_dive/attribute_errors.py \
        --cells_dir <dir with all_cells.npz, gt_cells.npz> --masks <dir>/masks.npz
"""
import argparse
import os

import h5py
import numpy as np

NWB = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/leifer_reexport/'
       'Leifer_NeRVE_Worm1_reexport_A2.nwb')
PSEUDO = 1000
UM_XYZ = np.array([0.3226, 0.3226, 1.5])  # NWB click order (x632, y600, z)
UM_ZXY = np.array([1.5, 0.3226, 0.3226])  # masks.npz centroid order
CATS = ['blank', 'dup', 'merged2', 'sliver', 'big', 'pseudo', 'clean']


def load_clicks(nwb, names):
    with h5py.File(nwb, 'r') as f:
        a = f['processing/CalciumActivity/NeuronCentroids']
        return np.stack([a[n]['data'][:] for n in names], 1)  # (T, N, 3)


def mask_table(m):
    key = m['t'].astype(np.int64) * 100000 + m['label']
    o = np.argsort(key)
    return key[o], o


def lookup(ks, o, t, lab):
    """Row of masks.npz for mask `lab` at frame t (arrays), -1 if absent."""
    q = t.astype(np.int64) * 100000 + lab.astype(np.int64)
    i = np.clip(np.searchsorted(ks, q), 0, len(ks) - 1)
    return np.where(ks[i] == q, o[i], -1)


def categorise(G, X, m, dup_um, sliver_vox, big_factor):
    T, N = G.shape
    ks, o = mask_table(m)
    tt, jj = np.nonzero(np.isfinite(G))
    lab = G[tt, jj].astype(int)
    row = lookup(ks, o, tt, lab)
    assert (row >= 0).all(), 'GT label missing from masks.npz'
    vox = m['voxels'][row]
    inside = m['clicks_inside'][row]
    nerve_t = np.bincount(m['t'][m['label'] < PSEUDO], minlength=T) > 0
    med = np.median(m['voxels'][m['label'] < PSEUDO])
    d = np.sqrt((((X[:, :, None] - X[:, None]) * UM_XYZ) ** 2).sum(-1))
    d[:, np.arange(N), np.arange(N)] = np.inf
    nearest = np.nanmin(np.where(np.isnan(d), np.inf, d), 2)[tt, jj]
    pseudo = lab >= PSEUDO
    rules = [~nerve_t[tt], nearest < dup_um, ~pseudo & (inside >= 2), pseudo & (vox < sliver_vox),
             ~pseudo & (vox > big_factor * med), pseudo]
    cat = np.full(len(tt), len(CATS) - 1)
    for k in reversed(range(len(rules))):
        cat[rules[k]] = k
    C = np.full(G.shape, -1)
    C[tt, jj] = cat
    return C


def wrong_kind(G, P, m):
    """Per GT cell: 0 correct, 1 miss, 2 onto another GT neuron's mask, 3 onto a non-GT detection."""
    K = np.full(G.shape, -1)
    for t in range(G.shape[0]):
        g = G[t]
        fin = np.isfinite(g)
        gt_set = set(g[fin].astype(int))
        for j in np.flatnonzero(fin):
            p = P[t, j]
            if p == g[j]:
                K[t, j] = 0
            elif not np.isfinite(p):
                K[t, j] = 1
            elif int(p) in gt_set:
                K[t, j] = 2
            else:
                K[t, j] = 3
    return K


def neighbour_distance(G, P, K, m):
    """um between the true mask and the wrongly chosen mask, for cells of kind 2 / 3."""
    ks, o = mask_table(m)
    tt, jj = np.nonzero(K >= 2)
    a = lookup(ks, o, tt, G[tt, jj])
    b = lookup(ks, o, tt, P[tt, jj])
    ok = (a >= 0) & (b >= 0)
    d = np.sqrt((((m['zxy'][a[ok]] - m['zxy'][b[ok]]) * UM_ZXY) ** 2).sum(1))
    return K[tt[ok], jj[ok]], d


def report(name, G, P, C, m):
    K = wrong_kind(G, P, m)
    fin = K >= 0
    n, err = fin.sum(), (K > 0).sum()
    print(f'\n=== {name}: {n} GT cells, accuracy {1 - err / n:.4f} ({err} errors)')
    print(f'{"category":>9} {"cells":>7} {"errors":>7} {"acc":>6} {"share of errors":>16} {"acc without":>12}')
    for k, c in enumerate(CATS):
        s = C == k
        e = (K[s] > 0).sum()
        rest = fin & ~s
        acc_wo = 1 - (K[rest] > 0).sum() / rest.sum()
        print(f'{c:>9} {s.sum():7d} {e:7d} {1 - e / max(s.sum(), 1):6.3f} {e / max(err, 1):16.3f} {acc_wo:12.4f}')
    clean = C == CATS.index('clean')
    flagged = fin & ~clean
    print(f'accuracy on clean cells only: {1 - (K[clean] > 0).sum() / clean.sum():.4f} '
          f'(all flagged categories dropped, {flagged.sum()} cells)')
    for kind, lbl in [(1, 'miss'), (2, 'onto another GT mask'), (3, 'onto a non-GT detection')]:
        print(f'  clean-cell errors {lbl}: {(K[clean] == kind).sum()}')
    kk, d = neighbour_distance(G, P, K, m)
    for kind, lbl in [(2, 'other GT'), (3, 'non-GT')]:
        if (kk == kind).any():
            print(f'  wrong mask is {lbl}: distance to the true mask um p10/50/90 '
                  f'{np.percentile(d[kk == kind], [10, 50, 90]).round(2)}')
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells_dir', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--nwb', default=NWB)
    ap.add_argument('--dup_um', type=float, default=1.5)
    ap.add_argument('--sliver_vox', type=int, default=30)
    ap.add_argument('--big_factor', type=float, default=2.0)
    args = ap.parse_args()
    m = dict(np.load(args.masks))
    runs = {}
    for s in ['all', 'gt', 'nerve']:
        f = os.path.join(args.cells_dir, f'{s}_cells.npz')
        if os.path.exists(f):
            runs[s] = np.load(f)
    names = list(next(iter(runs.values()))['names'])
    for r in runs.values():
        assert list(r['names']) == names
    G = runs['gt']['G'] if 'gt' in runs else runs['all']['G']
    X = load_clicks(args.nwb, names)
    C = categorise(G, X, m, args.dup_um, args.sliver_vox, args.big_factor)
    Ks = {}
    for s, r in runs.items():
        if s == 'nerve':
            continue  # different GT set (pseudo cells dropped); not comparable cell by cell
        assert np.array_equal(np.isfinite(r['G']), np.isfinite(G))
        Ks[s] = report(s, r['G'], r['P'], C, m)
    if {'all', 'gt'} <= set(Ks):
        a, g = Ks['all'] > 0, Ks['gt'] > 0
        print(f'\ncells wrong in all but right in gt: {(a & ~g).sum()}; wrong in gt but right in all: '
              f'{(~a & g).sum()}; wrong in both: {(a & g).sum()}')


if __name__ == '__main__':
    main()
