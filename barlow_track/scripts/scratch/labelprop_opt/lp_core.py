"""Parameterised re-implementation of the label-propagation tracker, for variant sweeps on cached embeddings.

Every stage of WormClusterTracker.track_using_label_propagation_clusterer is exposed as a function of one
flat variant dict (DEFAULTS). With the defaults each stage calls the library code path itself, so the
baseline variant reproduces the tracker exactly (checked by sweep_labelprop.py --check_baseline); other
values switch to local implementations here. No library file is modified.

Stages and the variant keys they read (a stage's memo key is its keys plus the previous stage's key):
  embed   svd_dim, svd_mode                         TruncatedSVD of the raw embedding (as eval_accuracy)
  graph   k, graph, drop_same_frame, edge_weight,   kNN graph -> (sparse adjacency, edge_index, norm)
          temporal_w, metric
  prop    num_seeds, seed_select, num_layers,       per-seed propagation -> top-k labels / probs
          alpha, class_norm, softmax, tau, top_k,
          batch_seeds
  sync    input_probability_threshold, prob_weight, spectral sync -> consensus (N, K) and perms
          per_run_unique, rounding, refine_iters,
          k_extra, run_weight, drop_worst_frac
  post    min_prob_threshold, temporal_smooth       per-volume Hungarian -> final labels
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))
if not os.path.isdir(os.path.join(_REPO, 'barlow_track')):
    # run from outside the repo (e.g. a scratchpad copy): fall back to the checkout
    _REPO = os.environ.get('BARLOW_TRACK_REPO', '/home/charles/Current_work/repos/barlow_track')
sys.path.insert(0, _REPO)

DEFAULTS = dict(
    # embed
    svd_dim=50, svd_mode='scaled',          # scaled = U*S (eval default); whiten = U; l2 = row-normalised U*S
    svd_center=False,                       # subtract the mean embedding first (PCA; TruncatedSVD does not)
    # graph
    k=20, metric='euclidean',
    knn_impl='nndescent',                   # nndescent (as now, approximate, seeded) | exact (chunked matmul)
    graph='knn',                            # knn (directed, as now) | sym (A + A^T, D^-1/2 A D^-1/2) | mutual
    drop_same_frame=False,                  # drop kNN edges between objects of one volume (cannot-link)
    edge_weight='uniform',                  # uniform (as now) | gauss (self-tuning kernel, sigma = 7th-NN dist)
    temporal_w=0.0,                         # > 0: add mutual-NN edges between adjacent volumes, this weight
    impl='lib',                             # lib: library code where the variant allows; local: always the
                                            # local graph/propagation code (equivalence checks)
    # prop
    num_seeds=25, seed_select='random',     # random (as now) | stratified (one per time bin) | complete
    num_layers=100,
    alpha=None,                             # None: hard clamping (as now); float: label spreading
    class_norm=False,                       # class-mass normalisation of the propagated scores
    frame_norm=False,                       # after each layer, cap each label's mass within a volume at 1
    softmax=False, tau=0.02, top_k=2,
    batch_seeds=1,                          # seeds propagated together in one SpMM (speed only)
    # sync
    input_probability_threshold=0.1,
    prob_weight=False,                      # keep propagated probabilities in the sync input (not 1.0)
    per_run_unique='greedy',                # greedy (as now) | hungarian: one label per volume within a run
    rounding='cpqr', refine_iters=10,
    k_extra=10,                             # consensus labels beyond the largest seed volume's count
    run_weight=None,                        # None | 'agree': weight runs by their agreement with consensus
    drop_worst_frac=0.0,                    # drop this fraction of runs with the lowest agreement
    # post
    min_prob_threshold=0.02,
    temporal_smooth=0.0,                    # > 0: add this times adjacent-volume mutual-NN consensus rows
    reprop_conf=0.0,                        # > 0: re-propagate from consensus labels with confidence >= this
    reprop_layers=30, reprop_w=1.0,         # layers of that propagation; weight of its result vs consensus
)

STAGES = dict(
    embed=('svd_dim', 'svd_mode', 'svd_center'),
    graph=('k', 'metric', 'knn_impl', 'graph', 'drop_same_frame', 'edge_weight', 'temporal_w', 'impl'),
    prop=('num_seeds', 'seed_select', 'num_layers', 'alpha', 'class_norm', 'frame_norm', 'softmax', 'tau',
          'top_k', 'batch_seeds'),
    sync=('input_probability_threshold', 'prob_weight', 'per_run_unique', 'rounding', 'refine_iters',
          'k_extra', 'run_weight', 'drop_worst_frac'),
    post=('min_prob_threshold', 'temporal_smooth', 'reprop_conf', 'reprop_layers', 'reprop_w'),
)


def full_variant(v):
    unknown = set(v) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown variant keys {sorted(unknown)}")
    return {**DEFAULTS, **v}


def stage_key(v, stage, seed):
    """Memo key of a stage: the variant values of this and every earlier stage, plus the RNG seed."""
    order = list(STAGES)
    keys = [k for s in order[:order.index(stage) + 1] for k in STAGES[s]]
    return (seed,) + tuple((k, v[k]) for k in keys)


def _is_default(v, stage):
    return all(v[k] == DEFAULTS[k] for k in STAGES[stage])


# ---------------------------------------------------------------- embed

def embed(X, v, seed):
    """eval_accuracy's TruncatedSVD (random_state = eval seed), plus whitening / L2 variants."""
    from sklearn.decomposition import TruncatedSVD
    if v['svd_center']:
        X = X - X.mean(axis=0, keepdims=True)
    if v['svd_dim'] in (None, 0):
        Xs = np.asarray(X, dtype=np.float64)
        sv = None
    else:
        svd = TruncatedSVD(n_components=min(int(v['svd_dim']), X.shape[1] - 1), random_state=seed)
        Xs = svd.fit_transform(X)
        sv = svd.singular_values_
    if v['svd_mode'] == 'whiten':
        Xs = Xs / sv
    elif v['svd_mode'] == 'l2':
        Xs = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)
    elif v['svd_mode'] != 'scaled':
        raise ValueError(f"svd_mode {v['svd_mode']!r}")
    return Xs


# ---------------------------------------------------------------- graph

def frame_of(time_to_lin, n):
    t = np.full(n, -1, dtype=np.int64)
    for k, idx in time_to_lin.items():
        t[np.asarray(list(idx), dtype=np.int64)] = k
    return t


def adjacent_mnn_pairs(Xs, time_to_lin, ratio=None):
    """Mutual nearest neighbours between consecutive volumes (sorted time keys): (i, j) index arrays."""
    keys = sorted(time_to_lin)
    I, J = [], []
    for a, b in zip(keys[:-1], keys[1:]):
        ia = np.asarray(list(time_to_lin[a]), dtype=np.int64)
        ib = np.asarray(list(time_to_lin[b]), dtype=np.int64)
        if len(ia) == 0 or len(ib) == 0:
            continue
        d = ((Xs[ia, None, :] - Xs[None, ib, :]) ** 2).sum(-1)
        ab = d.argmin(1)
        ba = d.argmin(0)
        m = np.flatnonzero(ba[ab] == np.arange(len(ia)))
        I.append(ia[m])
        J.append(ib[ab[m]])
    if not I:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(I), np.concatenate(J)


def build_graph(Xs, time_to_lin, v, seed, device=None, gt_obj=None):
    """Returns dict(adj=torch CSR on device, edge_index, norm, info) for propagation.

    Default (knn / uniform / no extras): the library's build_knn_graph + normalized_adjacency +
    sparse_adjacency, i.e. exactly what multi_seed_propagation builds.
    """
    import torch
    from barlow_track.utils import utils_label_propagation as ulp
    t0 = time.time()
    n = Xs.shape[0]
    info = {}
    if _is_default(v, 'graph'):
        edge_index = ulp.build_knn_graph(Xs, k=v['k'], random_state=seed)
        nbrs = edge_index[1].numpy().reshape(n, -1)
        info.update(_graph_stats(nbrs, time_to_lin, gt_obj))
        if device is not None:
            edge_index = edge_index.to(device)
        ei, norm = ulp.normalized_adjacency(edge_index)
        adj = ulp.sparse_adjacency(ei, norm, n)
        info['graph_s'] = round(time.time() - t0, 1)
        return dict(adj=adj, edge_index=ei, norm=norm, info=info)

    k = int(v['k'])
    extra = 2 * k if v['drop_same_frame'] else 0
    if v['knn_impl'] == 'exact':
        nbrs, dists = exact_knn(Xs, k + extra, metric=v['metric'], device=device)
    else:
        from pynndescent import NNDescent
        index = NNDescent(Xs, n_neighbors=k + extra, metric=v['metric'], random_state=seed)
        nbrs, dists = index.neighbor_graph
    nbrs = np.asarray(nbrs, dtype=np.int64)
    dists = np.asarray(dists, dtype=np.float64)
    if v['drop_same_frame']:
        fr = frame_of(time_to_lin, n)
        same = (fr[nbrs] == fr[:, None]) & (nbrs != np.arange(n)[:, None])
        # stable: keep the k nearest allowed neighbours of each row (self stays allowed)
        order = np.argsort(same, axis=1, kind='stable')[:, :k]
        nbrs = np.take_along_axis(nbrs, order, 1)
        dists = np.take_along_axis(dists, order, 1)
    info.update(_graph_stats(nbrs, time_to_lin, gt_obj))
    rows = np.repeat(np.arange(n), nbrs.shape[1])
    cols = nbrs.reshape(-1)
    d = dists.reshape(-1)
    if v['edge_weight'] == 'gauss':
        sig = np.maximum(dists[:, min(7, dists.shape[1] - 1)], 1e-12)
        w = np.exp(-d ** 2 / (sig[rows] * sig[cols]))
    elif v['edge_weight'] == 'uniform':
        w = np.ones_like(d)
    else:
        raise ValueError(f"edge_weight {v['edge_weight']!r}")
    import scipy.sparse as sp
    W = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    if v['graph'] == 'sym':
        W = W.maximum(W.T)
    elif v['graph'] == 'mutual':
        M = W.minimum(W.T)
        # keep each node's nearest neighbour so nothing is cut off
        nn1 = nbrs[:, 1] if nbrs.shape[1] > 1 else nbrs[:, 0]
        W = M.maximum(sp.csr_matrix((np.ones(n), (np.arange(n), nn1)), shape=(n, n)))
        W = W.maximum(W.T)
    elif v['graph'] != 'knn':
        raise ValueError(f"graph {v['graph']!r}")
    if v['temporal_w'] > 0:
        I, J = adjacent_mnn_pairs(Xs, time_to_lin)
        T = sp.csr_matrix((np.full(len(I), float(v['temporal_w'])), (I, J)), shape=(n, n))
        W = W + T + T.T
        info['temporal_edges'] = int(len(I))
    W = W + sp.identity(n, format='csr')  # self loop, as add_self_loops
    deg = np.asarray(W.sum(axis=1)).ravel()
    if v['graph'] == 'knn':
        Wn = sp.diags(1.0 / deg) @ W  # row mean, what the library's directed graph amounts to
    else:
        dis = 1.0 / np.sqrt(deg)
        Wn = sp.diags(dis) @ W @ sp.diags(dis)
    Wn = Wn.tocsr()
    adj = torch.sparse_csr_tensor(torch.from_numpy(Wn.indptr.astype(np.int64)),
                                  torch.from_numpy(Wn.indices.astype(np.int64)),
                                  torch.from_numpy(Wn.data.astype(np.float32)), size=(n, n))
    if device is not None:
        adj = adj.to(device)
    info['graph_s'] = round(time.time() - t0, 1)
    info['nnz'] = int(Wn.nnz)
    return dict(adj=adj, edge_index=None, norm=None, info=info)


def exact_knn(Xs, k, metric='euclidean', device=None, chunk=None):
    """Exact k nearest neighbours (self first, as NNDescent's neighbor_graph) by chunked matmul + topk."""
    import torch
    dev = device if device is not None else torch.device('cpu')
    X = torch.as_tensor(np.asarray(Xs, dtype=np.float32), device=dev)
    if metric == 'cosine':
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    elif metric != 'euclidean':
        raise ValueError(f"exact kNN supports euclidean / cosine, not {metric!r}")
    sq = (X * X).sum(1)
    n = X.shape[0]
    if chunk is None:
        chunk = max(256, int(2e8 // max(n, 1)))  # ~0.8 GB of distances per chunk
    I = np.empty((n, k), dtype=np.int64)
    D = np.empty((n, k), dtype=np.float64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d2 = sq[a:b, None] - 2.0 * X[a:b] @ X.T + sq[None, :]
        d2[torch.arange(b - a, device=dev), torch.arange(a, b, device=dev)] = -1.0  # self first
        val, idx = torch.topk(d2, k, dim=1, largest=False)
        I[a:b] = idx.cpu().numpy()
        D[a:b] = val.clamp_min(0).sqrt().cpu().numpy()
    if metric == 'cosine':
        D = D ** 2 / 2.0  # 1 - cos
    return I, D


def _graph_stats(nbrs, time_to_lin, gt_obj=None):
    n = nbrs.shape[0]
    extra = {} if gt_obj is None else dict(edge_purity=edge_purity(nbrs, gt_obj))
    fr = frame_of(time_to_lin, n)
    notself = nbrs != np.arange(n)[:, None]
    same = (fr[nbrs] == fr[:, None]) & notself
    indeg = np.bincount(nbrs[notself], minlength=n)
    return dict(self_in_knn=float((~notself).any(1).mean()), same_frame_edge_frac=float(same.sum() / max(notself.sum(), 1)),
                indeg_p99_over_mean=float(np.percentile(indeg, 99) / max(indeg.mean(), 1e-9)),
                indeg_zero_frac=float((indeg == 0).mean()), **extra)


def edge_purity(graph_nbrs, gt_obj):
    """Fraction of non-self edges between two GT objects that join the same GT neuron."""
    n = graph_nbrs.shape[0]
    rows = np.repeat(np.arange(n), graph_nbrs.shape[1])
    cols = graph_nbrs.reshape(-1)
    ok = (rows != cols) & (gt_obj[rows] >= 0) & (gt_obj[cols] >= 0)
    return float((gt_obj[rows[ok]] == gt_obj[cols[ok]]).mean()) if ok.any() else float('nan')


# ---------------------------------------------------------------- seeds + propagation

def select_seeds(time_to_lin, v, seed, num_frames):
    from barlow_track.utils.utils_tracking import select_seed_times
    from barlow_track.utils.utils_seeding import python_rng
    n = v['num_seeds']
    if n is None:
        n = min(100, int(0.5 * num_frames))
    if v['seed_select'] == 'random':
        return select_seed_times(time_to_lin, num_seeds=n, seed=seed)
    rng = python_rng(seed)
    keys = sorted(time_to_lin)
    if v['seed_select'] == 'stratified':
        bins = np.array_split(np.array(keys), n)
        return [int(rng.choice(list(b))) for b in bins if len(b)]
    if v['seed_select'] == 'complete':
        # volumes with at least the median object count (fewer missing neurons in the seed labels)
        counts = np.array([len(time_to_lin[k]) for k in keys])
        pool = [k for k, c in zip(keys, counts) if c >= np.median(counts)]
        rng.shuffle(pool)
        return pool[:n]
    raise ValueError(f"seed_select {v['seed_select']!r}")


def propagate(graph, time_to_lin, n, seed_times, v, device=None):
    """Per-seed top-k labels and probabilities (lists of numpy arrays), as multi_seed_propagation."""
    import torch
    from barlow_track.utils import utils_label_propagation as ulp
    lib_path = v['impl'] == 'lib' and v['alpha'] is None and not v['class_norm'] and not v['frame_norm'] \
        and int(v['batch_seeds']) == 1 \
        and graph['edge_index'] is not None
    labelings, probabilities = [], []
    if lib_path:
        for t in seed_times:
            y = ulp.make_seed_labels_no_dict(time_to_lin[t], num_timepoints=n)
            pred, probs = ulp.run_label_propagation(graph['edge_index'], y, edge_norm=graph['norm'], device=device,
                                                    adj=graph['adj'], return_top_k=v['top_k'],
                                                    num_layers=v['num_layers'], softmax=v['softmax'], tau=v['tau'])
            labelings.append(pred.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
        return labelings, probabilities
    adj = graph['adj']
    dev = adj.device
    B = max(int(v['batch_seeds']), 1)
    if v['frame_norm']:
        fr = torch.from_numpy(frame_of(time_to_lin, n)).to(dev)
        n_t = int(fr.max().item()) + 1
    for b0 in range(0, len(seed_times), B):
        block = seed_times[b0:b0 + B]
        seeds = [np.asarray(list(time_to_lin[t]), dtype=np.int64) for t in block]
        sizes = [len(s) for s in seeds]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        C = int(offs[-1])
        Y = torch.zeros(n, C, device=dev)
        rows = torch.from_numpy(np.concatenate(seeds)).to(dev)
        cols = torch.arange(C, device=dev)
        Y[rows, cols] = 1.0
        H = Y.clone()
        for _ in range(int(v['num_layers'])):
            H = adj @ H
            if v['frame_norm']:
                # a label cannot be more than one object of a volume: scale each (volume, label) block
                # whose mass exceeds 1 back to 1 (soft cannot-link)
                mass = torch.zeros(n_t, H.shape[1], device=dev).index_add_(0, fr, H)
                H = H / mass.clamp_min(1.0)[fr]
            if v['alpha'] is None:
                # hard clamp: each seed volume's rows back to one-hot within its own column block
                for s, (a, b) in zip(seeds, zip(offs[:-1], offs[1:])):
                    si = torch.from_numpy(s).to(dev)
                    H[si, a:b] = Y[si, a:b]
            else:
                H = float(v['alpha']) * H + (1.0 - float(v['alpha'])) * Y
        for s, (a, b) in zip(seeds, zip(offs[:-1], offs[1:])):
            out = H[:, a:b]
            if v['class_norm']:
                out = out / out.sum(0, keepdim=True).clamp_min(1e-12)
            lab, pr = _topk(out, v, n_seed=len(s))
            labelings.append(lab.cpu().numpy())
            probabilities.append(pr.cpu().numpy())
        del H, Y
    return labelings, probabilities


def _topk(out, v, n_seed):
    """run_label_propagation's tail (top_k > 1 branch)."""
    import torch
    prob_thresh = 1.1 * (1.0 / n_seed) if n_seed > 0 else 1.0
    if v['softmax']:
        probs = torch.softmax(out / v['tau'], dim=-1)
    else:
        rs = out.sum(dim=-1, keepdim=True)
        probs = torch.where(rs > 0, out / rs, torch.zeros_like(out))
    k = min(int(v['top_k']), probs.shape[1])
    tp, tl = torch.topk(probs, k, dim=-1)
    tl = tl + 1
    low = tp < prob_thresh
    tp[low] = 0
    tl[low] = -1
    return tl, tp


# ---------------------------------------------------------------- sync + post

def hungarian_top1_timewise(L, P, time_to_lin):
    """Like greedy_top1_timewise, but an optimal assignment of each volume's objects to labels."""
    from scipy.optimize import linear_sum_assignment
    N = L.shape[0]
    lo = np.full((N, 1), -1, dtype=np.int64)
    po = np.zeros((N, 1), dtype=P.dtype)
    for t, rows in time_to_lin.items():
        rows = np.asarray(list(rows))
        if len(rows) == 0:
            continue
        Lb, Pb = L[rows], P[rows]
        labs = np.unique(Lb[Lb > 0])
        if len(labs) == 0:
            continue
        col = {l: i for i, l in enumerate(labs)}
        W = np.zeros((len(rows), len(labs)))
        for m in range(len(rows)):
            for kk in range(Lb.shape[1]):
                if Lb[m, kk] > 0:
                    W[m, col[Lb[m, kk]]] = max(W[m, col[Lb[m, kk]]], Pb[m, kk] + 1e-9)
        r, c = linear_sum_assignment(-W)
        ok = W[r, c] > 0
        lo[rows[r[ok]], 0] = labs[c[ok]]
        po[rows[r[ok]], 0] = W[r[ok], c[ok]] - 1e-9
    return lo, po


def sync(labelings, probabilities, time_to_lin, v, verbose=False):
    """spectral_sync_from_topk split at the consensus, with the variant hooks. Returns dict(perms, cons, R, K)."""
    import scipy.sparse as sp
    from barlow_track.utils import utils_spectral_relabeling as ss
    t0 = time.time()
    R = len(labelings)
    N = labelings[0].shape[0]
    K = max([L.max() for L in labelings]) + int(v['k_extra'])
    # 1) per-run sparse matrices (same steps as prepare_A_list_from_labelings_probs)
    A_list = []
    for L, P in zip(labelings, probabilities):
        L = np.array(L, dtype=int)
        P = np.nan_to_num(np.array(P, dtype=float), nan=0)
        P[P < v['input_probability_threshold']] = 0.0
        if v['per_run_unique'] == 'greedy':
            L1, P1 = ss.greedy_top1_timewise(L, P, time_to_lin)
        elif v['per_run_unique'] == 'hungarian':
            L1, P1 = hungarian_top1_timewise(L, P, time_to_lin)
        else:
            raise ValueError(f"per_run_unique {v['per_run_unique']!r}")
        mask = L1 != -1
        rows = np.arange(N)[mask.ravel()]
        cols = L1[mask] - 1
        vals = P1[mask].astype(float)
        if not v['prob_weight']:
            rs = np.bincount(rows, weights=vals, minlength=N)
            rs[rs == 0] = 1.0
            vals = vals / rs[rows]
        A_list.append(sp.csr_matrix((vals, (rows, cols)), shape=(N, K)))
    # 2-4) weighting, SVD, rounding, refinement (library blocks, library defaults)
    w = ss.compute_column_weights(A_list, strategy='invsqrt-count')
    S = ss.build_S(A_list, w)
    V = ss.compute_topK_right_singular_vectors(S, K_dim=K, n_oversamples=20, n_iter=2, use_matrix_free=False)
    if v['rounding'] == 'cpqr':
        V = ss.rotate_to_anchors(V)
    perms = [ss.round_block_to_partial_perm(V[r * K:(r + 1) * K, :], null_cost=None) for r in range(R)]
    if v['refine_iters']:
        perms = ss.refine_perms(A_list, perms, K, n_iter=int(v['refine_iters']), verbose=verbose)
    weights = np.ones(R)
    if v['run_weight'] or v['drop_worst_frac']:
        agree = run_agreement(A_list, perms, K)
        if v['run_weight'] == 'agree':
            weights = weights * agree / max(agree.mean(), 1e-9)
        elif v['run_weight'] is not None:
            raise ValueError(f"run_weight {v['run_weight']!r}")
        if v['drop_worst_frac']:
            n_drop = int(round(v['drop_worst_frac'] * R))
            if n_drop:
                weights[np.argsort(agree)[:n_drop]] = 0.0
    cons = ss.aggregate_consensus(A_list, perms, K, normalize_rows=True, doweight_runs=weights)
    return dict(perms=perms, cons=cons, R=float(weights.sum()), K=K, sync_s=round(time.time() - t0, 1))


def run_agreement(A_list, perms, K):
    """Per run: fraction of its labelled objects whose mapped label is the consensus (of all runs) argmax."""
    from barlow_track.utils import utils_spectral_relabeling as ss
    cons = ss.aggregate_consensus(A_list, perms, K)
    top = cons.argmax(1)
    has = cons.max(1) > 0
    out = []
    for A, p in zip(A_list, perms):
        coo = A.tocoo()
        m = p[coo.col]
        ok = has[coo.row] & (coo.data > 0)
        out.append(float(np.mean(m[ok] == top[coo.row[ok]])) if ok.any() else 0.0)
    return np.array(out)


def finalize(sync_out, time_to_lin, v, Xs=None, graph=None):
    """Temporal-uniqueness Hungarian (library), optionally after smoothing the consensus along
    adjacent-volume mutual nearest neighbours, or after re-propagating confident consensus labels
    over the kNN graph. Returns (labels (N,), confidence (N,))."""
    from barlow_track.utils import utils_spectral_relabeling as ss
    cons = sync_out['cons']
    R = sync_out['R']
    if v['reprop_conf'] > 0:
        cons, R = repropagate(cons, R, time_to_lin, v, graph)
    if v['temporal_smooth'] > 0:
        I, J = adjacent_mnn_pairs(Xs, time_to_lin)
        sm = cons.copy()
        np.add.at(sm, I, v['temporal_smooth'] * cons[J])
        np.add.at(sm, J, v['temporal_smooth'] * cons[I])
        cons = sm
    labels, probs = ss.enforce_temporal_uniqueness_hungarian(cons, time_to_lin, R=R,
                                                            min_prob_threshold=v['min_prob_threshold'])
    return labels[:, 0], probs[:, 0]


def repropagate(cons, R, time_to_lin, v, graph):
    """One more propagation, seeded by every object whose consensus label is confident (from all volumes,
    not one seed volume), then mixed into the consensus: cons + reprop_w * R * P, R -> R (1 + reprop_w)."""
    import torch
    from barlow_track.utils import utils_spectral_relabeling as ss
    labels, probs = ss.enforce_temporal_uniqueness_hungarian(cons, time_to_lin, R=R,
                                                            min_prob_threshold=v['min_prob_threshold'])
    labels, probs = labels[:, 0], probs[:, 0]
    seeds = np.flatnonzero((labels > 0) & (probs >= v['reprop_conf']))
    adj = graph['adj']
    dev = adj.device
    N, K = cons.shape
    with torch.no_grad():
        Y = torch.zeros(N, K, device=dev)
        si = torch.from_numpy(seeds).to(dev)
        Y[si, torch.from_numpy(labels[seeds] - 1).to(dev)] = 1.0
        H = Y.clone()
        for _ in range(int(v['reprop_layers'])):
            H = adj @ H
            H[si] = Y[si]
        rs = H.sum(1, keepdim=True)
        P = torch.where(rs > 0, H / rs, torch.zeros_like(H)).cpu().numpy()
    w = float(v['reprop_w'])
    return cons + w * R * P, R * (1.0 + w)
