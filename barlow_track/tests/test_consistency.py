"""GT-free label-propagation consistency (utils_consistency) and its use as the shared sweep's objective."""
import numpy as np
import pytest

from barlow_track.utils.utils_consistency import labelprop_consistency, tracker_consistency
from barlow_track.utils.utils_tracking import WormClusterTracker


def _blobs(n_times=12, n_objects=6, dim=8, noise=0.3, seed=7):
    """One cluster per object, repeated over time (noise sets how separable the objects are)."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_objects, dim) * 4
    X, time_to_lin, lin_to_t_seg = [], {}, {}
    i = 0
    for t in range(n_times):
        for o in range(n_objects):
            X.append(centers[o] + rng.randn(dim) * noise)
            time_to_lin.setdefault(t, []).append(i)
            lin_to_t_seg[i] = (t, o, o)
            i += 1
    return np.vstack(X).astype(np.float32), time_to_lin, lin_to_t_seg


def test_agreement_maps_run_labels_through_the_permutation():
    # 4 objects in 2 volumes; consensus labels 1, 2 (= columns 0, 1)
    final = np.array([1, 2, 1, 2])
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    t = np.array([0, 0, 1, 1])
    # run A uses the consensus' own labels; run B swaps them, which its permutation undoes
    run_a = np.array([[1, 0], [2, 0], [1, 0], [2, 0]])
    run_b = np.array([[2, 0], [1, 0], [2, 0], [2, 0]])  # last object disagrees
    perms = [np.array([0, 1]), np.array([1, 0])]
    m = labelprop_consistency([run_a, run_b], perms, final, conf, t, n_frames=2)
    assert m['agree_mean'] == pytest.approx((1.0 + 0.75) / 2)
    assert m['conf_mean'] == pytest.approx(0.75)
    assert m['assigned_frac'] == 1.0 and m['dup_frac'] == 0.0
    assert m['complete_tracks_90'] == pytest.approx(1.0)  # both labels in both volumes, 2 objects/volume
    # unlabelled run objects (0) and unmapped columns (-1) never agree
    run_c = np.array([[0, 0], [2, 0], [1, 0], [2, 0]])
    m = labelprop_consistency([run_c], [np.array([0, -1])], final, conf, t, n_frames=2)
    assert m['agree_mean'] == pytest.approx(0.25)


def test_lin_to_t_may_be_the_trackers_mapping():
    final = np.array([1, 2, 1, -1])
    t = {0: (0, 5, 5), 1: (0, 6, 6), 2: (1, 5, 5), 3: (1, 7, 7)}
    m = labelprop_consistency([np.array([[1], [2], [1], [0]])], [np.array([0, 1])], final,
                              np.array([0.5, 0.5, 0.5, 0.0]), t, n_frames=2)
    assert m['assigned_frac'] == 0.75 and m['agree_mean'] == 1.0


def _consistency(noise):
    X, time_to_lin, lin_to_t_seg = _blobs(noise=noise)
    tracker = WormClusterTracker(X, dict(time_to_lin), linear_ind_to_t_and_seg_id=lin_to_t_seg, seed=0)
    assert tracker_consistency(tracker, lin_to_t_seg, 12) is None  # nothing tracked yet
    tracker.track_using_label_propagation_clusterer(num_seeds=6, num_neighbors=5, device='cpu')
    m = tracker_consistency(tracker, lin_to_t_seg, 12)
    assert m['sync'] == 'cpqr_refine10'
    return m


def test_separable_objects_track_consistently_and_overlapping_ones_do_not():
    # Clean blobs give ~0.88 (0.72 with the historical independent rounding), not 1: the blobs are
    # equal-sized, so the spectral sync's singular values tie and some runs stay misaligned. This is a
    # property of the synthetic data, not of the metric (Samuel 137 trial 63, historical rounding: 0.643
    # through the perms vs 0.664 for the best per-run Hungarian matching).
    clean, noisy = _consistency(0.3), _consistency(6.0)
    assert clean['n_runs'] == 6 and clean['n_objects'] == 72
    assert clean['agree_mean'] > 0.8
    assert noisy['agree_mean'] < clean['agree_mean'] - 0.2


def test_shared_sweep_regrets_the_chosen_metric():
    from barlow_track.scripts.optimize_shared_config import metric_value, regret_objective
    rec = dict(accuracy=0.9, consistency=dict(agree_mean=0.6, conf_mean=float('nan')))
    assert metric_value(rec, 'accuracy') == 0.9
    assert metric_value(rec, 'agree_mean') == 0.6
    assert metric_value(rec, 'conf_mean') is None  # NaN fails the cell, never reaches Ax
    assert metric_value(dict(accuracy=0.9, consistency=None), 'agree_mean') is None
    assert metric_value(None, 'accuracy') is None
    ref = dict(a=0.7, b=0.8)
    assert regret_objective(dict(a=0.6, b=0.8), ref, ['a', 'b'], q=0.0) == pytest.approx(-0.1)
    assert regret_objective(dict(a=0.6), ref, ['a', 'b']) is None


# ------------------------------------------------ spectral sync: rounding and refinement options
def _synthetic_runs(n_frames=20, n_obj=8, R=6, flip=0.1, seed=0):
    """R runs labelling the same objects with permuted labels; a fraction `flip` of labels is wrong."""
    rng = np.random.RandomState(seed)
    obj = np.tile(np.arange(n_obj), n_frames)
    t2l = {t: list(range(t * n_obj, (t + 1) * n_obj)) for t in range(n_frames)}
    labs, probs, true_perms = [], [], []
    for _ in range(R):
        perm = rng.permutation(n_obj)
        lab = perm[obj] + 1
        wrong = rng.rand(len(obj)) < flip
        lab[wrong] = rng.randint(1, n_obj + 1, wrong.sum())
        labs.append(np.stack([lab, np.full(len(lab), -1)], axis=1))
        probs.append(np.stack([np.ones(len(lab)), np.zeros(len(lab))], axis=1))
        true_perms.append(perm)
    return labs, probs, t2l, obj


def _purity(final, obj):
    return np.mean([np.mean(final[obj == o] == np.bincount(final[obj == o][final[obj == o] > 0]).argmax())
                    for o in np.unique(obj)])


def test_vectorised_consensus_matches_the_historical_loop():
    import scipy.sparse as sp
    from barlow_track.utils.utils_spectral_relabeling import aggregate_consensus
    rng = np.random.RandomState(0)
    N, K = 300, 20
    A_list = [sp.csr_matrix((rng.rand(250), (rng.choice(N, 250, replace=False), rng.randint(0, K, 250))),
                            shape=(N, K)) for _ in range(4)]
    perms = [np.where(rng.rand(K) < 0.2, -1, rng.permutation(K)) for _ in range(4)]
    expected = np.zeros((N, K))
    for A, perm in zip(A_list, perms):
        c = A.tocoo()
        for i, j, v in zip(c.row, c.col, c.data):
            if perm[j] >= 0:
                expected[i, perm[j]] += v
    np.testing.assert_allclose(aggregate_consensus(A_list, perms, K), expected)


def test_anchor_rotation_does_not_depend_on_the_svd_basis():
    from scipy.stats import ortho_group
    from barlow_track.utils.utils_spectral_relabeling import rotate_to_anchors, round_block_to_partial_perm
    rng = np.random.RandomState(1)
    V = np.linalg.qr(rng.randn(60, 6))[0]
    Q = ortho_group.rvs(6, random_state=2)
    a, b = rotate_to_anchors(V), rotate_to_anchors(V @ Q)
    # same subspace, any basis -> the same rows up to one fixed signed permutation of the columns
    for blk in range(10):
        pa = round_block_to_partial_perm(a[blk * 6:(blk + 1) * 6])
        pb = round_block_to_partial_perm(b[blk * 6:(blk + 1) * 6])
        if blk == 0:
            col_map = dict(zip(pa, pb))
        assert [col_map.get(x, x) for x in pa] == list(pb)


@pytest.mark.parametrize('kw', [dict(refine_iters=10), dict(rounding='cpqr', refine_iters=10)])
def test_refinement_tracks_noisy_runs_at_least_as_well(kw):
    from barlow_track.utils.utils_spectral_relabeling import spectral_sync_from_topk
    labs, probs, t2l, obj = _synthetic_runs(flip=0.25)
    base = spectral_sync_from_topk(labs, probs, t2l, verbose=False, rounding='independent',
                                   refine_iters=0)[1][:, 0]
    new = spectral_sync_from_topk(labs, probs, t2l, verbose=False, **kw)[1][:, 0]
    assert _purity(new, obj) >= _purity(base, obj) - 1e-9
    assert _purity(new, obj) > 0.9


def test_default_sync_is_cpqr_with_refinement():
    from barlow_track.utils.utils_spectral_relabeling import spectral_sync_from_topk
    labs, probs, t2l, _ = _synthetic_runs()
    a = spectral_sync_from_topk(labs, probs, t2l, verbose=False)
    b = spectral_sync_from_topk(labs, probs, t2l, verbose=False, rounding='cpqr', refine_iters=10)
    assert all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
    np.testing.assert_array_equal(a[1], b[1])
    with pytest.raises(ValueError):
        spectral_sync_from_topk(labs, probs, t2l, verbose=False, rounding='nope')
