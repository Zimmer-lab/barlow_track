"""Label propagation recipes (utils/labelprop_recipes.py): the library tracker's 'v2' recipe must give exactly
the labels of the harness variant it was validated as (labelprop_opt c_all_gh_s150_plain, research tree:
EVD - Confirmation of screened label propagation variants on held-out recordings), and 'legacy' exactly the
labels of the harness base (the tracker before 2026-11-07)."""
import os
import sys

import numpy as np
import pytest

from barlow_track.utils.labelprop_recipes import LABELPROP_RECIPES, eval_num_seeds, labelprop_recipe
from barlow_track.utils.utils_ground_truth import settings_match
from barlow_track.utils.utils_tracking import WormClusterTracker, select_seed_times

HARNESS = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'scratch', 'labelprop_opt')


def _lp_core():
    sys.path.insert(0, os.path.abspath(HARNESS))
    import lp_core
    return lp_core


def _drifting_blobs(n_times=30, n_objects=10, dim=12, seed=3):
    """Objects that drift slowly through embedding space, some missing in some volumes, plus distractors."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_objects, dim) * 3
    drift = rng.randn(n_objects, dim) * 0.05
    X, time_to_lin, lin_to_t_seg = [], {}, {}
    i = 0
    for t in range(n_times):
        objs = [o for o in range(n_objects) if rng.rand() > 0.1]
        n_extra = rng.randint(0, 3)
        for j, o in enumerate(objs + [-1] * n_extra):
            x = centers[o] + drift[o] * t + rng.randn(dim) * 0.6 if o >= 0 else rng.randn(dim) * 3
            X.append(x)
            time_to_lin.setdefault(t, []).append(i)
            lin_to_t_seg[i] = (t, j, j + 1)
            i += 1
    return np.vstack(X).astype(np.float64), time_to_lin, lin_to_t_seg


def _harness_labels(lp, Xs, t2l, variant, seed):
    v = lp.full_variant(variant)
    graph = lp.build_graph(Xs, t2l, v, seed)
    seeds = lp.select_seeds(t2l, v, seed, len(t2l))
    labs, probs = lp.propagate(graph, t2l, Xs.shape[0], seeds, v)
    syn = lp.sync(labs, probs, t2l, v)
    labels, _ = lp.finalize(syn, t2l, v, Xs=Xs, graph=graph)
    return seeds, labels


def _tracker_labels(Xs, t2l, l2t, seed, **kw):
    tr = WormClusterTracker(Xs, dict(t2l), linear_ind_to_t_and_seg_id=l2t, seed=seed, verbose=0)
    tr.track_using_label_propagation_clusterer(device='cpu', num_neighbors=8, **kw)
    return tr.labelprop_runs


@pytest.mark.parametrize('seed', [0, 1])
def test_v2_recipe_gives_the_harness_variant_it_was_validated_as(seed):
    lp = _lp_core()
    Xs, t2l, l2t = _drifting_blobs()
    # c_all_gh_s150_plain without its SVD (the caller's step), on a graph small enough for the toy data
    variant = dict(num_seeds=8, seed_select='stratified', edge_weight='gauss', per_run_unique='hungarian', k=8)
    seeds, ref = _harness_labels(lp, Xs, t2l, variant, seed)
    runs = _tracker_labels(Xs, t2l, l2t, seed, recipe='v2', num_seeds=8)
    assert runs['settings'] == dict(recipe='v2', num_seeds=8, seed_select='stratified', edge_weight='gauss',
                                    per_run_unique='hungarian')
    np.testing.assert_array_equal(runs['final_labels'], ref)


@pytest.mark.parametrize('seed', [0, 1])
def test_legacy_recipe_gives_the_old_tracker(seed):
    lp = _lp_core()
    Xs, t2l, l2t = _drifting_blobs()
    _, ref = _harness_labels(lp, Xs, t2l, dict(num_seeds=8, k=8), seed)
    runs = _tracker_labels(Xs, t2l, l2t, seed, recipe='legacy', num_seeds=8)
    np.testing.assert_array_equal(runs['final_labels'], ref)


def test_the_tracker_default_is_v2_unless_it_was_embedded_for_another_recipe():
    Xs, t2l, l2t = _drifting_blobs(n_times=12)
    assert _tracker_labels(Xs, t2l, l2t, 0, num_seeds=4)['settings']['recipe'] == 'v2'
    tr = WormClusterTracker(Xs, dict(t2l), linear_ind_to_t_and_seg_id=l2t, seed=0, verbose=0,
                            labelprop_recipe='legacy')
    tr.track_using_label_propagation_clusterer(device='cpu', num_neighbors=8, num_seeds=4)
    assert tr.labelprop_runs['settings']['recipe'] == 'legacy'
    assert tr.labelprop_runs['settings']['edge_weight'] == 'uniform'
    with pytest.raises(ValueError):
        labelprop_recipe('v3')


def test_stratified_seed_times_cover_the_recording():
    t2l = {t: [t] for t in range(100)}
    for seed in range(5):
        times = select_seed_times(t2l, num_seeds=10, seed=seed, mode='stratified')
        assert [t // 10 for t in times] == list(range(10))  # one per tenth of the recording
        assert times == select_seed_times(t2l, num_seeds=10, seed=seed, mode='stratified')
    # more seeds than volumes: every volume once
    assert sorted(select_seed_times({t: [t] for t in range(5)}, num_seeds=8, seed=0, mode='stratified')) == \
        list(range(5))


def test_hungarian_per_run_uniqueness_beats_greedy_where_greedy_blocks_itself():
    from barlow_track.utils.utils_spectral_relabeling import greedy_top1_timewise, hungarian_top1_timewise
    # one volume, two objects: greedy gives label 1 to object 0 (0.9) and leaves object 1 (only label 1, 0.8)
    # without a label; the assignment gives object 0 its second choice
    L = np.array([[1, 2], [1, -1]])
    P = np.array([[0.9, 0.7], [0.8, 0.0]])
    t2l = {0: [0, 1]}
    lg, _ = greedy_top1_timewise(L, P, t2l)
    lh, ph = hungarian_top1_timewise(L, P, t2l)
    assert lg[:, 0].tolist() == [1, -1]
    assert lh[:, 0].tolist() == [2, 1] and ph[:, 0].tolist() == pytest.approx([0.7, 0.8])


def test_eval_seed_volumes_and_record_settings():
    assert eval_num_seeds('legacy') == 25  # what eval_accuracy always passed
    assert eval_num_seeds('v2') == LABELPROP_RECIPES['v2']['num_seeds'] == 50
    assert eval_num_seeds('v2', 10) == 10
    old_record = dict(cluster='labelprop', num_seeds=25)  # written before the field existed
    assert settings_match(old_record, dict(labelprop='legacy', num_seeds=25))
    assert not settings_match(old_record, dict(labelprop='v2'))
    assert not settings_match(dict(old_record, labelprop='v2'), dict(labelprop='legacy'))
