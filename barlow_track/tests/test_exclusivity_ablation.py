"""Per-time exclusivity ablation (research TEST_PLAN T3, reviewer R2.3): the label propagation recipes that turn
off the per-run (E2) and final (E3) one-object-per-label-per-volume steps, and HDBSCAN with that exclusivity added.

The baselines must be untouched: v2 is the recorded recipe, and plain HDBSCAN keeps its historical labels."""
import numpy as np
import pytest

from barlow_track.tests.test_labelprop_recipes import _drifting_blobs
from barlow_track.utils.labelprop_recipes import LABELPROP_RECIPES
from barlow_track.utils.utils_spectral_relabeling import (consensus_argmax, enforce_temporal_uniqueness_hungarian,
                                                          spectral_sync_from_topk, top1_no_uniqueness)
from barlow_track.utils.utils_tracking import (WormClusterTracker, keep_most_confident_per_label,
                                               label_duplicate_stats)

T2L = {0: [0, 1], 1: [2, 3]}


def test_per_run_none_keeps_each_objects_top_label_even_when_repeated():
    labels = np.array([[1, 2], [1, 2], [2, -1], [-1, -1]])
    probs = np.array([[.9, .1], [.8, .2], [.7, 0], [0, 0]])
    L, P = top1_no_uniqueness(labels, probs, T2L)
    np.testing.assert_array_equal(L[:, 0], [1, 1, 2, -1])  # objects 0 and 1 share label 1 in volume 0
    np.testing.assert_allclose(P[:, 0], [.9, .8, .7, 0])


def test_consensus_argmax_allows_repeats_and_hungarian_does_not():
    cons = np.array([[2., 1.], [1.5, 0.], [0., 2.], [0., 1.]])  # R = 2 runs
    lab_a, prob_a = consensus_argmax(cons, T2L, R=2)
    np.testing.assert_array_equal(lab_a[:, 0], [1, 1, 2, 2])
    np.testing.assert_allclose(prob_a[:, 0], [1., .75, 1., .5])
    lab_h, _ = enforce_temporal_uniqueness_hungarian(cons, T2L, R=2)
    for idx in T2L.values():
        got = lab_h[idx, 0][lab_h[idx, 0] > 0]
        assert len(got) == len(set(got))


def test_duplicate_stats_and_most_confident_resolution():
    labels = np.array([3, 3, 5, -1])
    conf = np.array([.4, .9, .5, 0])
    stats = label_duplicate_stats(labels, conf, T2L)
    assert stats == dict(n_objects=4, n_labelled=3, n_duplicate_objects=2, duplicate_fraction=2 / 3, n_clusters=2)
    np.testing.assert_array_equal(keep_most_confident_per_label(labels, conf, T2L), [-1, 3, 5, -1])


def test_spectral_sync_rejects_unknown_final_step():
    with pytest.raises(ValueError):
        spectral_sync_from_topk([np.ones((4, 1), int)], [np.ones((4, 1))], T2L, final_unique='greedy')


def _track(recipe, seed=0):
    Xs, t2l, l2t = _drifting_blobs()
    tr = WormClusterTracker(Xs, dict(t2l), linear_ind_to_t_and_seg_id=l2t, seed=seed, verbose=0)
    df = tr.track_using_label_propagation_clusterer(device='cpu', num_neighbors=8, recipe=recipe, num_seeds=8)
    return tr, df


@pytest.mark.parametrize('recipe', ['v2_noE2', 'v2_noE3', 'v2_noexcl'])
def test_ablation_recipes_run_and_record_their_settings(recipe):
    tr, df = _track(recipe)
    s = tr.labelprop_runs['settings']
    assert (s['per_run_unique'], s['final_unique']) == (LABELPROP_RECIPES[recipe]['per_run_unique'],
                                                        LABELPROP_RECIPES[recipe]['final_unique'])
    assert tr.label_duplicates['n_objects'] == len(tr.X_svd)
    # whatever the consensus held, the track dataframe has one object per (volume, track)
    final = tr.labelprop_runs['final_labels']
    assert label_duplicate_stats(keep_most_confident_per_label(final, tr.labelprop_runs['confidence'],
                                                               tr.time_index_to_linear_feature_indices),
                                 tr.labelprop_runs['confidence'],
                                 tr.time_index_to_linear_feature_indices)['n_duplicate_objects'] == 0
    assert df.shape[0] > 0


def test_v2_has_no_repeated_labels():
    tr, _ = _track('v2')
    assert tr.label_duplicates['n_duplicate_objects'] == 0


def _global(**kw):
    Xs, t2l, l2t = _drifting_blobs(n_times=40)
    tr = WormClusterTracker(Xs, dict(t2l), linear_ind_to_t_and_seg_id=l2t, seed=0, verbose=0,
                            opt_db=dict(min_cluster_size=0.3, min_samples=0.1))
    df = tr.track_using_global_clusterer(umap_projection=False, **kw)
    return tr, df


def _labelled_objects(tr, df):
    """Linear indices of the objects the track dataframe holds."""
    lin = {(t, int(r)): i for i, (t, r) in
           ((i, (tr.dict_linear_index_to_time[i], tr.get_raw_neuron_ind_from_linear_ind(i))) for i in range(len(tr.X_svd)))}
    cols = [c for c in df.columns if c[1] == 'raw_neuron_ind_in_list']
    out = set()
    for c in cols:
        for t, v in df[c].dropna().items():
            out.add(lin[(int(t), int(v))])
    return out


def test_plain_hdbscan_keeps_its_hard_labels():
    tr, df = _global()
    hard = tr.global_clusterer.labels_
    assert _labelled_objects(tr, df) <= set(np.flatnonzero(hard >= 0))
    assert tr.label_duplicates['n_labelled'] == int((hard >= 0).sum())


@pytest.mark.parametrize('labels_from', ['hard', 'soft'])
def test_hdbscan_with_exclusivity_has_no_repeats(labels_from):
    tr, df = _global(exclusivity='hungarian', labels_from=labels_from)
    assert tr.label_duplicates['n_duplicate_objects'] == 0
    assert tr.label_duplicates['n_labelled'] > 0
    if labels_from == 'hard':  # noise stays unlabelled: exclusivity only among the objects HDBSCAN labelled
        assert _labelled_objects(tr, df) <= set(np.flatnonzero(tr.global_clusterer.labels_ >= 0))


def test_soft_labels_reach_objects_hdbscan_called_noise():
    tr, df = _global(labels_from='soft', duplicate_rule='confidence')
    noise = set(np.flatnonzero(tr.global_clusterer.labels_ < 0))
    assert noise, 'toy data should have noise objects'
    assert _labelled_objects(tr, df) & noise


def test_global_rejects_unknown_options():
    for kw in (dict(exclusivity='greedy'), dict(labels_from='both'), dict(duplicate_rule='last')):
        with pytest.raises(ValueError):
            _global(**kw)
