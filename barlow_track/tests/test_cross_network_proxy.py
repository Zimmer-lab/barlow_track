"""cross_network_proxy: independent-error networks -> the rank-one fit recovers each network's accuracy."""
import numpy as np

from barlow_track.scripts.scratch.gt_free_proxy.cross_network_proxy import (object_labels, pair_agreement,
                                                                            rank_one_accuracy)


def _tracks(true_label, n_t, n_neurons, acc, rng):
    """pred_all (n_t, n_neurons): each object keeps its true track with probability acc, else it swaps
    with another object of the same volume (a wrong but unique label); columns are permuted."""
    table = np.full((n_t, n_neurons), np.nan)
    for t in range(n_t):
        lab = true_label.copy()
        wrong = np.where(rng.random(n_neurons) > acc)[0]
        if len(wrong) > 1:
            lab[wrong] = np.roll(lab[wrong], 1)
        table[t, lab] = np.arange(n_neurons)  # object id = position in the volume
    return table[:, rng.permutation(n_neurons)]


def test_rank_one_recovers_accuracy():
    rng = np.random.default_rng(0)
    n_t, n_neurons = 300, 60
    accs = [0.98, 0.95, 0.90, 0.80]
    objs = [object_labels(_tracks(np.arange(n_neurons), n_t, n_neurons, a, rng)) for a in accs]
    n = len(accs)
    agree = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            agree[i, j] = agree[j, i] = pair_agreement(objs[i], objs[j])['long']
    est = rank_one_accuracy(agree)
    # a swapped pair is 2 errors, so the realized accuracy is a little below the nominal one
    assert np.all(np.abs(est - np.array(accs)) < 0.03), est
    assert np.all(np.diff(est) < 0)


def test_identical_networks_agree_fully_and_missing_objects_count():
    rng = np.random.default_rng(1)
    tab = _tracks(np.arange(20), 50, 20, 1.0, rng)
    a = object_labels(tab)
    assert pair_agreement(a, a) == dict(all=1.0, long=1.0)
    half = tab.copy()
    half[:, :10] = np.nan  # the second network tracks half the objects
    p = pair_agreement(a, object_labels(half))
    assert abs(p['all'] - 0.5) < 1e-9 and abs(p['long'] - 0.5) < 1e-9
    assert np.isnan(rank_one_accuracy(np.array([[np.nan, 0.9], [0.9, np.nan]]))).all()
