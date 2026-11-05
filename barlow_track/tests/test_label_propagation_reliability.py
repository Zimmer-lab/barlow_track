"""Regression tests for docs/pr_plan/PR02 (label propagation and spectral relabeling).

The fixes landed in #139 without tests; each test here pins one of the plan's acceptance
criteria on a tiny synthetic graph, so a refactor cannot silently bring the bug back.
"""
import numpy as np
import pytest
import torch

from barlow_track.utils import utils_label_propagation as lp
from barlow_track.utils.utils_spectral_relabeling import (enforce_temporal_uniqueness_hungarian,
                                                          greedy_top1_timewise)


def _edges(pairs, n):
    """Undirected edge_index from (i, j) pairs; nodes not in any pair are disconnected."""
    rows, cols = [], []
    for i, j in pairs:
        rows += [i, j]
        cols += [j, i]
    return torch.tensor([rows, cols], dtype=torch.long)


# --- clamped / run_label_propagation ---------------------------------------------------

def test_disconnected_nodes_never_get_ghost_label_zero():
    # Two seeded components (0-1, 2-3) and two isolated nodes (4, 5)
    edge_index = _edges([(0, 1), (2, 3)], 6)
    y = torch.tensor([1, -1, 2, -1, -1, -1])
    pred, probs = lp.run_label_propagation(edge_index, y, prob_thresh=0.0)
    assert 0 not in pred.tolist()
    assert pred[1].item() == 1 and pred[3].item() == 2
    labels_k, _ = lp.run_label_propagation(edge_index, y, return_top_k=2, prob_thresh=0.0)
    assert 0 not in labels_k.flatten().tolist()


def test_low_confidence_nodes_are_minus_one_when_thresholded():
    # Node 2 sits exactly between seeds 1 and 2: 50/50 after softmax-free propagation is
    # impossible to break, so its max prob is the lowest; a threshold above it must give -1.
    edge_index = _edges([(0, 2), (1, 2)], 3)
    y = torch.tensor([1, 2, -1])
    pred, probs = lp.run_label_propagation(edge_index, y, prob_thresh=0.0)
    assert pred[2].item() in (1, 2)
    pred_t, _ = lp.run_label_propagation(edge_index, y, prob_thresh=float(probs[2]) + 1e-6)
    assert pred_t[2].item() == -1
    # Seeds keep their labels (their probability is the clamped one-hot's softmax max)
    assert pred_t[0].item() == 1 and pred_t[1].item() == 2


def test_no_seeds_returns_all_minus_one_with_consistent_shapes():
    edge_index = _edges([(0, 1)], 3)
    y = torch.full((3,), -1, dtype=torch.long)
    pred, probs = lp.run_label_propagation(edge_index, y)
    assert pred.tolist() == [-1, -1, -1] and probs.shape == (3,)
    pred_k, probs_k = lp.run_label_propagation(edge_index, y, return_top_k=2)
    assert pred_k.shape == (3, 2) and probs_k.shape == (3, 2) and (pred_k == -1).all()


def test_seed_labels_are_one_based_and_skip_out_of_range_indices():
    y = lp.make_seed_labels_no_dict([3, 0, 7], num_timepoints=5)
    # index 7 is out of range: stops there, as before (the loop breaks)
    assert y.tolist() == [2, -1, -1, 1, -1]


def test_single_definition_of_run_label_propagation():
    import inspect
    src = inspect.getsource(lp)
    assert src.count('\ndef run_label_propagation(') == 1


# --- align_pair / align_all / fuse_labels_per_time ---------------------------------------

def test_zero_evidence_new_label_gets_a_fresh_id():
    ref = np.array([1, 1, 2, 2, -1, -1])
    # new label 7 overlaps ref 1; new label 9 overlaps nothing labeled in ref, only -1 positions
    new = np.array([7, 7, -1, -1, 9, 9])
    aligned, mapping = lp.align_pair(ref, new)
    assert mapping[7] == 1
    assert mapping[9] not in (1, 2)
    assert aligned.tolist()[:2] == [1, 1] and aligned[4] == aligned[5] == mapping[9]


def test_hungarian_does_not_force_zero_overlap_matches():
    # ref labels 1, 2; new labels 5 (overlaps 1) and 6 (co-occurs with ref only where ref is
    # also labeled, but with 0 overlap with ref 2 because they never share a node)
    ref = np.array([1, 1, 2, 2, -1])
    new = np.array([5, 5, -1, -1, 6])
    _, mapping = lp.align_pair(ref, new)
    assert mapping[5] == 1
    assert mapping[6] != 2  # must not steal label 2 with zero evidence


def test_fresh_ids_avoid_noncontiguous_reference_labels():
    ref = np.array([1, 5, -1, -1])
    new = np.array([3, 4, 8, 9])
    aligned, mapping = lp.align_pair(ref, new)
    fresh = {mapping[8], mapping[9]}
    assert len(fresh) == 2 and not fresh & {1, 5}


def test_align_all_empty_and_single_labelings():
    out = lp.align_all([], {0: [0]})
    assert len(out) == 3 and out[0] == []
    lab = np.array([1, -1, 2])
    aligned, ref, conf = lp.align_all([lab], {0: [0, 1, 2]})
    assert len(aligned) == 1
    assert ref.tolist() == [1, -1, 2]
    assert conf.tolist() == [1.0, 0.0, 1.0]


def test_fuse_labels_handles_a_time_point_with_only_minus_one_votes():
    labelings = [np.array([1, 2, -1, -1]), np.array([1, 2, -1, -1])]
    t2l = {0: [0, 1], 1: [2, 3], 2: []}
    final, conf = lp.fuse_labels_per_time(labelings, t2l)
    assert final.tolist() == [1, 2, -1, -1]
    assert conf.tolist() == [1.0, 1.0, 0.0, 0.0]


# --- spectral relabeling -----------------------------------------------------------------

def test_greedy_falls_back_to_top2_when_top1_is_taken():
    # Two objects at one time both prefer label 3; the weaker one should get its top-2 (4)
    labels = np.array([[3, 4], [3, 4]])
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    out, p = greedy_top1_timewise(labels, probs, {0: [0, 1]})
    assert out[:, 0].tolist() == [3, 4]
    assert p[1, 0] == pytest.approx(0.4)


def test_single_object_time_point_is_labeled():
    R = 4
    consensus = np.array([[0.0, 3.0, 0.0],   # t0: single object, label 2 with 3/4 of runs
                          [4.0, 0.0, 0.0],   # t1: two objects
                          [0.0, 0.0, 4.0]])
    labels, probs = enforce_temporal_uniqueness_hungarian(consensus, {0: [0], 1: [1, 2]}, R)[:2]
    assert labels[:, 0].tolist() == [2, 1, 3]
    assert probs[0, 0] == pytest.approx(0.75)


def test_null_option_is_scaled_by_number_of_runs():
    # Three objects, one label: two objects must stay unlabeled even with many runs (R large),
    # and the strongest object takes the label.
    R = 50
    consensus = np.array([[10.0], [40.0], [5.0]])
    labels, _ = enforce_temporal_uniqueness_hungarian(consensus, {0: [0, 1, 2]}, R)[:2]
    assert labels[:, 0].tolist() == [-1, 1, -1]


def test_cuda_oom_falls_back_to_cpu_with_the_same_labels(monkeypatch):
    """A CUDA OOM mid-propagation redoes every seed on CPU instead of failing the eval."""
    rng = np.random.default_rng(0)
    T, n = 6, 5
    X = np.concatenate([rng.normal(i * 10, 0.1, size=(T, 4)) for i in range(n)]).astype(np.float32)
    # object i of frame t is row i * T + t
    t2l = {t: [i * T + t for i in range(n)] for t in range(T)}
    kw = dict(k=4, random_state=0, return_top_k=1, num_layers=10, softmax=False, tau=0.02)
    ref, _ = lp.multi_seed_propagation(X, [0, 3], t2l, device=None, **kw)

    original = lp._propagate_all_seeds
    calls = []

    def fake(edge_index, num_nodes, slices, t2l_, device, **kwargs):
        calls.append(device)
        if device is not None:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return original(edge_index, num_nodes, slices, t2l_, device, **kwargs)

    monkeypatch.setattr(lp, '_propagate_all_seeds', fake)
    got, _ = lp.multi_seed_propagation(X, [0, 3], t2l, device='cuda', **kw)
    assert calls == ['cuda', None]
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(a, b)


# --- CSR adjacency (replaces torch_sparse.spmm's per-edge gather buffer) -------------------

def _spmm_reference(edge_index, y, num_layers):
    """The pre-CSR propagation loop, verbatim, as the reference."""
    from torch_sparse import spmm
    edge_index, norm = lp.normalized_adjacency(edge_index)
    mask = (y != -1)
    num_classes = int(y.max().item())
    Y = torch.zeros(y.size(0), num_classes)
    Y[mask, y[mask] - 1] = 1.0
    H = Y.clone()
    for _ in range(num_layers):
        H = spmm(edge_index, norm, H.size(0), H.size(0), H)
        H[mask] = Y[mask]
    return H


def test_csr_propagation_matches_spmm_on_a_knn_graph():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8)).astype(np.float32)
    edge_index = lp.build_knn_graph(X, k=10, random_state=0)
    y = torch.full((600,), -1, dtype=torch.long)
    y[rng.choice(600, 40, replace=False)] = torch.arange(1, 41)
    ref = _spmm_reference(edge_index, y, num_layers=30)
    out = lp.clamped_label_propagation(edge_index, y, num_layers=30, DEBUG=False)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-7)
    assert torch.equal(out.argmax(1), ref.argmax(1))


def test_shared_adjacency_gives_the_same_labels_as_per_call():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    edge_index = lp.build_knn_graph(X, k=8, random_state=0)
    ei, norm = lp.normalized_adjacency(edge_index)
    adj = lp.sparse_adjacency(ei, norm, 300)
    y = torch.full((300,), -1, dtype=torch.long)
    y[:20] = torch.arange(1, 21)
    a = lp.run_label_propagation(edge_index, y, return_top_k=2, softmax=False, num_layers=50)
    b = lp.run_label_propagation(ei, y, return_top_k=2, softmax=False, num_layers=50, edge_norm=norm, adj=adj)
    assert torch.equal(a[0], b[0]) and torch.allclose(a[1], b[1])
