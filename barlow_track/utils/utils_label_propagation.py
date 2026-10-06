from pynndescent import NNDescent
from tqdm.auto import tqdm
import torch
from torch_sparse import spmm
import plotly.express as px
import numpy as np
from scipy.optimize import linear_sum_assignment


def build_knn_graph(X, k=20, random_state=None):
    """
    X: np.ndarray (N, d) embeddings
    k: number of neighbors
    random_state: seed for NNDescent's initial random forest / random projections.
        None leaves pynndescent unseeded, i.e. the SAME input can give a
        different graph on every run (pynndescent only needs the seed to be
        fixed; the propagation downstream is deterministic given the graph).
    Returns PyTorch Geometric edge_index
    """
    index = NNDescent(X, n_neighbors=k, metric="euclidean", random_state=random_state)
    neighbors, _ = index.neighbor_graph

    rows, cols = [], []
    for i in range(len(neighbors)):
        for j in neighbors[i]:
            rows.append(i)
            cols.append(j)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index


# def make_seed_labels(time_index_to_linear_feature_indices, slice_t, num_timepoints):
#     """
#     time: np.ndarray (N,)
#     slice_t: which time slice to use as seeds
#     Returns y (torch tensor, N,), with -1 = unlabeled
#     """
#     y = -torch.ones(num_timepoints, dtype=torch.long)
#     mask = time_index_to_linear_feature_indices[slice_t]
#     for i, m in enumerate(mask):
#         if m >= len(y):
#             break
#         y[m] = i + 1 #torch.arange(mask.sum())  # unique cluster IDs per object
#     return y

def make_seed_labels_no_dict(mask, num_timepoints):
    """
    time: np.ndarray (N,)
    slice_t: which time slice to use as seeds
    Returns y (torch tensor, N,), with -1 = unlabeled
    """
    y = -torch.ones(num_timepoints, dtype=torch.long)
    for i, m in enumerate(mask):
        if m >= len(y):
            break
        y[m] = i + 1 #torch.arange(mask.sum())  # unique cluster IDs per object
    return y


def normalized_adjacency(edge_index):
    """Symmetric-normalized adjacency weights, on edge_index's device.

    Split out of clamped_label_propagation so multi-seed runs build it once
    instead of once per seed (identical values; the ops are deterministic).
    """
    from torch_geometric.utils import add_self_loops, degree

    edge_index, _ = add_self_loops(edge_index)
    row, col = edge_index
    deg = degree(row, dtype=torch.float)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    # deg_inv = deg.pow(-1)
    # deg_inv[deg_inv == float('inf')] = 0
    # norm = deg_inv[row]
    return edge_index, norm


def clamped_label_propagation(edge_index, y, num_layers=50, norm=None, DEBUG=True):
    """
    edge_index: graph edges
    y: seed labels (-1 for unlabeled)
    norm: precomputed adjacency weights from normalized_adjacency (same device
        as edge_index/y); rebuilt if None
    """
    if norm is None:
        edge_index, norm = normalized_adjacency(edge_index)

    # seed mask
    # NOTE: seed labels are 1-based (from make_seed_labels_no_dict). Use
    # zero-based columns internally so there is no unused "ghost" class 0
    # that disconnected nodes could argmax to. Column c corresponds to
    # label c+1, so callers must add 1 to argmax/topk indices.
    mask = (y != -1)
    if not bool(mask.any().item()):
        return torch.zeros(y.size(0), 0, device=y.device)
    num_classes = int(y.max().item())
    if num_classes <= 0:
        return torch.zeros(y.size(0), 0, device=y.device)
    Y = torch.zeros(y.size(0), num_classes, device=y.device)
    Y[mask, y[mask] - 1] = 1.0

    H = Y.clone()

    for _ in range(num_layers):
        # propagate: H <- normalized adjacency * H
        H = spmm(edge_index, norm, H.size(0), H.size(0), H)

        # clamp seeds back to one-hot
        H[mask] = Y[mask]
    if DEBUG:
        print(f"Final H shape: {H.shape}")
        print(f"Final H max: {H.max().item()}")
        print(f"Final H sum per row: {H.sum(dim=1)[:10]}")  # First 10 rows

    return H


# def multi_seed_propagation(X, slices, time_index_to_linear_feature_indices, k=20):
#     edge_index = build_knn_graph(X, k=k)
#     labelings = []
#     for t in tqdm(slices, desc="Propagating labels from seed times"):
#         y = make_seed_labels(time_index_to_linear_feature_indices, slice_t=t, num_timepoints=X.shape[0])
#         pred = run_label_propagation(edge_index, y)
#         labelings.append(pred.numpy())
#     return labelings


def run_label_propagation(edge_index, y, num_layers=50, alpha=0.95, return_top_k=1, prob_thresh=None, tau=0.01, softmax=True, DEBUG=False,
                          device=None, edge_norm=None):
    """
    edge_index: graph edges
    y: seed labels (-1 for unlabeled)
    device: torch device for propagation (None = keep inputs where they are, i.e. CPU backup)
    edge_norm: precomputed adjacency weights from normalized_adjacency (same
        device as edge_index); rebuilt per call if None
    """
    if device is not None:
        edge_index = edge_index.to(device)
        y = y.to(device)
    if edge_norm is None:
        edge_index, edge_norm = normalized_adjacency(edge_index)
    mask = (y != -1)  # seeds

    # Threshold purely chance labelings
    if prob_thresh is None:
        n_seeds = int(mask.sum().item())
        prob_thresh = 1.1 * (1.0 / n_seeds) if n_seeds > 0 else 1.0

    if DEBUG:
        print(f"Seeds found: {mask.sum().item()}")
        print(f"Unique labels: {torch.unique(y[mask]).tolist()}" if int(mask.sum().item()) > 0 else "Unique labels: []")
        print("Probability threshold: ", prob_thresh)

    if int(mask.sum().item()) == 0:
        N = y.size(0)
        if return_top_k == 1:
            return torch.full((N,), -1, dtype=torch.long, device=y.device), torch.zeros((N,), device=y.device)
        else:
            return (torch.full((N, return_top_k), -1, dtype=torch.long, device=y.device),
                    torch.zeros((N, return_top_k), device=y.device))

    # lp = LabelPropagation(num_layers=num_layers, alpha=alpha)
    # out = lp(y_filled, edge_index, mask=mask)  # (N, C)
    out = clamped_label_propagation(edge_index, y, num_layers=num_layers, norm=edge_norm, DEBUG=DEBUG)

    if return_top_k == 1:
        probs = torch.softmax(out, dim=-1)
        max_probs, pred_labels = torch.max(probs, dim=-1)

        # out columns are zero-based; seeds are 1-based, so shift to 1-based labels
        pred_labels = pred_labels + 1

        # Set low-confidence predictions to -1 (return the thresholded labels)
        pred_labels[max_probs < prob_thresh] = -1
        return pred_labels, max_probs
    else:
        if softmax:
            probs = torch.softmax(out / tau, dim=-1)
        else:
            # Generates very confident labels
            row_sums = out.sum(dim=-1, keepdims=True)
            probs = torch.where(row_sums > 0, out / row_sums, torch.zeros_like(out))

        top_probs, top_labels = torch.topk(probs, return_top_k, dim=-1)
        # out columns are zero-based; convert to 1-based labels before thresholding
        top_labels = top_labels + 1
        # Threshold small labels
        low_conf = top_probs < prob_thresh
        top_probs[low_conf] = 0
        top_labels[low_conf] = -1

        return top_labels, top_probs  # (N, k), (N, k)


def multi_seed_propagation(X, slices, time_index_to_linear_feature_indices, k=20, device=None,
                           random_state=None, **kwargs):
    """Propagate labels from each seed time; device=None keeps the CPU backup path.

    The kNN graph is built once on CPU; with a device, edge weights move there
    once (adjacency normalization hoisted out of the per-seed loop) and results
    come back as numpy either way.

    random_state is forwarded to the kNN graph builder: the graph is the only
    random step here, so fixing it makes the whole multi-seed run reproducible.
    """
    edge_index = build_knn_graph(X, k=k, random_state=random_state)
    if device is not None:
        edge_index = edge_index.to(device)
    edge_index, edge_norm = normalized_adjacency(edge_index)
    labelings = []
    probabilities = []
    for t in tqdm(slices, desc="Propagating labels from seed times", leave=False):
        y = make_seed_labels_no_dict(time_index_to_linear_feature_indices[t], num_timepoints=X.shape[0])
        pred, probs = run_label_propagation(edge_index, y, edge_norm=edge_norm, device=device, **kwargs)
        labelings.append(pred.detach().cpu().numpy())
        probabilities.append(probs.detach().cpu().numpy())
    return labelings, probabilities



def fuse_labels_per_time(aligned_labelings, time_index_to_linear_feature_indices, DEBUG=False):
    """
    aligned_labelings: list of np.arrays (N,) from different runs
    time_index_to_linear_feature_indices: dict indicating indices of each time point
    time_points: iterable of all unique time points
    Returns:
        final_labels: np.array (N,) final label assignment
        confidence: np.array (N,) number of runs agreeing on each label
    """
    N = len(aligned_labelings[0])
    final_labels = -np.ones(N, dtype=int)
    confidence = np.zeros(N, dtype=float)

    # process each time slice separately
    for t, idx in tqdm(time_index_to_linear_feature_indices.items(), desc="Aligning labels per time point", leave=False):
        # idx = np.where(times == t)[0]  # indices of this time point
        if len(idx) == 0:
            continue

        # Stack votes for this time
        votes = np.stack([y[idx] for y in aligned_labelings], axis=1)  # (M, R)
        num_objects, num_labelings = votes.shape
        if DEBUG:
            print(votes.shape)

        # Nothing to fuse if every vote is -1 (avoids empty vectorize/Hungarian)
        if not np.any(votes != -1):
            continue
        
        # Unique labels across all runs at this time slice
        unique_labels = np.unique(votes[votes != -1])
        num_labels = len(unique_labels)
        if num_labels == 0:
            continue
        label_to_col = {l: i for i, l in enumerate(unique_labels)}
        
        # Flatten object indices and their votes
        obj_idx = np.repeat(np.arange(num_objects), num_labelings)
        lab_vals = votes.ravel()
        
        valid = lab_vals != -1
        obj_idx = obj_idx[valid]
        lab_vals = lab_vals[valid]
        if len(lab_vals) == 0:
            continue
        
        lab_idx = np.vectorize(label_to_col.get, otypes=[np.int64])(lab_vals)
        
        cm = np.zeros((num_objects, num_labels), dtype=int)
        np.add.at(cm, (obj_idx, lab_idx), 1)


        # # Build confusion matrix: objects x labels
        # cm = np.zeros((num_objects, num_labels), dtype=int)
        # for i in range(num_objects):
        #     for j in range(num_labelings):
        #         v = votes[i, j]
        #         if v != -1:
        #             cm[i, label_to_col[v]] += 1

        # Hungarian matching: maximize total votes
        row_ind, col_ind = linear_sum_assignment(-cm)  # negative to maximize
        for i_object, i_label in zip(row_ind, col_ind):
            t_global = idx[i_object]
            final_labels[t_global] = unique_labels[i_label]
            confidence[t_global] = float(cm[i_object, i_label]) / float(num_labelings)
            
            if DEBUG:
                print(i_object, unique_labels[i_label], float(cm[i_object, i_label]) / float(num_labelings), confidence[t_global])
            
        if DEBUG:
            fig = px.imshow(cm)
            fig.show()
            break

    return final_labels, confidence



def align_pair(ref_labels, y_new):
    """
    Align y_new to ref_labels using Hungarian matching, adding new labels only if necessary.
    Both arrays must have the same length (N), -1 indicates unlabeled.
    
    Returns:
        y_new_aligned: np.array of same length as y_new
        mapping: dict mapping original y_new labels to global labels
    """
    if len(ref_labels) != len(y_new):
        raise ValueError(f"ref_labels and y_new must have same length: {len(ref_labels)} vs {len(y_new)}")
    
    # valid positions where both ref and new are labeled
    valid = (ref_labels != -1) & (y_new != -1)
    if not np.any(valid):
        # nothing to match, just assign new IDs for non -1
        y_new_aligned = y_new.copy()
        mapping = {}
        new_label_id = ref_labels.max() + 1 if np.any(ref_labels != -1) else 0
        for lb in np.unique(y_new):
            if lb != -1:
                mapping[lb] = new_label_id
                y_new_aligned[y_new == lb] = new_label_id
                new_label_id += 1
        return y_new_aligned, mapping
    # Flatten object indices and their votes
    # obj_idx = np.repeat(np.arange(num_objects), num_labelings)
    # lab_vals = votes.ravel()
    
    # valid = lab_vals != -1
    # obj_idx = obj_idx[valid]
    # lab_vals = lab_vals[valid]
    
    # lab_idx = np.vectorize(label_to_col.get)(lab_vals)
    
    # cm = np.zeros((num_objects, num_labels), dtype=int)
    # np.add.at(cm, (obj_idx, lab_idx), 1)

    # build confusion matrix for Hungarian matching
    labels_ref = np.unique(ref_labels[valid])
    # Include new labels that never co-occur with a valid reference label
    # (e.g. clusters existing only where the reference is -1); otherwise they
    # would fall through mapping.get(v, -1) and be silently discarded.
    labels_new_cooccurring = np.unique(y_new[valid])
    labels_new_all = np.unique(y_new[y_new != -1])
    labels_new = labels_new_cooccurring
    n_ref = len(labels_ref)
    n_new = len(labels_new)

    # mapping from label to matrix index
    ref_idx = {l: i for i, l in enumerate(labels_ref)}
    new_idx = {l: i for i, l in enumerate(labels_new)}

    cm = np.zeros((n_ref, n_new), dtype=int)
    for i in np.where(valid)[0]:
        r = ref_idx[ref_labels[i]]
        c = new_idx[y_new[i]]
        cm[r, c] += 1

    # Hungarian matching (maximize total votes), but do not force
    # zero-evidence matches: a complete matching may pair a new cluster with
    # an unrelated old label when cm[r, c] == 0. Route those through the
    # unmatched-new path so they get fresh global IDs.
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if cm[r, c] > 0:
            mapping[labels_new[c]] = labels_ref[r]

    # assign new label IDs for unmatched columns (including zero-overlap pairs)
    unmatched_new = set(labels_new) - set(mapping.keys())
    # plus labels that never co-occurred with the reference at all
    unmatched_new |= set(labels_new_all) - set(labels_new)
    new_label_id = ref_labels.max() + 1 if np.any(ref_labels != -1) else 0
    # Keep fresh IDs clear of any existing reference label (ref IDs need not
    # be contiguous).
    existing = set(np.unique(ref_labels[ref_labels != -1]).tolist()) | set(mapping.values())
    while new_label_id in existing:
        new_label_id += 1
    for lb in sorted(unmatched_new):
        mapping[lb] = new_label_id
        existing.add(new_label_id)
        new_label_id += 1
        while new_label_id in existing:
            new_label_id += 1

    # apply mapping
    y_new_aligned = np.array([mapping.get(v, -1) if v != -1 else -1 for v in y_new])

    return y_new_aligned, mapping


def align_all(labelings, time_index_to_linear_feature_indices):
    """
    Align multiple labelings using a rolling reference.
    Assumes all labelings are equal-length arrays (N,), -1 = unlabeled.

    Returns:
        aligned: list of np.arrays, each aligned to rolling reference
        ref_labels: np.array (N,) fused reference labeling
        ref_confidence: np.array (N,) per-node agreement fraction
    """
    if not labelings:
        return [], np.array([], dtype=int), np.array([], dtype=float)

    # Start with the first labeling as reference
    ref_labels = labelings[0].copy()
    all_aligned_labelings = [ref_labels]
    # Default confidence for the single-labeling case (no fusion yet).
    ref_confidence = (ref_labels != -1).astype(float)

    # Rolling alignment
    for y in tqdm(labelings[1:], desc="Aligning all labelings"):
        y_aligned, mapping = align_pair(ref_labels, y)
        all_aligned_labelings.append(y_aligned)

        # Do fusion of the reference with this new aligned labeling and all prior ones, to be used for the next iteration
        ref_labels, ref_confidence = fuse_labels_per_time(all_aligned_labelings, time_index_to_linear_feature_indices)

    return all_aligned_labelings, ref_labels, ref_confidence
