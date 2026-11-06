"""Variant batches for sweep_labelprop.py: each variant is a dict of lp_core.DEFAULTS overrides.

batch1  one-factor-at-a-time screen around the current tracker ('base' = what the sweep runs now).
        Names are the one change; see PLAN.md for the hypothesis behind each.
speed   variants that should not change accuracy (batch_seeds) or trade it for time (fewer layers).
"""

BATCH1 = {
    'base': {},
    # --- embedding / graph
    'whiten': dict(svd_mode='whiten'),            # U instead of U*S (paper features; +0.8 pt on one Flavell net)
    'l2': dict(svd_mode='l2'),                    # cosine-like geometry
    'svd100': dict(svd_dim=100),
    'pca': dict(svd_center=True),                 # centre before the SVD (first component = mean otherwise)
    'exact': dict(knn_impl='exact'),              # exact kNN: no graph randomness, no NNDescent JIT
    'k10': dict(k=10),
    'k40': dict(k=40),
    'sym': dict(graph='sym'),                     # undirected kNN, true symmetric normalisation
    'mutual': dict(graph='mutual'),               # mutual kNN (+ 1-NN), cuts hub edges
    'nosame': dict(drop_same_frame=True),         # cannot-link: no edges inside one volume
    'gauss': dict(edge_weight='gauss'),           # distance-weighted edges
    'temporal1': dict(temporal_w=1.0),            # + adjacent-volume mutual-NN edges
    'temporal5': dict(temporal_w=5.0),
    # --- seeds / propagation
    'layers30': dict(num_layers=30),
    'layers300': dict(num_layers=300),
    'spread99': dict(alpha=0.99),                 # label spreading instead of hard clamping
    'classnorm': dict(class_norm=True),           # class-mass normalisation
    'framenorm': dict(frame_norm=True),           # soft cannot-link inside each volume during propagation
    'seed_strat': dict(seed_select='stratified'),
    'seed_complete': dict(seed_select='complete'),
    'seeds50': dict(num_seeds=50),
    'seeds100': dict(num_seeds=100),             # the paper's count; cheap on GPU. Read err_fragment, not only accuracy
    # --- sync
    'probw': dict(prob_weight=True),              # keep propagated probabilities (not 1.0) in the sync
    'thr0': dict(input_probability_threshold=0.0),
    'thr03': dict(input_probability_threshold=0.3),
    'hungrun': dict(per_run_unique='hungarian'),
    'kextra40': dict(k_extra=40),
    'runw': dict(run_weight='agree'),
    'drop20': dict(drop_worst_frac=0.2),
    # --- post
    'tsmooth05': dict(temporal_smooth=0.5),
    'tsmooth2': dict(temporal_smooth=2.0),
    'reprop09': dict(reprop_conf=0.9),            # re-propagate from confident consensus labels
    'reprop07': dict(reprop_conf=0.7),
    'reprop09_w4': dict(reprop_conf=0.9, reprop_w=4.0),
}

SPEED = {
    'base': {},
    'local': dict(impl='local'),                  # local graph + propagation code; must equal base
    'exact': dict(knn_impl='exact'),
    'batch5': dict(batch_seeds=5),                # same result up to float order; one SpMM per 5 seeds
    'layers50': dict(num_layers=50),
    'layers30': dict(num_layers=30),
}

# Combined candidates for batch 2: fill in from the batch-1 winners (PLAN.md section 5)
BATCH2 = {
    'base': {},
}

BATCHES = dict(batch1=BATCH1, speed=SPEED, batch2=BATCH2)
