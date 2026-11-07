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
    'spread90': dict(alpha=0.90),                 # label spreading: votes discounted ~alpha^hops (horizon ~10
    'spread97': dict(alpha=0.97),                 # / ~33 hops), a smooth version of fewer layers; no clamp
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

# Batch 2 (confirmation, CONFIRM_CASES x seeds 0-2): the six batch-1 promotions singly, one neighbouring
# value of each tunable promotion, and cross-stage combinations (embed x seeds x graph x post).
_C = dict(svd_dim=100, num_seeds=50, seed_select='stratified', temporal_w=5.0, temporal_smooth=0.5)
BATCH2 = {
    'base': {},
    # promoted singly (batch 1: +0.34 to +0.74 pt, >= 9/12 wins)
    'svd100': dict(svd_dim=100),
    'seeds50': dict(num_seeds=50),
    'seeds100': dict(num_seeds=100),
    'seed_strat': dict(seed_select='stratified'),
    'temporal5': dict(temporal_w=5.0),
    'tsmooth05': dict(temporal_smooth=0.5),
    # neighbours
    'svd150': dict(svd_dim=150),
    'temporal2': dict(temporal_w=2.0),
    'temporal10': dict(temporal_w=10.0),
    # combinations
    'c_seeds': dict(num_seeds=50, seed_select='stratified'),
    'c_all': dict(_C),
    'c_all_s100': dict(_C, num_seeds=100),
    'c_all_notemp': dict(_C, temporal_w=0.0),
    'c_all_gh': dict(_C, edge_weight='gauss', per_run_unique='hungarian'),   # + the two near-misses
    'c_all_l30': dict(_C, num_layers=30),                                     # + the speed option
    # added 2026-11-06 22:30 after diag_chain (seed 0, cached consensus: Z1210 +1.6, S137 +0.5, F0330 +0.1 pt
    # at chain_w 0.5; worse above 2); run after the first 16 (resumable)
    'chain05': dict(chain_w=0.5),
    'chain1': dict(chain_w=1.0),
    'c_all_chain': dict(_C, temporal_smooth=0.0, chain_w=0.5),
    'c_all_gh_chain': dict(_C, temporal_smooth=0.0, chain_w=0.5, edge_weight='gauss', per_run_unique='hungarian'),
    # added 2026-11-07 03:00: Ax round 3 best on the dev cases (trial 23, 0.9549 vs c_all_gh 0.9520-0.9530)
    'c_all_gh_s150': dict(_C, svd_dim=150, edge_weight='gauss', per_run_unique='hungarian'),
    # added 2026-11-07 04:50: temporal edges cost the held-out Flavell F0324 0.4 pt (temporal5 -0.42,
    # c_all_gh -0.37, c_all_gh_s150 -0.60) while c_all_notemp is the only combination up on every case
    'c_all_gh_notemp': dict(_C, temporal_w=0.0, edge_weight='gauss', per_run_unique='hungarian'),
    'c_all_gh_s150_notemp': dict(_C, temporal_w=0.0, svd_dim=150, edge_weight='gauss', per_run_unique='hungarian'),
    # added 2026-11-07: the same without the temporal smoothing post-step (no temporal information at all)
    'c_all_gh_s150_plain': dict(_C, temporal_w=0.0, temporal_smooth=0.0, svd_dim=150, edge_weight='gauss',
                                per_run_unique='hungarian'),
}

BATCHES = dict(batch1=BATCH1, speed=SPEED, batch2=BATCH2)
