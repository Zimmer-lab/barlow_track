"""Named settings of the label propagation tracker (WormClusterTracker.track_using_label_propagation_clusterer).

No heavy imports, so the benchmark driver and the sweep can resolve a recipe's settings for their record
bookkeeping without loading the tracker.
"""

# Named label propagation settings, and the SVD dimension the embedding is reduced to before it (the
# callers do the SVD: eval_accuracy and track_using_barlow read svd_dim from here).
#   legacy: the tracker until 2026-11-07 (random seed volumes, uniform kNN edges, greedy per-run
#           uniqueness; 50 SVD components). num_seeds None = min(100, half the volumes); eval_accuracy and
#           the sweeps passed 25.
#   v2:     svd 150 (scaled), 50 seed volumes stratified over the recording, self-tuning Gaussian kNN edge
#           weights, Hungarian per-run uniqueness, no temporal terms. +0.88 accuracy points over legacy on six
#           held-out recordings x 3 seeds (16/18 wins, worst recording +0.11), +0.31 (Leifer StarDist) and
#           +0.86 (Zimmer 1210) over 100 RNG seeds, with a much smaller spread over seeds
#           (research tree: EVD - Confirmation of screened label propagation variants on held-out
#           recordings; harness barlow_track/scripts/scratch/labelprop_opt, variant c_all_gh_s150_plain).
LABELPROP_RECIPES = dict(
    legacy=dict(svd_dim=50, num_seeds=None, seed_select='random', edge_weight='uniform', per_run_unique='greedy'),
    v2=dict(svd_dim=150, num_seeds=50, seed_select='stratified', edge_weight='gauss', per_run_unique='hungarian'),
)
DEFAULT_LABELPROP_RECIPE = 'v2'


def labelprop_recipe(name=None):
    """The settings of a named recipe (None = DEFAULT_LABELPROP_RECIPE), as a new dict."""
    name = DEFAULT_LABELPROP_RECIPE if name is None else name
    if name not in LABELPROP_RECIPES:
        raise ValueError(f"label propagation recipe {name!r}; use one of {sorted(LABELPROP_RECIPES)}")
    return dict(LABELPROP_RECIPES[name])


# eval_accuracy (and through it the sweeps and the benchmark) ran the legacy tracker with 25 seed volumes
EVAL_LEGACY_NUM_SEEDS = 25


def eval_num_seeds(recipe, num_seeds=None):
    """Seed volumes an evaluation uses: num_seeds if given, else 25 for 'legacy' (what eval_accuracy always
    passed) and the recipe's own number otherwise. Recorded in each eval record as num_seeds."""
    if num_seeds is not None:
        return int(num_seeds)
    if recipe == 'legacy':
        return EVAL_LEGACY_NUM_SEEDS
    return int(labelprop_recipe(recipe)['num_seeds'])
