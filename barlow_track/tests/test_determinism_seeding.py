"""PR06 tests: one seed makes a run reproducible.

Everything here is synthetic (no test project needed) so the determinism
guarantees are checked even where the real data is unavailable. The claims
under test are the two rules in `barlow_track/utils/utils_seeding.py`:
private RNGs inside the pipeline, and `seed_all` only at entry points.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest barlow_track/tests/test_determinism_seeding.py -q --assert=plain
"""
import numpy as np
import pytest
import random
import torch

from barlow_track.utils.utils_seeding import (
    NUMPY_SEED_BITS,
    TORCH_SEED_BITS,
    derive_seed,
    numpy_rng,
    python_rng,
    seed_all,
)
from barlow_track.utils.utils_label_propagation import build_knn_graph
from barlow_track.utils.utils_tracking import WormClusterTracker, select_seed_times
from barlow_track.utils.volume_data import VolumeCoordsDataModule, VolumeCoordsDataset


# ------------------------------------------------------------- seed derivation
def test_derive_seed_is_stable_and_order_sensitive():
    assert derive_seed(0, 1, 2) == derive_seed(0, 1, 2)
    assert derive_seed(0, 1, 2) != derive_seed(0, 2, 1)
    assert derive_seed(0, 1) != derive_seed(1, 1)
    # Not the naive sum: base+epoch+index collisions would be invisible there
    assert derive_seed(1, 2) != derive_seed(2, 1)


def test_derive_seed_fits_the_requested_bit_width():
    for bits in (8, NUMPY_SEED_BITS, TORCH_SEED_BITS):
        s = derive_seed(7, 'torch', bits=bits)
        assert 0 <= s < 2 ** bits
    with pytest.raises(ValueError):
        derive_seed(1, bits=0)
    with pytest.raises(ValueError):
        derive_seed(1, bits=65)


def test_derive_seed_separates_ints_from_tags():
    # Call sites pass role tags ('torch', 'split', 'py') next to numbers; the
    # encoding must not let the tag and a number collapse onto one seed.
    assert derive_seed(1, 'torch') != derive_seed(1)
    assert derive_seed(1, '2') != derive_seed('1', 2)
    # torch seeds stay in the range torch.manual_seed is happy with
    assert derive_seed(3, 'torch', bits=TORCH_SEED_BITS) < 2 ** 31


def test_seed_all_makes_the_global_streams_reproducible():
    def draw():
        return (torch.randn(4).numpy(), np.random.rand(4), random.random())

    seed_all(1234)
    first = draw()
    seed_all(1234)
    second = draw()
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
    assert first[2] == pytest.approx(second[2])

    seed_all(4321)
    other = draw()
    assert not np.allclose(first[0], other[0])


def test_seed_all_does_not_hand_the_same_seed_to_every_stream():
    # Same integer in every library would make their samples correlated; each
    # stream gets its own derived seed (still a fixed function of --seed).
    seed_all(0)
    torch_draw, numpy_draw = torch.randn(200).numpy(), np.random.rand(200)
    seed_all(0)
    assert np.allclose(torch_draw, torch.randn(200).numpy())
    assert np.allclose(numpy_draw, np.random.rand(200))
    assert not np.array_equal(np.signbit(torch_draw), np.signbit(numpy_draw))


def test_rng_helpers_are_private_and_leave_globals_alone():
    assert python_rng(5).random() == python_rng(5).random()
    assert python_rng(5).random() != python_rng(6).random()
    assert numpy_rng(5).rand(3).tolist() == numpy_rng(5).rand(3).tolist()

    np_state, py_state = np.random.get_state(), random.getstate()
    numpy_rng(5).rand(3)
    python_rng(5).random()
    assert np.all(np.asarray(np_state[1]) == np.asarray(np.random.get_state()[1]))
    assert py_state == random.getstate()
    # None means unseeded, i.e. historical behaviour (never a silent 0)
    assert python_rng(None).random() != python_rng(None).random()


# ------------------------------------------------------- kNN graph / seed times
def _synthetic_embedding(n_times=12, n_objects=6, dim=8, seed=7):
    """Well separated blobs: one cluster per object per timepoint."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_objects, dim) * 4
    X, time_to_lin, lin_to_t_seg = [], {}, {}
    i = 0
    for t in range(n_times):
        for o in range(n_objects):
            X.append(centers[o] + rng.randn(dim) * 0.3)
            time_to_lin.setdefault(t, []).append(i)
            lin_to_t_seg[i] = (t, o, o)
            i += 1
    return np.vstack(X).astype(np.float32), time_to_lin, lin_to_t_seg


def test_knn_graph_is_seeded():
    X, _, _ = _synthetic_embedding()
    same = build_knn_graph(X, k=5, random_state=2)
    again = build_knn_graph(X, k=5, random_state=2)
    assert torch.equal(same, again)
    # random_state=None is the historical (unseeded) path, still accepted
    assert build_knn_graph(X, k=5, random_state=None) is not None
    # NOTE: no "different seed -> different graph" assertion. pynndescent only
    # randomizes the search; on well separated blobs every seed recovers the
    # same exact neighbours, which is why the reproducibility claim (same seed
    # => same graph) is the one worth locking in. The tracking test below does
    # check that a different seed can move the final tracks.


def test_select_seed_times_is_reproducible():
    _, time_to_lin, _ = _synthetic_embedding()
    chosen = select_seed_times(time_to_lin, num_seeds=4, seed=0)
    assert select_seed_times(time_to_lin, num_seeds=4, seed=0) == chosen
    assert select_seed_times(time_to_lin, num_seeds=4, seed=1) != chosen
    assert len(chosen) == len(set(chosen)) == 4
    assert set(chosen) <= set(time_to_lin.keys())
    # asking for more seeds than there are times cannot crash or repeat
    all_times = select_seed_times(time_to_lin, num_seeds=10 ** 6, seed=0)
    assert sorted(all_times) == sorted(time_to_lin.keys())


def test_tracker_seeds_umap_without_ignoring_the_config():
    X, time_to_lin, _ = _synthetic_embedding()
    seeded = WormClusterTracker(X, dict(time_to_lin), seed=3)
    assert seeded.opt_umap['random_state'] == 3
    # An explicit random_state in opt_umap is the user's, not ours to overwrite
    explicit = WormClusterTracker(X, dict(time_to_lin), seed=3,
                                  opt_umap=dict(random_state=99))
    assert explicit.opt_umap['random_state'] == 99
    # seed=None keeps the old (unseeded) options
    assert 'random_state' not in WormClusterTracker(X, dict(time_to_lin)).opt_umap


def test_label_propagation_tracking_is_reproducible():
    X, time_to_lin, lin_to_t_seg = _synthetic_embedding()
    kwargs = dict(num_seeds=4, num_neighbors=5)

    def run(seed):
        tracker = WormClusterTracker(X, dict(time_to_lin),
                                    linear_ind_to_t_and_seg_id=lin_to_t_seg, seed=seed)
        return tracker.track_using_label_propagation_clusterer(**kwargs)

    first, second = run(0), run(0)
    assert first.equals(second)
    # Same data, different seed -> the seed times/graph change, so results move
    assert not first.equals(run(1))


# ------------------------------------------------ volume augmentation pipeline
COLS = ['z', 'x', 'y', 'raw_segmentation_id']


class _FakeSegMeta:
    def __init__(self, points_by_time):
        self.points_by_time = points_by_time

    def get_all_neuron_metadata_for_single_time(self, t, as_dataframe=False):
        pts = self.points_by_time[t]
        # wbfm returns columns as a list of lists, matching column_names order
        return [list(pts[:, i]) for i in range(3)] + [list(range(len(pts)))], COLS


class _FakeProject:
    """Minimal stand-in for ProjectData: small volumes + centroid metadata.

    Module level on purpose: DataLoader workers pickle the dataset, and with
    num_workers>0 the project object travels with it.
    """

    def __init__(self, n_frames=3, shape=(40, 40, 40), n_objects=6):
        rng = np.random.RandomState(3)
        self.red_data = rng.rand(n_frames, *shape).astype(np.float32)
        self.num_frames = n_frames
        self.segmentation_metadata = _FakeSegMeta(
            {t: rng.uniform(6, 34, size=(n_objects, 3)) for t in range(n_frames)})


_AUG = dict(target_sz=(4, 16, 16),
            global_args=dict(max_degrees_z=180.0, p_flip=0.5),
            position_args=dict(jitter_std=1.0, dropout_p=0.4),
            photometric_args=dict(p_blur=0.3, p_noise=0.5))


def _items(ds):
    return [ds[i] for i in range(len(ds))]


def _all_equal(items_a, items_b):
    """Every tensor of every item matches (y1, y2, kpts1, kpts2, idx1, idx2)."""
    return (len(items_a) == len(items_b)
            and all(torch.equal(x, y)
                    for item_a, item_b in zip(items_a, items_b)
                    for x, y in zip(item_a, item_b)))


def _dataset(project, seed=0, epoch=0):
    return VolumeCoordsDataset(project, [0, 1, 2], seed=seed, epoch=epoch, **_AUG)


def test_views_are_identical_for_the_same_seed():
    project = _FakeProject()
    assert _all_equal(_items(_dataset(project, seed=0)), _items(_dataset(project, seed=0)))


def test_the_two_views_of_one_item_still_differ():
    # Per-item seeds must not collapse the pair into the same augmentation
    y1, y2, k1, k2 = _dataset(_FakeProject())[0][:4]
    assert not torch.allclose(y1, y2)
    assert not torch.allclose(k1, k2)


def test_augmentation_is_a_function_of_seed_and_epoch():
    project = _FakeProject()
    epoch0 = _items(_dataset(project, seed=0, epoch=0))
    assert _all_equal(epoch0, _items(_dataset(project, seed=0, epoch=0)))
    rotated = _dataset(project, seed=0, epoch=0)
    rotated.set_epoch(1)
    assert not torch.equal(epoch0[0][0], _items(rotated)[0][0])
    assert not torch.equal(epoch0[0][0], _items(_dataset(project, seed=5, epoch=0))[0][0])


def test_augmentation_does_not_depend_on_worker_count():
    # The historical bug: workers each fork the dataset's long-lived RandomState,
    # so num_workers silently changed which augmentation a volume got.
    project = _FakeProject()
    batches = {}
    for num_workers in (0, 2):
        dm = VolumeCoordsDataModule(project_data=project, num_frames=3, batch_size=1,
                                    train_fraction=0.66, val_fraction=0.33,
                                    seed=0, num_workers=num_workers, **_AUG)
        dm.setup()
        batches[num_workers] = list(dm.train_dataloader())
    assert len(batches[0]) == len(batches[2]) == 1
    assert all(torch.equal(a, b) for a, b in zip(batches[0][0], batches[2][0]))


def test_split_is_fixed_by_the_seed():
    project = _FakeProject()

    def split(seed):
        dm = VolumeCoordsDataModule(project_data=project, num_frames=3, batch_size=1,
                                    train_fraction=0.66, val_fraction=0.33, seed=seed, **_AUG)
        dm.setup()
        return (dm.train_dataset.indices, dm.val_dataset.indices, dm.test_dataset.indices)

    assert split(0) == split(0)
    assert split(0) != split(11)
