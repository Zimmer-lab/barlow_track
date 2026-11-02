"""Regression tests for docs/pr_plan/PR03 (tracking windows, resume artifacts, last frame).

The fixes landed in #140 without tests; each test pins one acceptance criterion of the plan on
tiny synthetic inputs (no project, no GPU).
"""
import os
import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest

from barlow_track.utils import utils_tracking
from barlow_track.utils.utils_tracking import WormClusterTracker


def _tracker(num_frames, per_frame=3, **kw):
    t2l, l2t, i = {}, {}, 0
    for t in range(num_frames):
        t2l[t] = list(range(i, i + per_frame))
        for j in range(per_frame):
            l2t[i + j] = (t, j + 1, j + 1)
        i += per_frame
    X = np.random.default_rng(0).normal(size=(i, 4))
    return WormClusterTracker(X, t2l, linear_ind_to_t_and_seg_id=l2t, verbose=0, **kw)


def test_num_frames_counts_the_last_frame():
    assert _tracker(7).num_frames == 7


@pytest.mark.parametrize('n,w,s', [(100, 20, 10), (100, 20, 7), (101, 20, 10), (20, 20, 5), (50, 10, 10), (37, 10, 9)])
def test_window_starts_cover_every_frame_and_stay_in_range(n, w, s):
    starts = _tracker(n, n_volumes_per_window=w, tracker_stride=s).all_start_volumes
    assert starts == sorted(set(starts)), "starts must be unique and increasing"
    assert starts[0] == 0
    assert starts[-1] == n - w, "last window must end exactly on the last frame"
    assert min(starts) >= 0
    covered = set()
    for st in starts:
        covered.update(range(st, st + w))
    assert covered == set(range(n))


def test_overlapping_windows_combine_includes_the_first_window(monkeypatch):
    tr = _tracker(30, n_volumes_per_window=10, tracker_stride=10)
    windows = []

    def fake_window(start):
        df = pd.DataFrame({('neuron', 'raw_neuron_ind_in_list'): np.arange(30, dtype=float)})
        df.attrs['start'] = start
        windows.append(start)
        return df, None

    monkeypatch.setattr(tr, 'multicluster_single_window', fake_window)
    monkeypatch.setattr(tr, 'build_streaming_clusterer', lambda: pd.DataFrame(
        {('neuron', 'raw_neuron_ind_in_list'): np.arange(30, dtype=float)}))
    monkeypatch.setattr(utils_tracking, 'fill_missing_indices_with_nan', lambda df, expected_max_t=None: (df, None))
    renamed = []
    monkeypatch.setattr(utils_tracking, 'rename_columns_using_matching',
                        lambda g, df, **k: (renamed.append(df.attrs.get('start')) or df, None, None))
    combined_inputs = []
    monkeypatch.setattr(utils_tracking, 'combine_dataframes_using_mode',
                        lambda dfs: combined_inputs.append(len(dfs)) or dfs[0])
    tr.track_using_overlapping_windows()
    assert windows == [0, 10, 20]
    assert renamed == [0, 10, 20], "every window, including the first, is renamed into the global namespace"
    assert combined_inputs == [1 + 3], "global + all windows reach the combination"


def test_label_propagation_with_umap_projection_feeds_the_projection(monkeypatch):
    tr = _tracker(6, seed=0)
    fake = types.ModuleType('umap')

    class UMAP:
        def __init__(self, **kw):
            pass

        def fit_transform(self, X):
            return X[:, :2] * 10.0

    fake.UMAP = UMAP
    monkeypatch.setitem(sys.modules, 'umap', fake)
    seen = {}

    def fake_msp(X, seed_times, t2l, **kw):
        seen['X'] = X
        raise RuntimeError('stop after propagation input')

    monkeypatch.setattr(utils_tracking, 'multi_seed_propagation', fake_msp)
    with pytest.raises(RuntimeError, match='stop after'):
        tr.track_using_label_propagation_clusterer(num_seeds=2, umap_projection=True)
    assert seen['X'].shape[1] == 2 and np.allclose(seen['X'], tr.X_umap)


def test_resume_artifacts_round_trip(tmp_path):
    """save_intermediate_results writes the files the resume branch of
    track_using_barlow_from_config reads (embedding.zarr, time index, linear_ind_to_t_and_seg_id)."""
    import inspect
    import zarr
    from barlow_track.utils import track_using_barlow as tub

    class Cfg:
        def resolve_relative_path(self, p):
            return str(tmp_path / p)

        def pickle_data_in_local_project(self, obj, p):
            full = tmp_path / p
            full.parent.mkdir(parents=True, exist_ok=True)
            with open(full, 'wb') as f:
                pickle.dump(obj, f)

    cfg = Cfg()
    proj = types.SimpleNamespace(project_config=cfg)
    X = np.random.default_rng(1).normal(size=(12, 8)).astype(np.float32)
    t2l = {0: [0, 1, 2], 1: [3, 4, 5], 2: list(range(6, 12))}
    l2t = {i: (t, i + 1, i + 1) for t, idx in t2l.items() for i in idx}
    (tmp_path / 'sub').mkdir()
    tub.save_intermediate_results(X, {}, l2t, cfg, proj, t2l, tracker={'dummy': 1}, tracker_no_svd=None,
                                  subfolder='sub')
    # The names the resume branch reads
    src = inspect.getsource(tub)
    for name in ['embedding.zarr', 'time_index_to_linear_feature_indices.pickle',
                 'linear_ind_to_t_and_seg_id.pickle', 'worm_tracker_barlow.pickle']:
        assert f"'{name}'" in src
        assert os.path.exists(tmp_path / 'sub' / name), name
    np.testing.assert_array_equal(np.array(zarr.open(str(tmp_path / 'sub' / 'embedding.zarr'))), X)
    with open(tmp_path / 'sub' / 'linear_ind_to_t_and_seg_id.pickle', 'rb') as f:
        assert pickle.load(f) == l2t
    with open(tmp_path / 'sub' / 'time_index_to_linear_feature_indices.pickle', 'rb') as f:
        assert pickle.load(f) == t2l
