"""PR01 regression tests: eval-accuracy correctness fixes.

Covers: --max_frames GT truncation, NaN-gap matching, fewer-pred-columns
matching, CPU/GPU normalization equivalence, calculate_accuracy key types,
and build_accuracy_dict atomic (ragged-free) appends.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from barlow_track.utils.utils_ground_truth import (
    align_gt_pred_time_index,
    build_accuracy_dict,
    calculate_accuracy,
    pad_with_nan_rows,
)


def _multiindex_df(n_rows, neurons, col='raw_segmentation_id', values=None):
    columns = pd.MultiIndex.from_product([neurons, [col]])
    if values is None:
        values = np.tile(np.arange(1, len(neurons) + 1, dtype=float), (n_rows, 1))
    return pd.DataFrame(values, columns=columns)


def test_max_frames_truncation_no_late_gt_miss():
    """GT rows beyond the predicted range must not count as misses."""
    from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
    n_total, n_pred = 100, 50
    df_gt = _multiindex_df(n_total, ['n0', 'n1'])
    df_pred = _multiindex_df(n_pred, ['p0', 'p1'])  # identical values, renamed below
    df_gt_a, df_pred_a = align_gt_pred_time_index(df_gt, df_pred)
    assert len(df_gt_a) == n_pred and len(df_pred_a) == n_pred
    assert (df_gt_a.index < n_pred).all()
    max_len = max(len(df_gt_a), len(df_pred_a))
    df_pred_r, _, _, _ = rename_columns_using_matching(
        pad_with_nan_rows(df_gt_a, max_len), pad_with_nan_rows(df_pred_a, max_len),
        column='raw_segmentation_id', try_to_fix_inf=True)
    col_gt = df_gt_a.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
    col_pr = df_pred_r.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
    stats = calculate_accuracy(col_gt, col_pr)
    assert float(stats['accuracy']) == pytest.approx(1.0)
    # Without truncation the same data scores ~n_pred/n_total (the reported bug).
    assert stats['total_ground_truth'] == col_gt.notna().sum().sum() == n_pred * 2


def test_align_disjoint_index_falls_back():
    df_gt = _multiindex_df(10, ['n0'])
    df_pred = _multiindex_df(10, ['n0'])
    df_gt.index = range(100, 110)
    df_pred.index = range(0, 10)
    g, p = align_gt_pred_time_index(df_gt, df_pred)
    assert len(g) == 10 and len(p) == 10  # unchanged, old behavior


def test_matching_with_nan_gaps_completes():
    from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
    df_gt = _multiindex_df(4, ['n0', 'n1'],
                           values=np.array([[1., 2.], [1., 2.], [np.nan, np.nan], [1., 2.]]))
    df_pr = _multiindex_df(4, ['p0', 'p1'],
                           values=np.array([[1., 2.], [np.nan, np.nan], [1., 2.], [1., 2.]]))
    for flag in (False, True):
        df_r, matches, _, _ = rename_columns_using_matching(
            df_gt, df_pr, column='raw_segmentation_id', try_to_fix_inf=flag)
        assert len(matches) == 2


def test_fewer_pred_columns_than_gt_no_crash():
    """Fewer predicted columns than GT must match + score without errors."""
    from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
    df_gt = _multiindex_df(4, ['n0', 'n1'])
    df_pr = _multiindex_df(4, ['p0'])
    df_r, _, _, _ = rename_columns_using_matching(
        df_gt, df_pr, column='raw_segmentation_id', try_to_fix_inf=True)
    if 'unmatched_neuron' in df_r.columns.get_level_values(0):
        df_r = df_r.drop(columns='unmatched_neuron')
    cgt = df_gt.droplevel(1, axis=1)
    cpr = df_r.droplevel(1, axis=1)
    stats = calculate_accuracy(cgt, cpr)  # must not raise duplicate-axis errors
    assert 0.0 <= float(stats['accuracy']) <= 1.0


def test_cpu_gpu_normalization_match():
    torch = pytest.importorskip('torch')
    tio = pytest.importorskip('torchio')
    from barlow_track.scripts.eval_accuracy import _rescale_gpu
    torch.manual_seed(0)
    x = (torch.rand(4, 8, 16, 16) * 1000).float()
    cpu_out = tio.RescaleIntensity(percentiles=(5, 99.5))(x.clone()).float()
    gpu_out = _rescale_gpu(x.clone())
    assert float((cpu_out - gpu_out).abs().max()) < 1e-5


@pytest.mark.parametrize('vol_shape, target', [((22, 90, 70), (8, 32, 32)), ((5, 40, 40), (8, 16, 16))])
def test_gpu_slide_crops_match_cpu_crops(vol_shape, target):
    """crop_window 'slide' reproduces data_loading's window exactly, edges and
    volumes thinner than the crop included; 'pad' differs at edges."""
    torch = pytest.importorskip('torch')
    from barlow_track.scripts.eval_accuracy import _extract_crops_gpu
    from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid
    rng = np.random.RandomState(0)
    vol = rng.rand(*vol_shape).astype(np.float32)
    pts = rng.uniform(0, np.array(vol_shape) - 0.51, size=(200, 3))
    Z, X, Y = vol_shape
    pts[:4] = [[0, 0, 0], [Z - 1, X - 1, Y - 1], [0, X // 2, 0], [Z // 2, 0, Y - 1]]
    sz = np.array([1, *vol_shape])
    cpu = np.stack([get_3d_crop_using_bbox_or_centroid(p, sz, np.array(target), vol)[0] for p in pts])
    gpu = _extract_crops_gpu(torch.from_numpy(vol), pts, target, round_centroids=True, crop_window='slide')
    assert np.array_equal(cpu, gpu.numpy())
    pad = _extract_crops_gpu(torch.from_numpy(vol), pts, target, round_centroids=True)
    assert not np.array_equal(cpu, pad.numpy())


def test_project_crop_centers_truncate_by_default_and_round_like_training():
    """The project path duplicates the centroid into a bbox (truncation);
    --round_crop_centers crops as training's extract_crops does."""
    torch = pytest.importorskip('torch')
    from barlow_track.scripts.eval_accuracy import _extract_crops_gpu
    from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid
    from barlow_track.utils.volume_data import extract_crops
    rng = np.random.RandomState(1)
    vol = rng.rand(22, 90, 70).astype(np.float32)
    target = np.array([8, 32, 32])
    pts = rng.uniform([0, 0, 0], [21.4, 89.4, 69.4], size=(200, 3))
    sz = np.array([1, *vol.shape])
    trunc_cpu = np.stack([get_3d_crop_using_bbox_or_centroid([*p, *p], sz, target, vol)[0] for p in pts])
    trunc_gpu = _extract_crops_gpu(torch.from_numpy(vol), pts, target, round_centroids=False, crop_window='slide')
    assert np.array_equal(trunc_cpu, trunc_gpu.numpy())
    train = extract_crops(vol, pts, target)
    round_gpu = _extract_crops_gpu(torch.from_numpy(vol), pts, target, round_centroids=True, crop_window='slide')
    assert np.array_equal(train, round_gpu.numpy())
    assert not np.array_equal(train, trunc_cpu)


def test_rescale_gpu_oversize_subsample_close_to_full():
    """Stacks above the CUDA quantile cap must rescale without error and
    stay close to the exact full-quantile result (round_flavell xy=64 evals
    crashed with 'quantile() input tensor is too large')."""
    torch = pytest.importorskip('torch')
    from barlow_track.scripts.eval_accuracy import _QUANTILE_MAX_ELEMENTS, _rescale_gpu
    torch.manual_seed(0)
    # > cap elements on any device (subsample path); ~40MB, CPU-safe.
    n = int(_QUANTILE_MAX_ELEMENTS) + 2_000_000
    x = (torch.rand(n) * 1000).float()
    out = _rescale_gpu(x.clone())
    assert out.shape == x.shape
    assert bool(((out >= 0) & (out <= 1)).all())
    # Reference: exact quantiles over all elements (CPU has no size cap).
    with torch.no_grad():
        q = torch.quantile(x.double().reshape(-1),
                           torch.tensor([0.05, 0.995], dtype=torch.float64))
        lo, hi = float(q[0]), float(q[1])
    ref = ((x.double() - lo) / max(hi - lo, 1e-12)).clamp(0, 1).float()
    assert float((out - ref).abs().max()) < 1e-3
    # Deterministic: same input, bit-identical output.
    assert torch.equal(out, _rescale_gpu(x.clone()))


def test_correlation_for_plot_empty_batch_returns_none():
    """A 0-object view must yield None (skip the plot), not crash the
    backbone with view(0, -1) (round_flavell trial_2 training crash)."""
    torch = pytest.importorskip('torch')
    from barlow_track.scripts.train_barlow_clusterer import _correlation_for_plot
    empty = (torch.empty(0, 1, 8, 8, 8), torch.empty(0, 1, 8, 8, 8),
             torch.empty(0, 4, 3), torch.empty(0, 4, 3))
    assert _correlation_for_plot(None, empty, 'cpu', True) is None
    assert _correlation_for_plot(None, empty, 'cpu', False) is None


def test_calculate_accuracy_key_types():
    df_gt = pd.DataFrame({'a': [1., 2., np.nan], 'b': [1., 2., 3.]})
    df_pred = pd.DataFrame({'a': [1., 9., np.nan], 'b': [1., 2., np.nan]})
    stats = calculate_accuracy(df_gt, df_pred)
    assert isinstance(stats['total_misses'], int)
    assert isinstance(stats['total_mismatches'], int)
    assert isinstance(stats['total_ground_truth'], int)
    assert isinstance(stats['misses_df'], pd.DataFrame)
    assert isinstance(stats['mismatches_df'], pd.DataFrame)
    # 5 valid GT detections: 1 miss (b t=2) + 1 mismatch (a t=1)
    assert stats['total_misses'] == 1
    assert stats['total_mismatches'] == 1
    assert stats['accuracy'] == pytest.approx(1 - 2 / 5)


class _FakeProj:
    def __init__(self, df):
        self._df = df

    def get_final_tracks_only_finished_neurons(self):
        return self._df, ['n0']


def _write_trial(trial_dir, num, with_config=True):
    tdir = os.path.join(trial_dir, f'trial_{num}')
    os.makedirs(os.path.join(tdir, 'log'), exist_ok=True)
    if with_config:
        with open(os.path.join(tdir, 'train_config.yaml'), 'w') as f:
            f.write('epochs: 1\nembedding_dim: 64\n')
        with open(os.path.join(tdir, 'log', 'stats.json'), 'w') as f:
            json.dump([{'epoch': 0, 'val_loss': 0.5}], f)
    return tdir


def test_build_accuracy_dict_atomic_appends(tmp_path, monkeypatch):
    import barlow_track.utils.utils_ground_truth as ug
    df_gt = _multiindex_df(4, ['n0'])
    monkeypatch.setattr(ug, 'ProjectData', type(
        'FakePD', (), {'load_final_project_data': staticmethod(lambda *a, **k: _FakeProj(df_gt))}))
    stats = {'accuracy': 0.75, 'accuracy_per_neuron': 0.75,
             'accuracy_per_timepoint': 0.75, 'misses_per_neuron_norm': 0.0,
             'misses_per_timepoint_norm': 0.0, 'mismatches_per_neuron_norm': 0.25,
             'mismatches_per_timepoint_norm': 0.25}
    monkeypatch.setattr(ug, 'process_trial', lambda trial, df, path: dict(stats))

    trial_dir = str(tmp_path / 'trials')
    project_dir = str(tmp_path / 'projects')
    os.makedirs(trial_dir)
    os.makedirs(project_dir)
    _write_trial(trial_dir, 0, with_config=True)   # missing project_config.yaml
    _write_trial(trial_dir, 1, with_config=False)  # missing train_config.yaml
    os.makedirs(os.path.join(project_dir, 'trial_0'))  # no project_config.yaml inside
    os.makedirs(os.path.join(project_dir, 'trial_1', ))
    open(os.path.join(project_dir, 'trial_1', 'project_config.yaml'), 'w').close()

    result, detailed = build_accuracy_dict('fake_gt', project_dir, trial_dir=trial_dir,
                                           check_if_training_finished=False)
    lengths = {len(v) for v in result.values()}
    assert lengths == {2}, {k: len(v) for k, v in result.items()}
    assert len({len(v) for v in detailed.values()}) == 1
    # trial_0: config present, project missing -> accuracy None, config kept
    assert result['trial'] == [0, 1]
    assert result['accuracy'][0] is None
    assert result['embedding_dim'][0] == 64
    # trial_1: config missing, project present -> config None, accuracy kept
    assert result['accuracy'][1] == 0.75
    assert result['embedding_dim'][1] is None
    assert detailed['per_neuron_accuracy'][1] == 0.75
