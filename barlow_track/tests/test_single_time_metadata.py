"""data_loading._single_time_metadata returns exactly what wbfm's per-frame accessor returns."""
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('wbfm')
from wbfm.utils.segmentation.util.utils_metadata import DetectedNeurons  # noqa: E402

from barlow_track.utils.data_loading import _single_time_metadata  # noqa: E402


def _frame(labels, rng):
    rows = {lab: [float(rng.randint(50, 500)), tuple(rng.uniform(0, 30, 3)), float(rng.uniform(1e3, 1e5)),
                  lab, None, float(rng.uniform(10, 900))] for lab in labels}
    return pd.DataFrame.from_dict(rows, orient='index',
                                  columns=['neuron_volume', 'centroids', 'total_brightness', 'label', 'bbox',
                                           'intensity_mean'])


def test_matches_wbfm_on_full_empty_and_missing_frames(tmp_path):
    rng = np.random.RandomState(0)
    fname = tmp_path / 'metadata.pickle'
    fname.write_bytes(b'')  # DetectedNeurons only checks that the file exists once a table is set
    table = {0: _frame([3, 1, 7, 12], rng), 1: _frame([], rng), 2: _frame([5, 2], rng)}
    sm = DetectedNeurons(str(fname), _segmentation_metadata=table)
    prj = types.SimpleNamespace(segmentation_metadata=sm)
    for t in (0, 1, 2):
        r1, c1 = _single_time_metadata(prj, t)
        r0, c0 = sm.get_all_neuron_metadata_for_single_time(t, as_dataframe=False)
        assert c1 == c0
        assert len(r1) == len(r0)
        for a, b in zip(r1, r0):
            np.testing.assert_array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
    assert _single_time_metadata(prj, 1) == ([], [])
    with pytest.raises(KeyError):  # a frame not in the table fails as in wbfm
        _single_time_metadata(prj, 9)


def test_falls_back_to_the_accessor_without_a_table():
    calls = []

    def accessor(t, as_dataframe):
        calls.append(t)
        return [[1.0], [2.0]], ['z', 'x']

    prj = types.SimpleNamespace(segmentation_metadata=types.SimpleNamespace(
        get_all_neuron_metadata_for_single_time=accessor))
    assert _single_time_metadata(prj, 4) == ([[1.0], [2.0]], ['z', 'x'])
    assert calls == [4]
