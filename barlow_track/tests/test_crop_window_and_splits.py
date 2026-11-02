"""Regression tests for docs/pr_plan/PR04 (crop windows at volume edges, split counts).

Synthetic volume whose voxel value encodes its own (z, x, y), so a crop proves which window
it came from.
"""
import numpy as np
import pytest

from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid
from barlow_track.utils.volume_data import _resolve_split_count

Z, X, Y = 12, 40, 30
VOL = (np.arange(Z)[:, None, None] * 10000 + np.arange(X)[None, :, None] * 100
       + np.arange(Y)[None, None, :]).astype(np.int64)
SZ = (1, Z, X, Y)
TARGET = (4, 16, 10)


def _origin(crop):
    v = int(crop[0, 0, 0])
    return v // 10000, (v // 100) % 100, v % 100


@pytest.mark.parametrize('c', [(0, 0, 0), (1, 3, 2), (6, 20, 15), (Z - 1, X - 1, Y - 1), (Z - 2, X - 4, 1)])
def test_crop_slides_inside_the_volume_and_is_never_padded(c):
    crop, bbox = get_3d_crop_using_bbox_or_centroid(c, SZ, TARGET, VOL)
    assert crop.shape == TARGET
    z0, x0, y0 = _origin(crop)
    # the crop is exactly the volume slice at its bbox: no zero padding anywhere
    np.testing.assert_array_equal(crop, VOL[z0:z0 + 4, x0:x0 + 16, y0:y0 + 10])
    assert bbox == [z0, x0, y0, z0 + 4, x0 + 16, y0 + 10]
    # the centroid lies inside the crop
    for m, o, t in zip(c, (z0, x0, y0), TARGET):
        assert o <= m < o + t


def test_interior_crop_is_centred():
    c = (6, 20, 15)
    crop, _ = get_3d_crop_using_bbox_or_centroid(c, SZ, TARGET, VOL)
    assert _origin(crop) == (6 - 2, 20 - 8, 15 - 5)


@pytest.mark.parametrize('c', [(-1, 5, 5), (Z, 5, 5), (3, X + 2, 5), (3, 5, -3)])
def test_out_of_range_centroid_is_skipped_not_zero_filled(c):
    crop, bbox = get_3d_crop_using_bbox_or_centroid(c, SZ, TARGET, VOL)
    assert crop is None and bbox is None


def test_volume_smaller_than_target_is_padded_only_on_that_axis():
    crop, _ = get_3d_crop_using_bbox_or_centroid((5, 20, 15), SZ, (16, 16, 10), VOL)
    assert crop.shape == (16, 16, 10)
    # z has 12 planes for a 16-plane target: all 12 kept, 4 planes of padding
    assert (crop[:4] == 0).all()
    np.testing.assert_array_equal(crop[4:], VOL[:, 12:28, 10:20])


@pytest.mark.parametrize('f,n,expected', [(0.5, 10, 5), (1.0, 10, 10), (0.0, 10, 0), (3, 10, 3), (25, 10, 10),
                                          (-0.2, 10, 0), ('bad', 10, 0)])
def test_split_count_fraction_and_absolute(f, n, expected):
    assert _resolve_split_count(f, n) == expected
