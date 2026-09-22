"""Unit tests for the vendored SuperGlue blocks (barlow_track.utils.superglue).

No test project needed; also guards the 'no new wbfm dependency' rule.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest barlow_track/tests/test_superglue_blocks.py -q --assert=plain
"""
import subprocess
import sys

import pytest
import torch

from barlow_track.utils.superglue import (
    AttentionalGNN,
    KeypointEncoder,
    arange_like,
    log_optimal_transport,
    normalize_keypoints,
    process_scores_into_matches,
)


def test_no_wbfm_import():
    code = ("import sys; import barlow_track.utils.superglue; "
            "assert not any(m == 'wbfm' or m.startswith('wbfm.') for m in sys.modules), "
            "'superglue module pulled in wbfm'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_normalize_keypoints_centers_volume():
    kpts = torch.tensor([[[[11.5, 300.0, 450.0]]]])  # center of (D=23, H=600, W=900)
    out = normalize_keypoints(kpts, (1, 1, 23, 600, 900))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_keypoint_encoder_shape():
    enc = KeypointEncoder(feature_dim=64, layers=[32, 64])
    kpts = torch.randn(2, 1, 10, 3)
    scores = torch.ones(2, 1, 10)  # (B, 1, N): already transposed, as in SuperGlue data dicts
    out = enc(kpts, scores)
    assert out.shape == (2, 64, 10)
    assert torch.isfinite(out).all()


def test_gnn_preserves_shape():
    gnn = AttentionalGNN(feature_dim=32, layer_names=['self', 'cross'] * 2)
    d0 = torch.randn(2, 32, 12)
    d1 = torch.randn(2, 32, 9)
    o0, o1 = gnn(d0, d1)
    assert o0.shape == d0.shape and o1.shape == d1.shape
    assert torch.isfinite(o0).all() and torch.isfinite(o1).all()


def test_optimal_transport_shape():
    scores = torch.randn(2, 5, 6)
    alpha = torch.tensor(1.0)
    Z = log_optimal_transport(scores, alpha, iters=10)
    assert Z.shape == (2, 6, 7)  # + dustbin row/col


def test_matches_from_peaked_scores():
    # 3x3 identity-like scores (log-space): point i matches point i
    scores = torch.full((1, 4, 4), -10.0)
    scores[0, :3, :3] = torch.eye(3) * 5.0
    idx0, idx1, ms0, ms1 = process_scores_into_matches(scores, match_threshold=0.2)
    assert idx0[0, :3].tolist() == [0, 1, 2]
    assert (ms0[0, :3] > 0.2).all()


def test_arange_like():
    x = torch.zeros(2, 5)
    assert arange_like(x, 1).tolist() == [0, 1, 2, 3, 4]
