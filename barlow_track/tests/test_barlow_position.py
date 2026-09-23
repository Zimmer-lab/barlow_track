"""Unit tests for BarlowWithPosition (fusion+norm) and BarlowVolumeAttention
(intra-volume self-attention). BarlowSuperGlue is a stub (pair-only matching
has no inference path) and is only checked for load/raise behavior.

Synthetic data + tiny backbone: fast on CPU, no test project needed.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest barlow_track/tests/test_barlow_position.py -q --assert=plain
"""
from types import SimpleNamespace

import pytest
import torch

from barlow_track.utils.barlow_superglue import (
    BarlowSuperGlue,
    BarlowVolumeAttention,
    BarlowWithPosition,
    both_correlation_matrices,
)
from barlow_track.utils.siamese import ResidualEncoder3D


def _args(**overrides):
    base = dict(
        embedding_dim=16,
        projector='32-32',
        projector_final=8,
        lambd=0.0051,
        lambd_obj=0.5,
        fusion='add',
        keypoint_encoder_layers=[16, 32],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _backbone_kwargs():
    import numpy as np
    # Note: embedding_dim is injected by the model itself; do not pass it here
    return dict(in_channels=1, num_levels=2, f_maps=2, crop_sz=np.array([4, 16, 16]))


@pytest.fixture
def batch():
    torch.manual_seed(0)
    n = 6
    y1 = torch.randn(n, 1, 4, 16, 16)
    y2 = torch.randn(n, 1, 4, 16, 16)
    k1 = torch.randn(n, 3) * 0.5
    k2 = torch.randn(n, 3) * 0.5
    return y1, y2, k1, k2


def test_correlation_matrices_shapes():
    z1, z2 = torch.randn(6, 8), torch.randn(6, 8)
    cf, co = both_correlation_matrices(z1, z2)
    assert cf.shape == (8, 8) and co.shape == (6, 6)


def test_fusion_add_forward_backward(batch):
    y1, y2, k1, k2 = batch
    model = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    loss, lo, lt = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.kenc.encoder[0].weight.grad is not None  # position path gets gradients


def test_fusion_concat_shapes(batch):
    y1, y2, k1, k2 = batch
    model = BarlowWithPosition(_args(fusion='concat'), backbone=ResidualEncoder3D, **_backbone_kwargs())
    z = model.embed_with_position(y1, k1)
    assert z.shape == (6, 8)
    loss, _, _ = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)


def test_encode_position_single_detection_falls_back(batch):
    y1, _, k1, _ = batch
    model = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        z = model.encode_position(k1[:1])
        assert z.shape == (1, 16)
        assert torch.allclose(z, torch.zeros_like(z))  # visual-only fallback, no crash
        fused = model.fused_descriptors(y1[:1], k1[:1])
        assert torch.allclose(fused, model.backbone(y1[:1]))


def test_position_changes_embedding(batch):
    y1, _, k1, _ = batch
    model = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        z_a = model.embed_with_position(y1, k1)
        k_perturbed = k1.clone()
        k_perturbed[0] += 0.5  # move a single neuron: relative geometry changes
        z_b = model.embed_with_position(y1, k_perturbed)
    assert not torch.allclose(z_a, z_b)  # keypoints actually matter


def test_explicit_scores(batch):
    y1, y2, k1, k2 = batch
    model = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    s = torch.ones(6)
    loss, _, _ = model(y1, y2, k1, k2, s, s)
    assert torch.isfinite(loss)


def test_superglue_stub_loads_but_forward_raises(batch):
    # Pair-only matching has no inference path: the class exists only so old
    # checkpoints still load. Instantiation is fine; forward() must refuse.
    y1, y2, k1, k2 = batch
    model = BarlowSuperGlue(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    with pytest.raises(NotImplementedError):
        model(y1, y2, k1, k2)


def _attn_args(**overrides):
    base = dict(self_layers=2)
    base.update(overrides)
    return _args(**base)


def test_fusion_norm_balances_branches(batch):
    y1, _, k1, _ = batch
    plain = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    normed = BarlowWithPosition(_args(fusion_norm='layernorm'), backbone=ResidualEncoder3D,
                                **_backbone_kwargs())
    plain.eval()
    normed.eval()
    with torch.no_grad():
        v = plain.backbone(y1)
        p = plain.encode_position(k1)
        gap_plain = p.std() / v.std()
        vn = normed.norm_visual(v)
        pn = normed.norm_pos(p)
        gap_normed = pn.std() / vn.std()
    assert gap_plain > 5.0  # documents the untrained scale mismatch
    assert 0.5 < gap_normed < 2.0  # layernorm balances the streams


def test_fusion_norm_invalid_kind(batch):
    with pytest.raises(ValueError):
        BarlowWithPosition(_args(fusion_norm='bogus'), backbone=ResidualEncoder3D, **_backbone_kwargs())


def test_contextualize_is_permutation_equivariant(batch):
    y1, _, k1, _ = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    perm = torch.randperm(6)
    with torch.no_grad():
        d = model.fused_descriptors(y1, k1)
        c1 = model.contextualize(d)
        c2 = model.contextualize(d[perm])
    assert torch.allclose(c2, c1[perm], atol=1e-5)


def test_contextualize_spreads_perturbation(batch):
    # Self-attention mixes neurons: moving one keypoint changes OTHERS' embeddings.
    # (Plain MLP fusion also leaks slightly via InstanceNorm stats; attention mixes directly.)
    y1, _, k1, _ = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        d = model.fused_descriptors(y1, k1)
        c_base = model.contextualize(d)
        k_pert = k1.clone()
        k_pert[0] += 1.0
        c_pert = model.contextualize(model.fused_descriptors(y1, k_pert))
    others_changed = ~torch.isclose(c_base[1:], c_pert[1:], atol=1e-5).all(dim=1)
    assert others_changed.any()


def test_contextualize_single_passthrough(batch):
    y1, _, k1, _ = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        d = model.fused_descriptors(y1[:1], k1[:1])
        assert torch.allclose(model.contextualize(d), d)  # no crash, no-op


def test_attention_forward_backward(batch):
    y1, y2, k1, k2 = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    loss, lo, lt = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.self_gnn.layers[0].attn.merge.weight.grad is not None


def test_attention_zero_self_layers_is_plain_fusion(batch):
    y1, y2, k1, k2 = batch
    torch.manual_seed(0)
    a = BarlowVolumeAttention(_attn_args(self_layers=0), backbone=ResidualEncoder3D, **_backbone_kwargs())
    torch.manual_seed(0)
    b = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    a.eval()
    b.eval()
    with torch.no_grad():
        assert torch.allclose(a.embed_with_position(y1, k1), b.embed_with_position(y1, k1))
