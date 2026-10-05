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
        # Default fusion_norm='layernorm': visual branch is normalized, and
        # the position branch must contribute nothing (no learned bias leak).
        assert torch.allclose(fused, model.norm_visual(model.backbone(y1[:1])))
        # Independence from trained norm_pos.bias
        if hasattr(model.norm_pos, 'bias') and model.norm_pos.bias is not None:
            model.norm_pos.bias.fill_(5.0)
            fused_biased = model.fused_descriptors(y1[:1], k1[:1])
            assert torch.allclose(fused, fused_biased)


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


def test_default_fusion_is_concat(batch):
    y1, y2, k1, k2 = batch
    a = _args()
    del a.fusion  # simulate configs predating the fusion flag
    model = BarlowWithPosition(a, backbone=ResidualEncoder3D, **_backbone_kwargs())
    assert model.fusion == 'concat'
    loss, _, _ = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)


def test_position_only_ignores_visual(batch):
    y1, y2, k1, k2 = batch
    model = BarlowWithPosition(_args(fusion='position_only'), backbone=ResidualEncoder3D,
                               **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        d = model.fused_descriptors(y1, k1)
        assert torch.allclose(d, model.norm_pos(model.encode_position(k1)))
        assert not torch.allclose(d, model.backbone(y1))  # no visual leakage
    loss, _, _ = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)


def test_position_only_with_attention(batch):
    y1, y2, k1, k2 = batch
    model = BarlowVolumeAttention(_attn_args(fusion='position_only'), backbone=ResidualEncoder3D,
                                  **_backbone_kwargs())
    loss, _, _ = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.self_gnn.layers[0].attn.merge.weight.grad is not None


def test_concat_fusion_has_no_bare_relu(batch):
    from torch import nn as _nn
    y1, y2, k1, k2 = batch
    model = BarlowWithPosition(_args(fusion='concat'), backbone=ResidualEncoder3D, **_backbone_kwargs())
    assert isinstance(model.fuse_mlp[-1], _nn.LayerNorm)  # post-fusion norm, condensed
    assert not any(isinstance(m, _nn.ReLU) for m in model.fuse_mlp)
    loss, _, _ = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)


def test_gated_attention_starts_near_identity(batch):
    import math
    y1, _, k1, _ = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    gate = model.attention_gate_value()
    assert gate == pytest.approx(1.0 / (1.0 + math.exp(4.0)), rel=0.01)
    assert gate < 0.05
    model.eval()
    with torch.no_grad():
        d = model.fused_descriptors(y1, k1)
        c = model.contextualize(d)
        # Near-identity residual + terminal LayerNorm: per-row unit-ish scale,
        # no huge shared offset injected.
        assert c.shape == d.shape
        assert torch.isfinite(c).all()
        row_norms = c.norm(dim=1)
        assert (row_norms > 0).all()


def test_custom_gate_init(batch):
    y1, _, k1, _ = batch
    model = BarlowVolumeAttention(_attn_args(attn_gate_init=0.0), backbone=ResidualEncoder3D,
                                  **_backbone_kwargs())
    assert model.attention_gate_value() == pytest.approx(0.5)
    model.eval()
    with torch.no_grad():
        c = model.contextualize(model.fused_descriptors(y1, k1))
        assert torch.isfinite(c).all()


def test_sweep_template_names_match_train_config():
    # The Ax runner merges sweep params flat over train_config.yaml, so every
    # swept name must be a top-level training key (nested dicts can't be swept).
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'barlow_track' / 'barlow_project_template'
    with open(root / 'train_config.yaml') as f:
        baseline = yaml.safe_load(f)
    with open(root / 'hyperparameter_search_template.yaml') as f:
        sweep = yaml.safe_load(f)
    names = [p['name'] for p in sweep['hyperparameters']]
    assert 'target_sz_z' not in names and 'target_sz_xy' not in names  # crop size fixed
    missing = [n for n in names if n not in baseline]
    assert not missing, f"sweep params missing from train_config.yaml: {missing}"


def test_descriptor_health_ranges(batch):
    from barlow_track.utils.barlow_superglue import descriptor_health
    y1, _, k1, _ = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        d = model.fused_descriptors(y1, k1)
        h = descriptor_health(d)
    n, dim = d.shape
    assert 1.0 <= h['eff_rank'] <= min(n, dim)
    assert 0.0 <= h['offdiag_corr'] <= 1.0
    assert 0.0 <= h['dead_frac'] <= 1.0
    assert h['mean_frac'] >= 0.0


def test_volume_diagnostics_keys_and_entropy(batch):
    from barlow_track.utils.barlow_superglue import volume_descriptor_diagnostics
    y1, _, k1, _ = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        diag = volume_descriptor_diagnostics(model, y1, k1)
    assert 'fused_eff_rank' in diag and 'contextual_eff_rank' in diag
    assert 'attn_gate' in diag and diag['attn_gate'] < 0.05
    assert 'position_jitter_sensitivity' in diag
    ent = diag['attention_entropy']
    assert ent != ent or 0.0 <= ent <= 1.0  # NaN (no probs yet ok) or normed range


def test_old_checkpoint_state_loads_with_new_params(batch):
    # Simulate a pre-gating checkpoint: strip the new keys, reload leniently.
    y1, y2, k1, k2 = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    full = model.state_dict()
    stripped = {k: v for k, v in full.items()
                if not (k.startswith('attn_gate') or k.startswith('norm_context.')
                        or k.startswith('fuse_mlp.2.') or k.startswith('fuse_mlp.3.')
                        or k.startswith('fuse_mlp.4.'))}
    assert stripped  # first Linear + backbone still there
    fresh = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    missing, unexpected = fresh.load_state_dict(stripped, strict=False)
    assert not unexpected
    assert set(missing) == set(full) - set(stripped)
    loss, _, _ = fresh(y1, y2, k1, k2)
    assert torch.isfinite(loss)


def test_projector_uses_layernorm_not_batchnorm(batch):
    # A batch is one volume: BatchNorm would erase volume-level signal before
    # the loss; LayerNorm (per-neuron) preserves inter-neuron structure.
    from torch import nn as _nn
    y1, y2, k1, k2 = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    assert not any(isinstance(m, _nn.BatchNorm1d) for m in model.modules())
    norms = [m for m in model.projector if isinstance(m, _nn.LayerNorm)]
    assert len(norms) == 2  # one per hidden projector layer
    loss, _, _ = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)
    loss.backward()


def test_old_batchnorm_checkpoint_loads_leniently(tmp_path, batch):
    # Simulate a pre-LayerNorm checkpoint: same keys plus stale BN buffers.
    # load_barlow_model must accept it (BN affine maps onto LayerNorm;
    # running stats ignored) since tracking never uses the projector.
    import pickle
    y1, _, k1, _ = batch
    model = BarlowVolumeAttention(_attn_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    sd = dict(model.state_dict())
    ln_idx = [i for i, m in enumerate(model.projector) if isinstance(m, torch.nn.LayerNorm)]
    assert ln_idx  # norm positions carry the stale BN buffers in old checkpoints
    for i in ln_idx:
        dim = model.projector[i].normalized_shape[0]
        sd[f'projector.{i}.running_mean'] = torch.zeros(dim)
        sd[f'projector.{i}.running_var'] = torch.ones(dim)
    sd[f'projector.{ln_idx[0]}.num_batches_tracked'] = torch.tensor(100)
    wpath = tmp_path / 'resnet50.pth'
    torch.save(sd, str(wpath))
    _a = _attn_args()
    _a.model_type = 'attention'
    _a.target_sz_z, _a.target_sz_xy = 4, 16
    _a.backbone_kwargs = dict(num_levels=2, f_maps=2)
    _a.fusion_norm = 'layernorm'
    _a.center_per_volume = True
    with open(tmp_path / 'args.pickle', 'wb') as f:
        pickle.dump(_a, f)
    from barlow_track.utils.barlow import load_barlow_model
    _, reloaded, _ = load_barlow_model(str(wpath))
    assert type(reloaded).__name__ == 'BarlowVolumeAttention'
    reloaded.eval()
    _dev = next(reloaded.parameters()).device
    y1, k1 = y1.to(_dev), k1.to(_dev)
    with torch.no_grad():
        d = reloaded.contextual_descriptors(y1, k1)
        assert torch.isfinite(d).all()
