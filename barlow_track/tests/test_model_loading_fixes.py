"""Checkpoint loading: visnet settings are architecture, and partial loads are visible."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from barlow_track.utils.barlow import compare_model_architecture, load_barlow_model
from barlow_track.utils.barlow_superglue import BarlowVolumeAttention
from barlow_track.utils.siamese import ResidualEncoder3D


def _visnet(**kw):
    a = dict(model_type='position', use_position=True, embedding_dim=16, position_encoder='visnet',
             fusion='add', fusion_norm='layernorm')
    a.update(kw)
    return SimpleNamespace(**a)


def test_visnet_settings_must_match_the_checkpoint():
    assert compare_model_architecture(_visnet(), _visnet()) == []
    # Explicit defaults and yaml types are the same architecture
    assert compare_model_architecture(_visnet(visnet_cutoff=0.35, visnet_hidden='64', visnet_cutoff_nn=None),
                                      _visnet()) == []
    got = compare_model_architecture(_visnet(visnet_rbf_rescale=True, visnet_cutoff_nn=6),
                                     _visnet(visnet_cutoff_nn=3))
    assert sorted(m.split(':')[0] for m in got) == ['visnet_cutoff_nn', 'visnet_rbf_rescale']
    # keypoint-MLP models carry no visnet fields to compare
    assert compare_model_architecture(_visnet(position_encoder='keypoint_mlp', visnet_cutoff=1.0),
                                      _visnet(position_encoder='keypoint_mlp')) == []


def _pre_gating_checkpoint(tmp_path):
    args = SimpleNamespace(embedding_dim=16, projector='32-32', projector_final=8, lambd=0.0051,
                           lambd_obj=0.5, fusion='add', keypoint_encoder_layers=[16, 32], self_layers=2,
                           model_type='attention', target_sz_z=4, target_sz_xy=16,
                           backbone_kwargs=dict(num_levels=2, f_maps=2), fusion_norm='layernorm',
                           center_per_volume=True, projector_norm='layernorm')
    model = BarlowVolumeAttention(args, backbone=ResidualEncoder3D, in_channels=1, num_levels=2, f_maps=2,
                                  crop_sz=np.array([4, 16, 16]))
    sd = {k: v for k, v in model.state_dict().items()
          if not (k.startswith('attn_gate') or k.startswith('norm_context.'))}
    wpath = tmp_path / 'resnet50.pth'
    torch.save(sd, str(wpath))
    with open(tmp_path / 'args.pickle', 'wb') as f:
        pickle.dump(args, f)
    return str(wpath)


def test_pre_gating_checkpoint_is_marked_or_refused(tmp_path, monkeypatch):
    wpath = _pre_gating_checkpoint(tmp_path)
    monkeypatch.delenv('BARLOW_STRICT_LOAD', raising=False)
    _, model, _ = load_barlow_model(wpath)
    assert model.partial_load == ['attn_gate', 'norm_context.bias', 'norm_context.weight']
    monkeypatch.setenv('BARLOW_STRICT_LOAD', '1')
    with pytest.raises(RuntimeError, match='Could not load weights'):
        load_barlow_model(wpath)


def test_unread_config_keys_are_named(caplog):
    import logging
    from barlow_track.scripts.train_barlow_clusterer import _warn_unread_config_keys
    with caplog.at_level(logging.WARNING):
        _warn_unread_config_keys(SimpleNamespace(training_fraction=0.5, batch_size=4, epochs=3))
    text = caplog.text
    assert 'training_fraction' in text and 'batch_size' in text and 'epochs' not in text
