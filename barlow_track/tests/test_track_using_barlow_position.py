"""Production tracking: position checkpoints embed as eval does; final_3d_tracks_df is never lost."""
import logging
from types import SimpleNamespace

import numpy as np
import torch

import barlow_track.utils.data_loading as data_loading
from barlow_track.utils import track_using_barlow as tub
from barlow_track.utils.volume_data import VolumeCoordsDataset

SHAPE = (10, 40, 40)
TARGET = (2, 4, 4)
# Two named objects and one untracked at t=0; one object at t=1 (too few to embed)
CENTROIDS = {0: {'neuron_002': [5., 30., 8.], 'neuron_001': [2., 10., 20.],
                 'untracked_time_0_0002_0007': [7., 20., 33.]},
             1: {'neuron_001': [3., 11., 21.]}}


def _fake_bbox(project_data, t, target_sz=None, include_untracked=False, skip_crops=False, **kw):
    cen = CENTROIDS[t]
    if skip_crops:
        return dict(cen), {}, None
    # Each crop is filled with its object's x, so rows can be traced through the forward
    return {n: np.full(TARGET, c[1], dtype=np.float32) + np.arange(np.prod(TARGET)).reshape(TARGET)
            for n, c in cen.items()}, {}, None


class _PositionModel(torch.nn.Module):
    """Descriptor = [mean crop intensity, keypoint x]; records each frame's inputs."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def backbone(self, crops):
        raise AssertionError("the position path must not embed crop by crop")

    def fused_descriptors(self, crops, kpts):
        self.calls.append((crops.clone(), kpts.clone()))
        return torch.stack([crops.flatten(1).mean(1), kpts[:, 1]], 1)


def _project():
    return SimpleNamespace(red_data=np.zeros((2,) + SHAPE), num_frames=2,
                           project_config=SimpleNamespace(logger=logging.getLogger('test')))


def test_position_checkpoint_embeds_whole_frames_with_aligned_keypoints(monkeypatch):
    monkeypatch.setattr(data_loading, 'get_bbox_data_for_volume_with_label', _fake_bbox)
    model = _PositionModel()
    emb = tub.embed_using_barlow(torch.device('cpu'), model, _project(), TARGET, False,
                                 descriptor_stage='auto')
    assert len(model.calls) == 1  # t=1 has one object: skipped as in eval
    crops, kpts = model.calls[0]
    assert crops.shape == (3, 1) + TARGET
    assert set(emb) == set(CENTROIDS[0])
    for name, (z, x, y) in CENTROIDS[0].items():
        assert list(emb[name]) == [0] and emb[name][0].shape == (1, 2)
        # The keypoint row that produced this name's descriptor is this name's own centroid
        own = VolumeCoordsDataset._normalize(torch.tensor([[z, x, y]]), SHAPE)[0, 1].item()
        assert np.isclose(emb[name][0][0, 1], own, atol=1e-6)
    # Rows stay in name order, so crops and keypoints are of the same object
    order = sorted(CENTROIDS[0])
    assert np.argsort([CENTROIDS[0][n][1] for n in order]).tolist() == \
        np.argsort(kpts[:, 1].numpy()).tolist() == np.argsort(crops.flatten(1).mean(1).numpy()).tolist()


def test_nonfinite_centroid_object_is_skipped_by_the_bbox_loader():
    # An empty mask's metadata centroid is NaN (Samuel 153 project); int(NaN) used to raise
    import pandas as pd
    rows = dict(raw_segmentation_id=[3, 7, 9], z=[2., 5., np.nan], x=[10., 30., np.nan],
                y=[20., 8., np.nan], raw_neuron_ind_in_list=[0, 1, 2])
    seg_meta = SimpleNamespace(
        get_all_neuron_metadata_for_single_time=lambda t, as_dataframe: (list(rows.values()), list(rows)),
        mask_index_to_i_in_array=lambda t, seg: [3, 7, 9].index(seg))
    tracks = pd.DataFrame({('neuron_001', 'raw_segmentation_id'): [3.]})
    project = SimpleNamespace(red_data=np.random.default_rng(0).random((1,) + SHAPE), final_tracks=tracks,
                              finished_neuron_names=lambda: ['neuron_001'], segmentation_metadata=seg_meta)
    for skip_crops in (False, True):
        dat, _, _ = data_loading.get_bbox_data_for_volume_with_label(
            project, 0, target_sz=TARGET, include_untracked=True, skip_crops=skip_crops)
        assert sorted(dat) == ['neuron_001', 'untracked_time_0_0001_0007']


def test_unset_stage_keeps_the_legacy_backbone_path(monkeypatch):
    seen = {}
    monkeypatch.setattr(tub, 'embed_frames_at_stage', lambda *a, **k: seen.setdefault('called', True))
    monkeypatch.setattr('barlow_track.utils.barlow.NeuronImageWithGTDataset',
                        lambda *a, **k: [])
    out = tub.embed_using_barlow(torch.device('cpu'), _PositionModel(), _project(), TARGET, False)
    assert out == {} and 'called' not in seen


class _TrackingConfig:
    def __init__(self, **config):
        self.config = dict(config)
        self.writes = 0

    def update_self_on_disk(self):
        self.writes += 1


def test_new_tracks_keep_the_replaced_path_in_history():
    log = logging.getLogger('test')
    cfg = _TrackingConfig(final_3d_tracks_df='3-tracking/manual_annotation/gt.h5')
    tub._point_final_tracks_at(cfg, '3-tracking/barlow/df_barlow_tracks.h5', log)
    tub._point_final_tracks_at(cfg, '3-tracking/barlow2/df_barlow_tracks.h5', log)
    assert cfg.config['final_3d_tracks_df'] == '3-tracking/barlow2/df_barlow_tracks.h5'
    assert cfg.config['final_3d_tracks_df_history'] == ['3-tracking/manual_annotation/gt.h5',
                                                        '3-tracking/barlow/df_barlow_tracks.h5']
    protected = _TrackingConfig(final_3d_tracks_df='gt.h5', protect_final_3d_tracks_df=True)
    tub._point_final_tracks_at(protected, 'new.h5', log)
    assert protected.config['final_3d_tracks_df'] == 'gt.h5' and protected.writes == 0
