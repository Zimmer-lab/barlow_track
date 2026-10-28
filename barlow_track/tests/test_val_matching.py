"""Frame-pair matching proxy (utils/val_matching.py): synthetic, fast."""
import numpy as np

from barlow_track.utils.val_matching import held_out_frames


def test_held_out_frames_skip_used_and_spread_evenly():
    frames = held_out_frames(100, used_frames=range(0, 100, 2), n=5)
    assert len(frames) == 5
    assert all(t % 2 == 1 for t in frames)
    assert frames[0] == 1 and frames[-1] == 99
    assert np.all(np.diff(frames) > 0)


def test_held_out_frames_off_or_exhausted():
    assert held_out_frames(10, [], 0) == []
    assert held_out_frames(3, [0, 1, 2], 5) == []
    assert held_out_frames(4, [1], 10) == [0, 2, 3]


def test_keypoint_proxy_embeds_metadata_centroids_without_volumes(monkeypatch):
    import torch
    import barlow_track.utils.volume_data as vd
    from barlow_track.utils.val_matching import _iter_keypoint_frames
    pts = {0: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1: np.array([[1.0, 1.0, 1.0]])}
    monkeypatch.setattr(vd, 'get_centroids_for_volume', lambda project_data, t: (pts[t], np.arange(len(pts[t])) + 7))
    project = type('P', (), {'red_data': np.zeros((2, 8, 40, 40), dtype=np.float32)})()
    frames = list(_iter_keypoint_frames(project, [0, 1]))
    assert [f['t'] for f in frames] == [0]  # < 2 objects skipped, like the crop path
    f = frames[0]
    want = vd.VolumeCoordsDataset._normalize(torch.from_numpy(pts[0].astype(np.float32)), (8, 40, 40))
    assert torch.equal(f['kpts'], want) and f['crops'].shape == (2, 1, 1, 1, 1)
    assert [seg for _, seg in f['meta']] == [7, 8]


def test_keypoint_proxy_refuses_the_image_stage():
    # batch1 of posonly_fast lost its whole proxy to a 'backbone' stage on placeholder crops,
    # swallowed by training's except; refuse it before embedding anything
    import pytest
    from barlow_track.utils.val_matching import frame_pair_matching_accuracy
    with pytest.raises(TypeError, match='backbone'):
        frame_pair_matching_accuracy(None, None, [0, 1], (2, 4, 4), 'cpu', stages=('auto', 'backbone'),
                                     keypoints_only=True)
