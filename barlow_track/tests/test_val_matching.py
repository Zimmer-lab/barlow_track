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
