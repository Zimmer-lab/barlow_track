"""The overlay independently preserves predictions and uses the saved scoring assignment."""
import numpy as np
import pandas as pd
import pytest

from barlow_track.scripts.paper_scripts.revision.export_supplementary_videos import (
    add_vector_targets, panel_xy, scored_identities, score_marks, volume_scores,
)


def test_markers_use_recording_match_and_available_annotation():
    pred = pd.DataFrame({'track7': [1., 15.], 'track2': [3., 6.],
                         'track9': [8., 13.], 'track5': [np.nan, 19.],
                         'track3': [2., 4.]}, index=[45, 46])
    matched = pred[['track9', 'track7', 'track2']].set_axis(['A', 'B', 'C'], axis=1)
    matched['missing'] = np.nan
    mapping = scored_identities(pred, matched)
    points = pd.DataFrame({'predicted_track_id': pred.columns, 'detection_id': pred.loc[46].values})
    # B and its detection agree. C is elsewhere. A is unannotated but its
    # assigned detection is known to belong to D. Neither endpoint of track5
    # is annotated. The unmatched track3 takes C's annotated detection.
    truth = pd.Series({'B': 15, 'C': 4, 'D': 13})
    score_marks(points, mapping, truth)
    assert points.status.tolist() == ['correct', 'incorrect', 'incorrect', 'no_ground_truth', 'incorrect']
    assert points.detection_id.tolist() == [15, 6, 13, 19, 4]


def test_clip_agreement_cannot_substitute_for_the_saved_track_mapping():
    pred = pd.DataFrame({'p': [1., 2.], 'q': [3., 4.]})
    with pytest.raises(ValueError, match='Inconsistent full-recording'):
        scored_identities(pred, pd.DataFrame({'gt_neuron': [1., 4.]}))


def test_volume_denominator_includes_misses_and_unmatched_reference_columns():
    gt = pd.DataFrame({'A': [8., np.nan], 'B': [3., np.nan], 'C': [5., np.nan]}, index=[52, 53])
    matched = pd.DataFrame({'B': [9., 2.], 'A': [8., 1.]}, index=[52, 53])
    scores = volume_scores(gt, matched)
    assert tuple(scores.loc[52]) == (1, 3, 1/3)
    assert scores.loc[53, 'annotated'] == 0 and pd.isna(scores.loc[53, 'accuracy'])


def test_vectors_end_at_expected_detection_even_without_a_prediction_there():
    det = pd.DataFrame({'z': [1., 5., 7.], 'x': [4., 13., 12.], 'y': [2., 9., 8.]}, index=[6, 8, 4])
    pts = pd.DataFrame({'detection_id': [8, 4, 12], 'expected_detection_id': [6, 4, np.nan],
                        'status': ['incorrect', 'correct', 'incorrect']}, index=[8, 4, 12])
    add_vector_targets(pts, det)
    target = pts.loc[8, ['target_z', 'target_x', 'target_y']].to_numpy(dtype=float)
    np.testing.assert_array_equal(target, [1., 4., 2.])
    assert pts.loc[[4, 12], 'target_x'].isna().all()
    assert panel_xy(target[1], target[2], True, 100, 10, (10, 100)) == (2, 4)
    assert panel_xy(target[1], target[2], False, 100, 10, (10, 100)) == (4, 2)
