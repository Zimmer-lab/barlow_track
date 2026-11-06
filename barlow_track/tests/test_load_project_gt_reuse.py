"""_load_project_gt reuses an already-loaded project when the GT path is the project path."""
import types

import pandas as pd

from barlow_track.scripts import eval_accuracy as ea


def _fake_project(df):
    return types.SimpleNamespace(get_final_tracks_only_finished_neurons=lambda: (df, None), final_tracks=None)


def test_reuses_loaded_project_for_same_path(tmp_path, monkeypatch):
    cfg = tmp_path / 'project_config.yaml'
    cfg.write_text('')
    df = pd.DataFrame({('AVAL', 'raw_segmentation_id'): [1.0, 2.0]})

    def no_load(*a, **k):
        raise AssertionError('project opened a second time')

    from wbfm.utils.projects.finished_project_data import ProjectData
    monkeypatch.setattr(ProjectData, 'load_final_project_data', staticmethod(no_load))
    out = ea._load_project_gt(str(cfg), project_data=_fake_project(df), project_path=str(tmp_path / '.' / cfg.name))
    assert out is df


def test_loads_when_paths_differ(tmp_path, monkeypatch):
    df = pd.DataFrame({('AVAL', 'raw_segmentation_id'): [1.0]})
    loaded = []

    def fake_load(path, **k):
        loaded.append(path)
        return _fake_project(df)

    from wbfm.utils.projects.finished_project_data import ProjectData
    monkeypatch.setattr(ProjectData, 'load_final_project_data', staticmethod(fake_load))
    out = ea._load_project_gt(str(tmp_path / 'gt.yaml'), project_data=_fake_project(None),
                              project_path=str(tmp_path / 'other.yaml'))
    assert out is df and loaded == [str(tmp_path / 'gt.yaml')]
