"""Results jsonl: one tag can hold differently measured cells (settings-aware resume/dedup)."""
import json
from types import SimpleNamespace

from barlow_track.utils.utils_ground_truth import (append_record, deduplicate_results,
                                                   read_results_by_tag, settings_match)


def _rec(**kw):
    base = dict(tag='round_x_trial0_flavell', source='nwb', gt='/data/flavell.nwb',
                descriptor_stage='auto', cluster='labelprop', num_seeds=25, seed=0,
                center_per_volume=False, l2_per_volume=False, accuracy=0.9)
    base.update(kw)
    return base


def _bench_args(**kw):
    a = dict(mode='trained', descriptor_stage='auto', cluster='labelprop', num_seeds=25,
             center_per_volume=False, l2_per_volume=False, max_frames=None,
             svd='scaled', gt_matching='nandist')
    a.update(kw)
    return SimpleNamespace(**a)


def test_in_sweep_project_record_does_not_count_as_the_benchmark_cell(tmp_path):
    import barlow_track.scripts.multiproject_scripts.run_trials_on_all_ground_truth as bench
    path = str(tmp_path / 'r.jsonl')
    append_record(path, _rec(source='project', gt='/p/2025_07_01/project_config.yaml', objective='accuracy'))
    want = bench.requested_settings('flavell', _bench_args())
    assert bench.recorded_tags(path, want) == set()
    assert bench.recorded_tags(path) == {'round_x_trial0_flavell'}  # tag alone still sees it


def test_smoke_run_and_other_stage_are_different_cells(tmp_path):
    path = str(tmp_path / 'r.jsonl')
    append_record(path, _rec(max_frames=50, accuracy=0.5))
    full = dict(source='nwb', gt='/data/flavell.nwb', descriptor_stage='auto', max_frames=None)
    assert read_results_by_tag(path, full) == {}
    append_record(path, _rec(descriptor_stage='backbone', accuracy=0.8))
    append_record(path, _rec(accuracy=0.9))
    assert read_results_by_tag(path, full)['round_x_trial0_flavell']['accuracy'] == 0.9
    assert read_results_by_tag(path, dict(full, descriptor_stage='backbone'))[
        'round_x_trial0_flavell']['accuracy'] == 0.8


def test_dedup_keeps_one_record_per_tag_and_settings(tmp_path):
    path = str(tmp_path / 'r.jsonl')
    for acc in (0.1, 0.2):
        append_record(path, _rec(accuracy=acc))
    append_record(path, _rec(descriptor_stage='backbone', accuracy=0.3))
    with open(path, 'a') as f:
        f.write(json.dumps(dict(accuracy=1.0)) + '\n')  # tagless junk is kept untouched
    assert deduplicate_results(path) == (4, 3)
    recs = [json.loads(line) for line in open(path)]
    assert sorted(r['accuracy'] for r in recs) == [0.2, 0.3, 1.0]


def test_legacy_defaults_and_paths_normalize():
    # records before nwb_objects / max_frames / gt existed
    legacy = dict(source='nwb', descriptor_stage='auto')
    assert settings_match(legacy, dict(source='nwb', nwb_objects='gt', max_frames=None,
                                       gt='/anything.nwb'))
    assert not settings_match(legacy, dict(nwb_objects='all'))
    assert settings_match(dict(gt='/p/proj/project_config.yaml'), dict(gt='/p/proj/'))


def test_svd_and_gt_matching_separate_cells_and_legacy_records_are_the_defaults(tmp_path):
    import barlow_track.scripts.multiproject_scripts.run_trials_on_all_ground_truth as bench
    path = str(tmp_path / 'r.jsonl')
    want = bench.requested_settings('flavell', _bench_args())
    append_record(path, _rec(gt=want['gt']))  # written before svd / gt_matching were recorded
    assert bench.recorded_tags(path, want) == {'round_x_trial0_flavell'}
    assert bench.recorded_tags(path, bench.requested_settings('flavell', _bench_args(svd='whiten'))) == set()
    append_record(path, _rec(gt=want['gt'], gt_matching='gt_valid', accuracy=0.95))
    assert read_results_by_tag(path, dict(want, gt_matching='gt_valid'))[
        'round_x_trial0_flavell']['accuracy'] == 0.95
    assert read_results_by_tag(path, want)['round_x_trial0_flavell']['accuracy'] == 0.9


def test_nwb_gt_mapping_separates_cells_and_defaults_to_centroid_for_all_objects():
    old_all = _rec(nwb_objects='all')
    assert settings_match(old_all, dict(nwb_objects='all', nwb_gt_mapping='centroid'))
    assert not settings_match(old_all, dict(nwb_gt_mapping='segmentation_id'))
    assert settings_match(_rec(), dict(nwb_objects='gt', nwb_gt_mapping=None))
