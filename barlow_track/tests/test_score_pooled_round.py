"""score_pooled_round.py on a fake pooled round and a fake reference round (no projects, no GPU)."""
import json
from types import SimpleNamespace

import pytest
import yaml

from barlow_track.scripts.scratch.transfer import score_pooled_round as spr

REF_Z = {'zimmer_1210': 4.615, 'flavell_2022033002': 1.0, 'zimmer_1128': 4.615}


def _jsonl(path, records):
    with open(path, 'a') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')


@pytest.fixture
def rounds(tmp_path):
    ref = tmp_path / 'round_shared_v1'
    for label in list(REF_Z) + ['zimmer_1128_rep1']:
        (ref / 'trial_59' / label).mkdir(parents=True)
        (ref / 'trial_59' / label / 'resnet50.pth').write_text('w')
    own = [dict(tag=f'round_shared_v1_trial59_{l}', lab=l, source='project', project=f'/p/{l}', gt=f'/p/{l}',
                seed=0, crop_window='slide', crop_center='trunc', accuracy=0.9,
                train_config=dict(visnet_z_scale=z, target_sz_z=16 if 'flavell' in l else 8, target_sz_xy=64))
           for l, z in REF_Z.items()]
    _jsonl(ref / 'exp_results.jsonl', own)
    _jsonl(ref / 'exp_results_v2.jsonl', [dict(r, accuracy=0.95) for r in own])
    pooled = tmp_path / 'round_pooled_v1'
    for n, (frames, seed, finished) in enumerate([(16, 1, True), (16, 2, True), (64, 1, True), (64, 2, False)]):
        d = pooled / f'trial_{n}'
        (d / 'log').mkdir(parents=True)
        cfg = dict(num_frames=frames, seed=seed, epochs=2, target_sz_z=8, target_sz_xy=64,
                   project_dir=str(d), project_paths=[dict(label='zimmer_1210', project_path='/p/a'),
                                                      dict(label='flavell_2022033002', project_path='/p/b')])
        (d / 'train_config.yaml').write_text(yaml.safe_dump(cfg))
        (d / 'log' / 'stats.json').write_text(json.dumps([{}, dict(epoch=1 if finished else 0)]))
        (d / 'resnet50.pth').write_text('w')
    return SimpleNamespace(ref=ref, pooled=pooled)


def _a(rounds, **kw):
    return SimpleNamespace(**{**dict(round_dir=str(rounds.pooled), ref_round_dir=None, ref_trial=59, trials=None,
                                     targets=None, tuning=['zimmer_1210', 'flavell_2022033002'], labelprop='v2',
                                     track_device='cpu', num_shards=1, out_jsonl=None, emb_dir=None), **kw})


def test_cells_are_every_finished_trial_on_every_reference_recording(rounds):
    cells = spr.cells(_a(rounds))
    assert len(cells) == 3 * 3  # trial_3 is unfinished; the _rep network is not a target
    by_tag = {c[0]: c for c in cells}
    tag = 'round_pooled_v1_trial2_on_flavell_2022033002'
    _, trial, target, weights, tgt, z, crop, cfg = by_tag[tag]
    assert (trial, target) == (2, 'flavell_2022033002')
    assert weights.endswith('round_pooled_v1/trial_2/resnet50.pth')
    assert z == 1.0  # the TARGET's z_scale
    assert crop == [8, 64, 64]  # the checkpoint's crop, not the target network's 16
    assert tgt['tag'] == 'round_shared_v1_trial59_flavell_2022033002'
    assert {c[2] for c in spr.cells(_a(rounds, targets=['zimmer_1128']))} == {'zimmer_1128'}


def test_run_shard_scores_with_the_target_settings_and_skips_done_cells(rounds, monkeypatch):
    import barlow_track.scripts.eval_accuracy as ea
    calls = []

    def fake_eval(weights, project, tag, results_jsonl, visnet_z_scale, extra_record, **kw):
        calls.append(dict(tag=tag, z=visnet_z_scale, project=project, **kw))
        rec = dict(tag=tag, accuracy=0.8, **extra_record)
        _jsonl(results_jsonl, [rec])
        return rec

    monkeypatch.setattr(ea, 'evaluate_trained_checkpoint', fake_eval)
    a = _a(rounds, trials=[0])
    assert spr.run_shard(a, 0) == 0
    assert len(calls) == 3
    c = next(c for c in calls if c['tag'].endswith('_on_zimmer_1128'))
    assert c['z'] == 4.615 and c['project'] == '/p/zimmer_1128'
    assert c['crop_window'] == 'slide' and c['labelprop'] == 'v2'
    rec = spr.latest_records(str(rounds.pooled / 'exp_results_pooled_v2.jsonl'))['round_pooled_v1_trial0_on_zimmer_1128']
    assert rec['pooled_num_frames'] == 16 and rec['pooled_labels'] == ['zimmer_1210', 'flavell_2022033002']
    spr.run_shard(a, 0)
    assert len(calls) == 3  # rerun: every cell done


def test_summary_means_over_seeds_and_over_tuning_vs_held_out():
    labels = ['a', 'b', 'c']
    cfgs = {0: dict(num_frames=16, seed=1), 1: dict(num_frames=16, seed=2), 2: dict(num_frames=64, seed=1)}
    acc = {(0, 'a'): 0.5, (0, 'b'): 0.7, (0, 'c'): 0.2,
           (1, 'a'): 0.7, (1, 'b'): 0.9, (1, 'c'): 0.4,
           (2, 'a'): 0.9}
    keys, trials, settings = spr.summarize(acc, cfgs, labels, tuning={'a', 'b'})
    assert keys == ['num_frames']
    assert trials[0]['tuning'] == pytest.approx(0.6) and trials[0]['held_out'] == pytest.approx(0.2)
    s16, s64 = settings
    assert s16['setting'] == (('num_frames', 16),) and s16['seeds'] == [1, 2]
    assert s16['acc'] == pytest.approx({'a': 0.6, 'b': 0.8, 'c': 0.3})
    assert s16['tuning'] == pytest.approx(0.7) and s16['held_out'] == pytest.approx(0.3)
    # sample SD over seeds of the per-seed tuning means (0.6, 0.8)
    assert s16['tuning_sd'] == pytest.approx(0.02 ** 0.5)
    assert (s64['n_tuning'], s64['n_held_out'], s64['held_out']) == (1, 0, None)


def test_report_runs_on_the_fake_round(rounds, monkeypatch, capsys):
    import barlow_track.scripts.eval_accuracy as ea
    monkeypatch.setattr(ea, 'evaluate_trained_checkpoint',
                        lambda weights, project, tag, results_jsonl, extra_record, **kw:
                        _jsonl(results_jsonl, [dict(tag=tag, accuracy=0.8)]) or dict(accuracy=0.8))
    spr.run_shard(_a(rounds), 0)
    spr.report(_a(rounds))
    out = capsys.readouterr().out
    assert 'num_frames=16' in out and 'num_frames=64' in out and 'own network (t59)' in out
