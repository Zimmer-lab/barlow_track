"""sensitivity_report.py: variants are the config's difference to the recipe; trials are paired by label and seed."""
import json

import pytest
import yaml

from barlow_track.scripts import sensitivity_report as sr

BASE = dict(lambd_obj=0.35, num_frames=371, seed=7, target_sz_z=8, global_augment=dict(p_flip=0.25, elastic_scale=0.3))


def test_variant_name_ignores_per_recording_keys():
    cfg = dict(BASE, target_sz_z=16, seed=99, project_path='/p', global_augment=dict(p_flip=0.25, elastic_scale=0.0))
    assert sr.variant_name(cfg, BASE) == 'global_augment.elastic_scale=0'
    assert sr.variant_name(dict(BASE, lambd_obj=0.35 + 1e-12), BASE) == ''
    assert sr.variant_name(dict(BASE, weight_decay='1e-06'), dict(BASE, weight_decay=1e-06)) == ''


def test_pairs_by_seed_and_label(tmp_path):
    d = tmp_path / 'round_sens_x_a'
    d.mkdir()
    recs = []
    for n, (extra, seed, acc) in enumerate([({}, 7, 0.90), ({}, 8, 0.92), ({'num_frames': 16}, 7, 0.80), ({'num_frames': 16}, 8, 0.86)]):
        (d / f'trial_{n}').mkdir()
        (d / f'trial_{n}' / 'train_config.yaml').write_text(yaml.safe_dump(dict(BASE, seed=seed, **extra)))
        recs.append(dict(tag=f'round_sens_x_a_trial{n}_a', accuracy=acc))
    (d / 'exp_results.jsonl').write_text('\n'.join(json.dumps(r) for r in recs))
    labels, rows = sr.summarize(sr.collect(str(tmp_path), 'round_sens_*', BASE))
    by = {r['variant']: r for r in rows}
    assert labels == ['a'] and rows[0]['variant'] == 'control'
    assert by['control']['accuracy']['a'] == pytest.approx(0.91)
    assert by['num_frames=16']['paired_diff'] == pytest.approx(-0.08) and by['num_frames=16']['n_pairs'] == 2
