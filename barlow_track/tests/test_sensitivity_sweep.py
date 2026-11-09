"""Variant bookkeeping of scripts/sensitivity_sweep.py (no projects, no GPU)."""
import os

import pytest
import yaml

from barlow_track.scripts import sensitivity_sweep as ss
from barlow_track.scripts import optimize_shared_config as osc

BASE = dict(epochs=30, num_frames=371, lr=2e-5, lambd_obj=0.35, embedding_dim=512, target_sz_z=8, target_sz_xy=64,
            fusion='add', use_position=True, seed=1, global_augment=dict(p_flip=0.25, max_degrees_z=41.0, elastic_scale=0.35,
                                                                           p_global_affine=1.0),
            crop_photometric=dict(p_noise=0.4, crop_degrees_z=35.0), position_augment=dict(dropout_p=0.06, spurious_p=0.17))


def fake_round_files(name, label, base, grid, z_scale_from_voxels=False):
    cfg = dict(base, project_path=f'/p/{label}', target_sz_z=16 if label.startswith('flavell') else 8)
    return cfg, dict(grid, objective='accuracy', objective_dataset=label)


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(osc, 'round_files', fake_round_files)
    monkeypatch.setattr(ss, 'round_files', fake_round_files, raising=False)
    monkeypatch.setattr(osc.oh, 'resolve_objective', lambda tpl: (None, None, {'gt': None}))
    d = tmp_path / 'round'
    d.mkdir()
    (d / 'base.yaml').write_text(yaml.safe_dump(BASE))
    tpl = dict(base_train_config='base.yaml', source_round='r', base_trial=1, base_params={'lambd_obj': 0.35, 'global_augment.max_degrees_z': 41.0},
               base_seed=100, replicates=3, tuning_labels=['zimmer_a', 'flavell_b'], heldout_labels=[],
               objective_metric='accuracy', reference_accuracy={'zimmer_a': 0.9, 'flavell_b': 0.9},
               variants=ss.default_variants())
    (d / 'sensitivity_template.yaml').write_text(yaml.safe_dump(tpl))
    return ss.SensitivitySweep(str(d))


def test_default_variants_are_unique_and_have_control():
    names = [v['name'] for v in ss.default_variants()]
    assert len(names) == len(set(names)) and ss.CONTROL in names
    assert {v['tier'] for v in ss.default_variants()} == {1, 2}


def test_cell_counts_and_paired_seeds(sweep):
    cells = sweep.cells()
    assert len(cells) == sum((v['replicates']) * 2 for v in sweep.variants)
    seeds = {c['rep']: {c2['seed'] for c2 in cells if c2['rep'] == c['rep']} for c in cells}
    assert all(len(s) == 1 for s in seeds.values())            # one seed per replicate across variants
    assert len({next(iter(s)) for s in seeds.values()}) == len(seeds)   # replicates differ
    assert seeds[0] == {100}                                   # replicate 0 keeps the trial seed
    tags = [osc.cell_tag(sweep.round_dir, c['trial'], c['label']) for c in cells]
    assert len(tags) == len(set(tags))
    assert [c['rep'] for c in cells] == sorted(c['rep'] for c in cells)   # replicate 0 first


def test_control_reproduces_the_frozen_config(sweep):
    c = [x for x in sweep.cells() if x['variant']['name'] == 'control' and x['label'] == 'flavell_b'][0]
    cfg = sweep.cell_config_variant(c['i'], 'flavell_b', c['variant'], c['seed'], 0)
    assert cfg['global_augment']['max_degrees_z'] == 41.0 and cfg['global_augment']['p_flip'] == 0.25
    assert cfg['target_sz_z'] == 16 and cfg['num_frames'] == 371 and cfg['project_path'] == '/p/flavell_b'


def test_overrides_and_multipliers(sweep):
    by = {v['name']: v for v in sweep.variants}
    cell = lambda name, label: sweep.cell_config_variant(sweep.variants.index(by[name]), label, by[name], 1, 0)
    z = cell('crop_z_half', 'flavell_b')
    assert z['target_sz_z'] == 8 and isinstance(z['target_sz_z'], int)          # 16 x 0.5 on Flavell
    assert cell('crop_z_half', 'zimmer_a')['target_sz_z'] == 4
    assert cell('crop_xy48', 'zimmer_a')['target_sz_xy'] == 48
    assert cell('n64_matched_steps', 'zimmer_a')['epochs'] == 174 and cell('n64_matched_steps', 'zimmer_a')['num_frames'] == 64
    assert cell('lr_x3', 'zimmer_a')['lr'] == pytest.approx(6e-5)
    assert cell('untrained', 'zimmer_a')['epochs'] == 0
    img = cell('img_lobj1_n16', 'zimmer_a')
    assert img['fusion'] == 'visual_only' and img['lambd_obj'] == 1.0 and img['num_frames'] == 16
    only = cell('aug_only_affine', 'zimmer_a')
    assert only['global_augment']['p_global_affine'] == 1.0 and only['global_augment']['elastic_scale'] == 0.0
    assert only['crop_photometric']['p_noise'] == 0.0 and only['position_augment']['dropout_p'] == 0.06
    none = cell('aug_none', 'zimmer_a')
    assert none['global_augment']['p_global_affine'] == 0.0 and none['position_augment']['spurious_p'] == 0.0


def test_filters(sweep):
    assert {c['variant']['tier'] for c in sweep.cells(tiers=[2])} == {2}
    assert {c['variant']['name'] for c in sweep.cells(names=['n128'])} == {'n128'}
    assert {c['label'] for c in sweep.cells(labels=['zimmer_a'])} == {'zimmer_a'}
    assert len(sweep.cells(names=['control'], replicates=2)) == 4


def test_report_pairs_against_control(sweep, tmp_path):
    import json
    with open(sweep.results_jsonl, 'w') as f:
        for c in sweep.cells(names=['control', 'n128']):
            acc = 0.9 + 0.01 * c['rep'] + (0.02 if c['variant']['name'] == 'n128' else 0.0)
            f.write(json.dumps(dict(tag=osc.cell_tag(sweep.round_dir, c['trial'], c['label']), accuracy=acc)) + '\n')
    rows = {r['variant']: r for r in sweep.report()}
    assert rows['n128']['paired_diff'] == pytest.approx(0.02) and rows['n128']['n_pairs'] == 6
    assert rows['control']['accuracy']['zimmer_a'] == pytest.approx(0.92)   # mean of 5 replicates 0.90..0.94
    assert rows['lr_x3']['mean'] is None
