"""run_trials_on_all_ground_truth.py command lines (synthetic, fast)."""
import os
from types import SimpleNamespace

import pytest

from barlow_track.scripts.multiproject_scripts import run_trials_on_all_ground_truth as drv


def _args():
    return SimpleNamespace(mode='trained', device='cpu', cluster='labelprop', num_seeds=25,
                           descriptor_stage='auto', max_frames=None,
                           center_per_volume=False, l2_per_volume=False)


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


@pytest.mark.parametrize('lab', [k for k, v in drv.DATASETS.items() if v['source'] == 'project'])
def test_project_datasets_score_against_their_own_gt(lab):
    cmd = drv.build_eval_cmd(lab, 'tag', 'w.pth', _args(), 'r.jsonl', 'emb')
    gt = _flag(cmd, '--gt')
    assert gt is not None
    assert os.path.dirname(gt) == drv.DATASETS[lab]['gt']
    assert os.path.dirname(_flag(cmd, '--project')) == drv.DATASETS[lab]['project']


@pytest.mark.parametrize('lab', [k for k, v in drv.DATASETS.items() if v['source'] == 'nwb'])
def test_nwb_datasets_pass_their_nwb_as_source_and_gt(lab):
    cmd = drv.build_eval_cmd(lab, 'tag', 'w.pth', _args(), 'r.jsonl', 'emb')
    assert _flag(cmd, '--nwb') == _flag(cmd, '--gt') == drv.DATASETS[lab]['nwb']
    # The record's gt then matches what a resumed run asks for, so finished cells are skipped
    assert drv.requested_settings(lab, _args())['gt'] == drv.DATASETS[lab]['nwb']
    assert _flag(cmd, '--lab') in ('flavell', 'samuel', 'leifer')


@pytest.mark.parametrize('device, gres', [('cuda', 'gpu:1'), ('cpu', None)])
def test_slurm_asks_for_a_gpu_only_when_embedding_on_one(monkeypatch, tmp_path, device, gres):
    import sys
    seen = {}

    class FakeExecutor:
        def __init__(self, folder, cluster):
            pass

        def update_parameters(self, **kw):
            seen.update(kw)
    monkeypatch.setitem(sys.modules, 'submitit', SimpleNamespace(AutoExecutor=FakeExecutor))
    args = SimpleNamespace(slurm_time='01:00:00', slurm_cpus=4, slurm_mem='8G', device=device, fail_fast=True)
    assert drv.run_slurm([], str(tmp_path / 'r.jsonl'), args) == []
    assert seen.get('slurm_gres') == gres


def test_stardist_gts_share_one_embedding_and_run_chained():
    """leifer_stardist_maskgt crops the same project as leifer_stardist: its cell loads that
    cell's embedding cache and runs right after it in the same worker."""
    labs = ['leifer_stardist', 'leifer_stardist_maskgt', 'zimmer_1128']
    tasks = []
    for trial in (3, 7):
        for lab in labs:
            tag = f'round_x_trial{trial}_{lab}'
            tasks.append((lab, trial, tag, drv.build_eval_cmd(lab, tag, 'w.pth', _args(), 'r.jsonl', 'emb')))
    for lab, trial, tag, cmd in tasks:
        if lab == 'zimmer_1128':
            assert '--emb_tag' not in cmd and '--skip_embed' not in cmd
        else:
            assert _flag(cmd, '--emb_tag') == f'round_x_trial{trial}_leifer_stardist'
            assert '--skip_embed' in cmd
    chained = drv.chain_shared_embeddings(tasks)
    assert [(lab, trial) for lab, trial, _, _ in chained] == [
        ('leifer_stardist', 3), ('zimmer_1128', 3), ('leifer_stardist', 7), ('zimmer_1128', 7)]
    sd3 = chained[0][3]
    assert [_flag(c, '--tag') for c in sd3] == ['round_x_trial3_leifer_stardist', 'round_x_trial3_leifer_stardist_maskgt']
    # A maskgt cell whose stardist cell is already recorded runs alone (and loads the cache)
    alone = drv.chain_shared_embeddings([t for t in tasks if t[0] != 'leifer_stardist'])
    assert sum(lab == 'leifer_stardist_maskgt' for lab, _, _, _ in alone) == 2


def test_run_one_runs_a_chain_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(drv.subprocess, 'call', lambda c, env: calls.append((c, env['CUDA_VISIBLE_DEVICES'])) or 0)
    assert drv.run_one([['a'], ['b']], '', 'tag') == 0
    assert calls == [(['a'], ''), (['b'], '')]
    assert drv.resolve_gpus('none') == ['']
