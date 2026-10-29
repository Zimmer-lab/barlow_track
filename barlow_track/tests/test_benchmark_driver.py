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
