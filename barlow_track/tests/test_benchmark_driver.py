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


def test_nwb_datasets_pass_their_nwb():
    cmd = drv.build_eval_cmd('flavell', 'tag', 'w.pth', _args(), 'r.jsonl', 'emb')
    assert _flag(cmd, '--nwb') == drv.DATASETS['flavell']['nwb']
    assert '--gt' not in cmd
