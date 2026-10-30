"""When training turns on DataLoader workers by itself (_auto_num_workers)."""
from types import SimpleNamespace

import numpy as np
import torch
import zarr

import barlow_track.scripts.train_barlow_clusterer as tbc


class _DataModule:
    """Stands in for VolumeCoordsDataModule: alldata.project_data.red_data, and a loader
    whose first batch draws from the global torch RNG (as a real worker start does)."""

    def __init__(self, red, fail=False):
        self.alldata = SimpleNamespace(project_data=SimpleNamespace(red_data=red))
        self.num_workers = 0
        self.fail = fail

    def train_dataloader(self):
        def batches():
            torch.rand(1)
            if self.fail:
                raise RuntimeError("DataLoader worker (pid 1) is killed by signal: Bus error")
            yield self.num_workers
        return batches()


def _args(**kw):
    return SimpleNamespace(**{'num_workers': 0, 'fusion': 'add', **kw})


def _zarr_red():
    return zarr.zeros((2, 3, 4, 4), chunks=(1, 3, 4, 4))


def test_slurm_gpu_job_gets_cpus_minus_two_and_keeps_the_rng(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    torch.manual_seed(0)
    expected = torch.rand(1)
    torch.manual_seed(0)
    assert tbc._auto_num_workers(_args(), _DataModule(_zarr_red())) == 6
    assert torch.equal(torch.rand(1), expected)


def test_auto_stays_off_outside_slurm_gpu_zarr(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    assert tbc._auto_num_workers(_args(), _DataModule(_zarr_red())) == 0  # local run
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
    assert tbc._auto_num_workers(_args(num_workers_auto=False), _DataModule(_zarr_red())) == 0
    assert tbc._auto_num_workers(_args(num_workers=3), _DataModule(_zarr_red())) == 3
    assert tbc._auto_num_workers(_args(), _DataModule(np.zeros((2, 3, 4, 4)))) == 0  # not zarr
    assert tbc._auto_num_workers(_args(fusion='position_only', keypoints_only=True),
                                 _DataModule(_zarr_red())) == 0
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)  # a CPU training job
    assert tbc._auto_num_workers(_args(), _DataModule(_zarr_red())) == 0


def test_failing_worker_start_falls_back_to_zero(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert tbc._auto_num_workers(_args(), _DataModule(_zarr_red(), fail=True)) == 0
