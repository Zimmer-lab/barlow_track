"""Load an NWB as ProjectData without reading its whole segmentation into memory.

wbfm's load_final_project_data_from_nwb calls da.squeeze on the raw h5py segmentation
dataset, which reads all of it (35 GB for the Leifer re-export A2) on every load; four
such loads at once tripped systemd-oomd twice on 2026-11-02. Until wbfm wraps the dataset
in da.from_array first, this does it for the duration of the load.
"""
import contextlib

import dask.array as da


@contextlib.contextmanager
def _lazy_squeeze():
    orig = da.squeeze

    def squeeze(a, *args, **kwargs):
        if not isinstance(a, da.Array) and hasattr(a, 'shape') and len(a.shape) > 1:
            a = da.from_array(a, chunks=(1,) + tuple(a.shape[1:]))
        return orig(a, *args, **kwargs)

    da.squeeze = squeeze
    try:
        yield
    finally:
        da.squeeze = orig


def load_nwb_project(path):
    from wbfm.utils.projects.finished_project_data import ProjectData
    with _lazy_squeeze():
        return ProjectData.load_final_project_data(path, allow_hybrid_loading=True, verbose=0)
