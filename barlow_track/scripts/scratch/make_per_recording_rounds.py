"""Write one sweep round per GT recording from one base config and one grid.

The final architecture is trained on each recording (self-supervised: a network per
dataset; the user, 2026-10-29), so a round compares candidate recipes recording by
recording. Each round directory gets train_config.yaml (the base with that recording's
project_path and crop depth) and hyperparameter_search_template.yaml (the grid, with the
objective scored on that recording's benchmark cell: objective_dataset is the benchmark
label, and NWB-source labels set objective_source / objective_gt, as round_posonly2_*).

    python make_per_recording_rounds.py --name round_final1 --base base.yaml --grid grid.yaml \
        --out barlow_track/scripts/scratch [--labels zimmer_1128 flavell ...] [--preflight]

grid.yaml holds the template's keys (hyperparameters, train_hours, objective_cpu_job, ...);
objective, objective_dataset, objective_source and objective_gt are set here.
"""
import argparse
import copy
import os

import yaml

from barlow_track.scripts.multiproject_scripts.run_trials_on_all_ground_truth import DATASETS

_PAPER_OTHER = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper"

# New wbfm projects built from the NWBs (scratch/wbfm_tracker_baseline): segmentation
# metadata recalculated from the masks, zero-brightness centroids fixed
_NEW_PROJECTS = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline"

# Benchmark label -> the wbfm project trained on (with segmentation metadata). Zimmer
# labels train on their GT project itself. Leifer trains on the project of its re-export
# (NeRVE's masks only, no GT-seeded masks) and is scored on the old Leifer NWB, as the
# benchmark's 'leifer' cell; samuel_137 / 184 train on projects built from their NWBs.
TRAIN_PROJECTS = {
    **{k: os.path.join(v['project'], 'project_config.yaml')
       for k, v in DATASETS.items() if k.startswith('zimmer_')},
    'flavell': os.path.join(_PAPER_OTHER, 'flavell_data/projects/2025_07_01/project_config.yaml'),
    'flavell_2022031603': os.path.join(_PAPER_OTHER, 'flavell_data/projects/for_Charlie2-2022-03-16-03/project_config.yaml'),
    'flavell_2022032402': os.path.join(_PAPER_OTHER, 'flavell_data/projects/forCharlie_2-2022-03-24-02/project_config.yaml'),
    'flavell_2022033001': os.path.join(_PAPER_OTHER, 'flavell_data/projects/forCharlie_2-2022-03-30-01/project_config.yaml'),
    'flavell_2022033002': os.path.join(_PAPER_OTHER, 'flavell_data/projects/forCharlie_2-2022-03-30-02/project_config.yaml'),
    'samuel': os.path.join(_PAPER_OTHER, 'samuel_data/projects/153/project_config.yaml'),
    'samuel_137': os.path.join(_NEW_PROJECTS, 'samuel_137/137/project_config.yaml'),
    'samuel_184': os.path.join(_NEW_PROJECTS, 'samuel_184/184-004/project_config.yaml'),
    'leifer': os.path.join(_NEW_PROJECTS, 'leifer_reexport/Leifer_NeRVE_Worm1_reexport_A/project_config.yaml'),
}

# Flavell neurons span median 8 / p90 12 / max 16 z-planes (AGENTS.md); z=8 truncates them
TARGET_SZ_Z = {'flavell': 16}

# Zimmer voxels (the projects' physical units are real; the other labs' are placeholders)
_ZIMMER_UM_XY, _ZIMMER_UM_Z = 0.325, 1.5


def voxel_aspect(label):
    """z / xy voxel size of a recording, from GT-free metadata: the NWB grid_spacing
    (Zimmer: the lab's voxel size). visnet_z_scale = this undoes the z compression of
    voxel-unit keypoints (batches 22-23)."""
    ds = DATASETS[label]
    if ds['source'] != 'nwb':
        return _ZIMMER_UM_Z / _ZIMMER_UM_XY
    import h5py
    import numpy as np
    with h5py.File(ds['nwb'], 'r') as h:
        opt = h['general/optophysiology']
        gs = [np.asarray(opt[k]['grid_spacing'][()], dtype=float) for k in opt if 'grid_spacing' in opt[k]][0]
    if not np.isclose(gs[0], gs[1]):
        raise ValueError(f"{label}: anisotropic xy grid_spacing {gs}")
    return float(gs[2] / gs[0])


def round_files(name, label, base, grid, z_scale_from_voxels=False):
    """(train_config dict, template dict) of one recording's round."""
    cfg = copy.deepcopy(base)
    cfg['project_path'] = TRAIN_PROJECTS[label]
    cfg['target_sz_z'] = TARGET_SZ_Z.get(label.split('_')[0], cfg.get('target_sz_z', 8))
    if z_scale_from_voxels:
        cfg['visnet_z_scale'] = round(voxel_aspect(label), 3)
    if 'wandb_name' in cfg:
        cfg['wandb_name'] = f"barlow_track_{name}_{label}"
    tpl = copy.deepcopy(grid)
    tpl.update(baseline_params_path=None, objective='accuracy', objective_dataset=label)
    ds = DATASETS[label]
    if ds['source'] == 'nwb':
        tpl.update(objective_source='nwb', objective_gt=ds['nwb'])
    return cfg, tpl


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--name', required=True)
    parser.add_argument('--base', required=True, help='base train_config.yaml')
    parser.add_argument('--grid', required=True, help='template keys (hyperparameters etc.)')
    parser.add_argument('--out', required=True, help='parent directory of the round directories')
    parser.add_argument('--labels', nargs='+', default=sorted(TRAIN_PROJECTS))
    parser.add_argument('--preflight', action='store_true',
                        help="run each round's objective preflight (reads the projects / NWBs)")
    parser.add_argument('--z_scale_from_voxels', action='store_true',
                        help="set visnet_z_scale to each recording's z / xy voxel size (voxel_aspect)")
    args = parser.parse_args()
    with open(args.base) as f:
        base = yaml.safe_load(f)
    with open(args.grid) as f:
        grid = yaml.safe_load(f)
    for label in args.labels:
        cfg, tpl = round_files(args.name, label, base, grid, args.z_scale_from_voxels)
        d = os.path.join(args.out, f"{args.name}_{label}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'train_config.yaml'), 'w') as f:
            f.write(f"# {args.name} on {label}: generated by make_per_recording_rounds.py from {os.path.basename(args.base)}\n")
            yaml.safe_dump(cfg, f, sort_keys=True)
        with open(os.path.join(d, 'hyperparameter_search_template.yaml'), 'w') as f:
            f.write(f"# {args.name} on {label}: generated by make_per_recording_rounds.py from {os.path.basename(args.grid)}\n")
            yaml.safe_dump(tpl, f, sort_keys=False)
        if args.preflight:
            from barlow_track.scripts.eval_accuracy import preflight_trained_eval
            summary = preflight_trained_eval(cfg['project_path'], gt=tpl.get('objective_gt'),
                                             source=tpl.get('objective_source', 'project'))
            print(f"{label}: preflight OK: {summary}", flush=True)
        print(d)


if __name__ == '__main__':
    main()
