# BarlowTrack
Using a modification of Barlow Twins (self-supervised learning) to track single-cell resolution microscopy data, specifically designed for _C. elegans_ neurons.
Please cite the preprint if you use this work:

```
@article{fieseler2025barlowtrack,
  title={BarlowTrack: A Self-Supervised Framework for Zero-Shot Multi-Object Cell Tracking},
  author={Fieseler, Charles and Lev, Itamar and Madhusudhanan, Jalaja and Zhai, Zihao and Schwartz, Siegfried and Zimmer, Manuel},
  journal={bioRxiv},
  pages={2025--11},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```


## Installation

### Install together with pytorch

Use the conda/mamba environment in the barlow_track.yaml file. Installation using mamba (and running) is tested on:
- Rocky Linux
- Ubuntu 22.04

### Install pytorch yourself

Depending on your gpu setup, it is easier for you to install pytorch yourself and then update the rest of the environment using our simplified yaml file, `barlow_track_without_torch.yaml`.
In this case, please install the following packages using instructions from their websites:
- pytorch
- torchaudio
- torchvision
- pytorch-cuda (if using conda/mamba)
- pytorch-lightning
- torchio
- torch_geometric
- torch-scatter
- torch-sparse


## Evaluating position encodings against ground truth

See [docs/accuracy_evaluation.md](docs/accuracy_evaluation.md) for the full handoff:
datasets, weights, eval scripts (`barlow_track/scripts/eval_accuracy.py`,
`eval_leifer_accuracy.py`), results, and environment gotchas.

## Training a network

See instructions in the [project folder](barlow_track/barlow_project_template/README.md)

## Visualizing augmentation (napari GUI)

Tune Step 1 (global-before-crop) augmentation by eye with a test project:

```
conda activate MY_ENV  # includes napari[all]
python barlow_track/scripts/view_volume_augmentation.py \
    --project_path /home/charles/Current_work/test_projects/barlow/worm4-2-2025-03-09/project_config.yaml \
    --frame 0
```

Options: `--frame` (timepoint), `--target_sz z x y` (default `8 64 64`), `--seed`.
On the cluster use the `/lisc/...` mirror of the same test project
(`.../wbfm/test_projects/barlow/worm4-2-2025-03-09/project_config.yaml`),
or set `BARLOW_TEST_PROJECT` (same convention as `test_step0_embed_and_augment.py`).

This opens two napari viewers (full volume + single-neuron crop, raw vs augmented
with points and crop box) plus a dock widget for all augmentation parameters;
press `Re-augment` to resample with the new settings.

## Tracking a BarlowTrack network

This is organized via the sibling repository: [wbfm](https://github.com/Zimmer-lab/wbfm).

The main instructions are found here: [Running the full pipeline](https://github.com/Zimmer-lab/wbfm/blob/main/docs/running_the_pipeline.md).
