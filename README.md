# CMWNet.pytorch

This project link: <https://github.com/lartpang/CMWNet.pytorch>

This project provides an unofficial pytorch implementation of the RGB-D SOD method for
'Cross-Modal Weighting Network for RGB-D Salient Object Detection', ECCV 2020.

- Paper Link: <https://arxiv.org/pdf/2007.04901.pdf>
- Official Caffe Code: <https://github.com/MathLee/CMWNet>

## Environment

- `torch` >= 1.5.0 for AMP.
- `torchvision`
- `numpy`
- `tqdm`
- `opencv-python`
- `pillow`
- `py_sod_metrics` for evaluating the results.

## Usage

1. Configure your datasets in `dataset_config.py`.
2. Set hyperparameters in `main.py`.
3. `python main.py`
4. All results will be saved into the `ckpt_path` (see `main.py`).
