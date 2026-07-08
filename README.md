# Prior-Guided SEM Image Segmentation

This repository contains the core implementation code for the Dataset123 perovskite SEM image segmentation experiments.

## Repository Structure

- `123combined_processing.py`: Data augmentation pipeline (14x augmentation: 1 original + 13 augmented variants) for SEM images, YOLO channels, and masks.
- `123compare_all_trainers_test.py`: Test-set evaluation and comparison across all nnUNet trainers on Dataset123_Perovskite.
- `123convert2nnunet.py`: Convert perovskite SEM grayscale images and 8-bit grayscale masks into nnUNet v2 standard format, with train/test split and visualization overlays.
- `123unified_predict.py`: Minimal nnUNet prediction pipeline for single-channel SEM images, producing segmentation masks, border overlays, and LabelMe-format JSON annotations.
- `trainers/nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost.py`: Custom nnUNet trainer integrating U-Mamba, active contour loss, dual-channel input, and semantic boost with a YOLO-based semantic adapter.

## Dataset

The AddTrain subset of the SEM image dataset used in this work is available on Zenodo:

- **Zenodo Record**: https://doi.org/10.5281/zenodo.21263625
- **DOI**: `10.5281/zenodo.21263625`

This subset corresponds to the `raw/addtrain/` directory and contains annotated SEM images for prior-guided SEM image segmentation.

## Requirements

For the custom trainer dependencies, see the original U-Mamba setup and install the required packages (e.g., `nnunetv2`, `monai`, `torch`).

## Usage

1. Prepare the dataset with `123convert2nnunet.py`.
2. Apply augmentation with `123combined_processing.py` if needed.
3. Train using the custom trainer `nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost`.
4. Evaluate and compare trainers with `123compare_all_trainers_test.py`.
5. Run inference with `123unified_predict.py`.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{...,
  title={Prior-Guided SEM Image Segmentation},
  author={...},
  journal={...},
  year={...}
}
```
