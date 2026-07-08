# Prior-Guided SEM Image Segmentation and MQS ranking

This repository contains the core implementation code for the perovskite SEM image segmentation and MQS ranking experiments.

## Repository Structure

- `123combined_processing.py`: Data augmentation pipeline (14x augmentation: 1 original + 13 augmented variants) for SEM images, YOLO channels, and masks.
- `123compare_all_trainers_test.py`: Test-set evaluation and comparison across all nnUNet trainers on Dataset123_Perovskite.
- `123convert2nnunet.py`: Convert perovskite SEM grayscale images and 8-bit grayscale masks into nnUNet v2 standard format, with train/test split and visualization overlays.
- `123unified_predict.py`: Minimal nnUNet prediction pipeline for single-channel SEM images, producing segmentation masks, border overlays, and LabelMe-format JSON annotations.
- `trainers/nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost.py`: Custom nnUNet trainer integrating U-Mamba, active contour loss, dual-channel input, and semantic boost with a YOLO-based semantic adapter.
- `add_yolo_info_features_v2.py`: Generate YOLO auxiliary channels (Channel 1) for nnUNet dual-channel input, supporting YOLOv8/YOLO11 detection models with Gaussian blob and confidence weighting.
- `analyze_perovskite.py`: Extract perovskite thin-film morphology descriptors (ABX3 grain size, PbI2 coverage, spatial uniformity, etc.) from SEM images and LabelMe JSON annotations.
- `update_train_features.py`: Merge the latest SEM morphology features into the training data file.
- `mqs_spearman4.py`: Train and validate a Spearman-4 Morphology Quality Score (MQS) model that links SEM descriptors to device PCE.
- `analyze_perovskite_sem_test_spearman4.py`: Extract SEM features from independent test sets and score them with the trained Spearman-4 MQS model.
- `evaluate_spearman4_test.py`: Evaluate the generalisation performance of Spearman-4 MQS against measured PCE on independent test datasets.

## Dataset

The AddTrain subset of the SEM image dataset used in this work is available on Zenodo:

- **Zenodo Record**: https://doi.org/10.5281/zenodo.21263625
- **DOI**: `10.5281/zenodo.21263625`

## Requirements

The code was developed and tested in the `umamba_pero` conda environment. Key dependencies include:

| Package | Version |
|---------|---------|
| torch | 2.9.0+cu128 |
| torchvision | 0.24.0+cu128 |
| torchaudio | 2.9.0+cu128 |
| pytorch-cuda | 11.8 |
| nnunetv2 | 2.1.1 |
| mamba-ssm | 2.2.6.post3 |
| monai | 1.3.0 |
| ultralytics | 8.4.53 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| scikit-image | 0.25.2 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |
| seaborn | 0.13.2 |
| pillow | 12.0.0 |
| opencv-python | 4.11.0.86 |
| batchgenerators | 0.25.1 |
| dynamic-network-architectures | 0.4.2 |
| albumentations | 2.0.8 |
| simpleitk | 2.5.2 |
| nibabel | 5.3.2 |
| tqdm | 4.65.2 |

A complete list of all ~255 packages in the `umamba_pero` environment is provided in [`umamba_pero_packages.txt`](umamba_pero_packages.txt).

## Usage

1. Prepare the dataset with `123convert2nnunet.py`.
2. Apply augmentation with `123combined_processing.py` if needed.
3. Train using the custom trainer `nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost`.
4. Evaluate and compare trainers with `123compare_all_trainers_test.py`.
5. Run inference with `123unified_predict.py`.

### Spearman-4 MQS pipeline (morphology-to-PCE scoring)

1. Extract training-set morphology features:
   ```bash
   python -B analyze_perovskite.py
   ```
2. Update the training data with the newly extracted features:
   ```bash
   python -B update_train_features.py
   ```
3. Train and validate the Spearman-4 MQS model:
   ```bash
   python -B mqs_spearman4.py
   ```
4. Extract features and score an independent test set:
   ```bash
   python -B analyze_perovskite_sem_test_spearman4.py [dataset_name]
   ```
5. Evaluate MQS against measured PCE:
   ```bash
   python -B evaluate_spearman4_test.py [dataset_name]
   ```


