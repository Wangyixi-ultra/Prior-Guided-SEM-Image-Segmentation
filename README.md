# Prior-Guided SEM Image Segmentation and MQS Ranking

Code for perovskite SEM image segmentation and morphology-quality-score (MQS) ranking.

## Pipeline

1. Build the nnUNet dataset.
2. Run YOLO detection and add the semantic prior channel.
3. Train the U-Mamba segmentation model with the prior channel.
4. Extract morphology features from segmented images.
5. Train and apply the Spearman-4 MQS model.

## Repository Structure

| File | Purpose |
|------|---------|
| `123convert2nnunet.py` | Convert raw SEM images and masks into nnUNet v2 format. |
| `add_yolo_info_features_v2.py` | Run a YOLO detector and write a Gaussian-blob prior channel. |
| `trainers/nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost.py` | Custom nnUNet trainer: U-Mamba + active contour + dual-channel input + semantic adapter. |
| `123combined_processing.py` | Offline training-set augmentation (1 original + 13 variants). |
| `123unified_predict.py` | Run inference on new SEM images and export masks / LabelMe JSON. |
| `analyze_perovskite.py` | Extract morphology descriptors from segmented training images. |
| `update_train_features.py` | Merge newly extracted features into the training spreadsheet. |
| `mqs_spearman4.py` | Train and validate the Spearman-4 MQS model. |
| `analyze_perovskite_sem_test_spearman4.py` | Extract features and score independent test images. |
| `evaluate_spearman4_test.py` | Compare MQS scores with measured PCE on a test set. |

## Requirements

The code was tested in a conda environment with:

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

A full package list is in [`umamba_pero_packages.txt`](umamba_pero_packages.txt).

## Usage

### 1. Build the nnUNet dataset

Edit the paths in `123convert2nnunet.py`, then run:

```bash
python 123convert2nnunet.py
```

Input: raw grayscale SEM images and 8-bit grayscale masks.  
Output: `Dataset123_Perovskite` in the nnUNet raw directory.

### 2. Add the YOLO prior channel

Use a trained YOLO detection model for PbI2 / ABO3 grains to generate the second input channel:

```bash
python add_yolo_info_features_v2.py \
    --weights /path/to/yolo/weights/best.pt \
    --dataset /path/to/U-Mamba/data/nnUNet_raw/Dataset123_Perovskite
```

### 3. Train the U-Mamba segmentation model

Use the custom trainer:

```bash
nnUNetv2_train Dataset123_Perovskite 2d 0 \
    -tr nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost
```

Optional: run `123combined_processing.py` before training to augment the training set.

### 4. Run inference

```bash
python 123unified_predict.py
```

Edit the input/output paths inside the script first.

### 5. Extract morphology features

For training images:

```bash
python -B analyze_perovskite.py
```

Then merge the new features into the training spreadsheet:

```bash
python -B update_train_features.py
```

### 6. Train the Spearman-4 MQS model

```bash
python -B mqs_spearman4.py
```

### 7. Score test images

```bash
python -B analyze_perovskite_sem_test_spearman4.py dataset1
```

### 8. Evaluate MQS against measured PCE

```bash
python -B evaluate_spearman4_test.py dataset1
```

## Data

The AddTrain SEM image subset used in this work is available on Zenodo:

- **Record:** https://doi.org/10.5281/zenodo.21263625
- **DOI:** `10.5281/zenodo.21263625`
