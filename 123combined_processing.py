#!/usr/bin/env python3
"""
Perovskite SEM data augmentation (14x version = 1 original + 13 augmentations)
Only the training set is augmented; output is written directly back to the original nnUNet directory.

Augmentation policy:
  Aug01~Aug07: geometric transformations (applied to SEM, YOLO channel, and mask simultaneously)
  Aug08~Aug12: pixel-level augmentations (applied to SEM only; YOLO channel and mask stay unchanged)
  Aug13: combined augmentation (geometric on all three + pixel-level on SEM only)
"""
import os
import cv2
from pathlib import Path
import albumentations as A
import numpy as np
import argparse

# ---------- 1. Original nnUNet paths ----------
TRAIN_IMG_DIR   = Path('U-Mamba/data/nnUNet_raw/Dataset123_Perovskite/imagesTr')
TRAIN_LABEL_DIR = Path('U-Mamba/data/nnUNet_raw/Dataset123_Perovskite/labelsTr')

# ---------- 2. Definition of 13 augmentation levels ----------

def get_aug_transforms():
    """Return the 13 augmentation transform definitions."""
    augs = []

    # Aug01: horizontal flip (geometric)
    augs.append(A.ReplayCompose([
        A.HorizontalFlip(p=1.0),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug02: vertical flip (geometric)
    augs.append(A.ReplayCompose([
        A.VerticalFlip(p=1.0),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug03: rotate 90° (geometric)
    augs.append(A.ReplayCompose([
        A.Rotate(limit=(90, 90), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug04: rotate 180° or 270° (geometric)
    augs.append(A.ReplayCompose([
        A.OneOf([
            A.Rotate(limit=(180, 180), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=(270, 270), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        ], p=1.0),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug05: small-angle rotation ±5°~±15° (geometric)
    augs.append(A.ReplayCompose([
        A.OneOf([
            A.Rotate(limit=(-15, -5), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=(5, 15), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        ], p=1.0),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug06: translation + scaling (geometric)
    augs.append(A.ReplayCompose([
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=0,
            interpolation=cv2.INTER_NEAREST,
            p=1.0
        ),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug07: random crop and resize (geometric)
    # Placeholder; size is passed dynamically in apply_augment
    augs.append({'type': 'RandomResizedCrop', 'scale': (0.80, 0.95), 'ratio': (0.9, 1.1)})

    # Aug08: brightness/contrast change (pixel, SEM only)
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ])
    })

    # Aug09: gamma correction (pixel, SEM only)
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.RandomGamma(gamma_limit=(75, 125), p=1.0),
        ])
    })

    # Aug10: CLAHE (pixel, SEM only)
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=1.0),
        ])
    })

    # Aug11: Gaussian noise (pixel, SEM only)
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.GaussNoise(var_limit=(1.0, 4.0), mean=0, p=1.0),
        ])
    })

    # Aug12: slight blur/sharpen (pixel, SEM only)
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
            ], p=1.0),
        ])
    })

    # Aug13: combined augmentation (geometric on all three + pixel-level on SEM only)
    augs.append({
        'geo': A.ReplayCompose([
            A.OneOf([
                A.Rotate(limit=(-8, -3), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                A.Rotate(limit=(3, 8), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
            ], p=1.0),
        ], additional_targets={'mask': 'mask', 'yolo': 'image'}),
        'pixel': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=1.0),
            A.GaussNoise(var_limit=(0.5, 2.0), mean=0, p=1.0),
        ])
    })

    return augs


# ---------- 3. Saving utilities ----------
def save_channel(img, stem, suffix, idx, output_dir, channel_idx):
    """Save a single-channel image."""
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    if idx == 0:
        img_name = f'{stem}_{channel_idx:04d}{suffix}'
    else:
        img_name = f'{stem}_aug{idx:02d}_{channel_idx:04d}{suffix}'
    cv2.imwrite(str(output_dir / img_name), img)


def save_triplet(sem_img, yolo_img, mask, stem, suffix, idx, img_dir, label_dir):
    """Save the SEM, YOLO channel, and mask triplet."""
    save_channel(sem_img, stem, suffix, idx, img_dir, 0)
    save_channel(yolo_img, stem, suffix, idx, img_dir, 1)
    if idx == 0:
        lab_name = f'{stem}.png'
    else:
        lab_name = f'{stem}_aug{idx:02d}.png'
    cv2.imwrite(str(label_dir / lab_name), mask)


# ---------- 4. Apply augmentations ----------
def apply_augment(sem_img, yolo_img, mask, aug, h, w):
    """
    Apply a single augmentation transform.
    Geometric transforms are applied to sem_img, yolo_img, and mask simultaneously.
    Pixel transforms are applied to sem_img only.
    Returns (aug_sem, aug_yolo, aug_mask).
    """
    # Aug07 placeholder: dynamically create RandomResizedCrop
    if isinstance(aug, dict) and aug.get('type') == 'RandomResizedCrop':
        geo = A.ReplayCompose([
            A.RandomResizedCrop(
                size=(h, w),
                scale=aug['scale'],
                ratio=aug['ratio'],
                interpolation=cv2.INTER_NEAREST,
                p=1.0
            ),
        ], additional_targets={'mask': 'mask', 'yolo': 'image'})
        res = geo(image=sem_img, mask=mask, yolo=yolo_img)
        return res['image'], res['yolo'], res['mask']

    if isinstance(aug, dict):
        geo = aug.get('geo')
        pixel = aug.get('pixel')
    else:
        geo = aug
        pixel = None

    # Geometric transform (applied to SEM, YOLO, and mask simultaneously)
    if geo is not None:
        res = geo(image=sem_img, mask=mask, yolo=yolo_img)
        aug_sem = res['image']
        aug_yolo = res['yolo']
        aug_mask = res['mask']
    else:
        aug_sem = sem_img.copy()
        aug_yolo = yolo_img.copy()
        aug_mask = mask.copy()

    # Pixel transform (applied to SEM only)
    if pixel is not None:
        pixel_res = pixel(image=aug_sem)
        aug_sem = pixel_res['image']

    return aug_sem, aug_yolo, aug_mask


# ---------- 5. Training data augmentation ----------
def augment_training_data():
    """Process training data: original + 13 augmentations (14x)."""
    augs = get_aug_transforms()
    supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    count = 0

    for img_path in sorted(TRAIN_IMG_DIR.rglob('*')):
        if img_path.suffix.lower() not in supported:
            continue

        # Only process original SEM images (*_0000.png)
        if '_0001' in img_path.name:
            continue

        # Corresponding YOLO auxiliary channel
        yolo_path = TRAIN_IMG_DIR / img_path.name.replace('_0000', '_0001')
        if not yolo_path.exists():
            print(f'Skipping {img_path.name}, no corresponding YOLO channel')
            continue

        # Corresponding label
        label_name = img_path.name.replace('_0000', '')
        label_path = TRAIN_LABEL_DIR / label_name
        if not label_path.exists():
            print(f'Skipping {img_path.name}, no corresponding label')
            continue

        sem = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        yolo = cv2.imread(str(yolo_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if sem is None or yolo is None or mask is None:
            print('Invalid file:', img_path, yolo_path, label_path)
            continue

        # Ensure consistent sizes
        h, w = sem.shape[:2]
        if yolo.shape != (h, w) or mask.shape != (h, w):
            print(f'Size mismatch, skipping: {img_path.name}')
            continue

        # Add channel dimension for albumentations
        sem = sem[..., None]   # (H,W,1)
        yolo = yolo[..., None] # (H,W,1)

        stem = img_path.stem.split('_')[0]
        suffix = img_path.suffix

        # Index 0: original image
        save_triplet(sem.squeeze(-1), yolo.squeeze(-1), mask, stem, suffix, 0, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

        # Indices 1~13: augmented
        for i in range(1, 14):
            aug = augs[i - 1]
            aug_sem, aug_yolo, aug_mask = apply_augment(sem, yolo, mask, aug, h, w)
            save_triplet(aug_sem.squeeze(-1), aug_yolo.squeeze(-1), aug_mask, stem, suffix, i, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

        count += 1

    total = count * 14
    print(f'✅ Training set augmentation complete: {count} original images → {total} cases')
    print('Images:', TRAIN_IMG_DIR.resolve())
    print('Labels:', TRAIN_LABEL_DIR.resolve())


# ---------- 6. Entry point ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='train',
                       help='Mode: train=training set only (default), test=test set only, both=training and test sets')
    args = parser.parse_args()

    if args.mode in ['train', 'both']:
        print('Starting training set processing...')
        augment_training_data()

    if args.mode in ['test', 'both']:
        print('Test set augmentation is disabled; handle test set augmentation separately if needed.')

    if args.mode == 'both':
        print('✅ Training data processing complete!')


if __name__ == '__main__':
    main()
