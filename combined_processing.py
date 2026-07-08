#!/usr/bin/env python3
"""
钙钛矿 SEM 数据增强（20倍版本 = 1 原图 + 19 增强）
支持训练集和测试集的相同处理
输出直接写回原始 nnUNet 目录
"""
import os
import cv2
from pathlib import Path
import albumentations as A
import numpy as np
import argparse

# ---------- 1. 原始 nnUNet 路径 ----------
TRAIN_IMG_DIR   = Path('U-Mamba/data/nnUNet_raw/Dataset107_Perovskite/imagesTr')
TRAIN_LABEL_DIR = Path('U-Mamba/data/nnUNet_raw/Dataset107_Perovskite/labelsTr')
TEST_IMG_DIR    = Path('/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset107_Perovskite/imagesTs')
TEST_LABEL_DIR  = Path('/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset107_Perovskite/labelsTs')

# ---------- 2. 递进式增强策略（5个强度级别） ----------
# Level 1: 最弱增强
level_1_transform = A.ReplayCompose([
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=0,
             interpolation=cv2.INTER_NEAREST, p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=60, sigma=60*0.05, interpolation=cv2.INTER_NEAREST, p=0.5),
        A.GridDistortion(num_steps=3, distort_limit=0.15, interpolation=cv2.INTER_NEAREST, p=0.5),
    ], p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=0.4),
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ], p=0.4),
], additional_targets={'mask': 'mask'})

# Level 2: 较弱增强
level_2_transform = A.ReplayCompose([
    A.Affine(scale=(0.92, 1.08), translate_percent=(-0.07, 0.07), rotate=0,
             interpolation=cv2.INTER_NEAREST, p=0.6),
    A.OneOf([
        A.ElasticTransform(alpha=80, sigma=80*0.05, interpolation=cv2.INTER_NEAREST, p=0.6),
        A.GridDistortion(num_steps=4, distort_limit=0.2, interpolation=cv2.INTER_NEAREST, p=0.6),
    ], p=0.4),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.MotionBlur(blur_limit=(3, 5), p=0.4),
    ], p=0.4),
    A.OneOf([
        A.GaussNoise(std_range=(0.05, 0.2), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    ], p=0.5),
], additional_targets={'mask': 'mask'})

# Level 3: 中等增强
level_3_transform = A.ReplayCompose([
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.08, 0.08), rotate=0,
             interpolation=cv2.INTER_NEAREST, p=0.7),
    A.OneOf([
        A.ElasticTransform(alpha=100, sigma=100*0.05, interpolation=cv2.INTER_NEAREST, p=0.7),
        A.GridDistortion(num_steps=4, distort_limit=0.25, interpolation=cv2.INTER_NEAREST, p=0.7),
        A.OpticalDistortion(distort_limit=0.2, interpolation=cv2.INTER_NEAREST, p=0.6),
    ], p=0.5),
    A.OneOf([
        A.GlassBlur(sigma=0.3, max_delta=1, iterations=1, p=0.6),
        A.GaussianBlur(blur_limit=(3, 7), p=0.6),
        A.MotionBlur(blur_limit=(3, 5), p=0.5),
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(std_range=(0.08, 0.3), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.7),
        A.RandomGamma(gamma_limit=(90, 110), p=0.6),
    ], p=0.6),
], additional_targets={'mask': 'mask'})

# Level 4: 较强增强
level_4_transform = A.ReplayCompose([
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=0,
             interpolation=cv2.INTER_NEAREST, p=0.8),
    A.OneOf([
        A.ElasticTransform(alpha=110, sigma=110*0.05, interpolation=cv2.INTER_NEAREST, p=0.8),
        A.GridDistortion(num_steps=5, distort_limit=0.28, interpolation=cv2.INTER_NEAREST, p=0.8),
        A.OpticalDistortion(distort_limit=0.28, interpolation=cv2.INTER_NEAREST, p=0.7),
    ], p=0.6),
    A.OneOf([
        A.GlassBlur(sigma=0.4, max_delta=1, iterations=2, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.7),
        A.MotionBlur(blur_limit=(3, 5), p=0.6),
    ], p=0.6),
    A.OneOf([
        A.GaussNoise(std_range=(0.1, 0.4), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.RandomGamma(gamma_limit=(85, 115), p=0.7),
    ], p=0.7),
], additional_targets={'mask': 'mask'})

# Level 5: 最强增强（原代码强度）
level_5_transform = A.ReplayCompose([
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=0,
             interpolation=cv2.INTER_NEAREST, p=0.8),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120*0.05, interpolation=cv2.INTER_NEAREST, p=0.9),
        A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_NEAREST, p=0.9),
        A.OpticalDistortion(distort_limit=0.3, interpolation=cv2.INTER_NEAREST, p=0.9),
    ], p=0.7),
    A.OneOf([
        A.GlassBlur(sigma=0.5, max_delta=2, iterations=2, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.8),
        A.MotionBlur(blur_limit=(3, 5), p=0.7),
    ], p=0.6),
    A.OneOf([
        A.GaussNoise(std_range=(0.1, 0.5), p=0.7),
    ], p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.RandomGamma(gamma_limit=(80, 120), p=0.8),
    ], p=0.7),
], additional_targets={'mask': 'mask'})

# 增强级别映射
AUGMENTATION_LEVELS = {
    1: level_1_transform,
    2: level_2_transform,
    3: level_3_transform,
    4: level_4_transform,
    5: level_5_transform
}

# ---------- 3. 保存工具 ----------
def save_image(img, stem, suffix, idx, output_dir):
    """保存图像到指定目录"""
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    img_name = f'{stem}_0000{suffix}' if idx == 0 else f'{stem}_aug{idx}_0000{suffix}'
    cv2.imwrite(str(output_dir / img_name), img)

def save_pair(img, mask, stem, suffix, idx, img_dir, label_dir):
    """保存图像和标签对"""
    save_image(img, stem, suffix, idx, img_dir)
    lab_name = f'{stem}.png' if idx == 0 else f'{stem}_aug{idx}.png'
    cv2.imwrite(str(label_dir / lab_name), mask)

# ---------- 4. 训练数据增强 ----------
def augment_training_data():
    """处理训练数据：原图 + 增强（20倍）"""
    supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for img_path in TRAIN_IMG_DIR.rglob('*'):
        if img_path.suffix.lower() not in supported:
            continue

        label_name = img_path.name.replace('_0000', '')
        label_path = TRAIN_LABEL_DIR / label_name
        if not label_path.exists():
            print(f'跳过 {img_path.name}，无对应标签')
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print('无效文件:', img_path, label_path)
            continue

        image = image[..., None]   # (H,W,1)
        stem  = img_path.stem.split('_')[0]
        suffix = img_path.suffix

        # 0 号：原图
        save_pair(image.squeeze(-1), mask, stem, suffix, 0, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

        # 1~19 号：增强（总共20倍）
        # 使用递进式增强：1-3号用level1, 4-6号用level2, 7-9号用level3, 10-12号用level4, 13-19号用level5
        for i in range(1, 20):
            if i <= 3:
                level = 1
            elif i <= 6:
                level = 2
            elif i <= 9:
                level = 3
            elif i <= 12:
                level = 4
            else:
                level = 5
            
            transform = AUGMENTATION_LEVELS[level]
            res = transform(image=image, mask=mask)
            save_pair(res['image'], res['mask'], stem, suffix, i, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

    print('✅ 训练集增强完成，数据已直接写回原始目录：')
    print('图像:', TRAIN_IMG_DIR.resolve())
    print('标签:', TRAIN_LABEL_DIR.resolve())

# ---------- 5. 测试数据增强 ----------
def augment_test_data():
    """处理测试数据：原图 + 增强（20倍）"""
    supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for img_path in TEST_IMG_DIR.rglob('*'):
        if img_path.suffix.lower() not in supported:
            continue

        label_name = img_path.name.replace('_0000', '')
        label_path = TEST_LABEL_DIR / label_name
        if not label_path.exists():
            print(f'跳过 {img_path.name}，无对应标签')
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print('无效文件:', img_path, label_path)
            continue

        image = image[..., None]   # (H,W,1)
        stem  = img_path.stem.split('_')[0]
        suffix = img_path.suffix

        # 0 号：原图
        save_pair(image.squeeze(-1), mask, stem, suffix, 0, TEST_IMG_DIR, TEST_LABEL_DIR)

        # 1~19 号：增强（总共20倍）
        # 使用递进式增强：1-3号用level1, 4-6号用level2, 7-9号用level3, 10-12号用level4, 13-19号用level5
        for i in range(1, 20):
            if i <= 3:
                level = 1
            elif i <= 6:
                level = 2
            elif i <= 9:
                level = 3
            elif i <= 12:
                level = 4
            else:
                level = 5
            
            transform = AUGMENTATION_LEVELS[level]
            res = transform(image=image, mask=mask)
            save_pair(res['image'], res['mask'], stem, suffix, i, TEST_IMG_DIR, TEST_LABEL_DIR)

    print('✅ 测试集增强完成，数据已直接写回原始目录：')
    print('图像:', TEST_IMG_DIR.resolve())
    print('标签:', TEST_LABEL_DIR.resolve())

# ---------- 6. 入口 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both',
                       help='处理模式：train=仅训练集, test=仅测试集, both=训练集和测试集')
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print('开始处理训练集...')
        augment_training_data()
    
    if args.mode in ['test', 'both']:
        print('开始处理测试集...')
        augment_test_data()
    
    if args.mode == 'both':
        print('✅ 所有数据处理完成！')

if __name__ == '__main__':
    main()
