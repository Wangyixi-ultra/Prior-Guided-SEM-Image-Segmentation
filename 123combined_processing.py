#!/usr/bin/env python3
"""
钙钛矿 SEM 数据增强（14倍版本 = 1 原图 + 13 增强）
仅增强训练集，输出直接写回原始 nnUNet 目录

增强策略：
  Aug01~Aug07：几何变换（同时作用于 SEM、YOLO通道、mask）
  Aug08~Aug12：像素增强（仅作用于 SEM，YOLO通道和mask不变）
  Aug13：复合增强（几何作用于三者 + 像素仅作用于SEM）
"""
import os
import cv2
from pathlib import Path
import albumentations as A
import numpy as np
import argparse

# ---------- 1. 原始 nnUNet 路径 ----------
TRAIN_IMG_DIR   = Path('U-Mamba/data/nnUNet_raw/Dataset123_Perovskite/imagesTr')
TRAIN_LABEL_DIR = Path('U-Mamba/data/nnUNet_raw/Dataset123_Perovskite/labelsTr')

# ---------- 2. 13 个增强等级定义 ----------

def get_aug_transforms():
    """返回 13 个增强变换定义"""
    augs = []

    # Aug01: 水平翻转（几何）
    augs.append(A.ReplayCompose([
        A.HorizontalFlip(p=1.0),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug02: 垂直翻转（几何）
    augs.append(A.ReplayCompose([
        A.VerticalFlip(p=1.0),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug03: 旋转 90°（几何）
    augs.append(A.ReplayCompose([
        A.Rotate(limit=(90, 90), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug04: 旋转 180° 或 270°（几何）
    augs.append(A.ReplayCompose([
        A.OneOf([
            A.Rotate(limit=(180, 180), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=(270, 270), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        ], p=1.0),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug05: 小角度旋转 ±5°~±15°（几何）
    augs.append(A.ReplayCompose([
        A.OneOf([
            A.Rotate(limit=(-15, -5), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=(5, 15), p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        ], p=1.0),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug06: 平移 + 缩放（几何）
    augs.append(A.ReplayCompose([
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=0,
            interpolation=cv2.INTER_NEAREST,
            p=1.0
        ),
    ], additional_targets={'mask': 'mask', 'yolo': 'image'}))

    # Aug07: 随机裁剪再缩放（几何）
    # 占位符，尺寸在 apply_augment 中动态传入
    augs.append({'type': 'RandomResizedCrop', 'scale': (0.80, 0.95), 'ratio': (0.9, 1.1)})

    # Aug08: 亮度/对比度变化（像素，仅 SEM）
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ])
    })

    # Aug09: Gamma 校正（像素，仅 SEM）
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.RandomGamma(gamma_limit=(75, 125), p=1.0),
        ])
    })

    # Aug10: CLAHE（像素，仅 SEM）
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=1.0),
        ])
    })

    # Aug11: 高斯噪声（像素，仅 SEM）
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.GaussNoise(var_limit=(1.0, 4.0), mean=0, p=1.0),
        ])
    })

    # Aug12: 轻微模糊/锐化（像素，仅 SEM）
    augs.append({
        'geo': None,
        'pixel': A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
            ], p=1.0),
        ])
    })

    # Aug13: 复合增强（几何作用于三者 + 像素仅作用于 SEM）
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


# ---------- 3. 保存工具 ----------
def save_channel(img, stem, suffix, idx, output_dir, channel_idx):
    """保存单通道图像"""
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    if idx == 0:
        img_name = f'{stem}_{channel_idx:04d}{suffix}'
    else:
        img_name = f'{stem}_aug{idx:02d}_{channel_idx:04d}{suffix}'
    cv2.imwrite(str(output_dir / img_name), img)


def save_triplet(sem_img, yolo_img, mask, stem, suffix, idx, img_dir, label_dir):
    """保存 SEM、YOLO通道、mask 三元组"""
    save_channel(sem_img, stem, suffix, idx, img_dir, 0)
    save_channel(yolo_img, stem, suffix, idx, img_dir, 1)
    if idx == 0:
        lab_name = f'{stem}.png'
    else:
        lab_name = f'{stem}_aug{idx:02d}.png'
    cv2.imwrite(str(label_dir / lab_name), mask)


# ---------- 4. 应用增强 ----------
def apply_augment(sem_img, yolo_img, mask, aug, h, w):
    """
    应用单个增强变换
    几何变换同时作用于 sem_img、yolo_img、mask
    像素变换仅作用于 sem_img
    返回 (aug_sem, aug_yolo, aug_mask)
    """
    # Aug07 占位符：动态创建 RandomResizedCrop
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

    # 几何变换（同时作用于 SEM、YOLO、mask）
    if geo is not None:
        res = geo(image=sem_img, mask=mask, yolo=yolo_img)
        aug_sem = res['image']
        aug_yolo = res['yolo']
        aug_mask = res['mask']
    else:
        aug_sem = sem_img.copy()
        aug_yolo = yolo_img.copy()
        aug_mask = mask.copy()

    # 像素变换（仅作用于 SEM）
    if pixel is not None:
        pixel_res = pixel(image=aug_sem)
        aug_sem = pixel_res['image']

    return aug_sem, aug_yolo, aug_mask


# ---------- 5. 训练数据增强 ----------
def augment_training_data():
    """处理训练数据：原图 + 13 增强（14倍）"""
    augs = get_aug_transforms()
    supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    count = 0

    for img_path in sorted(TRAIN_IMG_DIR.rglob('*')):
        if img_path.suffix.lower() not in supported:
            continue

        # 只处理原始 SEM 图（*_0000.png）
        if '_0001' in img_path.name:
            continue

        # 对应 YOLO 辅助通道
        yolo_path = TRAIN_IMG_DIR / img_path.name.replace('_0000', '_0001')
        if not yolo_path.exists():
            print(f'跳过 {img_path.name}，无对应 YOLO 通道')
            continue

        # 对应标签
        label_name = img_path.name.replace('_0000', '')
        label_path = TRAIN_LABEL_DIR / label_name
        if not label_path.exists():
            print(f'跳过 {img_path.name}，无对应标签')
            continue

        sem = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        yolo = cv2.imread(str(yolo_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if sem is None or yolo is None or mask is None:
            print('无效文件:', img_path, yolo_path, label_path)
            continue

        # 确保尺寸一致
        h, w = sem.shape[:2]
        if yolo.shape != (h, w) or mask.shape != (h, w):
            print(f'尺寸不匹配，跳过: {img_path.name}')
            continue

        # 为 albumentations 添加通道维度
        sem = sem[..., None]   # (H,W,1)
        yolo = yolo[..., None] # (H,W,1)

        stem = img_path.stem.split('_')[0]
        suffix = img_path.suffix

        # 0 号：原图
        save_triplet(sem.squeeze(-1), yolo.squeeze(-1), mask, stem, suffix, 0, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

        # 1~13 号：增强
        for i in range(1, 14):
            aug = augs[i - 1]
            aug_sem, aug_yolo, aug_mask = apply_augment(sem, yolo, mask, aug, h, w)
            save_triplet(aug_sem.squeeze(-1), aug_yolo.squeeze(-1), aug_mask, stem, suffix, i, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)

        count += 1

    total = count * 14
    print(f'✅ 训练集增强完成：{count} 张原图 → {total} 张 cases')
    print('图像:', TRAIN_IMG_DIR.resolve())
    print('标签:', TRAIN_LABEL_DIR.resolve())


# ---------- 6. 入口 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='train',
                       help='处理模式：train=仅训练集(默认), test=仅测试集, both=训练集和测试集')
    args = parser.parse_args()

    if args.mode in ['train', 'both']:
        print('开始处理训练集...')
        augment_training_data()

    if args.mode in ['test', 'both']:
        print('测试集增强功能已禁用，如需增强测试集请单独处理')

    if args.mode == 'both':
        print('✅ 训练数据处理完成！')


if __name__ == '__main__':
    main()
