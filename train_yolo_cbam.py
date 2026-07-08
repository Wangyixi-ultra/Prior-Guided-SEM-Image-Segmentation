#!/usr/bin/env python3
"""
YOLO11x Detect with CBAM 训练脚本
现在 CBAM 已直接注册到 Ultralytics tasks.py 中
"""

from ultralytics import YOLO


def train_custom_arch():
    # 1. 初始化自定义模型
    model = YOLO('yolo11x-detect-cbam.yaml')

    # 2. 加载预训练权重 (迁移学习)
    try:
        model.load('yolo11x.pt')
        print("Successfully loaded partial weights from yolo11x.pt")
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")

    # 3. 训练
    results = model.train(
        data='raw/cropped_yolo_det_dataset/data.yaml',
        
        # --- 训练参数 ---
        imgsz=128,
        epochs=300,
        batch=160, #160、192\128
        device=[0, 3, 2, 4],
        patience=50,
        workers=8,
        
        # --- 数据增强 ---
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        scale=0.1,
        shear=5,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,
        
        # --- 损失权重（分类更重要） ---
        box=2.0,
        cls=5.0,
        dfl=1.5,
        
        # --- 训练策略 ---
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        
        project='perovskite_grains_opt',
        name='yolo11x_cbam_detect',
        exist_ok=True
    )


if __name__ == '__main__':
    train_custom_arch()
