#!/usr/bin/env python3
"""
为 nnUNet 生成双通道数据的 YOLO 辅助通道 (Channel 1)
=================================================================
适配 YOLOv8 / YOLO11 / YOLO12 / YOLO26 等最新检测模型
优化点:
  1. 命令行参数化, 不再硬编码路径
  2. Batch + Stream 推理, 速度提升 3~5x
  3. 高斯 Blob 与置信度加权, 保留空间不确定性信息
  4. 更鲁棒的类别映射 (PbI2/ABO3 自动识别)
  5. 支持训练集/测试集灵活选择

使用方法:
  # 基础用法 (处理 imagesTr + imagesTs)
  python add_yolo_info_features_v2.py \
      --weights perovskite_grains_opt/yolo26x_cbam_detect/weights/best.pt \
      --dataset U-Mamba/data/nnUNet_raw/Dataset123_Perovskite

  # 只处理测试集, 使用高斯模式+置信度加权
  python add_yolo_info_features_v2.py \
      --weights yolo12x_cbam_detect/weights/best.pt \
      --dataset U-Mamba/data/nnUNet_raw/Dataset122_Perovskite \
      --splits Ts \
      --mode gaussian --use-confidence

  # 批量大小和设备控制
  python add_yolo_info_features_v2.py \
      --weights best.pt \
      --dataset Dataset123_Perovskite \
      --batch-size 32 --device 0
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def get_class_mapping(model) -> dict:
    """从 YOLO 模型中动态提取 PbI2 / ABO3 的类别 ID"""
    names = model.names
    mapping = {}
    for cls_id, cls_name in names.items():
        name_lower = cls_name.lower()
        if "pbi" in name_lower:
            mapping["pbi2"] = cls_id
        elif "abo" in name_lower:
            mapping["abo3"] = cls_id
    return mapping


def create_mask_from_result(result, h: int, w: int, class_map: dict,
                            mode: str = "gaussian", sigma: float = 4.0,
                            use_confidence: bool = False) -> np.ndarray:
    """
    将单张 YOLO 检测结果转换为辅助通道 mask

    Args:
        result: ultralytics Results 对象
        h, w: 输出图像尺寸
        class_map: {"pbi2": id, "abo3": id}
        mode: "gaussian" (高斯blob) 或 "dot" (硬圆点)
        sigma: 高斯标准差 (或圆点半径基准)
        use_confidence: 是否用检测置信度加权强度
    Returns:
        uint8 单通道 mask [H, W]
    """
    mask = np.zeros((h, w), dtype=np.float32)

    if result.boxes is None or len(result.boxes) == 0:
        return mask.astype(np.uint8)

    # 类别基准强度
    base_intensity = {
        class_map.get("pbi2", 0): 100,
        class_map.get("abo3", 1): 200,
    }

    if mode == "gaussian":
        # 预计算基础高斯核 (3-sigma 截断)
        radius = int(3 * sigma)
        ks = 2 * radius + 1
        y_grid, x_grid = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        base_gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        xyxy = box.xyxy[0].cpu().numpy()
        cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)

        peak = base_intensity.get(cls_id, 50)
        if use_confidence:
            peak *= conf

        if mode == "gaussian":
            # 裁剪到图像边界
            x_start, x_end = cx - radius, cx + radius + 1
            y_start, y_end = cy - radius, cy + radius + 1

            pad_left = max(0, -x_start)
            pad_top = max(0, -y_start)
            pad_right = max(0, x_end - w)
            pad_bottom = max(0, y_end - h)

            img_x1, img_x2 = x_start + pad_left, x_end - pad_right
            img_y1, img_y2 = y_start + pad_top, y_end - pad_bottom

            kern_x1 = pad_left
            kern_y1 = pad_top
            kern_x2 = ks - pad_right
            kern_y2 = ks - pad_bottom

            if img_x2 > img_x1 and img_y2 > img_y1:
                kernel_patch = base_gaussian[kern_y1:kern_y2, kern_x1:kern_x2] * peak
                roi = mask[img_y1:img_y2, img_x1:img_x2]
                mask[img_y1:img_y2, img_x1:img_x2] = np.maximum(roi, kernel_patch)
        else:
            radius_dot = max(2, int(sigma))
            cv2.circle(mask, (cx, cy), radius_dot, peak, -1)

    return np.clip(mask, 0, 255).astype(np.uint8)


def process_split(image_dir: Path, model, class_map: dict, args) -> int:
    """处理一个数据 split (imagesTr 或 imagesTs)"""
    if not image_dir.exists():
        print(f"[SKIP] {image_dir} does not exist")
        return 0

    img_files = sorted(image_dir.glob("*_0000.png"))
    if not img_files:
        print(f"[WARN] No *_0000.png files found in {image_dir}")
        return 0

    print(f"[INFO] Processing {len(img_files)} images in {image_dir.name} ...")

    count = 0
    # Stream + batch 推理: 内存友好且高效
    results_gen = model.predict(
        [str(p) for p in img_files],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch_size,
        stream=True,
        verbose=False,
    )

    for img_path, result in tqdm(zip(img_files, results_gen), total=len(img_files),
                                 desc=f"  {image_dir.name}", ncols=80):
        h, w = result.orig_shape[:2]

        mask = create_mask_from_result(
            result, h, w, class_map,
            mode=args.mode,
            sigma=args.sigma,
            use_confidence=args.use_confidence,
        )

        out_path = img_path.parent / img_path.name.replace("_0000.png", "_0001.png")
        cv2.imwrite(str(out_path), mask)
        count += 1

    print(f"[DONE] Wrote {count} channel-1 masks to {image_dir}")
    return count


def update_dataset_json(dataset_path: Path, channel_name: str = "YOLO_Gaussian_Class_Mask"):
    """更新 dataset.json 的通道名称"""
    json_path = dataset_path / "dataset.json"
    if not json_path.exists():
        print(f"[WARN] {json_path} not found, skipping metadata update")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ch_names = data.setdefault("channel_names", {})
    if ch_names.get("1") == channel_name:
        print(f"[INFO] dataset.json already has '{channel_name}'")
    else:
        ch_names["1"] = channel_name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[UPDATED] dataset.json channel 1 -> '{channel_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO auxiliary channel for nnUNet dual-channel training"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="YOLO 检测模型权重路径 (.pt)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="nnUNet 数据集根目录 (包含 imagesTr, imagesTs, dataset.json)"
    )
    parser.add_argument(
        "--splits", type=str, default="Tr,Ts",
        help="要处理的数据划分, 逗号分隔: Tr,Ts (默认), 或 Tr, 或 Ts"
    )
    parser.add_argument(
        "--mode", type=str, default="gaussian", choices=["gaussian", "dot"],
        help="mask 生成模式: gaussian (高斯blob, 默认) 或 dot (硬圆点)"
    )
    parser.add_argument(
        "--sigma", type=float, default=4.0,
        help="高斯 sigma (或圆点半径基准), 默认 4.0"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="YOLO 置信度阈值, 默认 0.25"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="YOLO NMS IoU 阈值, 默认 0.45"
    )
    parser.add_argument(
        "--imgsz", type=int, default=128,
        help="推理图像尺寸, 默认 128"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="推理 batch size, 默认 16"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="计算设备: cuda id (如 0 或 0,1) 或 cpu, 默认 0"
    )
    parser.add_argument(
        "--use-confidence", action="store_true",
        help="用检测置信度加权像素强度 (高置信度更亮)"
    )
    parser.add_argument(
        "--channel-name", type=str, default="YOLO_Gaussian_Class_Mask",
        help="写入 dataset.json 的通道名称"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    dataset_path = Path(args.dataset)

    if not weights_path.exists():
        print(f"[ERROR] 权重文件不存在: {weights_path}")
        sys.exit(1)
    if not dataset_path.exists():
        print(f"[ERROR] 数据集目录不存在: {dataset_path}")
        sys.exit(1)

    print(f"[INFO] Loading YOLO model: {weights_path}")
    model = YOLO(str(weights_path))

    # 动态类别映射 (只打印一次)
    class_map = get_class_mapping(model)
    print(f"[INFO] Class mapping: {class_map}")
    if "pbi2" not in class_map or "abo3" not in class_map:
        print("[WARN] 未自动识别 PbI2/ABO3, 使用 fallback 0/1")
        class_map = {"pbi2": 0, "abo3": 1}

    # 更新元数据
    update_dataset_json(dataset_path, args.channel_name)

    # 处理各 split
    splits = [s.strip() for s in args.splits.split(",")]
    total = 0
    for split in splits:
        img_dir = dataset_path / f"images{split}"
        n = process_split(img_dir, model, class_map, args)
        total += n

    print(f"\n[ALL DONE] Total {total} images processed.")


if __name__ == "__main__":
    main()
