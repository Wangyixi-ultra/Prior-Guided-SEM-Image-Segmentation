#!/usr/bin/env python3
"""
从 addtrain/ 的 Labelme JSON 标注生成 YOLO 检测数据集
=============================================================
发现: addtrain/ 下 86 张大图全部已有 JSON polygon 标注！
      无需模板匹配，直接从 polygon 外包框转 YOLO bbox 即可。

使用方法:
    python create_full_image_yolo_dataset.py
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# 配置
# =============================================================================

SOURCE_IMAGE_DIR = Path("/home/chen/seg6/raw/addtrain")
OUTPUT_DIR = Path("/home/chen/seg6/raw/addtrain_yolo_det_dataset")
TRAIN_RATIO = 0.8

# 标签映射: 统一大小写/别名，过滤不关心的类别
LABEL_MAP = {
    "ABO₃": 0,
    "ABX3": 0,      # ABX3 是 ABO₃ 的别名/笔误
    "PbI₂": 1,
    "PBI2": 1,      # PBI2 是 PbI₂ 的大小写别名
    # "defect": 2,  # 如需保留缺陷类，取消注释此行
}
VALID_CLASSES = {"ABO₃", "ABX3", "PbI₂", "PBI2"}  # 只保留这两类

# =============================================================================
# 核心函数
# =============================================================================

def polygon_to_bbox(points):
    """将 polygon 点列表转为外接矩形 (xmin, ymin, xmax, ymax)。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def convert_json_to_yolo(json_path, image_path):
    """
    读取 Labelme JSON，将 polygon 转为 YOLO bbox 格式。
    
    Returns:
        list of str: YOLO 格式行
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_h = data.get("imageHeight")
    img_w = data.get("imageWidth")

    # 如果 JSON 中没有图像尺寸，读图获取
    if img_h is None or img_w is None:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [WARN] 无法读取图像: {image_path}")
            return []
        img_h, img_w = img.shape[:2]

    labels = []
    skipped = {"bad_label": 0, "defect": 0}

    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()

        # 过滤不在目标类别中的标签
        if label not in VALID_CLASSES:
            if label == "defect":
                skipped["defect"] += 1
            else:
                skipped["bad_label"] += 1
            continue

        class_id = LABEL_MAP[label]
        points = shape.get("points", [])
        if len(points) < 3:
            continue

        xmin, ymin, xmax, ymax = polygon_to_bbox(points)

        # 边界保护
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_w, xmax)
        ymax = min(img_h, ymax)

        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue

        # YOLO 格式：归一化
        xc = (xmin + w / 2) / img_w
        yc = (ymin + h / 2) / img_h
        bw = w / img_w
        bh = h / img_h

        # 裁剪到 [0, 1] 区间
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        labels.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return labels, skipped


def create_dataset():
    # 收集所有配对 (image, json)
    pairs = []
    for img_path in sorted(SOURCE_IMAGE_DIR.iterdir()):
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        json_path = SOURCE_IMAGE_DIR / (img_path.stem + ".json")
        if not json_path.exists():
            print(f"[WARN] 缺失 JSON: {img_path.name}")
            continue
        pairs.append((img_path, json_path))

    print(f"[INFO] 共找到 {len(pairs)} 张带标注的大图")

    if not pairs:
        print("[ERROR] 没有可用的图像-标注对！")
        return

    # 划分 train/val
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # 创建目录
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    total_defect_skipped = 0

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, json_path in split_pairs:
            # 复制图像
            dest_img = OUTPUT_DIR / "images" / split_name / img_path.name
            shutil.copy2(img_path, dest_img)

            # 转换标注
            labels, skipped = convert_json_to_yolo(json_path, img_path)
            total_defect_skipped += skipped.get("defect", 0)

            # 保存标注
            label_name = img_path.stem + ".txt"
            label_path = OUTPUT_DIR / "labels" / split_name / label_name
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(labels))

            total_boxes += len(labels)
            print(f"  [{split_name}] {img_path.stem}: {len(labels)} boxes")

    # 生成 data.yaml
    yaml_content = f"""path: {OUTPUT_DIR.absolute()}
train: images/train
val: images/val
test:  # 可选

# 类别
nc: 2  # 类别数量
names: ['ABO₃', 'PbI₂']  # 类别名称
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"[DONE] 数据集创建完成!")
    print(f"       输出路径: {OUTPUT_DIR}")
    print(f"       训练大图: {len(train_pairs)} 张")
    print(f"       验证大图: {len(val_pairs)} 张")
    print(f"       总 bbox 数: {total_boxes}")
    print(f"       跳过 defect 标注: {total_defect_skipped} 个")
    print(f"       配置文件: {yaml_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_dataset()
