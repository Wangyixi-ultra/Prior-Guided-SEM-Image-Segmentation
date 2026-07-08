#!/usr/bin/env python3
"""
YOLO Detect with CBAM 训练脚本 (最新版兼容，无需修改源码)
=================================================================
 ultralytics >= 8.4.0 已内置 CBAM 模块，不再需要修改 tasks.py。
 本脚本支持 YOLO11x / YOLO12x / YOLO26x 三种 backbone，自动选择最新架构。

 使用方法:
   # 使用最新 YOLO26x (默认)
   python train_yolo_cbam_latest.py

   # 使用 YOLO12x
   python train_yolo_cbam_latest.py --model yolo12x

   # 使用 YOLO11x (兼容旧权重)
   python train_yolo_cbam_latest.py --model yolo11x
"""

import argparse
import sys
from pathlib import Path
from packaging import version

import ultralytics
from ultralytics import YOLO

# 将 CBAM 注入 ultralytics.nn.tasks 的命名空间，使 parse_model 可以识别
from ultralytics.nn.modules import CBAM
import ultralytics.nn.tasks as tasks
tasks.CBAM = CBAM

# =============================================================================
# 版本检查
# =============================================================================
MIN_VERSION = "8.4.0"
try:
    if version.parse(ultralytics.__version__) < version.parse(MIN_VERSION):
        print(f"[ERROR] 当前 ultralytics 版本 {ultralytics.__version__} 过低")
        print(f"        CBAM 内置模块需要 >= {MIN_VERSION}")
        print(f"        请升级: pip install -U ultralytics")
        sys.exit(1)
    else:
        print(f"[INFO] ultralytics 版本: {ultralytics.__version__} (符合要求)")
except Exception:
    # 某些旧版本 packaging 可能报错，简单跳过
    pass

# =============================================================================
# YAML 配置模板
# =============================================================================

YOLO11X_CBAM_YAML = """# YOLO11x-detect-CBAM
# Compatible with ultralytics >= 8.4.0 (CBAM is built-in, no source patch needed)
nc: 2
scale: x
scales:
  x: [1.0, 1.25, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4 (P3/8)
  - [-1, 1, CBAM, [640, 7]]            # 5 <-- CBAM after P3 (512*1.25=640)
  - [-1, 1, Conv, [512, 3, 2]]         # 6-P4/16
  - [-1, 2, C3k2, [512, True]]         # 7 (P4/16)
  - [-1, 1, CBAM, [640, 7]]            # 8 <-- CBAM after P4 (512*1.25=640)
  - [-1, 1, Conv, [1024, 3, 2]]        # 9-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 10 (P5/32)
  - [-1, 2, C2PSA, [1024]]             # 11

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 7], 1, Concat, [1]]          # cat backbone P4
  - [-1, 2, C3k2, [512, False]]        # 14

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]          # cat backbone P3
  - [-1, 2, C3k2, [256, False]]        # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]         # cat head P4
  - [-1, 2, C3k2, [512, False]]        # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]         # cat head P5
  - [-1, 2, C3k2, [1024, True]]        # 23 (P5/32-large)

  - [[17, 20, 23], 1, Detect, [nc]]    # Detect(P3, P4, P5)
"""

YOLO12X_CBAM_YAML = """# YOLO12x-detect-CBAM
# Compatible with ultralytics >= 8.4.0 (CBAM is built-in, no source patch needed)
nc: 2
scale: x
scales:
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4 (P3/8)
  - [-1, 1, CBAM, [768, 7]]            # 5 <-- CBAM after P3 (512*1.5=768)
  - [-1, 1, Conv, [512, 3, 2]]         # 6-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]     # 7 (P4/16)
  - [-1, 1, CBAM, [768, 7]]            # 8 <-- CBAM after P4 (512*1.5=768)
  - [-1, 1, Conv, [1024, 3, 2]]        # 9-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]    # 10 (P5/32)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 7], 1, Concat, [1]]          # cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]]   # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]          # cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]]   # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]         # cat head P4
  - [-1, 2, A2C2f, [512, False, -1]]   # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]         # cat head P5
  - [-1, 2, C3k2, [1024, True]]        # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]    # Detect(P3, P4, P5)
"""

YOLO26X_CBAM_YAML = """# YOLO26x-detect-CBAM
# Compatible with ultralytics >= 8.4.0 (CBAM is built-in, no source patch needed)
# YOLO26 is the latest architecture as of ultralytics 8.4.53
nc: 2
end2end: True
reg_max: 1
scale: x
scales:
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4 (P3/8)
  - [-1, 1, CBAM, [768, 7]]            # 5 <-- CBAM after P3 (512*1.5=768)
  - [-1, 1, Conv, [512, 3, 2]]         # 6-P4/16
  - [-1, 2, C3k2, [512, True]]         # 7 (P4/16)
  - [-1, 1, CBAM, [768, 7]]            # 8 <-- CBAM after P4 (512*1.5=768)
  - [-1, 1, Conv, [1024, 3, 2]]        # 9-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 10 (P5/32)
  - [-1, 1, SPPF, [1024, 5, 3, True]]  # 11
  - [-1, 2, C2PSA, [1024]]             # 12

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 7], 1, Concat, [1]]          # cat backbone P4
  - [-1, 2, C3k2, [512, True]]         # 14

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]          # cat backbone P3
  - [-1, 2, C3k2, [256, True]]         # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]         # cat head P4
  - [-1, 2, C3k2, [512, True]]         # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]         # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 23 (P5/32-large)

  - [[17, 20, 23], 1, Detect, [nc]]    # Detect(P3, P4, P5)
"""

# =============================================================================
# 训练主函数
# =============================================================================

MODEL_CONFIGS = {
    "yolo11x": {
        "yaml": YOLO11X_CBAM_YAML,
        "pretrained": "yolo11x.pt",
        "proj_name": "yolo11x_cbam_detect_v2",
        "desc": "YOLO11x + CBAM (兼容旧权重)",
    },
    "yolo12x": {
        "yaml": YOLO12X_CBAM_YAML,
        "pretrained": "yolo12x.pt",
        "proj_name": "yolo12x_cbam_detect",
        "desc": "YOLO12x + CBAM (较新架构)",
    },
    "yolo26x": {
        "yaml": YOLO26X_CBAM_YAML,
        "pretrained": "yolo26x.pt",
        "proj_name": "yolo26x_cbam_detect",
        "desc": "YOLO26x + CBAM (最新架构，默认推荐)",
    },
}


def train(model_type: str = "yolo11x", data: str = "/home/chen/seg6/raw/addtrain_yolo_det_dataset/data.yaml"):
    """Train YOLO detect model with CBAM attention."""

    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"不支持的模型类型: {model_type}\n"
            f"请选择: {', '.join(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_type]

    # 生成/保存 YAML
    cfg_dir = Path("perovskite_grains_opt/cbam_configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = cfg_dir / f"{model_type}-detect-cbam.yaml"
    yaml_path.write_text(cfg["yaml"], encoding="utf-8")

    print(f"[INFO] 使用架构: {cfg['desc']}")
    print(f"[INFO] 已生成模型配置: {yaml_path.resolve()}")

    # 初始化模型
    model = YOLO(str(yaml_path))

    # 加载预训练权重 (迁移学习)
    pretrained = cfg["pretrained"]
    try:
        model.load(pretrained)
        print(f"[INFO] 成功加载预训练权重: {pretrained}")
    except Exception as e:
        print(f"[WARN] 无法加载权重 {pretrained}: {e}")
        print("[WARN] 将使用随机初始化继续训练")

    # 开始训练
    print("[INFO] 开始训练...")
    results = model.train(
        data=data,

        # --- 图像与计算 ---
        imgsz=640,
        epochs=300,
        batch=8,
        device=[0],
        workers=4,
        patience=50,

        # --- 数据增强 (针对晶粒图像优化) ---
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        scale=0.1,
        shear=5,
        perspective=0.0,
        mosaic=0.0,      # 晶粒图像不适合 mosaic
        mixup=0.0,       # 晶粒图像不适合 mixup
        hsv_h=0.0,       # SEM 图像无颜色
        hsv_s=0.0,
        hsv_v=0.05,

        # --- 损失权重 (分类更重要) ---
        box=2.0,
        cls=5.0,
        dfl=1.5,

        # --- 优化器策略 ---
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        # --- 输出目录 ---
        project="perovskite_grains_opt",
        name=cfg["proj_name"],
        exist_ok=True,
    )

    return results


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO detect model with CBAM (no source code modification required)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11x",
        choices=list(MODEL_CONFIGS.keys()),
        help="选择模型架构 (默认: yolo26x)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/home/chen/seg6/raw/addtrain_yolo_det_dataset/data.yaml",
        help="数据集 data.yaml 路径",
    )
    args = parser.parse_args()

    train(model_type=args.model, data=args.data)
