#!/usr/bin/env python3
"""
Generate YOLO auxiliary channel (Channel 1) for nnUNet dual-channel data
=================================================================
Compatible with latest detection models such as YOLOv8 / YOLO11 / YOLO12 / YOLO26
Optimizations:
  1. Command-line argument based, no hard-coded paths
  2. Batch + Stream inference, 3~5x speed improvement
  3. Gaussian blob and confidence weighting, preserving spatial uncertainty information
  4. More robust class mapping (automatic PbI2/ABO3 recognition)
  5. Flexible selection of training/test sets

Usage:
  # Basic usage (process imagesTr + imagesTs)
  python add_yolo_info_features_v2.py \
      --weights perovskite_grains_opt/yolo26x_cbam_detect/weights/best.pt \
      --dataset U-Mamba/data/nnUNet_raw/Dataset123_Perovskite

  # Process only test set, use gaussian mode + confidence weighting
  python add_yolo_info_features_v2.py \
      --weights yolo12x_cbam_detect/weights/best.pt \
      --dataset U-Mamba/data/nnUNet_raw/Dataset122_Perovskite \
      --splits Ts \
      --mode gaussian --use-confidence

  # Batch size and device control
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
    """Dynamically extract class IDs for PbI2 / ABO3 from the YOLO model"""
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
    Convert a single YOLO detection result into an auxiliary channel mask

    Args:
        result: ultralytics Results object
        h, w: output image size
        class_map: {"pbi2": id, "abo3": id}
        mode: "gaussian" (gaussian blob) or "dot" (hard dot)
        sigma: gaussian standard deviation (or base dot radius)
        use_confidence: whether to weight intensity by detection confidence
    Returns:
        uint8 single-channel mask [H, W]
    """
    mask = np.zeros((h, w), dtype=np.float32)

    if result.boxes is None or len(result.boxes) == 0:
        return mask.astype(np.uint8)

    # Base intensity per class
    base_intensity = {
        class_map.get("pbi2", 0): 100,
        class_map.get("abo3", 1): 200,
    }

    if mode == "gaussian":
        # Pre-compute base gaussian kernel (3-sigma truncation)
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
            # Clip to image boundary
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
    """Process one data split (imagesTr or imagesTs)"""
    if not image_dir.exists():
        print(f"[SKIP] {image_dir} does not exist")
        return 0

    img_files = sorted(image_dir.glob("*_0000.png"))
    if not img_files:
        print(f"[WARN] No *_0000.png files found in {image_dir}")
        return 0

    print(f"[INFO] Processing {len(img_files)} images in {image_dir.name} ...")

    count = 0
    # Stream + batch inference: memory-friendly and efficient
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
    """Update channel name in dataset.json"""
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
        help="Path to YOLO detection model weights (.pt)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Root directory of nnUNet dataset (contains imagesTr, imagesTs, dataset.json)"
    )
    parser.add_argument(
        "--splits", type=str, default="Tr,Ts",
        help="Data splits to process, comma-separated: Tr,Ts (default), or Tr, or Ts"
    )
    parser.add_argument(
        "--mode", type=str, default="gaussian", choices=["gaussian", "dot"],
        help="Mask generation mode: gaussian (gaussian blob, default) or dot (hard dot)"
    )
    parser.add_argument(
        "--sigma", type=float, default=4.0,
        help="Gaussian sigma (or base dot radius), default 4.0"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="YOLO confidence threshold, default 0.25"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="YOLO NMS IoU threshold, default 0.45"
    )
    parser.add_argument(
        "--imgsz", type=int, default=128,
        help="Inference image size, default 128"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Inference batch size, default 16"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Compute device: cuda id (e.g. 0 or 0,1) or cpu, default 0"
    )
    parser.add_argument(
        "--use-confidence", action="store_true",
        help="Weight pixel intensity by detection confidence (higher confidence is brighter)"
    )
    parser.add_argument(
        "--channel-name", type=str, default="YOLO_Gaussian_Class_Mask",
        help="Channel name written to dataset.json"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    dataset_path = Path(args.dataset)

    if not weights_path.exists():
        print(f"[ERROR] Weight file does not exist: {weights_path}")
        sys.exit(1)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory does not exist: {dataset_path}")
        sys.exit(1)

    print(f"[INFO] Loading YOLO model: {weights_path}")
    model = YOLO(str(weights_path))

    # Dynamic class mapping (print once)
    class_map = get_class_mapping(model)
    print(f"[INFO] Class mapping: {class_map}")
    if "pbi2" not in class_map or "abo3" not in class_map:
        print("[WARN] PbI2/ABO3 not automatically recognized, using fallback 0/1")
        class_map = {"pbi2": 0, "abo3": 1}

    # Update metadata
    update_dataset_json(dataset_path, args.channel_name)

    # Process each split
    splits = [s.strip() for s in args.splits.split(",")]
    total = 0
    for split in splits:
        img_dir = dataset_path / f"images{split}"
        n = process_split(img_dir, model, class_map, args)
        total += n

    print(f"\n[ALL DONE] Total {total} images processed.")


if __name__ == "__main__":
    main()
