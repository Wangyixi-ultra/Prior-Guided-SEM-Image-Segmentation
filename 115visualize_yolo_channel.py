import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- 1. 配置路径 ---
dataset_root = Path('/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset119_Perovskite')
output_root = Path('/home/chen/seg6/yolo_channel_vis')

# --- 2. 颜色定义 (BGR for OpenCV) ---
# Mask 像素值 -> BGR 颜色
# 0: Background (Black)
# 1: PbI2 (Red)
# 2: ABO3 (Green)
# 3: Defect (Blue)
PALETTE = {
    0: (0, 0, 0),       # Background
    1: (0, 0, 255),     # PbI2
    2: (0, 255, 0),     # ABO3
    3: (255, 0, 0)      # Defect
}

LABELS_TEXT = {
    1: "1: PbI2",
    2: "2: ABO3",
    3: "3: Defect"
}

def apply_palette(mask):
    """将单通道 Label Mask 转换为 3通道 BGR 图像"""
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label in PALETTE:
            vis[mask == label] = PALETTE[label]
            
    return vis

def draw_legend(img):
    """在左上角绘制图例"""
    # 简单的图例绘制
    # Red: PbI2
    # Green: ABO3
    # Blue: Defect
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    y = 30
    for label_id, text in LABELS_TEXT.items():
        color = PALETTE[label_id]
        cv2.putText(img, text, (10, y), font, font_scale, color, thickness)
        y += 30
    return img

def process_subset(subset_name):
    print(f"\nProcessing {subset_name}...")
    src_dir = dataset_root / subset_name
    dst_dir = output_root / subset_name
    
    if not src_dir.exists():
        print(f"Source directory {src_dir} does not exist.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有的 _0001.png (YOLO Channel)
    mask_files = sorted(list(src_dir.glob('*_0001.png')))
    print(f"Found {len(mask_files)} channel 1 files.")

    for mask_path in tqdm(mask_files):
        # 1. 读取 Mask (YOLO预测结果)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading {mask_path}")
            continue
            
        # 2. 读取 原图 (SEM)
        img_path = str(mask_path).replace('_0001.png', '_0000.png')
        img = cv2.imread(img_path)
        
        # 3. Mask 可视化 (彩色化)
        color_mask = apply_palette(mask)
        
        # 4. 叠加 (Overlay)
        if img is not None:
            # 确保尺寸匹配 (防患未然)
            if img.shape[:2] != color_mask.shape[:2]:
                color_mask = cv2.resize(color_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 使用 addWeighted 混合: 0.7 * 原图 + 0.3 * Mask
            overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
            
            # 添加图例到叠加图
            overlay = draw_legend(overlay)
            
            # 5. 拼接显示: [原图, 彩色Mask, 叠加结果]
            # 为了美观，并在每张图上写个标题
            border = np.zeros((img.shape[0], 5, 3), dtype=np.uint8) + 255 # 白色分割线
            combined = np.hstack([img, border, color_mask, border, overlay])
        else:
            # 如果没有原图，只保存Mask
            combined = color_mask

        # 保存
        save_path = dst_dir / mask_path.name.replace('_0001.png', '_vis.jpg')
        cv2.imwrite(str(save_path), combined)
        
    print(f"Saved visualizations to {dst_dir}")

def main():
    print(f"Visualizing YOLO Channels from {dataset_root}")
    print(f"Output to {output_root}")
    
    process_subset('imagesTr')
    process_subset('imagesTs')
    
    print("\nAll done!")

if __name__ == "__main__":
    main()
