#!/usr/bin/env python3
"""
One-click conversion of perovskite SEM grayscale images + 8-bit grayscale masks into nnUNet v2 standard format,
and generate overlay visualizations (red=defect) for visual inspection of coverage.
python convert2nnunet_vis.py
Adds a test set for final model evaluation.
"""
import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# ========== User configuration ==========
raw_img    = '/home/chen/seg6/raw/addtrainimage'          # raw grayscale images
raw_mask   = '/home/chen/seg6/raw/addtrain_mask_only_gray'  # 0/120/180/255 grayscale mask
out_root   = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/dataset'                              # temporary directory
task_id    = 123
task_name  = f'Dataset{task_id}_Perovskite'
nnunet_out = os.path.join(os.path.dirname(out_root), task_name)
# ========================================

# Create visualization directory
vis_dir = 'vis'
os.makedirs(vis_dir, exist_ok=True)

# 1. Build nnUNet directory structure (Tr=training set, Ts=test set)
# Note: the validation set is automatically split from the training set by nnUNet during training; no need to define it here
for split in ['Tr', 'Ts']:
    os.makedirs(f'{out_root}/images{split}', exist_ok=True)
    os.makedirs(f'{out_root}/labels{split}', exist_ok=True)

# ---------- 2. Train/test split (based only on filename rules) ----------
# Support multiple image formats
supported_extensions = ['.png', '.tiff', '.tif','.jpg', '.jpeg', '.bmp']
file_list = []
for ext in supported_extensions:
    file_list.extend([f for f in os.listdir(raw_img) if f.lower().endswith(ext)])

names = [os.path.splitext(n)[0] for n in file_list]
# Rule: filenames containing 02, 04, 06, 10, 16, 22, 24, 26, 30, 38, 44, 48 go to the test set; the rest go to training
test_names = {'02','06','24','37','38','41','50','65','66','67','82','84'}
train_names = [n for n in names if not any(t in n for t in test_names)]
test_names  = [n for n in names if     any(t in n for t in test_names)]
print(f'Training set: {len(train_names)} images, Test set: {len(test_names)} images')

# ---------- 3. Processing function (reuses your existing augmentation + visualization) ----------
def read_image_with_pil(path):
    """Read image with PIL; supports multiple formats."""
    try:
        img = Image.open(path)
        if img.mode != 'L':  # if not grayscale
            img = img.convert('L')  # convert to grayscale
        return np.array(img)
    except Exception as e:
        print(f"Failed to read image {path}: {e}")
        return None

def process_split(split, name_list):
    for name in tqdm(name_list, desc=f'Processing {split}'):
        # 3.1 Image: force single-channel 8-bit
        # Try multiple extensions
        img_path = None
        for ext in supported_extensions:
            test_path = os.path.join(raw_img, f'{name}{ext}')
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            print(f"Warning: image file not found {name}")
            continue
            
        # Read image with PIL to support multiple formats
        gray = read_image_with_pil(img_path)
        if gray is None:
            continue

        # Force resize to 1024x768
        target_w, target_h = 1024, 768
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Use the original image directly, no augmentation
        gray_enh = gray

        # 3.2 mask: grayscale values -> contiguous class IDs
        # Try multiple extensions
        msk_path = None
        for ext in supported_extensions:
            test_path = os.path.join(raw_mask, f'{name}{ext}')
            if os.path.exists(test_path):
                msk_path = test_path
                break
        
        if msk_path is None:
            print(f"Warning: mask file not found {name}")
            continue
            
        msk = read_image_with_pil(msk_path)
        if msk is None:
            continue

        # Force resize to 1024x768 (use nearest-neighbor interpolation to preserve label values)
        msk = cv2.resize(msk, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        out_msk = np.zeros_like(msk)
        out_msk[msk == 120] = 1   # PbI₂
        out_msk[msk == 180] = 2   # ABO₃
        #out_msk[msk == 255] = 3   # defect

        # ---------- Visualization ----------
        # Create pseudo-color image for visualization
        pseudo = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        overlay = pseudo.copy()
        #overlay[msk == 255] = [0, 0, 255]
        overlay[msk == 120] = [180, 180, 0]
        overlay[msk == 180] = [0, 180, 180]
        cv2.imwrite(os.path.join(vis_dir, f'{name}_overlay.png'), overlay)

        # Save in nnUNet format
        cv2.imwrite(os.path.join(out_root, f'images{split}', f'{name}_0000.png'), gray_enh)
        cv2.imwrite(os.path.join(out_root, f'labels{split}', f'{name}.png'), out_msk)

# ---------- 4. Generate Tr/Ts at once ----------
process_split('Tr', train_names)
process_split('Ts', test_names)

# ---------- 5. dataset.json ----------
dataset_json = {
    "channel_names": {"0": "SEM"},
    "labels": {"background": 0, "PbI2": 1, "ABO3": 2},
    "numTraining": len(train_names),
    "file_ending": ".png"  # nnUNet output still uses PNG format
}
with open(f'{out_root}/dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)

# ---------- 6. Move to nnUNet directory ----------
if os.path.exists(nnunet_out):
    shutil.rmtree(nnunet_out)
shutil.move(out_root, nnunet_out)

print(f'\n✅ Dataset split complete! Tr:{len(train_names)}  Ts:{len(test_names)}')
print(f'nnUNet directory: {nnunet_out}')
print(f'Overlay visualization directory: {os.path.abspath(vis_dir)}')
print(f'Note: the test set will be automatically split from the training set by the nnUNet framework during training (default 5-fold cross-validation)')