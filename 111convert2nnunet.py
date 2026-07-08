#!/usr/bin/env python3
"""
一键把钙钛矿 SEM 灰度图 + 8-bit 灰度 mask 转成 nnUNet v2 标准格式
并生成可视化叠加图（红色=defect）用于肉眼检查是否被遮盖
python 111convert2nnunet.py
增加测试集，用于最终模型评估
修改：文件名统一改为按数字顺序编号（如 01_0000.png），不再使用原始文件名
"""
import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# ========== 用户配置 ==========
raw_img    = '/home/chen/seg6/raw/img_dir_resized'          # 原始灰度图
raw_mask   = '/home/chen/seg6/raw/mask_only_gray_dir_nodefect_resized'  # 0/120/180/255 灰度 mask
out_root   = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/dataset'                              # 临时目录
task_id    = 111
task_name  = f'Dataset{task_id}_Perovskite'
nnunet_out = os.path.join(os.path.dirname(out_root), task_name)
# ========================================

# 建立可视化目录
vis_dir = 'vis'
os.makedirs(vis_dir, exist_ok=True)

# 1. 建立 nnUNet 目录结构（Tr=训练集, Ts=测试集）
# Note: Delete old temporary dir if exists to avoid mixing
if os.path.exists(out_root):
    shutil.rmtree(out_root)

for split in ['Tr', 'Ts']:
    os.makedirs(f'{out_root}/images{split}', exist_ok=True)
    os.makedirs(f'{out_root}/labels{split}', exist_ok=True)

# ---------- 2. 训练/测试拆分（仅按文件名规则） ----------
supported_extensions = ['.png', '.tiff', '.tif']
file_list = []
# Ensure sorted order for reproducibility
if os.path.exists(raw_img):
    for f in sorted(os.listdir(raw_img)):
        if any(f.lower().endswith(ext) for ext in supported_extensions):
            file_list.append(f)
else:
    print(f"Error: {raw_img} does not exist.")
    file_list = []

names = [os.path.splitext(n)[0] for n in file_list]
# 规则：文件名里带以下关键字的进测试集，其余训练集
test_keywords = {'02','06','10','22','24','37','38','41'}

train_names = []
test_names = []
for n in names:
    # 检查是否包含任何关键字
    is_test = False
    for k in test_keywords:
        if k in n:
            is_test = True
            break
    if is_test:
        test_names.append(n)
    else:
        train_names.append(n)

# 排序以保证编号稳定
train_names.sort()
test_names.sort()

print(f'训练集: {len(train_names)} 张, 测试集: {len(test_names)} 张')

# 记录映射关系
name_mapping = {}

# ---------- 3. 处理函数 ----------
def read_image_with_pil(path):
    """使用PIL读取图片，支持多种格式"""
    try:
        img = Image.open(path)
        if img.mode != 'L':  # 如果不是灰度图
            img = img.convert('L')  # 转换为灰度
        return np.array(img)
    except Exception as e:
        print(f"读取图片失败 {path}: {e}")
        return None

def process_split(split, name_list, start_index):
    """
    start_index: 起始编号
    return: 处理后的下一个可用编号
    """
    current_idx = start_index
    
    for name in tqdm(name_list, desc=f'Processing {split}'):
        # 3.1 图像
        img_path = None
        for ext in supported_extensions:
            test_path = os.path.join(raw_img, f'{name}{ext}')
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            print(f"警告: 找不到图片文件 {name}")
            continue
            
        gray = read_image_with_pil(img_path)
        if gray is None:
            continue

        target_w, target_h = 1024, 768
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        gray_enh = gray

        # 3.2 mask
        msk_path = None
        for ext in supported_extensions:
            test_path = os.path.join(raw_mask, f'{name}{ext}')
            if os.path.exists(test_path):
                msk_path = test_path
                break
        
        if msk_path is None:
            print(f"警告: 找不到mask文件 {name}，跳过")
            continue
            
        msk = read_image_with_pil(msk_path)
        if msk is None:
            print(f"读取mask失败 {name}，跳过")
            continue

        msk = cv2.resize(msk, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        out_msk = np.zeros_like(msk)
        out_msk[msk == 120] = 1   # PbI₂
        out_msk[msk == 180] = 2   # ABO₃
        #out_msk[msk == 255] = 3   # defect

        # 构造新文件名：2位数字 (如 05_0000.png)
        new_name = f'{current_idx:02d}'
        
        # 记录映射
        name_mapping[new_name] = name

        # 可视化文件名也用新的，方便对应
        pseudo = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        overlay = pseudo.copy()
        overlay[msk == 120] = [180, 180, 0]
        overlay[msk == 180] = [0, 180, 180]
        cv2.imwrite(os.path.join(vis_dir, f'{new_name}_overlay_{name}.png'), overlay)

        # 保存 nnUNet
        cv2.imwrite(os.path.join(out_root, f'images{split}', f'{new_name}_0000.png'), gray_enh)
        cv2.imwrite(os.path.join(out_root, f'labels{split}', f'{new_name}.png'), out_msk)
        
        current_idx += 1
        
    return current_idx

# ---------- 4. 执行处理 ----------
# 训练集从 01 开始
next_idx = process_split('Tr', train_names, start_index=1)
# 测试集接在训练集后面
process_split('Ts', test_names, start_index=next_idx)

# ---------- 5. 保存映射表 ----------
with open(f'{out_root}/name_mapping.json', 'w') as f:
    json.dump(name_mapping, f, indent=2)

# ---------- 6. dataset.json ----------
dataset_json = {
    "channel_names": {"0": "SEM"},
    "labels": {"background": 0, "PbI2": 1, "ABO3": 2},
    "numTraining": len(train_names),
    "file_ending": ".png"
}
with open(f'{out_root}/dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)

# ---------- 7. 移到 nnUNet 目录 ----------
if os.path.exists(nnunet_out):
    shutil.rmtree(nnunet_out)
shutil.move(out_root, nnunet_out)

print(f'\n✅ 数据集已拆好！Tr:{len(train_names)}  Ts:{len(test_names)}')
print(f'nnUNet 目录：{nnunet_out}')
print(f'名称映射已保存至：{nnunet_out}/name_mapping.json')
print(f'可视化叠加图目录：{os.path.abspath(vis_dir)}')
