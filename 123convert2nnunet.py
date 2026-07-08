#!/usr/bin/env python3
"""
一键把钙钛矿 SEM 灰度图 + 8-bit 灰度 mask 转成 nnUNet v2 标准格式
并生成可视化叠加图（红色=defect）用于肉眼检查是否被遮盖
python convert2nnunet_vis.py
增加测试集，用于最终模型评估
"""
import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# ========== 用户配置 ==========
raw_img    = '/home/chen/seg6/raw/addtrainimage'          # 原始灰度图
raw_mask   = '/home/chen/seg6/raw/addtrain_mask_only_gray'  # 0/120/180/255 灰度 mask
out_root   = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/dataset'                              # 临时目录
task_id    = 123
task_name  = f'Dataset{task_id}_Perovskite'
nnunet_out = os.path.join(os.path.dirname(out_root), task_name)
# ========================================

# 建立可视化目录
vis_dir = 'vis'
os.makedirs(vis_dir, exist_ok=True)

# 1. 建立 nnUNet 目录结构（Tr=训练集, Ts=测试集）
# 注意：验证集由nnUNet框架在训练时自动从训练集中划分，无需在此定义
for split in ['Tr', 'Ts']:
    os.makedirs(f'{out_root}/images{split}', exist_ok=True)
    os.makedirs(f'{out_root}/labels{split}', exist_ok=True)

# ---------- 2. 训练/测试拆分（仅按文件名规则） ----------
# 支持多种图片格式
supported_extensions = ['.png', '.tiff', '.tif','.jpg', '.jpeg', '.bmp']
file_list = []
for ext in supported_extensions:
    file_list.extend([f for f in os.listdir(raw_img) if f.lower().endswith(ext)])

names = [os.path.splitext(n)[0] for n in file_list]
# 规则：文件名里带 02、04、06、10、16、22、24、26、30、38、44、48 的进测试集，其余训练集
test_names = {'02','06','24','37','38','41','50','65','66','67','82','84'}
train_names = [n for n in names if not any(t in n for t in test_names)]
test_names  = [n for n in names if     any(t in n for t in test_names)]
print(f'训练集: {len(train_names)} 张, 测试集: {len(test_names)} 张')

# ---------- 3. 处理函数（完全复用你已有的增强+可视化） ----------
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

def process_split(split, name_list):
    for name in tqdm(name_list, desc=f'Processing {split}'):
        # 3.1 图像：强制单通道 8-bit
        # 尝试多种扩展名
        img_path = None
        for ext in supported_extensions:
            test_path = os.path.join(raw_img, f'{name}{ext}')
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            print(f"警告: 找不到图片文件 {name}")
            continue
            
        # 使用PIL读取图片以支持多种格式
        gray = read_image_with_pil(img_path)
        if gray is None:
            continue

        # 强制缩放到 1024x768
        target_w, target_h = 1024, 768
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 直接使用原图，不进行增强
        gray_enh = gray

        # 3.2 mask：灰度值 → 连续类别 ID
        # 尝试多种扩展名
        msk_path = None
        for ext in supported_extensions:
            test_path = os.path.join(raw_mask, f'{name}{ext}')
            if os.path.exists(test_path):
                msk_path = test_path
                break
        
        if msk_path is None:
            print(f"警告: 找不到mask文件 {name}")
            continue
            
        msk = read_image_with_pil(msk_path)
        if msk is None:
            continue

        # 强制缩放到 1024x768 (使用最近邻插值保持标签值)
        msk = cv2.resize(msk, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        out_msk = np.zeros_like(msk)
        out_msk[msk == 120] = 1   # PbI₂
        out_msk[msk == 180] = 2   # ABO₃
        #out_msk[msk == 255] = 3   # defect

        # ---------- 可视化 ----------
        # 创建伪彩色图用于可视化
        pseudo = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        overlay = pseudo.copy()
        #overlay[msk == 255] = [0, 0, 255]
        overlay[msk == 120] = [180, 180, 0]
        overlay[msk == 180] = [0, 180, 180]
        cv2.imwrite(os.path.join(vis_dir, f'{name}_overlay.png'), overlay)

        # 保存 nnUNet 格式文件
        cv2.imwrite(os.path.join(out_root, f'images{split}', f'{name}_0000.png'), gray_enh)
        cv2.imwrite(os.path.join(out_root, f'labels{split}', f'{name}.png'), out_msk)

# ---------- 4. 一次性生成 Tr/Ts ----------
process_split('Tr', train_names)
process_split('Ts', test_names)

# ---------- 5. dataset.json ----------
dataset_json = {
    "channel_names": {"0": "SEM"},
    "labels": {"background": 0, "PbI2": 1, "ABO3": 2},
    "numTraining": len(train_names),
    "file_ending": ".png"  # nnUNet输出仍使用png格式
}
with open(f'{out_root}/dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)

# ---------- 6. 移到 nnUNet 目录 ----------
if os.path.exists(nnunet_out):
    shutil.rmtree(nnunet_out)
shutil.move(out_root, nnunet_out)

print(f'\n✅ 数据集已拆好！Tr:{len(train_names)}  Ts:{len(test_names)}')
print(f'nnUNet 目录：{nnunet_out}')
print(f'可视化叠加图目录：{os.path.abspath(vis_dir)}')
print(f'注意：测试集将在训练时由nnUNet框架自动从训练集中划分（默认5折交叉测试）')