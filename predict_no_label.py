#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnUNet 推理 → 原图 + 多类别轮廓线 叠加图（单阶段输出）
"""
import os
import glob
import subprocess
from imageio.v2 import imread
import numpy as np
import cv2

# --------------- 参数 ---------------
dataset_id = 102
fold       = 0
config     = '2d'
trainer    = 'nnUNetTrainer'
in_dir     = '/home/chen/seg6/predict_no_label/experiment/in/output'
out_dir    = '/home/chen/seg6/predict_no_label/experiment/out'
border_dir = '/home/chen/seg6/predict_no_label/experiment/border'
checkpoint = 'checkpoint_best.pth'
tmp_dir    = '/home/chen/seg6/predict_no_label/experiment/temp_nnUNet'  # 临时目录
# ------------------------------------

def parse_cli():
    import sys
    global trainer
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ('-tr', '--trainer'):
            if i + 1 >= len(args):
                raise SystemExit('缺少 trainer 值')
            trainer = args[i + 1]
            i += 2
        else:
            i += 1

def prep_input():
    """将原始图像转换为nnUNet格式"""
    import shutil
    from pathlib import Path
    
    # 清理并创建临时目录
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 查找所有图像文件
    img_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
        img_files.extend(glob.glob(os.path.join(in_dir, ext)))
        img_files.extend(glob.glob(os.path.join(in_dir, ext.upper())))
    
    if not img_files:
        raise ValueError(f"在 {in_dir} 中未找到图像文件")
    
    # 转换并复制到临时目录，重命名为nnUNet格式
    for idx, img_path in enumerate(sorted(img_files)):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"警告：无法读取 {img_path}，跳过")
            continue
        
        # 转换为单通道灰度图
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 保存为nnUNet格式：caseXXX_0000.png
        new_name = f'case{idx:03d}_0000.png'
        cv2.imwrite(os.path.join(tmp_dir, new_name), img)
        print(f'已准备：{img_path} -> {new_name}')
    
    return tmp_dir

def run_predict():
    # 首先准备输入数据
    prep_input()
    
    cmd = ['nnUNetv2_predict',
           '-i', tmp_dir, '-o', out_dir,
           '-d', str(dataset_id), '-c', config,
           '-f', str(fold), '-chk', checkpoint, '-tr', trainer]
    print('=== 1. 推理（无标签）===')
    subprocess.check_call(cmd)

def gen_contour_overlay():
    print('=== 2. 生成"原图+多类别轮廓线"叠加图 ===')
    os.makedirs(border_dir, exist_ok=True)

    # 类别颜色表（B,G,R），需要就继续加
    class_color = {1: (0, 140, 255),  # 橙
                   2: (0, 255, 0),    # 绿
                   3: (255, 0, 255),  # 洋红
                   }

    for mask_path in sorted(glob.glob(os.path.join(out_dir, '*.png'))):
        name = os.path.splitext(os.path.basename(mask_path))[0]
        img_path = os.path.join(tmp_dir, f'{name}_0000.png')  # 从临时目录读取
        if not os.path.isfile(img_path):
            print(f'[SKIP] {name}：找不到原图')
            continue
        try:
            img  = imread(img_path)
            mask = imread(mask_path).astype(np.uint8)

            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            for cls_id in np.unique(mask):
                if cls_id == 0:
                    continue
                cls_bin = ((mask == cls_id) * 255).astype(np.uint8)
                contours, _ = cv2.findContours(cls_bin,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
                color = class_color.get(cls_id, (255, 255, 255))
                cv2.drawContours(overlay, contours, -1, color, thickness=6)

            save_path = os.path.join(border_dir, f'{name}_contour.png')
            cv2.imwrite(save_path, overlay)
            print(f'[ OK ] {save_path} 已保存')
        except Exception as e:
            print(f'[FAIL] {name}：{e}')

def main():
    import shutil
    
    parse_cli()
    
    try:
        run_predict()
        gen_contour_overlay()
        print(f'=== 全部完成！轮廓线叠加图已输出至 {border_dir} ===')
    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            print(f'已清理临时目录：{tmp_dir}')

if __name__ == '__main__':
    main()