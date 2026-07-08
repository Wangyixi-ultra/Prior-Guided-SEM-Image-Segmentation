#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEM图片裁剪工具
用于批量裁剪SEM图片下方的拍摄信息部分，便于后续晶粒图分割处理
"""

import os
from PIL import Image
import numpy as np

# ==================== 用户配置区域 ====================
# 输入图片目录路径（需要裁剪的SEM图片所在文件夹）
INPUT_DIR = "/home/chen/seg6/predict_no_label/experiment/anneal/image"  # 请修改为实际的输入路径

# 输出图片目录路径（裁剪后图片保存的文件夹）
OUTPUT_DIR = "/home/chen/seg6/predict_no_label/experiment/anneal/image/resized"  # 请修改为实际的输出路径

# 裁剪区域高度（从底部裁剪的像素数，根据实际图片调整）
CROP_HEIGHT = 170  # 默认裁剪底部100像素，可根据实际情况修改

# 支持的图片格式
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
# ===================================================

def create_output_directory(output_dir):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

def get_image_files(input_dir):
    """获取输入目录中所有支持的图片文件"""
    image_files = []
    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_FORMATS:
                image_files.append(filename)
    return sorted(image_files)

def crop_image_bottom(image_path, crop_height):
    """
    裁剪图片底部指定高度的区域
    
    参数:
        image_path: 图片文件路径
        crop_height: 从底部裁剪的像素高度
    
    返回:
        裁剪后的图片对象
    """
    with Image.open(image_path) as img:
        width, height = img.size
        
        # 检查裁剪高度是否超过图片高度
        if crop_height >= height:
            raise ValueError(f"裁剪高度({crop_height}px)超过图片高度({height}px)")
        
        # 计算裁剪区域 (左, 上, 右, 下)
        # 保留从顶部到 (高度 - crop_height) 的区域
        crop_box = (0, 0, width, height - crop_height)
        cropped_img = img.crop(crop_box)
        
        return cropped_img

def process_images():
    """批量处理图片"""
    print("=" * 50)
    print("SEM图片裁剪工具")
    print("=" * 50)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"裁剪高度: {CROP_HEIGHT}像素")
    print("=" * 50)
    
    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        print("请修改 INPUT_DIR 变量为正确的路径")
        return
    
    # 创建输出目录
    create_output_directory(OUTPUT_DIR)
    
    # 获取所有图片文件
    image_files = get_image_files(INPUT_DIR)
    
    if not image_files:
        print(f"警告: 在 {INPUT_DIR} 中没有找到支持的图片文件")
        print(f"支持的格式: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    print("开始处理...")
    
    success_count = 0
    fail_count = 0
    
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            # 裁剪图片
            cropped_img = crop_image_bottom(input_path, CROP_HEIGHT)
            
            # 保存裁剪后的图片
            cropped_img.save(output_path)
            
            # 获取原始和裁剪后的尺寸
            with Image.open(input_path) as original_img:
                orig_width, orig_height = original_img.size
            crop_width, crop_height = cropped_img.size
            
            print(f"[{i}/{len(image_files)}] 成功: {filename}")
            print(f"  原始尺寸: {orig_width}x{orig_height} -> 裁剪后: {crop_width}x{crop_height}")
            
            success_count += 1
            
        except Exception as e:
            print(f"[{i}/{len(image_files)}] 失败: {filename}")
            print(f"  错误: {str(e)}")
            fail_count += 1
    
    print("=" * 50)
    print("处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    process_images()