#!/usr/bin/env python3
"""
将YOLO分类数据集转换为检测数据集格式
每张图片生成一个伪标注框（整张图作为一个目标）
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_cls_to_det(src_dir, dst_dir):
    """
    将分类数据集转换为检测格式
    
    Args:
        src_dir: 源分类数据集目录 (包含 train/val 文件夹，每个文件夹下有类别子文件夹)
        dst_dir: 目标检测数据集目录
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # 获取类别列表
    classes = []
    for split in ['train', 'val']:
        split_path = src_path / split
        if split_path.exists():
            classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
            break
    
    if not classes:
        print("错误：未找到类别文件夹")
        return
    
    print(f"发现 {len(classes)} 个类别: {classes}")
    
    # 创建目标目录结构
    for split in ['train', 'val']:
        (dst_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 类别到索引的映射
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # 处理每个 split
    for split in ['train', 'val']:
        src_split_path = src_path / split
        if not src_split_path.exists():
            continue
        
        print(f"\n处理 {split} 集...")
        
        dst_img_path = dst_path / 'images' / split
        dst_lbl_path = dst_path / 'labels' / split
        
        # 遍历每个类别
        for class_name in tqdm(classes, desc=f"处理类别"):
            class_path = src_split_path / class_name
            if not class_path.exists():
                continue
            
            class_idx = class_to_idx[class_name]
            
            # 处理该类别下的所有图片
            for img_file in class_path.glob('*'):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    continue
                
                # 复制图片到目标目录
                new_img_name = f"{class_name}_{img_file.name}"
                dst_img_file = dst_img_path / new_img_name
                shutil.copy2(img_file, dst_img_file)
                
                # 生成标注文件
                # 伪标注：目标占据整张图片，中心点(0.5, 0.5)，宽高(1.0, 1.0)
                lbl_file = dst_lbl_path / (new_img_name.rsplit('.', 1)[0] + '.txt')
                with open(lbl_file, 'w') as f:
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
    
    # 生成 data.yaml
    yaml_content = f"""# YOLO 检测数据集配置
path: {dst_path.absolute()}  # 数据集根目录

train: images/train
val: images/val
test:  # 可选

# 类别
nc: {len(classes)}  # 类别数量
names: {classes}  # 类别名称
"""
    
    yaml_path = dst_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ 转换完成！")
    print(f"数据集保存到: {dst_path}")
    print(f"配置文件: {yaml_path}")
    
    # 统计信息
    for split in ['train', 'val']:
        img_dir = dst_path / 'images' / split
        lbl_dir = dst_path / 'labels' / split
        if img_dir.exists():
            n_img = len(list(img_dir.glob('*')))
            n_lbl = len(list(lbl_dir.glob('*')))
            print(f"  {split}: {n_img} 张图片, {n_lbl} 个标注文件")


if __name__ == '__main__':
    # 源分类数据集
    src_dir = 'raw/cropped_yolo_dataset'
    # 目标检测数据集
    dst_dir = 'raw/cropped_yolo_det_dataset'
    
    convert_cls_to_det(src_dir, dst_dir)
