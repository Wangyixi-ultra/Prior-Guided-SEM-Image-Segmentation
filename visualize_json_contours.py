#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据JSON标注文件生成原图和边界轮廓叠加图
只输出原图和轮廓叠加图，简洁明了
"""

import cv2
import json
import base64
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_labelme_json(json_path):
    """加载LabelMe格式的JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def decode_image_data(image_data):
    """解码base64图像数据"""
    if image_data is None:
        return None
    try:
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"图像解码失败: {e}")
        return None

def load_image_from_path(image_path):
    """从文件路径加载图像"""
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        return None
    except Exception as e:
        print(f"图像加载失败: {e}")
        return None

def create_contour_overlay(image, shapes):
    """创建轮廓叠加图"""
    overlay = image.copy()
    
    # 类别颜色映射
    class_colors = {
        "PbI₂": (255, 140, 0),    # 橙色
        "ABO₃": (0, 255, 0),      # 绿色
        "defect": (255, 0, 255),  # 紫色
    }
    
    # 为每个shape绘制轮廓
    for shape in shapes:
        label = shape.get("label", "unknown")
        points = shape.get("points", [])
        
        if len(points) < 3:  # 多边形至少需要3个点
            continue
            
        # 转换点为numpy数组
        pts = np.array(points, dtype=np.int32)
        
        # 获取类别颜色
        color = class_colors.get(label, (255, 255, 255))  # 默认白色
        
        # 绘制轮廓线
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=6)
    
    return overlay

def create_mask_from_shapes(image_shape, shapes):
    """从JSON shapes创建掩模图像"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # 类别标签到像素值的映射
    class_values = {
        "PbI₂": 1,
        "ABO₃": 2,
        "defect": 3,
    }
    
    # 为每个shape填充掩模
    for shape in shapes:
        label = shape.get("label", "unknown")
        points = shape.get("points", [])
        
        if len(points) < 3:  # 多边形至少需要3个点
            continue
            
        # 转换点为numpy数组
        pts = np.array(points, dtype=np.int32)
        
        # 获取类别像素值
        pixel_value = class_values.get(label, 0)  # 默认背景为0
        
        # 填充多边形区域
        cv2.fillPoly(mask, [pts], color=pixel_value)
    
    return mask

def visualize_json_contours(json_dir, output_dir, use_embedded_image=True):
    """可视化JSON标注文件的轮廓"""
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 获取所有JSON文件
    json_files = list(json_dir.glob("*.json"))
    
    if not json_files:
        print("未找到JSON标注文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON标注文件")
    
    for json_file in json_files:
        print(f"处理: {json_file.name}")
        
        # 加载JSON数据
        data = load_labelme_json(json_file)
        
        # 获取图像数据
        if use_embedded_image and "imageData" in data and data["imageData"]:
            image = decode_image_data(data["imageData"])
            if image is None:
                print(f"无法解码嵌入图像，尝试从文件加载")
                image_path = json_dir / data.get("imagePath", "")
                image = load_image_from_path(image_path)
        else:
            image_path = json_dir / data.get("imagePath", "")
            image = load_image_from_path(image_path)
        
        if image is None:
            print(f"无法加载图像: {json_file.name}")
            continue
        
        # 获取标注信息
        shapes = data.get("shapes", [])
        
        # 创建轮廓叠加图
        contour_overlay = create_contour_overlay(image, shapes)
        
        # 创建掩模
        mask = create_mask_from_shapes(image.shape, shapes)
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))  # 改为3个子图
        
        # 原图
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 轮廓叠加图
        axes[1].imshow(contour_overlay)
        axes[1].set_title('Contour Overlay', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 掩模图
        axes[2].imshow(mask, cmap='tab10', vmin=0, vmax=3)  # 使用合适的颜色映射
        axes[2].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(255/255, 140/255, 0/255), label='PbI₂'),
            Patch(facecolor=(0/255, 255/255, 0/255), label='ABO₃'),
            Patch(facecolor=(255/255, 0/255, 255/255), label='defect'),
        ]
        fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)
        
        plt.tight_layout()
        
        # 保存结果
        base_name = json_file.stem
        output_path = output_dir / f"{base_name}_contour_overlay.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # 单独保存轮廓叠加图
        contour_path = output_dir / f"{base_name}_overlay.png"
        cv2.imwrite(str(contour_path), cv2.cvtColor(contour_overlay, cv2.COLOR_RGB2BGR))
        
        # 单独保存掩模
        mask_path = output_dir / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        print(f"已保存: {output_path.name}")
        print(f"  - 已生成掩模: {mask_path.name}")
        
        # 统计类别
        class_stats = {}
        for shape in shapes:
            label = shape.get("label", "unknown")
            class_stats[label] = class_stats.get(label, 0) + 1
        
        if class_stats:
            print(f"  - 标注统计:")
            for label, count in class_stats.items():
                print(f"    {label}: {count} 个区域")
        else:
            print("  - 无标注信息")

def main():
    """主函数"""
    # JSON标注文件目录
    json_dir = "/home/chen/seg6/predict_no_label/wu/json"
    
    # 输出目录
    output_dir = "/home/chen/seg6/predict_no_label/wu/json_contours"
    
    # 是否使用嵌入的图像数据（如果可用）
    use_embedded_image = True
    
    print("开始生成轮廓叠加图...")
    visualize_json_contours(json_dir, output_dir, use_embedded_image)
    print("轮廓叠加图生成完成！")

if __name__ == "__main__":
    main()