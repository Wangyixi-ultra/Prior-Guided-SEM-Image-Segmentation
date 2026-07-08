#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只计算每个ID的class1与class2面积比例
"""
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

border_dir = '/home/chen/seg6/predict_no_label/wu/border'   # 轮廓图目录
mask_dir   = '/home/chen/seg6/predict_no_label/wu/out'      # 对应掩模目录
output_dir = '/home/chen/seg6/predict_no_label/wu/border_results'  # 结果输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

def measure_area_ratio(contour_path):
    """输入一张轮廓图路径，返回class1与class2面积比例"""
    name = os.path.splitext(os.path.basename(contour_path))[0].replace('_contour', '')
    mask_path = os.path.join(mask_dir, f'{name}.png')
    if not os.path.exists(mask_path):
        print(f'[SKIP] 找不到对应 mask: {mask_path}'); return None

    mask = imread(mask_path).astype(np.uint8)
    unique_cls = [cls for cls in np.unique(mask) if cls in {1, 2}]
    cls1_total_area = 0  # class1的总面积
    cls2_total_area = 0  # class2的总面积

    for cls_id in unique_cls:
        bin_mask = ((mask == cls_id) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 计算对应类别的总面积
        if cls_id == 1:
            for cnt in contours:
                cls1_total_area += cv2.contourArea(cnt)
        elif cls_id == 2:
            for cnt in contours:
                cls2_total_area += cv2.contourArea(cnt)
    
    # 计算class2与class1的面积比例（转换为百分比）
    cls2_to_cls1_area_ratio = (cls2_total_area / cls1_total_area * 100) if cls1_total_area > 0 else 0
    
    return name, cls2_to_cls1_area_ratio, cls1_total_area, cls2_total_area

def main():
    contour_files = sorted(glob.glob(os.path.join(border_dir, '*_contour.png')))
    if not contour_files:
        print('未找到 *_contour.png'); return

    # 存储每个ID的面积比例结果
    area_ratios = []
    
    for cfile in contour_files:
        ret = measure_area_ratio(cfile)
        if ret is None: continue
        name, ratio, cls1_area, cls2_area = ret
        
        # 提取文件ID
        import re
        numbers = re.findall(r'\d+', str(name))
        file_id = int(numbers[0]) if numbers else 0
        
        area_ratios.append({
            'id': file_id,
            'name': name,
            'cls1_area': cls1_area,
            'cls2_area': cls2_area,
            'ratio': ratio
        })
        
        print(f'ID {file_id}: {name} - Class1面积: {cls1_area:.2f}, Class2面积: {cls2_area:.2f}, 比例: {ratio:.2f}%')

    if not area_ratios:
        print('未找到有效的面积比例数据')
        return
    
    # 按ID排序
    area_ratios.sort(key=lambda x: x['id'])
    
    # 只保留前两个ID
    if len(area_ratios) > 2:
        area_ratios = area_ratios[:2]
        print(f'只处理前两个ID的数据')
    
    # 绘制面积比例图
    if len(area_ratios) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 提取数据
        ids = [item['id'] for item in area_ratios]
        ratios = [item['ratio'] for item in area_ratios]
        names = [item['name'] for item in area_ratios]
        
        # 创建柱状图
        bars = ax.bar(range(len(ids)), ratios, alpha=0.8, color=(224/255, 210/255, 229/255))
        ax.set_title('ABX₃/PbI₂ Area Ratio', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('ABX₃/PbI₂', fontsize=12)
        
        # 设置横坐标标签
        x_labels = ['control aged', '2dpa aged']
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(x_labels)
        
        # 在柱状图上添加数值标签
        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ratios)*0.01,
                   f'{ratio:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        ratio_plot_path = os.path.join(output_dir, 'class2_to_class1_area_ratio.png')
        plt.savefig(ratio_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f'\n已保存面积比例图 → {ratio_plot_path}')
        
        # 打印最终结果
        print('\n======== 最终结果 ========')
        for item in area_ratios:
            print(f'ID {item["id"]} ({item["name"]}): ABX₃/PbI₂ = {item["ratio"]:.2f}%')
    else:
        print(f'需要至少2个ID才能生成图表，当前只有 {len(area_ratios)} 个ID')

if __name__ == '__main__':
    main()