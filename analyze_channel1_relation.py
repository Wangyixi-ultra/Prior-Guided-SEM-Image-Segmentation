#!/usr/bin/env python3
"""
分析 Dataset122_Perovskite 中 channel 1 (YOLO_Feature_Class) 的灰度值与分类的关系
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import glob

# 数据集路径
data_dir = "U-Mamba/data/nnUNet_raw/Dataset122_Perovskite"
images_dir = os.path.join(data_dir, "imagesTr")
labels_dir = os.path.join(data_dir, "labelsTr")

# 获取所有图像文件
image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))

print("=" * 80)
print("分析 Dataset122_Perovskite 中 Channel 1 (YOLO_Feature_Class) 与分类的关系")
print("=" * 80)

# 存储每个类别的灰度值
class_pixel_values = defaultdict(list)
class_names = {0: "background", 1: "PbI2", 2: "ABO3"}

# 分析所有训练图像
for img_path in image_files:
    filename = os.path.basename(img_path)
    
    # 解析文件名获取基础名称
    # 文件名格式: XX_0000.png (channel 0) 或 XX_0001.png (channel 1)
    if "_0001.png" in filename:
        # 这是 channel 1 (YOLO_Feature_Class)
        base_name = filename.replace("_0001.png", "")
        
        # 找到对应的 label 文件
        # 尝试不同的 label 文件名格式
        label_path = None
        possible_labels = [
            os.path.join(labels_dir, f"{base_name}.png"),
            os.path.join(labels_dir, f"{base_name.replace('_aug1', '')}.png"),
            os.path.join(labels_dir, f"{base_name.replace('_aug2', '')}.png"),
            os.path.join(labels_dir, f"{base_name.replace('_aug3', '')}.png"),
            os.path.join(labels_dir, f"{base_name.replace('_aug4', '')}.png"),
            os.path.join(labels_dir, f"{base_name.replace('_aug5', '')}.png"),
        ]
        
        for lp in possible_labels:
            if os.path.exists(lp):
                label_path = lp
                break
        
        if label_path is None:
            continue
        
        # 读取 channel 1 图像和标签
        channel1_img = np.array(Image.open(img_path))
        label_img = np.array(Image.open(label_path))
        
        # 确保尺寸匹配
        if channel1_img.shape != label_img.shape:
            continue
        
        # 对每个类别收集像素值
        for class_id in [0, 1, 2]:
            mask = (label_img == class_id)
            if mask.any():
                pixel_values = channel1_img[mask]
                class_pixel_values[class_id].extend(pixel_values.tolist())

# 统计结果
print("\n" + "=" * 80)
print("统计结果")
print("=" * 80)

for class_id in [0, 1, 2]:
    values = class_pixel_values[class_id]
    if len(values) > 0:
        values_array = np.array(values)
        print(f"\n{class_names[class_id]} (Class {class_id}):")
        print(f"  像素总数: {len(values_array):,}")
        print(f"  最小值: {values_array.min()}")
        print(f"  最大值: {values_array.max()}")
        print(f"  平均值: {values_array.mean():.2f}")
        print(f"  中位数: {np.median(values_array):.2f}")
        print(f"  标准差: {values_array.std():.2f}")
        print(f"  25%分位数: {np.percentile(values_array, 25):.2f}")
        print(f"  75%分位数: {np.percentile(values_array, 75):.2f}")
    else:
        print(f"\n{class_names[class_id]} (Class {class_id}): 无数据")

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 灰度值分布直方图
ax1 = axes[0, 0]
colors = ['gray', 'blue', 'red']
for class_id in [0, 1, 2]:
    values = class_pixel_values[class_id]
    if len(values) > 0:
        ax1.hist(values, bins=50, alpha=0.5, label=f"{class_names[class_id]} (n={len(values):,})", 
                 color=colors[class_id], density=True)
ax1.set_xlabel("Channel 1 Pixel Value")
ax1.set_ylabel("Density")
ax1.set_title("Channel 1 Gray Value Distribution by Class")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 箱线图
ax2 = axes[0, 1]
data_for_box = []
labels_for_box = []
for class_id in [0, 1, 2]:
    values = class_pixel_values[class_id]
    if len(values) > 0:
        data_for_box.append(values)
        labels_for_box.append(f"{class_names[class_id]}\n(n={len(values):,})")

if data_for_box:
    bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.set_ylabel("Channel 1 Pixel Value")
    ax2.set_title("Channel 1 Gray Value Boxplot by Class")
    ax2.grid(True, alpha=0.3)

# 3. 小提琴图
ax3 = axes[1, 0]
if data_for_box:
    parts = ax3.violinplot(data_for_box, positions=range(1, len(data_for_box) + 1), 
                           showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors[:len(data_for_box)]):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    ax3.set_xticks(range(1, len(labels_for_box) + 1))
    ax3.set_xticklabels(labels_for_box)
    ax3.set_ylabel("Channel 1 Pixel Value")
    ax3.set_title("Channel 1 Gray Value Violin Plot by Class")
    ax3.grid(True, alpha=0.3)

# 4. 统计摘要表格
ax4 = axes[1, 1]
ax4.axis('off')
table_data = []
for class_id in [0, 1, 2]:
    values = class_pixel_values[class_id]
    if len(values) > 0:
        values_array = np.array(values)
        table_data.append([
            class_names[class_id],
            f"{len(values_array):,}",
            f"{values_array.min()}",
            f"{values_array.max()}",
            f"{values_array.mean():.2f}",
            f"{np.median(values_array):.2f}",
            f"{values_array.std():.2f}"
        ])

table = ax4.table(cellText=table_data,
                  colLabels=['Class', 'Count', 'Min', 'Max', 'Mean', 'Median', 'Std'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0.3, 1, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax4.set_title("Statistical Summary", y=0.95)

plt.tight_layout()
plt.savefig("channel1_class_relation_analysis.png", dpi=150, bbox_inches='tight')
print("\n" + "=" * 80)
print("可视化已保存到: channel1_class_relation_analysis.png")
print("=" * 80)

# 分析不同类别之间的灰度值区分度
print("\n" + "=" * 80)
print("类别间灰度值区分度分析")
print("=" * 80)

bg_values = np.array(class_pixel_values[0]) if class_pixel_values[0] else None
pb_values = np.array(class_pixel_values[1]) if class_pixel_values[1] else None
abo_values = np.array(class_pixel_values[2]) if class_pixel_values[2] else None

if bg_values is not None and pb_values is not None:
    print(f"\nBackground vs PbI2:")
    print(f"  Background 均值: {bg_values.mean():.2f}, PbI2 均值: {pb_values.mean():.2f}")
    print(f"  均值差: {abs(bg_values.mean() - pb_values.mean()):.2f}")
    
if bg_values is not None and abo_values is not None:
    print(f"\nBackground vs ABO3:")
    print(f"  Background 均值: {bg_values.mean():.2f}, ABO3 均值: {abo_values.mean():.2f}")
    print(f"  均值差: {abs(bg_values.mean() - abo_values.mean()):.2f}")
    
if pb_values is not None and abo_values is not None:
    print(f"\nPbI2 vs ABO3:")
    print(f"  PbI2 均值: {pb_values.mean():.2f}, ABO3 均值: {abo_values.mean():.2f}")
    print(f"  均值差: {abs(pb_values.mean() - abo_values.mean()):.2f}")

# 检查灰度值是否直接对应类别
print("\n" + "=" * 80)
print("灰度值与类别直接对应关系检查")
print("=" * 80)

# 统计每个灰度值对应的类别分布
gray_class_distribution = defaultdict(lambda: defaultdict(int))
for class_id in [0, 1, 2]:
    values = class_pixel_values[class_id]
    for v in values:
        gray_class_distribution[v][class_id] += 1

# 打印前10个最常见的灰度值及其类别分布
print("\n最常见的灰度值及其类别分布 (前20):")
sorted_gray = sorted(gray_class_distribution.items(), key=lambda x: sum(x[1].values()), reverse=True)[:20]
for gray_val, class_dist in sorted_gray:
    total = sum(class_dist.values())
    bg_pct = class_dist[0] / total * 100 if total > 0 else 0
    pb_pct = class_dist[1] / total * 100 if total > 0 else 0
    abo_pct = class_dist[2] / total * 100 if total > 0 else 0
    print(f"  灰度值 {gray_val:3d}: 总计 {total:7,} | Background: {bg_pct:5.1f}%, PbI2: {pb_pct:5.1f}%, ABO3: {abo_pct:5.1f}%")

plt.show()
