import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

img_dir  = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset110_Perovskite/imagesTr'
mask_dir = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset110_Perovskite/labelsTr'
out_dir  = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset110_Perovskite/view'
os.makedirs(out_dir, exist_ok=True)

# 8 条高对比 BGR 颜色（OpenCV 用 BGR），多于 8 类会循环
CATEGORY_COLORS_BGR = [
    (  0,   0, 255),   # 红
    (  0, 255,   0),   # 绿
    (255,   0,   0),   # 蓝
    (  0, 255, 255),   # 黄
    (255,   0, 255),   # 品红
    (255, 255,   0),   # 青
    (255, 255, 255),   # 白
    (128, 128, 128),   # 灰
]

ok_count = 0
skip_count = 0

for mask_name in sorted(os.listdir(mask_dir)):
    if not mask_name.endswith('.png'):
        continue
    name, _ = os.path.splitext(mask_name)
    img_path  = os.path.join(img_dir, f'{name}_0000.png')
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.isfile(img_path):
        print(f'[SKIP] {name}：找不到对应灰度图')
        skip_count += 1
        continue

    try:
        img  = imread(img_path)          # H×W 灰度或 RGB
        mask = imread(mask_path)         # H×W 单通道，像素值 = 类别 id

        # 统一转成 3 通道
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img[:, :, :3]

        # 为了 OpenCV 处理，转 BGR
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 逐类画轮廓
        for cls_id in np.unique(mask):
            if cls_id == 0:               # 假设 0 是背景，跳过
                continue
            binary = (mask == cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = CATEGORY_COLORS_BGR[cls_id % len(CATEGORY_COLORS_BGR)]
            cv2.drawContours(vis, contours, -1, color, thickness=2)  # thickness 可调

        # 转回 RGB 保存
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        save_path = os.path.join(out_dir, f'{name}_border.png')
        plt.imsave(save_path, vis_rgb)

        print(f'[ OK ] {name} -> {save_path}')
        ok_count += 1
    except Exception as e:
        print(f'[FAIL] {name}：{e}')
        skip_count += 1

print(f'全部处理完成！成功 {ok_count} 张，跳过/失败 {skip_count} 张。')