import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.tasks
from ultralytics.nn.modules import CBAM
import json
from tqdm import tqdm

# --- 1. 配置路径 ---
# 目标数据集路径 (需要生成 channel 1 的地方)
dataset_root = Path('/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset115_Perovskite')
images_tr_dir = dataset_root / 'imagesTr'
images_ts_dir = dataset_root / 'imagesTs' # 如果有测试集也要处理

# 模型路径
detector_path = '/home/chen/runs/detect/train18/weights/best.pt'
classifier_path = '/home/chen/seg6/perovskite_grains_opt/yolo_cbam_s_128/weights/best.pt'

# --- 2. 注册 CBAM 模块 (必须与训练代码一致) ---
ultralytics.nn.tasks.CBAM = CBAM

# --- 3. 定义标签映射 ---
# U-Mamba/nnUNet dataset.json 中的标签定义:
# "background": 0, "PbI2": 1, "ABO3": 2, "defect": 3
# 我们需要将分类器的结果映射到这些 ID
LABEL_MAP = {
    'background': 0,
    'pbi2': 1,
    'abo3': 2,
    'defect': 3
}

# 辅助函数：标准化类名 (处理下标 unicode 等)
def normalize_name(name):
    name = name.lower()
    if 'pbi' in name: return 'pbi2'  # 处理 PbI₂
    if 'abo' in name: return 'abo3'  # 处理 ABO₃
    return name

def process_images(image_dir, detector, classifier):
    if not image_dir.exists():
        print(f"Directory {image_dir} does not exist, skipping.")
        return

    image_files = sorted(list(image_dir.glob('*_0000.png')))
    print(f"Found {len(image_files)} images in {image_dir}")

    for img_path in tqdm(image_files):
        # 读取原始图像
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # 创建空白的 channel 1 (mask)
        # 用 0 (background) 初始化
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. 检测 (Detection)
        # 调低 conf 阈值以尽可能多地找到目标，分类器会进行二次确认
        det_results = detector(img, verbose=False, conf=0.25) 
        
        boxes = []
        if len(det_results) > 0:
            boxes = det_results[0].boxes

        # 2. 对每个检测到的框进行分类 (Classification)
        for box in boxes:
            # 获取坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 边界保护
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            # 抠图 crop
            crop = img[y1:y2, x1:x2]
            
            # 分类
            # verbose=False 防止刷屏
            cls_results = classifier(crop, verbose=False)
            
            if len(cls_results) > 0:
                # 获取预测类别名称
                # probs = cls_results[0].probs
                # top1_idx = probs.top1
                # class_name = cls_results[0].names[top1_idx]
                
                # 更稳健的方法: 获取置信度最高的类别
                top1_idx = cls_results[0].probs.top1
                class_name = cls_results[0].names[top1_idx]
                
                norm_name = normalize_name(class_name)
                label_id = LABEL_MAP.get(norm_name, 0)
                
                if label_id > 0:
                    # 将框内的区域填充为对应的 label_id
                    # 也可以选择只填充中心部分，或者使用高斯热图，这里简单填充矩形
                    # 注意：如果有重叠，这里是覆盖式（后画的覆盖先画的）
                    mask[y1:y2, x1:x2] = label_id

        # 3. 保存为 _0001.png
        out_path = img_path.parent / img_path.name.replace('_0000.png', '_0001.png')
        if out_path == img_path: # 防止命名错误覆盖原图
             print("Filename error, skipping save")
             continue
             
        cv2.imwrite(str(out_path), mask)

def main():
    # 加载模型
    print(f"Loading Detector: {detector_path}")
    detector = YOLO(detector_path)
    
    print(f"Loading Classifier: {classifier_path}")
    classifier = YOLO(classifier_path)

    # 处理 Training data
    print("Processing Training Images...")
    process_images(images_tr_dir, detector, classifier)
    
    # 处理 Test data (如果存在)
    if images_ts_dir.exists():
        print("Processing Test Images...")
        process_images(images_ts_dir, detector, classifier)

    print("Done! Channel 1 generated.")

if __name__ == "__main__":
    main()
