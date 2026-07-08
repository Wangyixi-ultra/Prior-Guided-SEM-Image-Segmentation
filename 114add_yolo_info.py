import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import glob

def update_dataset_json(dataset_path):
    """
    更新dataset.json文件，添加YOLO分类通道信息
    """
    json_path = os.path.join(dataset_path, 'dataset.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 检查是否已经添加过YOLO分类通道
    if "1" in data['channel_names'] and data['channel_names']["1"] == "YOLO_Class":
        print("dataset.json already updated.")
    else:
        # 添加新通道
        data['channel_names']["1"] = "YOLO_Class"
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print("Updated dataset.json with new channel.")

def process_images(image_dir, model):
    """
    对图像目录中的所有图像进行YOLO分类预测，并生成对应的分类结果图作为新通道
    """
    # 查找所有通道0的图像（nnU-Net命名规范 *_0000.png）
    img_files = glob.glob(os.path.join(image_dir, "*_0000.png"))
    
    # 动态确定映射关系，基于model.names
    # 数据集标签: {0: 'background', 1: 'PbI2', 2: 'ABO3', 3: 'defect'}
    
    # 定义期望的YOLO类别，将YOLO预测结果映射到数据集标签
    target_mapping = {
        'PbI2': 1,      # 钙钛矿材料PbI2
        'ABO3': 2,      # 钙钛矿材料ABO3
        'PbI₂': 1,      # 处理下标字符
        'ABO₃': 2,      # 处理下标字符
        'defect': 3     # 缺陷类别
    }
    
    print(f"Model classes: {model.names}")
    
    # 构建从YOLO类别索引到数据集标签值的映射
    yolo_to_mask_map = {}
    for idx, name in model.names.items():
        if name in target_mapping:
            yolo_to_mask_map[idx] = target_mapping[name]
        else:
            # 如果YOLO类别不在目标映射中，则映射到背景(0)
            print(f"Warning: YOLO class '{name}' not found in target mapping. It will map to 0.")
            yolo_to_mask_map[idx] = 0

    print(f"YOLO Index -> Pixel Value Map: {yolo_to_mask_map}")

    print(f"Processing {len(img_files)} images in {image_dir}...")
    
    for img_path in tqdm(img_files, desc=f"Processing {os.path.basename(image_dir)}"):
        # 使用YOLO模型进行预测
        results = model(img_path, verbose=False)
        
        # 获取最高概率的类别
        top1_class = results[0].probs.top1
        
        # 映射到对应的像素值
        fill_value = yolo_to_mask_map.get(top1_class, 0)
        
        # 读取原始图像以获取尺寸
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # 创建新通道图像（全图使用同一像素值）
        new_channel = np.full((h, w), fill_value, dtype=np.uint8)
        
        # 保存为*_0001.png格式（nnU-Net第二通道命名规范）
        out_path = img_path.replace("_0000.png", "_0001.png")
        cv2.imwrite(out_path, new_channel)

def main():
    """
    主函数：加载YOLO模型，更新数据集配置，处理训练和测试图像
    """
    # 配置参数
    dataset_root = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset114_Perovskite'
    yolo_weights = '/home/chen/seg6/perovskite_grains_opt/train29/weights/best.pt'
    
    # 检查YOLO权重文件是否存在
    if not os.path.exists(yolo_weights):
        print(f"Error: YOLO weights not found at {yolo_weights}")
        return

    print(f"Loading YOLO model from {yolo_weights}")
    model = YOLO(yolo_weights)
    
    # 1. 更新dataset.json文件，添加YOLO分类通道
    update_dataset_json(dataset_root)
    
    # 2. 处理训练图像
    train_dir = os.path.join(dataset_root, 'imagesTr')
    if os.path.exists(train_dir):
        process_images(train_dir, model)
    
    # 3. 处理测试图像（如果存在）
    test_dir = os.path.join(dataset_root, 'imagesTs')
    if os.path.exists(test_dir):
        process_images(test_dir, model)

if __name__ == "__main__":
    main()