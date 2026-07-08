import os
import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def convert_mask_to_yolo_format(mask, original_class_id, yolo_class_id):
    """
    将掩码转换为YOLO分割格式（归一化的像素坐标）
    """
    # 获取图像尺寸
    height, width = mask.shape[:2]
    
    # 查找轮廓 - 使用原始类别ID从掩码中提取对象
    contours, _ = cv2.findContours((mask == original_class_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_annotations = []
    for contour in contours:
        if len(contour) < 3:  # 至少需要3个点
            continue
            
        # 将轮廓点展平为一维数组
        points = contour.reshape(-1, 2)
        
        # 归一化坐标
        normalized_points = []
        for x, y in points:
            normalized_x = x / width
            normalized_y = y / height
            normalized_points.extend([normalized_x, normalized_y])
        
        # 添加YOLO类别ID和归一化点
        if len(normalized_points) >= 6:  # 至少3个点对才能构成多边形
            yolo_annotations.append([yolo_class_id] + normalized_points)
    
    return yolo_annotations

def process_dataset(source_dir, target_dir):
    """
    处理整个数据集，将nnUNet格式转换为YOLO格式
    """
    # 创建目标目录
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建images和labels子目录
    train_images_dir = target_dir / "images" / "train"
    train_labels_dir = target_dir / "labels" / "train"
    val_images_dir = target_dir / "images" / "val"
    val_labels_dir = target_dir / "labels" / "val"
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 读取原始数据集信息
    with open(os.path.join(source_dir, "dataset.json"), "r") as f:
        dataset_info = json.load(f)
    
    # 创建YOLO格式的数据集配置文件
    class_names = []
    for label_name in dataset_info["labels"]:
        if label_name != "background":
            class_names.append(label_name)
    
    # 写入YOLO数据集配置文件
    data_yaml_content = f"""path: {target_dir.absolute()}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(class_names):
        data_yaml_content += f"  {i}: {name}\n"
    
    with open(target_dir / "data.yaml", "w") as f:
        f.write(data_yaml_content)
    
    # 处理训练集标签
    process_labels(
        os.path.join(source_dir, "labelsTr"),
        os.path.join(source_dir, "imagesTr"),
        train_labels_dir,
        train_images_dir,
        dataset_info["labels"]
    )
    
    # 处理验证集标签
    process_labels(
        os.path.join(source_dir, "labelsTs"),
        os.path.join(source_dir, "imagesTs"),
        val_labels_dir,
        val_images_dir,
        dataset_info["labels"]
    )

def process_labels(labels_src_dir, images_src_dir, labels_dst_dir, images_dst_dir, label_mapping):
    """
    处理标签文件
    """
    # 创建从原始ID到YOLO ID的映射
    # YOLO ID从0开始，按字母顺序排列类别
    non_background_classes = [(name, id) for name, id in label_mapping.items() if name != "background"]
    non_background_classes.sort()  # 按名称排序
    
    original_to_yolo_id = {}
    for yolo_id, (name, original_id) in enumerate(non_background_classes):
        original_to_yolo_id[original_id] = yolo_id
        print(f"映射类别: {name} -> 原始ID: {original_id}, YOLO ID: {yolo_id}")
    
    # 获取所有标签文件
    label_files = list(Path(labels_src_dir).glob("*.png"))
    
    for label_file in label_files:
        # 读取标签图像
        mask = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"无法读取标签文件: {label_file}")
            continue
        
        # 生成YOLO格式的标注
        yolo_lines = []
        
        # 对每个非背景类别的标签进行处理
        for label_name, original_class_id in label_mapping.items():
            if label_name == "background":
                continue
                
            yolo_class_id = original_to_yolo_id[original_class_id]
            annotations = convert_mask_to_yolo_format(mask, original_class_id, yolo_class_id)
            for ann in annotations:
                # 格式化为字符串
                line = " ".join(map(str, ann))
                yolo_lines.append(line)
        
        # 写入YOLO标签文件
        label_filename = label_file.stem + ".txt"
        with open(os.path.join(labels_dst_dir, label_filename), "w") as f:
            f.write("\n".join(yolo_lines))
        
        # 复制对应的图像文件
        # 根据标签文件名找到对应的图像文件
        image_filename_base = label_file.stem
        image_file_src = os.path.join(images_src_dir, image_filename_base + "_0000.png")
        
        if os.path.exists(image_file_src):
            image_file_dst = os.path.join(images_dst_dir, label_file.name)  # 使用与标签相同的名称，但保持原格式
            img_data = cv2.imread(image_file_src)
            cv2.imwrite(os.path.join(images_dst_dir, label_file.name), img_data)
        else:
            print(f"找不到对应的图像文件: {image_file_src}")

def main():
    source_dataset_path = "/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset110_Perovskite"
    target_dataset_path = "/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset111_Perovskite"
    
    print("开始转换数据集...")
    process_dataset(source_dataset_path, target_dataset_path)
    print("数据集转换完成!")

if __name__ == "__main__":
    main()
