import cv2
import numpy as np
import os
import glob
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima # Added for deep groove detection
from skimage.segmentation import watershed
from scipy import ndimage
import json
from ultralytics import YOLO

def process_single_mask(mask_path, output_dir, original_img_dir, crop_output_dir, model, min_instance_area=50, min_distance=10):
    """
    处理单个掩码：
    1. 读取灰度掩码
    2. 使用距离变换+分水岭算法分割粘连实例（近似凸包划分）
    3. 提取质心和ID
    4. 从原图中切出对应的实例 (Crop)
    5. 使用YOLO分类模型对Crop进行推理
    """
    filename = os.path.basename(mask_path)
    
    # 构建原图文件名: mask '06.png' -> image '06_0000.png'
    # 假设mask文件名格式为 {id}.png
    name_no_ext = os.path.splitext(filename)[0]
    original_img_name = f"{name_no_ext}_0000.png"
    original_img_path = os.path.join(original_img_dir, original_img_name)

    # 读取原图
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Warning: 无法读取原图 {original_img_path}，将跳过Crop步骤")
    
    # 读取图片 (假设背景为0，前景>0)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: 无法读取 {mask_path}")
        return []

    # 二值化
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if binary is None:
         _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # --- 核心算法：距离变换 + 分水岭 ---
    
    # 1. 计算距离变换 (每个前景像素到最近背景像素的距离)
    # 这能体现物体的形态，中心位置像素值最高
    distance = ndimage.distance_transform_edt(binary)

    # 2. 修改策略：仅对凹进去很深的沟壑（Deep Grooves）进行分割
    # 使用 h-maxima 变换，h 参数控制分割的"深度"敏感度。
    # 距离变换的值近似于半径。h=4 意味着半径收缩至少4个像素的沟壑才会触发分割。
    h_depth = 4 # 深度阈值，仅分割非常明显的粘连
    peak_mask = h_maxima(distance, h=h_depth)
    
    # --- 兜底逻辑：防止小物体丢失 ---
    # h-maxima 可能会过滤掉高度不足 h 的独立小物体（半径小且没有深沟壑），必须找回它们
    num_labels, labels_im = cv2.connectedComponents(binary.astype(np.uint8))
    
    # 检查哪些连通域已经有了种子点
    # peak_mask 是 bool 矩阵
    has_seed = np.unique(labels_im[peak_mask])
    
    # 找到没有种子的连通域 (排除背景 0)
    all_labels = np.arange(1, num_labels)
    missing_labels = np.setdiff1d(all_labels, has_seed)
    
    if len(missing_labels) > 0:
        for lab_id in missing_labels:
            # 找到该连通域(ROI)的位置
            mask_locs = np.where(labels_im == lab_id)
            if len(mask_locs[0]) == 0: continue
            
            # 在该区域内找距离变换最大的点作为种子
            vals = distance[mask_locs]
            max_idx = np.argmax(vals)
            
            # 还原坐标
            py = mask_locs[0][max_idx]
            px = mask_locs[1][max_idx]
            peak_mask[py, px] = True

    # 3. 生成种子标记 (Markers)
    # peak_mask现在已经包含了h-maxima找到的显著峰值 + 补回的小物体中心
    markers, _ = ndimage.label(peak_mask)

    # 4. 执行分水岭算法
    # 使用 -distance 作为地形图，使峰值成为盆地底部
    # mask=binary 保证分水岭只在前景区域生长，不溢出到背景
    labels_ws = watershed(-distance, markers, mask=binary)

    # --- 后处理与数据提取 ---
    
    instances_data = []
    
    # 获取所有非0的标签
    unique_labels = np.unique(labels_ws)
    unique_labels = unique_labels[unique_labels != 0]
    
    # 用于保存可视化的结果 (Label ID Map)
    final_labeled_mask = np.zeros_like(labels_ws, dtype=np.int32)
    current_export_id = 1
    
    for label_id in unique_labels:
        # 提取单个实例的掩码
        instance_mask = (labels_ws == label_id).astype(np.uint8)
        area = np.sum(instance_mask)
        
        # 过滤过小的凸包/噪点
        if area < min_instance_area:
            continue
            
        # 计算质心 (Centroid)
        M = cv2.moments(instance_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            
        # 记录到最终mask图层
        final_labeled_mask[instance_mask > 0] = current_export_id
        
        # --- Crop 切割 ---
        bbox = cv2.boundingRect(instance_mask) # x, y, w, h
        x, y, w, h = bbox
        
        crop_filename = f"{name_no_ext}_id{current_export_id}.png"
        crop_path = os.path.join(crop_output_dir, crop_filename)

        yolo_result = None
        
        if original_img is not None:
            # 切割原图
            crop_img = original_img[y:y+h, x:x+w].copy()
            # 切割Mask
            crop_mask = instance_mask[y:y+h, x:x+w]
            
            # 使用Mask将背景置黑 (可选，如果只要矩形框里的内容可以注释掉下面三行)
            # 创建3通道Mask
            crop_mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])
            # 应用Mask (只保留Mask区域内的像素，其余为黑)
            crop_img = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)
            
            # 保存Crop
            cv2.imwrite(crop_path, crop_img)

            # --- YOLO 推理 ---
            if model is not None:
                # 运行推理
                # verbose=False 不要打印每张图的结果
                results = model(crop_img, verbose=False) 
                
                # 获取Top-1 结果 (假设是多分类)
                if len(results) > 0:
                    probs = results[0].probs # Classification probabilities
                    if probs is not None:
                        top1_index = int(probs.top1)
                        top1_conf = float(probs.top1conf)
                        # 获取类别名称
                        class_name = results[0].names[top1_index]
                        
                        yolo_result = {
                            "class_id": top1_index,
                            "class_name": class_name,
                            "confidence": top1_conf
                        }
        
        # 收集元数据
        record = {
            "id": int(current_export_id),
            "original_filename": filename,
            "crop_filename": crop_filename,
            "centroid_xy": [int(cX), int(cY)],  # 用于放回原图
            "area": int(area),
            "bbox_xywh": bbox 
        }
        
        if yolo_result:
            record["yolo_prediction"] = yolo_result
            
        instances_data.append(record)
        
        current_export_id += 1

    # 保存新的实例分割图 (uint16以支持超过255个实例)
    # 图片名保持一致，方便对应
    output_filename = os.path.join(output_dir, filename)
    cv2.imwrite(output_filename, final_labeled_mask.astype(np.uint16))

    return instances_data

def main():
    # 配置路径
    pred_path = "/home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset109_Perovskite/testset_comparison/nnUNetTrainerUMambaBotEdgeAttention_1__nnUNetPlans__2d_predictions"
    
    # 修正后的原图路径
    original_img_base = "/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset109_Perovskite/imagesTs"
    
    # YOLO 模型路径
    yolo_weights = "/home/chen/runs/classify/train20/weights/best.pt"
    
    # 输出路径
    output_base_dir = "processed_instances_output"
    output_masks_dir = os.path.join(output_base_dir, "instance_masks")
    output_crops_dir = os.path.join(output_base_dir, "instance_crops")
    
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_crops_dir, exist_ok=True)

    # 加载YOLO模型
    print(f"正在加载 YOLO 模型: {yolo_weights} ...")
    try:
        model = YOLO(yolo_weights)
        print("模型加载成功!")
    except Exception as e:
        print(f"Error: 加载模型失败 - {e}")
        model = None
    
    # 获取所有PNG文件
    mask_files = glob.glob(os.path.join(pred_path, "*.png"))
    all_metadata = []
    
    print(f"开始处理 {len(mask_files)} 张掩码...")
    print(f"原图路径: {original_img_base}")
    
    for mask_file in mask_files:
        results = process_single_mask(
            mask_file, 
            output_masks_dir,
            original_img_base,
            output_crops_dir,
            model,
            min_instance_area=50, # 忽略小于50像素的碎片
            min_distance=10       # 两个晶粒中心的最小距离
        )
        all_metadata.extend(results)
        print(f"已处理: {os.path.basename(mask_file)} - 提取实例数: {len(results)}")
            
    # 将所有坐标信息的元数据保存为JSON
    json_path = os.path.join(output_base_dir, "instance_centroids.json")
    with open(json_path, 'w') as f:
        json.dump(all_metadata, f, indent=4)
        
    print(f"\n处理完成!")
    print(f"分割后的Mask保存在: {output_masks_dir}")
    print(f"切割出的实例(Crops)保存在: {output_crops_dir}")
    print(f"质心坐标及ID保存在: {json_path}")

if __name__ == "__main__":
    main()
