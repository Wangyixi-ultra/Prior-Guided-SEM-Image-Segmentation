import os
import json
import numpy as np
import cv2
import glob
from ultralytics import YOLO
from tqdm import tqdm

def update_dataset_json(dataset_path):
    json_path = os.path.join(dataset_path, 'dataset.json')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Update channel name to reflect the box mask nature
    channel_name = "YOLO_Gaussian_Class_Mask"
    
    if "1" in data['channel_names'] and data['channel_names']["1"] == channel_name:
        print(f"dataset.json already updated for {channel_name}.")
    else:
        data['channel_names']["1"] = channel_name
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Updated dataset.json with new channel name '{channel_name}'.")

def process_images(image_dir, model):
    if not os.path.exists(image_dir):
        print(f"Skipping {image_dir} (does not exist)")
        return

    img_files = glob.glob(os.path.join(image_dir, "*_0000.png"))
    print(f"Processing {len(img_files)} images in {image_dir}...")
    
    # 动态获取 YOLO 模型的类别映射
    names = model.names
    pbi2_id = None
    abo3_id = None
    
    # 查找 PbI2 和 ABO3 对应的 ID (不区分大小写，处理 unicode 下标)
    for cls_id, cls_name in names.items():
        name_lower = cls_name.lower()
        if 'pbi' in name_lower:
            pbi2_id = cls_id
        elif 'abo' in name_lower:
            abo3_id = cls_id
            
    if pbi2_id is None or abo3_id is None:
        print(f"Wait! Model classes {names} do not contain expected 'PbI2' or 'ABO3'. Using defaults 0/1.")
        # Fallback (unsafe but better than crash)
        pbi2_id = 0
        abo3_id = 1
    else:
        print(f"Info: Using dynamic class mapping from model: PbI2={pbi2_id}, ABO3={abo3_id}")

    # Class Intensity Mapping
    # Dynamic Mapping based on model
    class_intensity = {}
    if pbi2_id is not None:
        class_intensity[pbi2_id] = 100  # PbI2 always gets 100
    if abo3_id is not None:
        class_intensity[abo3_id] = 200  # ABO3 always gets 200
    
    # Strategy A: Small solid circles for clear visibility
    dot_radius = 3  # 3px radius = 7x7 pixel solid dot
    
    for img_path in tqdm(img_files):
        # Read original image to get dimensions
        img_cv = cv2.imread(img_path) 
        if img_cv is None:
            continue
        h, w = img_cv.shape[:2]
        
        # Run inference (Detection)
        results = model.predict(img_path, conf=0.25, iou=0.45, verbose=False)
        
        # Create blank mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        num_boxes = 0
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            num_boxes = len(boxes)
            
            for box in boxes:
                # Get Class ID
                cls_id = int(box.cls[0].item())
                
                # Get Coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                
                # Calculate Centroid
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Get intensity for this class
                intensity = class_intensity.get(cls_id, 50)
                
                # Draw solid circle (clearly visible)
                cv2.circle(mask, (cx, cy), dot_radius, intensity, -1)
        
        # Debug: print number of detections for first few images
        if img_path == img_files[0] or num_boxes > 0:
            print(f"  {os.path.basename(img_path)}: {num_boxes} detections")
        
        # Convert to uint8 for saving
        mask_uint8 = np.clip(mask, 0, 255).astype(np.uint8)
        
        # Save as channel 1 (_0001.png)
        out_path = img_path.replace("_0000.png", "_0001.png")
        cv2.imwrite(out_path, mask_uint8)

def main():
    # Configuration
    dataset_root = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset123_Perovskite'
    yolo_weights = '/home/chen/seg6/perovskite_grains_opt/yolo11x_grain_optimized2/weights/best.pt'

    
    print(f"Loading YOLO model from {yolo_weights}")
    model = YOLO(yolo_weights)
    
    # 1. Update dataset.json
    update_dataset_json(dataset_root)
    
    # 2. Process Training Images
    print("Skipping training images to preserve dataset integrity...")
    process_images(os.path.join(dataset_root, 'imagesTr'), model)
    
    # 3. Process Test Images
    print("Processing Test Images only...")
    process_images(os.path.join(dataset_root, 'imagesTs'), model)

if __name__ == "__main__":
    main()
