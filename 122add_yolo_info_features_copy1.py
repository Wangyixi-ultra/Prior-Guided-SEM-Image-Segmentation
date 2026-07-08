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
    
    # Gaussian Config
    sigma = 4.0 # Controls the "spread" or softness
    patch_radius = int(3 * sigma) # 3-sigma rule covers 99% of area
    
    # Pre-compute Gaussian Kernels for each class to speed up
    kernels = {}
    patch_size = 2 * patch_radius + 1
    y_grid, x_grid = np.ogrid[-patch_radius:patch_radius+1, -patch_radius:patch_radius+1]
    dist_sq = x_grid**2 + y_grid**2
    base_gaussian = np.exp(-dist_sq / (2 * sigma**2))
    
    for cls_id, peak_val in class_intensity.items():
        kernels[cls_id] = (base_gaussian * peak_val).astype(np.float32)

    for img_path in tqdm(img_files):
        # Read original image to get dimensions
        img_cv = cv2.imread(img_path) 
        if img_cv is None:
            continue
        h, w = img_cv.shape[:2]
        
        # Run inference (Detection)
        results = model.predict(img_path, conf=0.25, iou=0.45, verbose=False)
        
        # Create blank mask (Float for accurate accumulation/max)
        mask = np.zeros((h, w), dtype=np.float32)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                # Get Class ID
                cls_id = int(box.cls[0].item())
                
                # Get Coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                
                # Calculate Centroid
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Get Pre-computed kernel
                # Default to 0 intensity kernel if class unknown (shouldn't happen)
                kernel = kernels.get(cls_id, None)
                if kernel is None: continue
                
                # Determine ROI bounds
                x_start = cx - patch_radius
                y_start = cy - patch_radius
                x_end = cx + patch_radius + 1
                y_end = cy + patch_radius + 1
                
                # Clip to image boundaries
                pad_left = max(0, -x_start)
                pad_top = max(0, -y_start)
                pad_right = max(0, x_end - w)
                pad_bottom = max(0, y_end - h)
                
                # Slices for Image
                img_x1 = x_start + pad_left
                img_y1 = y_start + pad_top
                img_x2 = x_end - pad_right
                img_y2 = y_end - pad_bottom
                
                # Slices for Kernel
                kern_x1 = pad_left
                kern_y1 = pad_top
                kern_x2 = patch_size - pad_right
                kern_y2 = patch_size - pad_bottom
                
                # Apply Max (Gaussian "Soft" Stamping)
                if img_x2 > img_x1 and img_y2 > img_y1:
                    patch = kernel[kern_y1:kern_y2, kern_x1:kern_x2]
                    
                    # Update mask with maximum intensity at this location
                    mask[img_y1:img_y2, img_x1:img_x2] = np.maximum(
                        mask[img_y1:img_y2, img_x1:img_x2], 
                        patch
                    )
        
        # Convert to uint8 for saving
        mask_uint8 = np.clip(mask, 0, 255).astype(np.uint8)
        
        # Save as channel 1 (_0001.png)
        out_path = img_path.replace("_0000.png", "_0001.png")
        cv2.imwrite(out_path, mask_uint8)

def main():
    # Configuration
    dataset_root = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset122_Perovskite'
    yolo_weights = '/home/chen/seg6/yolo_cls_optimization/train21_l_aug5/weights/best.pt'
    # yolo_weights = '/home/chen/seg6/perovskite_grains_opt/yolo_cbam_s_128/weights/best.pt'
    
    print(f"Loading YOLO model from {yolo_weights}")
    model = YOLO(yolo_weights)
    
    # 1. Update dataset.json
    update_dataset_json(dataset_root)
    
    # 2. Process Training Images
    # print("Skipping training images to preserve dataset integrity...")
    # process_images(os.path.join(dataset_root, 'imagesTr'), model)
    
    # 3. Process Test Images
    print("Processing Test Images only...")
    process_images(os.path.join(dataset_root, 'imagesTs'), model)

if __name__ == "__main__":
    main()
