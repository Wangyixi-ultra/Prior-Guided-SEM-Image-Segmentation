import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import glob

def update_dataset_json(dataset_path):
    json_path = os.path.join(dataset_path, 'dataset.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if channel 1 already exists
    if "1" in data['channel_names'] and data['channel_names']["1"] == "YOLO_Class":
        print("dataset.json already updated.")
    else:
        # Add new channel
        data['channel_names']["1"] = "YOLO_Class"
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print("Updated dataset.json with new channel.")

def process_images(image_dir, model):
    # Find all channel 0 images
    img_files = glob.glob(os.path.join(image_dir, "*_0000.png"))
    
    # YOLO Classes: {0: 'ABO3', 1: 'PbI2'}
    # Dataset Labels: {0: 'background', 1: 'PbI2', 2: 'ABO3'}
    # Mapping YOLO class index to semantic value (optional, but helpful to match semantics)
    # Let's use: YOLO 0 (ABO3) -> 2
    #            YOLO 1 (PbI2) -> 1
    
    yolo_to_mask_map = {0: 2, 1: 1}

    print(f"Processing {len(img_files)} images in {image_dir}...")
    
    for img_path in tqdm(img_files):
        # Predict
        # Load image for YOLO (it handles paths directly)
        results = model(img_path, verbose=False)
        result = results[0]

        # Read original image to get shape
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        
        # Create new channel image (initialized to 0 - background)
        new_channel = np.zeros((h, w), dtype=np.uint8)
        
        # Process detections
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Clip to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                fill_value = yolo_to_mask_map.get(cls_id, 0)
                
                if fill_value != 0:
                    # Fill the box area with the class value
                    cv2.rectangle(new_channel, (x1, y1), (x2, y2), fill_value, -1)
        
        # Save as _0001.png
        out_path = img_path.replace("_0000.png", "_0001.png")
        cv2.imwrite(out_path, new_channel)

def main():
    # Configuration
    dataset_root = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset110_Perovskite'
    yolo_weights = '/home/chen/seg6/yolo_cls_optimization/train21_l_aug5/weights/best.pt'
    
    print(f"Loading YOLO model from {yolo_weights}")
    model = YOLO(yolo_weights)
    
    # Log classes
    print(f"Model classes: {model.names}")
    
    # 1. Update dataset.json
    update_dataset_json(dataset_root)
    
    # 2. Process Training Images
    train_dir = os.path.join(dataset_root, 'imagesTr')
    if os.path.exists(train_dir):
        process_images(train_dir, model)
    
    # 3. Process Test Images (if exist)
    test_dir = os.path.join(dataset_root, 'imagesTs')
    if os.path.exists(test_dir):
        process_images(test_dir, model)

if __name__ == "__main__":
    main()
