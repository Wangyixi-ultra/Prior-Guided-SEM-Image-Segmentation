import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from tqdm import tqdm
import glob

# Global list to store features from hooks
captured_features = []

def get_activation(name):
    def hook(model, input, output):
        # Detach and move to CPU immediately to save GPU memory if needed, 
        # but for resize usually keeping on device is better.
        # output is likely a Tensor.
        captured_features.append(output)
    return hook

def update_dataset_json(dataset_path):
    json_path = os.path.join(dataset_path, 'dataset.json')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if channel 1 already exists
    # We are changing the semantic of channel 1 from Class ID to Feature Map
    if "1" in data['channel_names'] and data['channel_names']["1"] == "YOLO_Feature":
        print("dataset.json already updated for YOLO_Feature.")
    else:
        # Update or Overwrite channel 1
        data['channel_names']["1"] = "YOLO_Feature"
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print("Updated dataset.json with new channel name 'YOLO_Feature'.")

def process_images(image_dir, model):
    # Find all channel 0 images
    img_files = glob.glob(os.path.join(image_dir, "*_0000.png"))
    
    print(f"Processing {len(img_files)} images in {image_dir}...")
    
    # Attach hooks to layers 16, 19, 22 (The outputs of the Neck, inputs to Detect Head)
    # Based on architecture inspection:
    # 16: P3 (8x downsample)
    # 19: P4 (16x downsample)
    # 22: P5 (32x downsample)
    
    # Note: access internal pytorch model via model.model.model
    # We clear handles later just in case, but for script run it's fine.
    
    # Ideally we'd remove hooks if we ran this multiple times, but once on startup is fine.
    model.model.model[16].register_forward_hook(get_activation('layer16'))
    model.model.model[19].register_forward_hook(get_activation('layer19'))
    model.model.model[22].register_forward_hook(get_activation('layer22'))
    
    for img_path in tqdm(img_files):
        # Clear previous features
        captured_features.clear()
        
        # Original Image to get shape
        img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            print(f"Could not read {img_path}")
            continue
            
        h, w = img_cv.shape[:2]
        
        # Run inference
        # visualize=False, embed=None. 
        # Passing numpy or path triggers prediction.
        # We assume the model puts it on the correct device.
        results = model(img_path, verbose=False)
        
        # Now captured_features should contain [feat16, feat19, feat22]
        # Verify
        if not captured_features:
            print(f"Warning: No features captured for {img_path}")
            continue
            
        # Resize all features to image size (h, w) and aggregate
        # Strategy: Averages of mean-channel activation
        
        accumulated_map = None
        
        for feat in captured_features:
            # feat shape: [Batch, Channel, H_feat, W_feat]
            # Since we pass 1 image, Batch=1.
            
            # 1. Collapse channels: Mean activation
            # [1, C, Hf, Wf] -> [1, 1, Hf, Wf]
            mean_activation = torch.mean(feat, dim=1, keepdim=True)
            
            # 2. Resize to Original Image Size
            # bilinear interpolation suitable for features
            upsampled = F.interpolate(mean_activation, size=(h, w), mode='bilinear', align_corners=False)
            
            # 3. Accumulate
            if accumulated_map is None:
                accumulated_map = upsampled
            else:
                accumulated_map += upsampled
                
        # Average over the number of layers
        final_map = accumulated_map / len(captured_features)
        
        # Convert to numpy [H, W]
        final_map_np = final_map.squeeze().cpu().detach().numpy()
        
        # Normalize to 0-255 for PNG saving
        # Min-Max Normalization per image to maximize contrast?
        # Or fixed scale? Features are unbounded. Min-Max is safer for visualization/usage as 8-bit.
        
        f_min, f_max = final_map_np.min(), final_map_np.max()
        if f_max - f_min > 1e-6:
            norm_map = (final_map_np - f_min) / (f_max - f_min) * 255.0
        else:
            norm_map = np.zeros_like(final_map_np)
            
        norm_map = norm_map.astype(np.uint8)
        
        # Save as _0001.png
        out_path = img_path.replace("_0000.png", "_0001.png")
        cv2.imwrite(out_path, norm_map)

def main():
    # Configuration
    dataset_root = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset110_Perovskite copy'
    yolo_weights = '/home/chen/seg6/yolo_cls_optimization/train21_l_aug5/weights/best.pt'
    
    print(f"Loading YOLO model from {yolo_weights}")
    model = YOLO(yolo_weights)
    
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
