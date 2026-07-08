import json
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def calculate_dice(pred, gt, class_id):
    p = (pred == class_id).astype(np.float32)
    g = (gt == class_id).astype(np.float32)
    
    intersection = np.sum(p * g)
    union = np.sum(p) + np.sum(g)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0 # Handle empty case
    
    return 2.0 * intersection / union

def main():
    # Paths
    centroids_path = '/home/chen/seg6/processed_instances_output/instance_centroids.json'
    masks_dir = '/home/chen/seg6/processed_instances_output/instance_masks'
    gt_dir = '/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset107_Perovskite/labelsTs'
    output_dir = '/home/chen/seg6/processed_instances_output/yolo_mask_visualization'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO results
    print(f"Loading {centroids_path}...")
    with open(centroids_path, 'r') as f:
        instances = json.load(f)
        
    # Group instances by filename
    instances_by_file = {}
    for inst in instances:
        fname = inst['original_filename']
        if fname not in instances_by_file:
            instances_by_file[fname] = []
        instances_by_file[fname].append(inst)
        
    # Class mapping
    # YOLO prediction strings
    # "PbI\u2082" -> 1
    # "ABO\u2083" -> 2
    
    # Check what specific strings are in the json
    # Because u2082 is subscript 2, u2083 is subscript 3
    CLASS_MAP = {
        "PbI\u2082": 1,
        "ABO\u2083": 2,
        "PbI2": 1, 
        "ABO3": 2
    }
    
    dice_scores = {1: [], 2: []}
    
    # Process each mask file found
    mask_files = sorted(glob(os.path.join(masks_dir, '*.png')))
    
    print(f"Found {len(mask_files)} mask files to process.")
    
    for mask_path in tqdm(mask_files):
        filename = os.path.basename(mask_path)
        
        # Load instance mask (contains IDs like 1, 2, 3...)
        inst_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if inst_mask is None:
            print(f"Could not read {mask_path}")
            continue
            
        # Create empty semantic mask
        pred_semantic = np.zeros_like(inst_mask, dtype=np.uint8)
        
        # Get instances for this file
        file_instances = instances_by_file.get(filename, [])
        
        # Fill semantic mask
        for inst in file_instances:
            inst_id = inst['id']
            # Sometimes instance IDs in mask might not match exactly if not handled carefully, 
            # but usually they correspond to connected components analysis.
            # Let's assume inst_id maps to pixel value in inst_mask.
            
            yolo_class_name = inst['yolo_prediction']['class_name']
            class_id = CLASS_MAP.get(yolo_class_name)
            
            if class_id is not None:
                # Mask out this instance and assign class_id
                pred_semantic[inst_mask == inst_id] = class_id
            else:
                print(f"Warning: Unknown class {yolo_class_name} in {filename} instance {inst_id}")
                
        # Load GT
        gt_path = os.path.join(gt_dir, filename)
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            
            # Ensure shapes match
            if pred_semantic.shape != gt_mask.shape:
               print(f"Shape mismatch for {filename}: Pred {pred_semantic.shape} vs GT {gt_mask.shape}")
               continue
            
            # Calculate Dice
            d1 = calculate_dice(pred_semantic, gt_mask, 1)
            d2 = calculate_dice(pred_semantic, gt_mask, 2)
            
            dice_scores[1].append(d1)
            dice_scores[2].append(d2)
            
            # Visualization
            # Create RGB image
            # Background: Black
            # PbI2 (1): Red
            # ABO3 (2): Green
            # GT overlay or side-by-side? Side by side is easier to read.
            
            vis_img = np.zeros((pred_semantic.shape[0], pred_semantic.shape[1], 3), dtype=np.uint8)
            vis_img[pred_semantic == 1] = [0, 0, 255] # Red (BGR)
            vis_img[pred_semantic == 2] = [0, 255, 0] # Green
            
            gt_vis = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
            gt_vis[gt_mask == 1] = [0, 0, 255]
            gt_vis[gt_mask == 2] = [0, 255, 0]
            
            # Combine side-by-side: Pred | GT
            combined = np.hstack([vis_img, gt_vis])
            
            # Add text
            cv2.putText(combined, f"Pred (Dice PbI2: {d1:.2f}, ABO3: {d2:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Ground Truth", (vis_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite(os.path.join(output_dir, filename), combined)
            
        else:
            print(f"Ground truth not found for {filename}")
            
    # Report Average Dice
    print("-" * 30)
    print("Evaluation Results:")
    if len(dice_scores[1]) > 0:
        avg_d1 = np.mean(dice_scores[1])
        print(f"Average Dice PbI2 (Class 1): {avg_d1:.4f}")
    
    if len(dice_scores[2]) > 0:
        avg_d2 = np.mean(dice_scores[2])
        print(f"Average Dice ABO3 (Class 2): {avg_d2:.4f}")
        
if __name__ == "__main__":
    main()
