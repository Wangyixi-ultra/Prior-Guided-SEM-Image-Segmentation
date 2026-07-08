#!/usr/bin/env python3
"""
Compare all trainers on the Dataset123_Perovskite test set.
Automatically discovers all trainer models, runs predictions on the test set, and computes metrics.
"""

import os
import sys
sys.path.insert(0, "/home/chen/seg6/U-Mamba/umamba")

os.environ['DISABLE_FOURIER_INFERENCE'] = 'True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
# Fix NumExpr thread limit issue
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import cv2
import numpy as np
import traceback
import re
import glob
import shutil
from ultralytics import YOLO
import ultralytics.nn.tasks
from ultralytics.nn.modules import CBAM

ultralytics.nn.tasks.CBAM = CBAM

# nnUNet-related modules
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference import data_iterators
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from typing import List, Union

# Monkey patch: avoid multiprocessing issues
from nnunetv2.inference import predict_from_raw_data

def preprocessing_iterator_fromfiles_synchronous(
    list_of_lists: List[List[str]],
    list_of_segs_from_prev_stage_files: Union[None, List[str]],
    output_filenames_truncated: Union[None, List[str]],
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_processes: int,
    pin_memory: bool = False,
    verbose: bool = False
):
    label_manager = plans_manager.get_label_manager(dataset_json)
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    
    list_of_segs = list_of_segs_from_prev_stage_files or [None] * len(list_of_lists)
    output_files = output_filenames_truncated or [None] * len(list_of_lists)
    
    for data_files, seg_prev, ofile in zip(list_of_lists, list_of_segs, output_files):
        data, seg, data_properties = preprocessor.run_case(
            data_files, seg_prev, plans_manager, configuration_manager, dataset_json
        )
        if seg_prev is not None:
            seg_onehot = convert_labelmap_to_one_hot(
                seg[0], label_manager.foreground_labels, data.dtype
            )
            data = np.vstack((data, seg_onehot))
        
        data = torch.from_numpy(data).contiguous().float()
        item = {'data': data, 'data_properties': data_properties, 'ofile': ofile}
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item

data_iterators.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous
predict_from_raw_data.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous

# ========== Unified evaluation metric configuration ==========
val_evaluator = dict(
    type='CrackIoUMetric', 
    iou_metrics=['mIoU', 'mDice'],
    # Adhesion penalty parameters (soft mode)
    penalty_scale=0.18,      # penalty scaling factor
    max_penalty=0.30,        # maximum penalty value (30%)
    min_component_size=10    # minimum connected component size
)

# ========== Configuration ==========
DATASET_NAME = "Dataset123_Perovskite"
NNUNET_RAW = "/home/chen/seg6/U-Mamba/data/nnUNet_raw"
NNUNET_RESULTS = "/home/chen/seg6/U-Mamba/data/nnUNet_results"
RAW_TEST_IMAGES = f"{NNUNET_RAW}/{DATASET_NAME}/imagesTs"
TEST_LABELS = f"{NNUNET_RAW}/{DATASET_NAME}/labelsTs"
OUTPUT_DIR = f"{NNUNET_RESULTS}/{DATASET_NAME}/testset_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOLD = 0
CHECKPOINT = "checkpoint_final.pth"
ALLOW_AGGRESSIVE_OOM_FALLBACK = False

CLASS_COLORS = {1: (0, 255, 0), 2: (0, 165, 255)}
CLASS_NAMES = {1: "PbI2", 2: "ABO3"}
VALID_CLASS_IDS = [1, 2]

YOLO_DETECTOR = '/home/chen/runs/detect/perovskite_grains_opt/yolo11x_cbam_detect_v2/weights/best.pt'

LABEL_MAP = {'background': 0, 'PbI2': 1, 'ABO3': 2}

# ========== Utility functions ==========
def get_model_info(predictor):
    """Get model parameter count and computational cost."""
    try:
        network = predictor.network
        
        # Calculate parameter count
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        params_m = total_params / 1e6  # convert to millions
        
        # Calculate FLOPs (estimated from input size)
        try:
            cm = predictor.configuration_manager
            if hasattr(cm, 'patch_size'):
                patch_size = cm.patch_size
            elif hasattr(cm, 'configuration') and 'patch_size' in cm.configuration:
                patch_size = cm.configuration['patch_size']
            else:
                # Infer default size from network structure
                conv_op = getattr(network, 'conv_op', torch.nn.Conv2d)
                is_2d = conv_op == torch.nn.Conv2d
                patch_size = [256, 256] if is_2d else [128, 128, 128]
            
            # Get number of input channels
            num_channels = get_input_channels(predictor)
            
            # Prefer thop library for FLOPs (more accurate)
            input_shape = [num_channels] + list(patch_size)
            thop_result = get_model_info_thop(network, input_shape)
            
            if thop_result is not None:
                return thop_result
            
            # Use simplified FLOPs estimation
            gflops = estimate_gflops_simple(network, num_channels, patch_size)
        except Exception as e:
            print(f"  Warning: FLOPs calculation failed - {e}")
            gflops = 0.0
        
        return {
            'params_m': round(params_m, 2),
            'gflops': round(gflops, 2),
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    except Exception as e:
        print(f"  Warning: unable to get model info - {e}")
        return {'params_m': 0.0, 'gflops': 0.0, 'total_params': 0, 'trainable_params': 0}


def estimate_gflops_simple(network, in_channels, input_size):
    """
    Simplified GFLOPs estimation - no hooks, computed directly from layer parameters.
    """
    total_flops = 0
    
    # Helper function for convolution FLOPs
    def calc_conv_flops(module, input_spatial_size):
        if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d)):
            k = module.kernel_size[0] if isinstance(module.kernel_size, (tuple, list)) else module.kernel_size
            c_in = module.in_channels
            c_out = module.out_channels
            groups = module.groups
            
            # Calculate output spatial size
            if len(input_spatial_size) == 3:  # 3D
                h_out, w_out, d_out = input_spatial_size
                kernel_ops = k ** 3 * c_in * c_out // groups
                output_ops = h_out * w_out * d_out
            else:  # 2D
                h_out, w_out = input_spatial_size[:2]
                kernel_ops = k ** 2 * c_in * c_out // groups
                output_ops = h_out * w_out
            
            flops = kernel_ops * output_ops * 2  # multiply-add
            return flops, [h_out, w_out] if len(input_spatial_size) == 2 else [h_out, w_out, d_out]
        elif isinstance(module, torch.nn.Linear):
            flops = module.in_features * module.out_features * 2
            return flops, input_spatial_size
        elif isinstance(module, torch.nn.BatchNorm3d):
            # BN: 2 * C * H * W * D (multiplication and addition)
            flops = 2 * module.num_features * input_spatial_size[0] * input_spatial_size[1] * input_spatial_size[2]
            return flops, input_spatial_size
        elif isinstance(module, torch.nn.BatchNorm2d):
            flops = 2 * module.num_features * input_spatial_size[0] * input_spatial_size[1]
            return flops, input_spatial_size
        return 0, input_spatial_size
    
    # Estimate size changes during forward propagation
    # Use encoder depth to estimate downsampling steps
    current_size = list(input_size)
    is_3d = len(input_size) == 3
    
    # Iterate over all modules and estimate FLOPs in order
    for name, module in network.named_modules():
        # Skip container modules
        if len(list(module.children())) > 0:
            continue
        
        flops, new_size = calc_conv_flops(module, current_size)
        total_flops += flops
        
        # If downsampling layer (conv with stride > 1), update size
        if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d)) and hasattr(module, 'stride'):
            stride = module.stride[0] if isinstance(module.stride, (tuple, list)) else module.stride
            if stride > 1:
                current_size = [max(s // stride, 1) for s in current_size]
    
    # If no FLOPs were computed, roughly estimate from parameter count
    if total_flops == 0:
        total_params = sum(p.numel() for p in network.parameters())
        # Assume each parameter participates in ~100 operations on average
        total_flops = total_params * 100
    
    return total_flops / 1e9  # convert to GFLOPs


def get_model_info_thop(network, input_shape):
    """
    Use the thop library to compute model parameters and FLOPs (more accurate).
    Deep-copy the model to isolate thop's side effects and avoid polluting the original model.
    """
    try:
        import copy
        from thop import profile, clever_format
        
        # Deep-copy model to avoid thop hooks polluting the original model
        network_copy = copy.deepcopy(network)
        network_copy.eval()
        
        # Create dummy input
        device = next(network_copy.parameters()).device
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # Calculate FLOPs and parameters
        flops, params = profile(network_copy, inputs=(dummy_input,), verbose=False)
        
        # Delete copied model to free memory
        del network_copy
        torch.cuda.empty_cache()
        
        return {
            'params_m': params / 1e6,
            'gflops': flops / 1e9,
            'total_params': int(params),
            'trainable_params': sum(p.numel() for p in network.parameters() if p.requires_grad)
        }
    except Exception as e:
        print(f"  thop calculation failed: {e}, using simplified estimation")
        return None

def get_input_channels(predictor):
    """Get the number of model input channels."""
    network = predictor.network
    if hasattr(network, 'encoder') and hasattr(network.encoder, 'stem'):
        return network.encoder.stem[0].conv1.in_channels
    if hasattr(network, 'input_channels'):
        return network.input_channels
    params = list(network.parameters())
    return params[0].shape[1] if params else 1

def shrink_patch_size_for_oom(predictor, scale=0.8, min_size=64):
    """Reduce patch_size when OOM occurs."""
    cm = predictor.configuration_manager
    if not hasattr(cm, 'num_pool_per_axis') or not hasattr(cm, 'patch_size'):
        return False
    
    current = list(cm.patch_size)
    divs = [2 ** n for n in cm.num_pool_per_axis]
    new, changed = [], False
    
    for size, d in zip(current, divs):
        target = max((int(size * scale) // d) * d, d)
        target = max(target, (min_size // d) * d if min_size // d > 0 else d)
        if target >= size:
            target = size - d if size - d >= d else size
        if target != size:
            changed = True
        new.append(int(target))
    
    if not changed:
        return False
    
    cm.configuration['patch_size'] = new
    try:
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        compute_gaussian.cache_clear()
    except:
        pass
    print(f"  ✅ Reduced patch_size: {current} -> {new}")
    return True

def has_cuda_only_ops(network):
    """Check whether the network contains CUDA-only operators."""
    for m in network.modules():
        name, mod = type(m).__name__.lower(), type(m).__module__.lower()
        if 'mamba' in name or 'mamba' in mod or 'causal_conv1d' in mod:
            return True
    return False

def normalize_name(name):
    name = name.lower()
    return 'PbI2' if 'pbi' in name else 'ABO3' if 'abo' in name else name

# ========== Core functionality ==========
def get_all_trainers():
    """Get all trainer models."""
    dataset_dir = Path(f"{NNUNET_RESULTS}/{DATASET_NAME}")
    if not dataset_dir.exists():
        return []
    
    checkpoint_priority = ["checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"]
    fold_priority = [f"fold_{FOLD}", "fold_all"]
    trainers = []
    
    for item in dataset_dir.iterdir():
        if not (item.is_dir() and item.name.startswith("nnUNetTrainer")):
            continue
        
        best_fold, best_checkpoint, best_score = None, None, -1
        for fold_idx, fold_name in enumerate(fold_priority):
            fold_dir = item / fold_name
            if not fold_dir.exists():
                continue
            for ckpt_idx, ckpt_name in enumerate(checkpoint_priority):
                if (fold_dir / ckpt_name).exists():
                    score = fold_idx * 10 + ckpt_idx
                    if score > best_score:
                        best_score = score
                        best_fold, best_checkpoint = fold_dir, fold_dir / ckpt_name
        
        if best_fold and best_checkpoint:
            trainers.append({
                "name": item.name,
                "path": str(item),
                "fold_dir": str(best_fold),
                "checkpoint": str(best_checkpoint),
                "fold_name": best_fold.name,
                "checkpoint_name": best_checkpoint.name
            })
    
    return sorted(trainers, key=lambda x: x["name"])


def create_predictor(trainer_info):
    """Create a predictor with OOM fallback."""
    for perform_on_device in [True, False]:
        try:
            return nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=perform_on_device,
                device=torch.device('cuda'),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower() or perform_on_device is False:
                raise e
            print("  ⚠️ Insufficient GPU memory, switching to CPU/GPU hybrid mode...")
    return None

def build_input_list(input_folder, num_channels):
    """Build input file list."""
    files_0000 = sorted([f for f in os.listdir(input_folder) if f.endswith('_0000.png')])
    case_ids = [f.replace('_0000.png', '') for f in files_0000]
    
    list_of_lists, missing = [], []
    for case_id in case_ids:
        case_files = []
        for i in range(num_channels):
            fpath = os.path.join(input_folder, f"{case_id}_{i:04d}.png")
            if not os.path.exists(fpath):
                missing.append(fpath)
            case_files.append(fpath)
        list_of_lists.append(case_files)
    
    return list_of_lists, missing

def apply_oom_strategy(predictor, attempt, max_retries):
    """Apply OOM recovery strategy."""
    strategies = [
        ("Switch to CPU/GPU hybrid mode", lambda: setattr(predictor, 'perform_everything_on_device', False)),
        ("Disable TTA and increase step size", lambda: (setattr(predictor, 'use_mirroring', False), 
                                       setattr(predictor, 'tile_step_size', 0.9))),
        ("Disable Gaussian weight blending", lambda: setattr(predictor, 'use_gaussian', False) 
            if ALLOW_AGGRESSIVE_OOM_FALLBACK else None),
        ("Reduce patch_size", lambda: shrink_patch_size_for_oom(predictor) 
            if ALLOW_AGGRESSIVE_OOM_FALLBACK else None),
        ("Clear cache", lambda: (torch.cuda.empty_cache(), __import__('gc').collect())),
    ]
    
    if attempt < len(strategies):
        name, action = strategies[attempt]
        if action() is not False:
            print(f"  Strategy {attempt+1}: {name}...")
            return True
    
    # Last resort: CPU mode
    if attempt >= max_retries - 2 and not has_cuda_only_ops(predictor.network):
        print("  Strategy: force switch to CPU mode...")
        predictor.perform_everything_on_device = False
        predictor.device = torch.device('cpu')
        predictor.network = predictor.network.to('cpu')
        torch.cuda.empty_cache()
        return True
    
    return attempt < max_retries - 1

def predict_with_trainer(trainer_info, output_folder, input_folder):
    """Run prediction with the specified trainer."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {trainer_info['name']}")
    print(f"  Fold: {trainer_info['fold_name']}, Checkpoint: {trainer_info['checkpoint_name']}")
    print(f"{'='*60}")
    
    model_info = None
    
    try:
        fold_str = trainer_info['fold_name']
        use_folds = ['all'] if fold_str == 'fold_all' else [int(re.match(r'fold_(\d+)', fold_str).group(1))]
        
        predictor = create_predictor(trainer_info)
        
        # Initialize model (weights not loaded yet)
        predictor.initialize_from_trained_model_folder(
            trainer_info["path"], use_folds=use_folds, checkpoint_name=trainer_info["checkpoint_name"]
        )
        
        # Check whether the network loaded successfully
        if hasattr(predictor, 'network') and predictor.network is not None:
            print(f"  ✅ Model loaded successfully")
        else:
            print(f"  ❌ Model loading failed")
            return False, None
        
        # Get model info (parameters and FLOPs)
        model_info = get_model_info(predictor)
        print(f"  Model parameters: {model_info['params_m']:.2f}M, GFLOPs: {model_info['gflops']:.2f}G")
        
        num_channels = get_input_channels(predictor)
        print(f"  Model expected input channels: {num_channels}")
        
        list_of_lists, missing = build_input_list(input_folder, num_channels)
        if not list_of_lists:
            print(f"  ❌ No test images found")
            return False
        if missing:
            print(f"  ❌ Missing {len(missing)} channel files")
            return False
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"  Starting prediction for {len(list_of_lists)} samples...")
        
        for attempt in range(6):
            try:
                predictor.predict_from_files(
                    list_of_lists, output_folder, save_probabilities=False, overwrite=True,
                    num_processes_preprocessing=1, num_processes_segmentation_export=1,
                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
                )
                print(f"  Prediction complete: {output_folder}")
                return True, model_info
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_str = str(e).lower()
                if 'out of memory' not in err_str and 'expected x.is_cuda()' not in err_str:
                    if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
                        print(f"  Warning: key mismatch when loading model weights, trying non-strict loading...")
                        print(f"  Error details: {str(e)[:500]}...")
                        # If key mismatch occurs, skip this prediction and continue to the next model
                        return False
                    raise e
                if 'expected x.is_cuda()' in err_str:
                    print("  ❌ Contains CUDA-only operators, cannot use CPU mode")
                    return False
                if not apply_oom_strategy(predictor, attempt, 6):
                    print("  ❌ All OOM strategies failed")
                    return False, None
    except Exception as e:
        if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
            print(f"  Warning: key mismatch when loading model weights, skipping this model...")
            print(f"  Error details: {str(e)[:500]}...")
            return False, None
        print(f"  ❌ Prediction failed: {e}")
        traceback.print_exc()
        return False, None

def visualize_prediction_mask(pred_mask, raw_image_path, output_path):
    """Visualize prediction result."""
    raw = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return False
    
    overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if len(raw.shape) == 2 else raw.copy()
    
    for class_id, color in CLASS_COLORS.items():
        mask = (pred_mask == class_id).astype(np.uint8)
        if np.any(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(overlay, contours, -1, color, 2)
    
    cv2.imwrite(output_path, overlay)
    return True

def visualize_errors(trainer_name, pred_folder, raw_images_dir):
    """Generate error visualizations."""
    vis_dir = f"{OUTPUT_DIR}/{trainer_name}_error_visualizations"
    pred_vis_dir = f"{OUTPUT_DIR}/{trainer_name}_prediction_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_vis_dir, exist_ok=True)
    
    for filename in os.listdir(TEST_LABELS):
        if not filename.endswith('.png'):
            continue
        
        case_name = filename[:-4]
        gt_path = os.path.join(TEST_LABELS, filename)
        pred_path = os.path.join(pred_folder, filename)
        raw_path = os.path.join(raw_images_dir, f"{case_name}_0000.png")
        
        if not (os.path.exists(pred_path) and os.path.exists(raw_path)):
            continue
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        overlay = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        
        if gt is None or pred is None or overlay is None:
            continue
        
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        errors = (gt != pred) & (gt > 0)
        overlay[errors] = [0, 0, 255]
        
        for class_id in VALID_CLASS_IDS:
            gt_mask = (gt == class_id).astype(np.uint8) * 255
            pred_mask = (pred == class_id).astype(np.uint8) * 255
            if np.any(gt_mask):
                contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            if np.any(pred_mask):
                contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)
        
        cv2.imwrite(os.path.join(vis_dir, f"{case_name}_error_map.png"), overlay)
        visualize_prediction_mask(pred, raw_path, os.path.join(pred_vis_dir, f"{case_name}_prediction_vis.png"))

class CrackIoUMetric:
    """
    Crack IoU evaluation metric.
    Unified evaluator supporting adhesion penalty and small-component filtering.
    """
    
    def __init__(self, iou_metrics=['mIoU', 'mDice'], penalty_scale=0.18, max_penalty=0.30, min_component_size=10):
        """
        Args:
            iou_metrics: list of metrics to compute, supports 'mIoU', 'mDice'
            penalty_scale: adhesion penalty scaling factor (default 0.18)
            max_penalty: maximum penalty value (default 0.30 = 30%)
            min_component_size: minimum connected component size; components smaller than this are ignored
        """
        self.iou_metrics = iou_metrics
        self.penalty_scale = penalty_scale
        self.max_penalty = max_penalty
        self.min_component_size = min_component_size
        self.results = {}
    
    def compute_iou(self, pred, target, class_id):
        """Compute IoU for a single class."""
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union
    
    def compute_dice(self, pred, target, class_id):
        """Compute Dice for a single class."""
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()
        
        if pred_sum + target_sum == 0:
            return 1.0 if intersection == 0 else 0.0
        return 2.0 * intersection / (pred_sum + target_sum)
    
    def filter_small_components(self, mask, class_id):
        """Filter connected components smaller than min_component_size."""
        class_mask = (mask == class_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # start from 1 to skip background
            if stats[i, cv2.CC_STAT_AREA] >= self.min_component_size:
                filtered_mask[labels == i] = class_id
        
        return filtered_mask
    
    def compute_adhesion_penalty(self, pred, target, class_id, penalty_scale=0.18, max_penalty=0.30):
        """
        Compute adhesion penalty score (soft mode).

        Logic:
        1. If Pred_components >= GT_components: Adhesion_Ratio = 0 (no adhesion)
        2. Otherwise: Adhesion_Ratio = (GT_components - Pred_components) / GT_components
        3. Adhesion_Penalty = min(Adhesion_Ratio * penalty_scale, max_penalty)

        Args:
            pred: predicted mask
            target: ground-truth mask
            class_id: class ID
            penalty_scale: penalty scaling factor (default 0.18)
            max_penalty: maximum penalty value (default 0.30)
        """
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        
        # Run connected component analysis on prediction and GT
        pred_num_labels, _ = cv2.connectedComponents(pred_mask, connectivity=8)[:2]
        target_num_labels, _ = cv2.connectedComponents(target_mask, connectivity=8)[:2]
        
        # Subtract background
        pred_components = pred_num_labels - 1
        target_components = target_num_labels - 1
        
        # If no GT components or no predicted components, no penalty
        if target_components == 0 or pred_components == 0:
            return 0.0
        
        # Compute adhesion ratio
        if pred_components >= target_components:
            adhesion_ratio = 0.0  # no adhesion
        else:
            adhesion_ratio = (target_components - pred_components) / target_components
        
        # Compute penalty
        adhesion_penalty = adhesion_ratio * penalty_scale
        adhesion_penalty = min(adhesion_penalty, max_penalty)
        
        return adhesion_penalty
    
    def compute_metrics_on_case(self, pred_path, target_path):
        """Compute metrics for a single case."""
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        if pred is None or target is None:
            return None
        
        # Ensure sizes match
        if pred.shape != target.shape:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        case_metrics = {}
        
        for class_id in VALID_CLASS_IDS:
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            
            # Apply small-component filtering
            if self.min_component_size > 0:
                filtered_pred = self.filter_small_components(pred, class_id)
            else:
                filtered_pred = pred
            
            # Basic metrics
            iou = self.compute_iou(filtered_pred, target, class_id)
            dice = self.compute_dice(filtered_pred, target, class_id)
            
            # Adhesion penalty (soft mode)
            adhesion_penalty = self.compute_adhesion_penalty(
                filtered_pred, target, class_id, 
                penalty_scale=self.penalty_scale, 
                max_penalty=self.max_penalty
            )
            
            # Apply penalty: Final = Original * (1 - Adhesion_Penalty)
            penalized_iou = iou * (1 - adhesion_penalty)
            penalized_dice = dice * (1 - adhesion_penalty)
            
            case_metrics[class_id] = {
                'IoU': iou,
                'Dice': dice,
                'AdhesionPenalty': adhesion_penalty,
                'PenalizedIoU': penalized_iou,
                'PenalizedDice': penalized_dice
            }
        
        return case_metrics
    
    def compute_metrics_on_folder(self, folder_pred, folder_ref):
        """
        Compute metrics for an entire folder.
        Compatible with nnUNet's compute_metrics_on_folder interface.
        """
        pred_files = sorted([f for f in os.listdir(folder_pred) if f.endswith('.png')])
        
        if not pred_files:
            return {}
        
        all_case_metrics = []
        
        for pred_file in pred_files:
            pred_path = os.path.join(folder_pred, pred_file)
            target_path = os.path.join(folder_ref, pred_file)
            
            if not os.path.exists(target_path):
                continue
            
            case_metrics = self.compute_metrics_on_case(pred_path, target_path)
            if case_metrics:
                all_case_metrics.append(case_metrics)
        
        if not all_case_metrics:
            return {}
        
        # Compute average metrics
        mean_metrics = {'mean': {}}
        
        for class_id in VALID_CLASS_IDS:
            class_metrics = [m[class_id] for m in all_case_metrics if class_id in m]
            
            if not class_metrics:
                continue
            
            mean_metrics['mean'][class_id] = {
                'IoU': np.mean([m['IoU'] for m in class_metrics]),
                'Dice': np.mean([m['Dice'] for m in class_metrics]),
                'AdhesionPenalty': np.mean([m['AdhesionPenalty'] for m in class_metrics]),
                'PenalizedIoU': np.mean([m['PenalizedIoU'] for m in class_metrics]),
                'PenalizedDice': np.mean([m['PenalizedDice'] for m in class_metrics])
            }
        
        self.results = mean_metrics
        return mean_metrics


def evaluate_predictions(pred_folder, trainer_info, model_info=None):
    """Evaluate predictions - using unified evaluation metric."""
    try:
        if not [f for f in os.listdir(pred_folder) if f.endswith('.png')]:
            return None
        
        # Use unified CrackIoUMetric evaluator (soft adhesion penalty mode)
        evaluator = CrackIoUMetric(
            iou_metrics=val_evaluator['iou_metrics'],
            penalty_scale=val_evaluator.get('penalty_scale', 0.18),
            max_penalty=val_evaluator.get('max_penalty', 0.30),
            min_component_size=val_evaluator['min_component_size']
        )
        
        metrics = evaluator.compute_metrics_on_folder(
            folder_pred=pred_folder, 
            folder_ref=TEST_LABELS
        )
        
        if not metrics or 'mean' not in metrics:
            print(f"  ⚠️ CrackIoUMetric evaluation failed, trying standard evaluation...")
            # Fall back to standard evaluation
            try:
                from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
                reader = SimpleITKIO()
            except ImportError:
                reader = None
            
            metrics = compute_metrics_on_folder(
                folder_ref=TEST_LABELS, folder_pred=pred_folder,
                output_file=f"{pred_folder}/summary.json", image_reader_writer=reader,
                file_ending='.png', regions_or_labels=VALID_CLASS_IDS,
                ignore_label=None, num_processes=3, chill=True
            )
        
        mean_metrics = metrics.get("mean", {})
        
        # Use penalized metrics as primary evaluation indicators
        dice_sum = sum(
            (mean_metrics.get(cid) or mean_metrics.get(str(cid), {})).get("PenalizedDice", 
                (mean_metrics.get(cid) or mean_metrics.get(str(cid), {})).get("Dice", 0))
            for cid in VALID_CLASS_IDS
        )
        iou_sum = sum(
            (mean_metrics.get(cid) or mean_metrics.get(str(cid), {})).get("PenalizedIoU",
                (mean_metrics.get(cid) or mean_metrics.get(str(cid), {})).get("IoU", 0))
            for cid in VALID_CLASS_IDS
        )
        
        avg_dice = dice_sum / len(VALID_CLASS_IDS)
        avg_iou = iou_sum / len(VALID_CLASS_IDS)
        
        # Compute raw metrics
        raw_dice_sum = sum(
            (mean_metrics.get(cid) or mean_metrics.get(str(cid), {})).get("Dice", 0)
            for cid in VALID_CLASS_IDS
        )
        raw_iou_sum = sum(
            (mean_metrics.get(cid) or mean_metrics.get(str(cid), {})).get("IoU", 0)
            for cid in VALID_CLASS_IDS
        )
        avg_raw_dice = raw_dice_sum / len(VALID_CLASS_IDS)
        avg_raw_iou = raw_iou_sum / len(VALID_CLASS_IDS)
        
        results = {
            "trainer": trainer_info['name'],
            "fold": trainer_info['fold_name'],
            "checkpoint": trainer_info['checkpoint_name'],
            "timestamp": datetime.now().isoformat(),
            "model_type": os.path.basename(trainer_info['path']),
            "model_info": model_info or {'params_m': 0.0, 'gflops': 0.0},
            "overall": {
                "dice": avg_dice,  # penalized Dice
                "iou": avg_iou,    # penalized IoU
                "raw_dice": avg_raw_dice,  # raw Dice
                "raw_iou": avg_raw_iou     # raw IoU
            },
            "per_class": {}
        }
        
        for cid in VALID_CLASS_IDS:
            cm = mean_metrics.get(cid, {}) or mean_metrics.get(str(cid), {})
            class_name = CLASS_NAMES.get(cid, f"class_{cid}")
            results["per_class"][f"{class_name} ({cid})"] = {
                "dice": cm.get("Dice", 0),
                "iou": cm.get("IoU", 0),
                "penalized_dice": cm.get("PenalizedDice", cm.get("Dice", 0)),
                "penalized_iou": cm.get("PenalizedIoU", cm.get("IoU", 0)),
                "adhesion_penalty": cm.get("AdhesionPenalty", 0)
            }
        
        # Save detailed evaluation results
        with open(f"{pred_folder}/crack_iou_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  Raw metrics - avg Dice: {avg_raw_dice:.4f}, avg IoU: {avg_raw_iou:.4f}")
        print(f"  Penalized metrics - avg Dice: {avg_dice:.4f}, avg IoU: {avg_iou:.4f}")
        return results
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        traceback.print_exc()
        return None

def save_comparison_results(all_results):
    """Save comparison results."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        return None
    
    with open(f"{OUTPUT_DIR}/comparison_results.json", 'w') as f:
        json.dump(valid, f, indent=2)
    
    df_data = []
    for r in valid:
        model_info = r.get("model_info", {})
        
        # Compute average adhesion penalty
        adhesion_penalties = []
        for ck, cv in r["per_class"].items():
            if "adhesion_penalty" in cv:
                adhesion_penalties.append(cv["adhesion_penalty"])
        avg_adhesion_penalty = np.mean(adhesion_penalties) if adhesion_penalties else 0.0
        
        row = {
            "Trainer": r["trainer"], 
            "Model Type": r.get("model_type", "Unknown"),
            "Fold": r["fold"], 
            "Checkpoint": r["checkpoint"],
            "Params(M)": model_info.get("params_m", 0.0),
            "GFLOPs(G)": model_info.get("gflops", 0.0),
            "Raw Dice": r["overall"].get("raw_dice", r["overall"]["dice"]),  # raw Dice
            "Penalized Dice": r["overall"]["dice"],  # penalized Dice
            "Dice Drop": r["overall"].get("raw_dice", r["overall"]["dice"]) - r["overall"]["dice"],  # Dice drop
            "Adhesion Penalty": avg_adhesion_penalty,  # average adhesion penalty
            "Avg IoU": r["overall"]["iou"]
        }
        for ck, cv in r["per_class"].items():
            class_name = ck.split(" (")[0]
            row[f"{class_name} Raw Dice"] = cv.get("dice", 0)
            row[f"{class_name} Penalized Dice"] = cv.get("penalized_dice", cv.get("dice", 0))
            row[f"{class_name} Adhesion Penalty"] = cv.get("adhesion_penalty", 0)
            row[f"{class_name} IoU"] = cv["iou"]
        df_data.append(row)
    
    df = pd.DataFrame(df_data).sort_values("Penalized Dice", ascending=False)
    df.to_csv(f"{OUTPUT_DIR}/comparison_results.csv", index=False)
    
    print(f"\n{'='*140}")
    print("Model performance ranking (sorted by penalized Dice):")
    print(f"{'='*140}")
    # Adjust display column order
    col_order = ["Trainer", "Params(M)", "GFLOPs(G)", "Raw Dice", "Penalized Dice", "Dice Drop", "Adhesion Penalty", "Avg IoU"]
    other_cols = [c for c in df.columns if c not in col_order]
    display_df = df[col_order + other_cols]
    print(display_df.to_string(index=False))
    
    # Also print a simplified table (key metrics only)
    print(f"\n{'='*100}")
    print("Simplified comparison table:")
    print(f"{'='*100}")
    simple_cols = ["Trainer", "Params(M)", "Raw Dice", "Penalized Dice", "Dice Drop", "Adhesion Penalty"]
    print(df[simple_cols].to_string(index=False))
    
    return df

def main():
    print(f"{'='*80}")
    print("Dataset123_Perovskite test set comparison evaluation")
    print(f"{'='*80}")
    
    if not os.path.exists(RAW_TEST_IMAGES):
        print("❌ Critical directory does not exist")
        return
    
    # Prepare data
    # prepared_dir = os.path.join(OUTPUT_DIR, "test_input_prepared")
    # print("\n[0/3] Preparing YOLO auxiliary channel data...")
    # input_dir = prepared_dir if prepare_yolo_input(RAW_TEST_IMAGES, prepared_dir) else RAW_TEST_IMAGES
    print("\n[0/3] Using raw test set data (assumed already processed by 123 script)...")
    input_dir = RAW_TEST_IMAGES
    print(f"✅ Data directory: {input_dir}")
    
    # Get trainers
    print("\n[1/3] Discovering trainer models...")
    trainers = get_all_trainers()
    if not trainers:
        print("❌ No trainer found")
        return
    print(f"Found {len(trainers)} trainers")
    
    # Evaluate
    print(f"\n[2/3] Starting evaluation...")
    all_results = []
    
    for trainer in tqdm(trainers, desc="Evaluation progress"):
        pred_folder = f"{OUTPUT_DIR}/{trainer['name']}_predictions"
        
        success, model_info = predict_with_trainer(trainer, pred_folder, input_dir)
        if success:
            result = evaluate_predictions(pred_folder, trainer, model_info)
            if result:
                all_results.append(result)
                visualize_errors(trainer['name'], pred_folder, input_dir)
    
    save_comparison_results(all_results)
    print(f"\n✅ All done!")

if __name__ == "__main__":
    main()
