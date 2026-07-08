#!/usr/bin/env python3
"""
对Dataset122_Perovskite数据集下的所有trainer进行测试集对比评估
自动发现所有trainer模型，对测试集进行预测并计算指标
"""

import os
import sys
sys.path.insert(0, "/home/chen/seg6/U-Mamba/umamba")

os.environ['DISABLE_FOURIER_INFERENCE'] = 'True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
# 解决NumExpr线程限制问题
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

# nnunet相关模块
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference import data_iterators
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from typing import List, Union

# Monkey Patch: 避免多进程问题
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

# ========== 统一评估指标配置 ==========
val_evaluator = dict(
    type='CrackIoUMetric', 
    iou_metrics=['mIoU', 'mDice'],
    # 粘连惩罚参数 (Soft模式)
    penalty_scale=0.18,      # 惩罚缩放因子
    max_penalty=0.30,        # 最大惩罚值 (30%)
    min_component_size=10    # 最小连通组件大小
)

# ========== 配置 ==========
DATASET_NAME = "Dataset122_Perovskite"
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

YOLO_DETECTOR = '/home/chen/seg6/yolo_cls_optimization/train21_l_aug5/weights/best.pt'
#YOLO_CLASSIFIER = '/home/chen/seg6/perovskite_grains_opt/yolo_cbam_s_128/weights/best.pt'
LABEL_MAP = {'background': 0, 'PbI2': 1, 'ABO3': 2}

# ========== 工具函数 ==========
def get_model_info(predictor):
    """获取模型的参数量和计算量"""
    try:
        network = predictor.network
        
        # 计算参数量
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        params_m = total_params / 1e6  # 转换为M
        
        # 计算FLOPs (使用输入尺寸估算)
        try:
            cm = predictor.configuration_manager
            if hasattr(cm, 'patch_size'):
                patch_size = cm.patch_size
            elif hasattr(cm, 'configuration') and 'patch_size' in cm.configuration:
                patch_size = cm.configuration['patch_size']
            else:
                # 根据网络结构推断默认尺寸
                conv_op = getattr(network, 'conv_op', torch.nn.Conv2d)
                is_2d = conv_op == torch.nn.Conv2d
                patch_size = [256, 256] if is_2d else [128, 128, 128]
            
            # 获取输入通道数
            num_channels = get_input_channels(predictor)
            
            # 优先使用thop库计算FLOPs (更精确)
            input_shape = [num_channels] + list(patch_size)
            thop_result = get_model_info_thop(network, input_shape)
            
            if thop_result is not None:
                return thop_result
            
            # 使用简化估算计算FLOPs
            gflops = estimate_gflops_simple(network, num_channels, patch_size)
        except Exception as e:
            print(f"  警告: FLOPs计算失败 - {e}")
            gflops = 0.0
        
        return {
            'params_m': round(params_m, 2),
            'gflops': round(gflops, 2),
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    except Exception as e:
        print(f"  警告: 无法获取模型信息 - {e}")
        return {'params_m': 0.0, 'gflops': 0.0, 'total_params': 0, 'trainable_params': 0}


def estimate_gflops_simple(network, in_channels, input_size):
    """
    简化估算GFLOPs - 不使用hook，直接根据层参数计算
    """
    total_flops = 0
    
    # 计算卷积层FLOPs的辅助函数
    def calc_conv_flops(module, input_spatial_size):
        if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d)):
            k = module.kernel_size[0] if isinstance(module.kernel_size, (tuple, list)) else module.kernel_size
            c_in = module.in_channels
            c_out = module.out_channels
            groups = module.groups
            
            # 计算输出空间尺寸
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
            # BN: 2 * C * H * W * D (乘法和加法)
            flops = 2 * module.num_features * input_spatial_size[0] * input_spatial_size[1] * input_spatial_size[2]
            return flops, input_spatial_size
        elif isinstance(module, torch.nn.BatchNorm2d):
            flops = 2 * module.num_features * input_spatial_size[0] * input_spatial_size[1]
            return flops, input_spatial_size
        return 0, input_spatial_size
    
    # 估算前向传播过程中的尺寸变化
    # 使用encoder的层数来估算下采样次数
    current_size = list(input_size)
    is_3d = len(input_size) == 3
    
    # 遍历所有模块，按顺序估算FLOPs
    for name, module in network.named_modules():
        # 跳过容器模块
        if len(list(module.children())) > 0:
            continue
        
        flops, new_size = calc_conv_flops(module, current_size)
        total_flops += flops
        
        # 如果是下采样层(卷积且stride>1)，更新尺寸
        if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d)) and hasattr(module, 'stride'):
            stride = module.stride[0] if isinstance(module.stride, (tuple, list)) else module.stride
            if stride > 1:
                current_size = [max(s // stride, 1) for s in current_size]
    
    # 如果没有计算到FLOPs，使用参数量进行粗略估算
    if total_flops == 0:
        total_params = sum(p.numel() for p in network.parameters())
        # 假设每个参数平均参与100次运算
        total_flops = total_params * 100
    
    return total_flops / 1e9  # 转换为GFLOPs


def get_model_info_thop(network, input_shape):
    """
    使用thop库计算模型的参数量和计算量 (更精确)
    使用深拷贝隔离thop的影响，避免污染原始模型
    """
    try:
        import copy
        from thop import profile, clever_format
        
        # 深拷贝模型，避免thop的hooks污染原始模型
        network_copy = copy.deepcopy(network)
        network_copy.eval()
        
        # 创建dummy input
        device = next(network_copy.parameters()).device
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # 计算FLOPs和参数量
        flops, params = profile(network_copy, inputs=(dummy_input,), verbose=False)
        
        # 删除拷贝的模型，释放内存
        del network_copy
        torch.cuda.empty_cache()
        
        return {
            'params_m': params / 1e6,
            'gflops': flops / 1e9,
            'total_params': int(params),
            'trainable_params': sum(p.numel() for p in network.parameters() if p.requires_grad)
        }
    except Exception as e:
        print(f"  thop计算失败: {e}, 使用简化估算")
        return None

def get_input_channels(predictor):
    """获取模型输入通道数"""
    network = predictor.network
    if hasattr(network, 'encoder') and hasattr(network.encoder, 'stem'):
        return network.encoder.stem[0].conv1.in_channels
    if hasattr(network, 'input_channels'):
        return network.input_channels
    params = list(network.parameters())
    return params[0].shape[1] if params else 1

def shrink_patch_size_for_oom(predictor, scale=0.8, min_size=64):
    """OOM时缩小patch_size"""
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
    print(f"  ✅ 已降低 patch_size: {current} -> {new}")
    return True

def has_cuda_only_ops(network):
    """检查是否包含仅CUDA的算子"""
    for m in network.modules():
        name, mod = type(m).__name__.lower(), type(m).__module__.lower()
        if 'mamba' in name or 'mamba' in mod or 'causal_conv1d' in mod:
            return True
    return False

def normalize_name(name):
    name = name.lower()
    return 'PbI2' if 'pbi' in name else 'ABO3' if 'abo' in name else name

# ========== 核心功能 ==========
def get_all_trainers():
    """获取所有trainer模型"""
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
    """创建预测器，带OOM回退"""
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
            print("  ⚠️ 显存不足，切换到CPU/GPU混合模式...")
    return None

def build_input_list(input_folder, num_channels):
    """构建输入文件列表"""
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
    """应用OOM恢复策略"""
    strategies = [
        ("切换到CPU/GPU混合模式", lambda: setattr(predictor, 'perform_everything_on_device', False)),
        ("禁用TTA并增大步长", lambda: (setattr(predictor, 'use_mirroring', False), 
                                       setattr(predictor, 'tile_step_size', 0.9))),
        ("关闭高斯权重融合", lambda: setattr(predictor, 'use_gaussian', False) 
            if ALLOW_AGGRESSIVE_OOM_FALLBACK else None),
        ("缩小patch_size", lambda: shrink_patch_size_for_oom(predictor) 
            if ALLOW_AGGRESSIVE_OOM_FALLBACK else None),
        ("清理缓存", lambda: (torch.cuda.empty_cache(), __import__('gc').collect())),
    ]
    
    if attempt < len(strategies):
        name, action = strategies[attempt]
        if action() is not False:
            print(f"  策略{attempt+1}: {name}...")
            return True
    
    # 最后手段: CPU模式
    if attempt >= max_retries - 2 and not has_cuda_only_ops(predictor.network):
        print("  策略: 强制切换到CPU模式...")
        predictor.perform_everything_on_device = False
        predictor.device = torch.device('cpu')
        predictor.network = predictor.network.to('cpu')
        torch.cuda.empty_cache()
        return True
    
    return attempt < max_retries - 1

def predict_with_trainer(trainer_info, output_folder, input_folder):
    """使用指定trainer进行预测"""
    print(f"\n{'='*60}")
    print(f"正在评估: {trainer_info['name']}")
    print(f"  Fold: {trainer_info['fold_name']}, Checkpoint: {trainer_info['checkpoint_name']}")
    print(f"{'='*60}")
    
    model_info = None
    
    try:
        fold_str = trainer_info['fold_name']
        use_folds = ['all'] if fold_str == 'fold_all' else [int(re.match(r'fold_(\d+)', fold_str).group(1))]
        
        predictor = create_predictor(trainer_info)
        
        # 初始化模型，但暂时不加载权重
        predictor.initialize_from_trained_model_folder(
            trainer_info["path"], use_folds=use_folds, checkpoint_name=trainer_info["checkpoint_name"]
        )
        
        # 检查网络是否成功加载
        if hasattr(predictor, 'network') and predictor.network is not None:
            print(f"  ✅ 模型成功加载")
        else:
            print(f"  ❌ 模型加载失败")
            return False, None
        
        # 获取模型信息 (参数量和FLOPs)
        model_info = get_model_info(predictor)
        print(f"  模型参数量: {model_info['params_m']:.2f}M, GFLOPs: {model_info['gflops']:.2f}G")
        
        num_channels = get_input_channels(predictor)
        print(f"  模型期望输入通道数: {num_channels}")
        
        list_of_lists, missing = build_input_list(input_folder, num_channels)
        if not list_of_lists:
            print(f"  ❌ 未找到测试图像")
            return False
        if missing:
            print(f"  ❌ 缺少{len(missing)}个通道文件")
            return False
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"  开始对{len(list_of_lists)}个样本进行预测...")
        
        for attempt in range(6):
            try:
                predictor.predict_from_files(
                    list_of_lists, output_folder, save_probabilities=False, overwrite=True,
                    num_processes_preprocessing=1, num_processes_segmentation_export=1,
                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
                )
                print(f"  预测完成: {output_folder}")
                return True, model_info
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_str = str(e).lower()
                if 'out of memory' not in err_str and 'expected x.is_cuda()' not in err_str:
                    if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
                        print(f"  警告: 模型权重加载时出现键不匹配，尝试非严格加载...")
                        print(f"  错误详情: {str(e)[:500]}...")
                        # 如果遇到键不匹配问题，跳过本次预测，继续下一个模型
                        return False
                    raise e
                if 'expected x.is_cuda()' in err_str:
                    print("  ❌ 包含CUDA-only算子，无法使用CPU模式")
                    return False
                if not apply_oom_strategy(predictor, attempt, 6):
                    print("  ❌ 所有OOM策略均失败")
                    return False, None
    except Exception as e:
        if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
            print(f"  警告: 模型权重加载时出现键不匹配，跳过此模型...")
            print(f"  错误详情: {str(e)[:500]}...")
            return False, None
        print(f"  ❌ 预测失败: {e}")
        traceback.print_exc()
        return False, None

def visualize_prediction_mask(pred_mask, raw_image_path, output_path):
    """可视化预测结果"""
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
    """生成错误可视化"""
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
    裂缝IoU评估指标
    支持粘连惩罚和小组件过滤的统一评估器
    """
    
    def __init__(self, iou_metrics=['mIoU', 'mDice'], penalty_scale=0.18, max_penalty=0.30, min_component_size=10):
        """
        Args:
            iou_metrics: 要计算的指标列表，支持 'mIoU', 'mDice'
            penalty_scale: 粘连惩罚缩放因子 (默认0.18)
            max_penalty: 最大惩罚值 (默认0.30 = 30%)
            min_component_size: 最小连通组件大小，小于此值的组件将被忽略
        """
        self.iou_metrics = iou_metrics
        self.penalty_scale = penalty_scale
        self.max_penalty = max_penalty
        self.min_component_size = min_component_size
        self.results = {}
    
    def compute_iou(self, pred, target, class_id):
        """计算单个类别的IoU"""
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union
    
    def compute_dice(self, pred, target, class_id):
        """计算单个类别的Dice"""
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()
        
        if pred_sum + target_sum == 0:
            return 1.0 if intersection == 0 else 0.0
        return 2.0 * intersection / (pred_sum + target_sum)
    
    def filter_small_components(self, mask, class_id):
        """过滤小于min_component_size的连通组件"""
        class_mask = (mask == class_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # 从1开始，跳过背景
            if stats[i, cv2.CC_STAT_AREA] >= self.min_component_size:
                filtered_mask[labels == i] = class_id
        
        return filtered_mask
    
    def compute_adhesion_penalty(self, pred, target, class_id, penalty_scale=0.18, max_penalty=0.30):
        """
        计算粘连惩罚分数 (Soft模式)
        
        逻辑:
        1. 如果 Pred_components >= GT_components: Adhesion_Ratio = 0 (无粘连)
        2. 否则: Adhesion_Ratio = (GT_components - Pred_components) / GT_components
        3. Adhesion_Penalty = min(Adhesion_Ratio × penalty_scale, max_penalty)
        
        Args:
            pred: 预测掩码
            target: GT掩码
            class_id: 类别ID
            penalty_scale: 惩罚缩放因子 (默认0.18)
            max_penalty: 最大惩罚值 (默认0.30)
        """
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        
        # 对预测和GT进行连通组件分析
        pred_num_labels, _ = cv2.connectedComponents(pred_mask, connectivity=8)[:2]
        target_num_labels, _ = cv2.connectedComponents(target_mask, connectivity=8)[:2]
        
        # 减去背景
        pred_components = pred_num_labels - 1
        target_components = target_num_labels - 1
        
        # 如果没有GT组件或没有预测组件，不计算惩罚
        if target_components == 0 or pred_components == 0:
            return 0.0
        
        # 计算粘连度 (Adhesion Ratio)
        if pred_components >= target_components:
            adhesion_ratio = 0.0  # 无粘连
        else:
            adhesion_ratio = (target_components - pred_components) / target_components
        
        # 计算惩罚
        adhesion_penalty = adhesion_ratio * penalty_scale
        adhesion_penalty = min(adhesion_penalty, max_penalty)
        
        return adhesion_penalty
    
    def compute_metrics_on_case(self, pred_path, target_path):
        """计算单个案例的指标"""
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        if pred is None or target is None:
            return None
        
        # 确保尺寸一致
        if pred.shape != target.shape:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        case_metrics = {}
        
        for class_id in VALID_CLASS_IDS:
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            
            # 应用小组件过滤
            if self.min_component_size > 0:
                filtered_pred = self.filter_small_components(pred, class_id)
            else:
                filtered_pred = pred
            
            # 基础指标
            iou = self.compute_iou(filtered_pred, target, class_id)
            dice = self.compute_dice(filtered_pred, target, class_id)
            
            # 粘连惩罚 (Soft模式)
            adhesion_penalty = self.compute_adhesion_penalty(
                filtered_pred, target, class_id, 
                penalty_scale=self.penalty_scale, 
                max_penalty=self.max_penalty
            )
            
            # 应用惩罚: Final = Original × (1 - Adhesion_Penalty)
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
        计算整个文件夹的指标
        兼容nnUNet的compute_metrics_on_folder接口
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
        
        # 计算平均指标
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
    """评估预测结果 - 使用统一评估指标"""
    try:
        if not [f for f in os.listdir(pred_folder) if f.endswith('.png')]:
            return None
        
        # 使用统一的CrackIoUMetric评估器 (Soft粘连惩罚模式)
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
            print(f"  ⚠️ CrackIoUMetric评估失败，尝试使用标准评估...")
            # 回退到标准评估
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
        
        # 使用惩罚后的指标作为主要评估指标
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
        
        # 计算原始指标
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
                "dice": avg_dice,  # 惩罚后的Dice
                "iou": avg_iou,    # 惩罚后的IoU
                "raw_dice": avg_raw_dice,  # 原始Dice
                "raw_iou": avg_raw_iou     # 原始IoU
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
        
        # 保存详细评估结果
        with open(f"{pred_folder}/crack_iou_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  原始指标 - 平均 Dice: {avg_raw_dice:.4f}, 平均 IoU: {avg_raw_iou:.4f}")
        print(f"  惩罚后指标 - 平均 Dice: {avg_dice:.4f}, 平均 IoU: {avg_iou:.4f}")
        return results
    except Exception as e:
        print(f"  ❌ 评估失败: {e}")
        traceback.print_exc()
        return None

def save_comparison_results(all_results):
    """保存对比结果"""
    valid = [r for r in all_results if r is not None]
    if not valid:
        return None
    
    with open(f"{OUTPUT_DIR}/comparison_results.json", 'w') as f:
        json.dump(valid, f, indent=2)
    
    df_data = []
    for r in valid:
        model_info = r.get("model_info", {})
        
        # 计算平均粘连惩罚
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
            "Raw Dice": r["overall"].get("raw_dice", r["overall"]["dice"]),  # 原始Dice
            "Penalized Dice": r["overall"]["dice"],  # 惩罚后Dice
            "Dice Drop": r["overall"].get("raw_dice", r["overall"]["dice"]) - r["overall"]["dice"],  # Dice下降值
            "Adhesion Penalty": avg_adhesion_penalty,  # 平均粘连惩罚
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
    print("模型性能排名 (按惩罚后Dice排序):")
    print(f"{'='*140}")
    # 调整显示列顺序
    col_order = ["Trainer", "Params(M)", "GFLOPs(G)", "Raw Dice", "Penalized Dice", "Dice Drop", "Adhesion Penalty", "Avg IoU"]
    other_cols = [c for c in df.columns if c not in col_order]
    display_df = df[col_order + other_cols]
    print(display_df.to_string(index=False))
    
    # 同时打印简化版表格（仅关键指标）
    print(f"\n{'='*100}")
    print("简化对比表:")
    print(f"{'='*100}")
    simple_cols = ["Trainer", "Params(M)", "Raw Dice", "Penalized Dice", "Dice Drop", "Adhesion Penalty"]
    print(df[simple_cols].to_string(index=False))
    
    return df

def main():
    print(f"{'='*80}")
    print("Dataset122_Perovskite 测试集对比评估")
    print(f"{'='*80}")
    
    if not os.path.exists(RAW_TEST_IMAGES):
        print("❌ 关键目录不存在")
        return
    
    # 准备数据
    # prepared_dir = os.path.join(OUTPUT_DIR, "test_input_prepared")
    # print("\n[0/3] 准备YOLO辅助通道数据...")
    # input_dir = prepared_dir if prepare_yolo_input(RAW_TEST_IMAGES, prepared_dir) else RAW_TEST_IMAGES
    print("\n[0/3] 使用原始测试集数据 (假设已由122脚本处理)...")
    input_dir = RAW_TEST_IMAGES
    print(f"✅ 数据目录: {input_dir}")
    
    # 获取trainer
    print("\n[1/3] 发现trainer模型...")
    trainers = get_all_trainers()
    if not trainers:
        print("❌ 未找到trainer")
        return
    print(f"发现 {len(trainers)} 个trainer")
    
    # 评估
    print(f"\n[2/3] 开始评估...")
    all_results = []
    
    for trainer in tqdm(trainers, desc="评估进度"):
        pred_folder = f"{OUTPUT_DIR}/{trainer['name']}_predictions"
        
        success, model_info = predict_with_trainer(trainer, pred_folder, input_dir)
        if success:
            result = evaluate_predictions(pred_folder, trainer, model_info)
            if result:
                all_results.append(result)
                visualize_errors(trainer['name'], pred_folder, input_dir)
    
    save_comparison_results(all_results)
    print(f"\n✅ 全部完成!")

if __name__ == "__main__":
    main()
