#!/usr/bin/env python3
"""
对Dataset112_Perovskite数据集下的所有trainer进行测试集对比评估
自动发现所有trainer模型，对测试集进行预测并计算指标
优化逻辑：支持自动检测模型输入通道数，正确处理单通道预测 (基于119优化主要逻辑)
优化内容：修复类别不匹配、优化Avg Dice/IoU计算、统一错误处理、合并可视化函数、OOM处理
"""

import os
import sys
# 添加U-Mamba源码路径到sys.path
sys.path.insert(0, "/home/chen/seg6/U-Mamba/umamba")

# 设置环境变量以减小显存占用
os.environ['DISABLE_FOURIER_INFERENCE'] = 'True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

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

# 导入nnunet相关模块（使用安全导入）
try:
    from nnunetv2.paths import nnUNet_raw, nnUNet_results
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
    from batchgenerators.utilities.file_and_folder_operations import save_json
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
except ImportError as e:
    print(f"nnunetv2模块导入错误: {e}")
    # 定义一些基本的占位符，以便脚本可以继续执行
    nnUNet_raw = "/not/available"
    nnUNet_results = "/not/available"
    nnUNetPredictor = None
    recursive_find_python_class = None
    PlansManager = None
    ConfigurationManager = None
    compute_metrics_on_folder = None

# 尝试导入正确的图像读写类
try:
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    class NaturalImage2DIO(SimpleITKIO):
        pass
except ImportError:
    try:
        from nnunetv2.imageio.natural_image2d_reader_writer import NaturalImage2DIO
    except ImportError:
        # 如果都没有，创建一个基本的类
        class NaturalImage2DIO:
            def read_images(self, image_fnames):
                images = []
                properties = {}
                for fname in image_fnames:
                    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        img = cv2.imread(fname)
                    images.append(img)
                properties['shape'] = images[0].shape
                return np.stack(images), properties

            def write_seg(self, seg, output_fname, properties):
                cv2.imwrite(output_fname, seg.astype(np.uint8))

# ========== 统一评估指标配置 ==========
val_evaluator = dict(
    type='CrackIoUMetric', 
    iou_metrics=['mIoU', 'mDice'],
    # 粘连惩罚参数 (Soft模式)
    penalty_scale=0.18,      # 惩罚缩放因子
    max_penalty=0.30,        # 最大惩罚值 (30%)
    min_component_size=10    # 最小连通组件大小
)

# ==================================================================================================
# MONKEY PATCH: Bypass multiprocessing for data loading to avoid BrokenPipeError
# ==================================================================================================
from nnunetv2.inference import data_iterators
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from typing import List, Union

def preprocessing_iterator_fromfiles_synchronous(list_of_lists: List[List[str]],
                                     list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    
    label_manager = plans_manager.get_label_manager(dataset_json)
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)

    if list_of_segs_from_prev_stage_files is None:
        list_of_segs_from_prev_stage_files = [None] * len(list_of_lists)
    if output_filenames_truncated is None:
        output_filenames_truncated = [None] * len(list_of_lists)

    for idx, (data_files, seg_prev, ofile) in enumerate(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)):
        data, seg, data_properties = preprocessor.run_case(data_files,
                                                           seg_prev,
                                                           plans_manager,
                                                           configuration_manager,
                                                           dataset_json)
        if seg_prev is not None:
             seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
             data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data).contiguous().float()
        
        item = {'data': data, 'data_properties': data_properties,
                'ofile': ofile}
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item

# Apply the monkey patch only if nnunetv2 modules are successfully imported
if PlansManager is not None:  # Only apply if the modules were successfully imported
    data_iterators.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous
    # Also patch the function in predict_from_raw_data module where it is imported
    import nnunetv2.inference.predict_from_raw_data

    # FORCE patch the module attribute
    nnunetv2.inference.predict_from_raw_data.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous
else:
    print("DEBUG: Skipping monkey patch due to missing nnunetv2 modules")
# ==================================================================================================

# ========== 配置 ==========
DATASET_NAME = "Dataset112_Perovskite"
NNUNET_RAW = "/home/chen/seg6/U-Mamba/data/nnUNet_raw"
NNUNET_RESULTS = "/home/chen/seg6/U-Mamba/data/nnUNet_results"

# 原始测试集路径
RAW_TEST_IMAGES = f"{NNUNET_RAW}/{DATASET_NAME}/imagesTs"
TEST_LABELS = f"{NNUNET_RAW}/{DATASET_NAME}/labelsTs"

# 输出结果目录
OUTPUT_DIR = f"{NNUNET_RESULTS}/{DATASET_NAME}/testset_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预测参数
FOLD = 0  # 默认使用哪个fold的模型，但也会检查fold_all
CHECKPOINT = "checkpoint_final.pth"  # 默认使用最终checkpoint

# OOM 兜底策略开关（影响性能/精度的操作默认关闭）
ALLOW_AGGRESSIVE_OOM_FALLBACK = False

# 类别配置 - 只包含实际存在的类别
CLASS_COLORS = {
    1: (0, 255, 0),    # PbI2 - 绿色
    2: (0, 165, 255)   # ABO3 - 橙色 (BGR格式)
}

CLASS_NAMES = {
    1: "PbI2",
    2: "ABO3"
}

# 有效的类别ID列表
VALID_CLASS_IDS = [1, 2]

# ========== CrackIoUMetric 评估器 (统一评估指标) ==========
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
        """计算单个类别的IoU: TP / (TP + FP + FN)"""
        pred_mask = (pred == class_id)  # 保持布尔类型
        target_mask = (target == class_id)  # 保持布尔类型
        
        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, np.logical_not(target_mask)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), target_mask).sum()
        
        # 如果预测和标签都没有该类别，视为完美匹配
        if tp + fp + fn == 0:
            return 1.0
        iou = tp / (tp + fp + fn)
        return iou
    
    def compute_dice(self, pred, target, class_id):
        """计算单个类别的Dice: 2*TP / (2*TP + FP + FN)"""
        pred_mask = (pred == class_id)  # 保持布尔类型
        target_mask = (target == class_id)  # 保持布尔类型
        
        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, np.logical_not(target_mask)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), target_mask).sum()
        
        # 如果预测和标签都没有该类别，视为完美匹配
        if 2 * tp + fp + fn == 0:
            return 1.0
        dice = 2 * tp / (2 * tp + fp + fn)
        return dice
    
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


# ========== 辅助函数 ==========
def get_all_trainers():
    """获取Dataset112_Perovskite下所有的trainer模型"""
    dataset_dir = Path(f"{NNUNET_RESULTS}/{DATASET_NAME}")
    trainers = []
    
    # Checkpoint优先级
    checkpoint_priority = [
        "checkpoint_final.pth",
        "checkpoint_best.pth",
        "checkpoint_latest.pth"
    ]
    
    # Fold优先级
    fold_priority = [
        f"fold_{FOLD}",
        "fold_all"
    ]
    
    if not dataset_dir.exists():
        print(f"警告: 数据集目录不存在: {dataset_dir}")
        return trainers
    
    for item in dataset_dir.iterdir():
        if item.is_dir() and item.name.startswith("nnUNetTrainer"):
            trainer_path = item
            
            # 查找最优的fold和checkpoint组合
            best_fold = None
            best_checkpoint = None
            best_score = -1
            
            for fold_idx, fold_name in enumerate(fold_priority):
                fold_dir = trainer_path / fold_name
                if fold_dir.exists():
                    for ckpt_idx, ckpt_name in enumerate(checkpoint_priority):
                        checkpoint_file = fold_dir / ckpt_name
                        if checkpoint_file.exists():
                            # 计算评分
                            score = (fold_idx * 10) + ckpt_idx
                            if score > best_score:
                                best_score = score
                                best_fold = fold_dir
                                best_checkpoint = checkpoint_file
            
            if best_fold and best_checkpoint:
                trainers.append({
                    "name": trainer_path.name,
                    "path": str(trainer_path),
                    "fold_dir": str(best_fold),
                    "checkpoint": str(best_checkpoint),
                    "fold_name": best_fold.name,
                    "checkpoint_name": best_checkpoint.name
                })
    
    return sorted(trainers, key=lambda x: x["name"])

def detect_model_type_from_trainer(trainer_path):
    """从trainer类信息中检测模型类型"""
    try:
        # 读取debug.json文件
        debug_file = None
        for fold_dir in glob.glob(f"{trainer_path}/fold_*"):
            debug_path = os.path.join(fold_dir, "debug.json")
            if os.path.exists(debug_path):
                debug_file = debug_path
                break
        
        if debug_file and os.path.exists(debug_file):
            with open(debug_file, 'r') as f:
                debug_info = json.load(f)
                trainer_class_name = debug_info.get('trainer_class_name', '')
                if trainer_class_name:
                    return trainer_class_name
        
        return os.path.basename(trainer_path)
    except Exception as e:
        return os.path.basename(trainer_path)

def load_trainer_class(trainer_name):
    """动态加载trainer类"""
    try:
        # 自定义trainer目录
        custom_trainer_path = "/home/chen/seg6/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer"
        if os.path.exists(custom_trainer_path) and recursive_find_python_class is not None:
            trainer_class = recursive_find_python_class(
                [custom_trainer_path],
                trainer_name,
                "nnunetv2.training.nnUNetTrainer"
            )
            if trainer_class:
                return trainer_class
        
        # 标准nnU-Net trainer目录
        if recursive_find_python_class is not None:
            import nnunetv2.training.nnUNetTrainer
            standard_trainer_path = os.path.dirname(nnunetv2.training.nnUNetTrainer.__file__)
            trainer_class = recursive_find_python_class(
                [standard_trainer_path],
                trainer_name,
                "nnunetv2.training.nnUNetTrainer"
            )
        
        return trainer_class
    except Exception as e:
        print(f"  加载trainer类失败: {str(e)}")
        return None

def shrink_patch_size_for_oom(predictor, scale=0.8, min_size=64):
    """在OOM时缩小patch_size（按pool倍数对齐）"""
    try:
        if predictor is None or predictor.configuration_manager is None:
            return False
        cm = predictor.configuration_manager
        if not hasattr(cm, 'num_pool_per_axis') or not hasattr(cm, 'patch_size'):
            return False
        current = list(cm.patch_size)
        divs = [2 ** n for n in cm.num_pool_per_axis]
        new = []
        changed = False
        for size, d in zip(current, divs):
            target = int(size * scale)
            target = max(target, d)
            target = (target // d) * d
            if target < d:
                target = d
            if target < min_size:
                min_aligned = (min_size // d) * d
                target = max(target, min_aligned if min_aligned > 0 else d)
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
        except Exception:
            pass

        print(f"  ✅ 已降低 patch_size: {current} -> {new}")
        return True
    except Exception as e:
        print(f"  ⚠️ 缩小 patch_size 失败: {e}")
        return False

def predict_with_trainer(trainer_info, output_folder, input_folder):
    """
    使用指定trainer对测试集进行预测
    优化：主动检测模型通道数，并构造对应的文件列表
    """
    print(f"\n{'='*60}")
    print(f"正在评估: {trainer_info['name']}")
    print(f"  Fold: {trainer_info['fold_name']}")
    print(f"  Checkpoint: {trainer_info['checkpoint_name']}")
    print(f"{'='*60}")
    
    if nnUNetPredictor is None:
        print("  ❌ nnUNetPredictor 不可用，跳过预测")
        return False
    
    try:
        # 1. 提取Fold ID
        fold_str = trainer_info['fold_name']
        if fold_str == 'fold_all':
            use_folds = ['all']
        else:
            fold_match = re.match(r'fold_(\d+)', fold_str)
            if fold_match:
                use_folds = [int(fold_match.group(1))]
            else:
                use_folds = [FOLD]
        
        # 2. 尝试加载Trainer Class (如果需要)
        trainer_class = None
        # 有些特殊的trainer可能需要显示指定class，这里简单处理，如果加载失败会回退到默认
        potential_class_name = trainer_info['name'].split('__')[0]
        if "nnUNetTrainer" in potential_class_name:
             trainer_class = load_trainer_class(potential_class_name)

        # 3. 初始化预测器
        # 显存优化: 如果显存不足，自动切换 perform_everything_on_device=False
        try:
             predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True, # 尝试在GPU上进行所有操作
                device=torch.device('cuda'),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
        except RuntimeError as e:
             if 'out of memory' in str(e).lower():
                  print("  ⚠️ 显存不足，切换到CPU/GPU混合模式...")
                  predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=True,
                    perform_everything_on_device=False,
                    device=torch.device('cuda'),
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=True
                )
             else:
                  raise e

        # 4. 加载模型权重和配置
        predictor.initialize_from_trained_model_folder(
            trainer_info["path"],
            use_folds=use_folds,
            checkpoint_name=trainer_info["checkpoint_name"]
        )

        # 5. 【关键优化】检测模型期望的输入通道数
        if hasattr(predictor.network, 'encoder') and hasattr(predictor.network.encoder, 'stem'):
             # 对于 U-Net 结构
             num_input_channels = predictor.network.encoder.stem[0].conv1.in_channels
        elif hasattr(predictor.network, 'input_channels'):
             num_input_channels = predictor.network.input_channels
        else:
             # 尝试从第一个卷积层获取
             try:
                 params = list(predictor.network.parameters())
                 if len(params) > 0:
                      # 假设第一个卷积核是 [out, in, k, k]
                      num_input_channels = params[0].shape[1]
                 else:
                      num_input_channels = 1 # fallback
             except:
                 num_input_channels = 1 # fallback
        
        print(f"  ℹ️ 模型期望输入通道数: {num_input_channels}")

        # 6. 【关键优化】构建输入文件列表 (List[List[str]])
        if not os.path.exists(input_folder):
             print(f"  ❌ 输入目录不存在: {input_folder}")
             return False

        case_files_0000 = sorted([f for f in os.listdir(input_folder) if f.endswith('_0000.png')])
        case_ids = [f.replace('_0000.png', '') for f in case_files_0000]

        if not case_ids:
            print(f"  ❌ 未找到测试图像 (_0000.png)")
            return False

        list_of_lists = []
        missing_files = []

        for case_id in case_ids:
            case_file_list = []
            for i in range(num_input_channels):
                # 构建文件名 case_0000.png, case_0001.png ...
                fname = f"{case_id}_{i:04d}.png"
                fpath = os.path.join(input_folder, fname)
                
                if not os.path.exists(fpath):
                    missing_files.append(fpath)
                
                case_file_list.append(fpath)
            
            list_of_lists.append(case_file_list)

        if missing_files:
            print(f"  ❌ 缺少必要的通道文件 (共 {len(missing_files)} 个):")
            print(f"     示例: {missing_files[0]}")
            return False

        # 7. 执行预测
        os.makedirs(output_folder, exist_ok=True)
        print(f"  开始对 {len(list_of_lists)} 个样本进行预测...")

        # 尝试多次，处理OOM
        max_retries = 6
        for attempt in range(max_retries):
            try:
                predictor.predict_from_files(
                    list_of_lists,
                    output_folder,
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=1,
                    num_processes_segmentation_export=1,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0
                )
                break # 成功则退出循环
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                 # Catch generic RuntimeError too because sometimes OOM comes as RuntimeError
                 if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
                     # CPU fallback may fail for CUDA-only ops (e.g., Mamba/causal_conv1d)
                     if 'expected x.is_cuda()' in str(e).lower():
                         print("  ❌ 当前模型包含仅CUDA算子，CPU模式不可用，跳过该trainer。")
                         return False
                     raise e

                 print(f"  ⚠️ 显存不足 (Attempt {attempt+1}/{max_retries})")
                 
                 # Strategy 1: Switch to Mixed Mode
                 if predictor.perform_everything_on_device:
                     print("  策略1: 切换到 CPU/GPU 混合模式...")
                     predictor.perform_everything_on_device = False
                 
                 # Strategy 2: Disable Mirroring (TTA)
                 elif predictor.use_mirroring:
                     print("  策略2: 禁用镜像增强 (TTA) 并增大步长...")
                     predictor.use_mirroring = False
                     predictor.tile_step_size = 0.9

                 # Strategy 3: Disable Gaussian weighting (may affect quality)
                 elif ALLOW_AGGRESSIVE_OOM_FALLBACK and predictor.use_gaussian:
                     print("  策略3: 关闭高斯权重融合...")
                     predictor.use_gaussian = False

                 # Strategy 4: Shrink patch size (may affect quality)
                 elif ALLOW_AGGRESSIVE_OOM_FALLBACK and shrink_patch_size_for_oom(predictor, scale=0.8, min_size=64):
                     print("  策略4: 缩小 patch_size 并重试...")
                 
                 # Strategy 5: Clear Cache
                 elif attempt < max_retries - 2: 
                     print("  策略5: 清理缓存并重试...")
                     torch.cuda.empty_cache()
                     import gc
                     gc.collect()

                 # Strategy 6: FORCE CPU (Last Resort)
                 else:
                     # 避免对包含 CUDA-only 算子的模型强制CPU
                     has_cuda_only_ops = False
                     if hasattr(predictor, 'network'):
                         for _m in predictor.network.modules():
                             name = type(_m).__name__.lower()
                             mod = type(_m).__module__.lower()
                             if 'mamba' in name or 'mamba' in mod or 'causal_conv1d' in mod:
                                 has_cuda_only_ops = True
                                 break
                     if has_cuda_only_ops:
                         print("  策略6: 检测到CUDA-only算子，跳过CPU回退。")
                     else:
                         print("  策略6: ⛔ 强制切换到 CPU 模式 (速度较慢，但更可靠)...")
                         predictor.perform_everything_on_device = False
                         predictor.device = torch.device('cpu')
                         # Move model to CPU
                         if hasattr(predictor, 'network'):
                            predictor.network = predictor.network.to('cpu')
                         
                         # Clean GPU cache completely
                         torch.cuda.empty_cache()
                         import gc
                         gc.collect()

                 if attempt == max_retries - 1:
                      print("  ❌ 所有策略均失败，预测终止。")
                      return False
            except Exception as e:
                print(f"  ❌ 预测失败: {e}")
                traceback.print_exc()
                return False
        
        print(f"  预测完成，结果保存在: {output_folder}")
        return True

    except Exception as e:
        print(f"  ❌ 预测失败: {str(e)}")
        print(f"  错误详情: {traceback.format_exc()}")
        return False

def visualize_prediction_mask(pred_mask, raw_image_path, output_path):
    """可视化: 橙色=ABO3, 绿色=PbI2"""
    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
    if raw_image is None:
        return False
    
    if len(raw_image.shape) == 2:
        overlay = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = raw_image.copy()
    
    for class_id, color in CLASS_COLORS.items():
        class_mask = (pred_mask == class_id).astype(np.uint8)
        if np.any(class_mask):
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(overlay, contours, -1, color, 2)
    
    cv2.imwrite(output_path, overlay)
    return True

def visualize_errors_for_trainer(trainer_name, pred_folder, raw_images_dir):
    """生成错误可视化图"""
    print(f"正在生成 {trainer_name} 的错误可视化图...")
    vis_output_dir = f"{OUTPUT_DIR}/{trainer_name}_error_visualizations"
    os.makedirs(vis_output_dir, exist_ok=True)
    pred_vis_output_dir = f"{OUTPUT_DIR}/{trainer_name}_prediction_visualizations"
    os.makedirs(pred_vis_output_dir, exist_ok=True)
    
    if not os.path.exists(TEST_LABELS):
        print("  ⚠️ 测试标签目录不存在，跳过错误可视化")
        return

    for filename in os.listdir(TEST_LABELS):
        if filename.endswith('.png'):
            case_name = filename[:-4]
            gt_path = os.path.join(TEST_LABELS, filename)
            pred_path = os.path.join(pred_folder, filename)
            # 使用 raw_images_dir 确保找到 _0000.png
            raw_img_path = os.path.join(raw_images_dir, f"{case_name}_0000.png")
            
            if not os.path.exists(pred_path) or not os.path.exists(raw_img_path):
                continue
            
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            overlay = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
            
            if gt is None or pred is None or overlay is None:
                continue
                
            if len(overlay.shape) == 2:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            
            errors = (gt != pred) & (gt > 0)
            overlay[errors] = [0, 0, 255]
            
            for class_id in VALID_CLASS_IDS:
                gt_mask = (gt == class_id).astype(np.uint8) * 255
                pred_mask = (pred == class_id).astype(np.uint8) * 255
                
                if np.any(gt_mask) or np.any(pred_mask):
                    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 1)
                    
                    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 1)
            
            cv2.imwrite(os.path.join(vis_output_dir, f"{case_name}_error_map.png"), overlay)
            visualize_prediction_mask(pred, raw_img_path, os.path.join(pred_vis_output_dir, f"{case_name}_prediction_vis.png"))
    
    print(f"错误可视化图已保存到: {vis_output_dir}")

def evaluate_predictions(pred_folder, trainer_info):
    """评估预测结果 - 使用统一评估指标 (CrackIoUMetric)"""
    try:
        pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.png')]
        if not pred_files:
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
        
        # 如果CrackIoUMetric失败，回退到标准评估
        if not metrics or 'mean' not in metrics:
            print(f"  ⚠️ CrackIoUMetric评估失败，尝试使用标准评估...")
            if compute_metrics_on_folder is None:
                return None
            try:
                from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
                image_reader_writer = SimpleITKIO()
            except ImportError:
                image_reader_writer = None
            
            metrics = compute_metrics_on_folder(
                folder_ref=TEST_LABELS,
                folder_pred=pred_folder,
                output_file=f"{pred_folder}/summary.json",
                image_reader_writer=image_reader_writer,
                file_ending='.png',
                regions_or_labels=VALID_CLASS_IDS,
                ignore_label=None,
                num_processes=3,
                chill=True
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
        
        # 计算原始指标 (无惩罚)
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
            "model_type": detect_model_type_from_trainer(trainer_info['path']),
            "overall": {
                "dice": avg_dice,              # 惩罚后的Dice (主要指标)
                "iou": avg_iou,                # 惩罚后的IoU
                "raw_dice": avg_raw_dice,      # 原始Dice
                "raw_iou": avg_raw_iou         # 原始IoU
            },
            "per_class": {}
        }
        
        for cid in VALID_CLASS_IDS:
            class_name = CLASS_NAMES.get(cid, f"class_{cid}")
            cm = mean_metrics.get(cid, {}) or mean_metrics.get(str(cid), {})
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
        print(f"  ❌ 评估失败: {str(e)}")
        traceback.print_exc()
        return None

def save_comparison_results(all_results):
    """保存对比结果"""
    if not all_results:
        return None
    
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        return None
    
    # 保存JSON
    with open(f"{OUTPUT_DIR}/comparison_results.json", 'w') as f:
        json.dump(valid_results, f, indent=2)
    
    # 保存CSV
    df_data = []
    for result in valid_results:
        # 计算平均粘连惩罚
        adhesion_penalties = []
        for ck, cv in result["per_class"].items():
            if "adhesion_penalty" in cv:
                adhesion_penalties.append(cv["adhesion_penalty"])
        avg_adhesion_penalty = np.mean(adhesion_penalties) if adhesion_penalties else 0.0
        
        row = {
            "Trainer": result["trainer"],
            "Model Type": result.get("model_type", "Unknown"),
            "Fold": result["fold"],
            "Checkpoint": result["checkpoint"],
            "Raw Dice": result["overall"].get("raw_dice", result["overall"]["dice"]),  # 原始Dice
            "Penalized Dice": result["overall"]["dice"],  # 惩罚后Dice (排序依据)
            "Dice Drop": result["overall"].get("raw_dice", result["overall"]["dice"]) - result["overall"]["dice"],  # Dice下降值
            "Adhesion Penalty": avg_adhesion_penalty,  # 平均粘连惩罚
            "Avg IoU": result["overall"]["iou"]
        }
        for class_key, class_data in result["per_class"].items():
            class_name = class_key.split(" (")[0]
            row[f"{class_name} Raw Dice"] = class_data.get("dice", 0)
            row[f"{class_name} Penalized Dice"] = class_data.get("penalized_dice", class_data.get("dice", 0))
            row[f"{class_name} Adhesion Penalty"] = class_data.get("adhesion_penalty", 0)
            row[f"{class_name} IoU"] = class_data["iou"]
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df = df.sort_values("Penalized Dice", ascending=False)
    
    csv_file = f"{OUTPUT_DIR}/comparison_results.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*140}")
    print("模型性能排名 (按惩罚后Dice排序):")
    print(f"{'='*140}")
    # 调整显示列顺序
    col_order = ["Trainer", "Model Type", "Fold", "Checkpoint", "Raw Dice", "Penalized Dice", "Dice Drop", "Adhesion Penalty", "Avg IoU"]
    other_cols = [c for c in df.columns if c not in col_order]
    display_df = df[col_order + other_cols]
    print(display_df.to_string(index=False))
    
    # 同时打印简化版表格（仅关键指标）
    print(f"\n{'='*100}")
    print("简化对比表:")
    print(f"{'='*100}")
    simple_cols = ["Trainer", "Raw Dice", "Penalized Dice", "Dice Drop", "Adhesion Penalty"]
    print(df[simple_cols].to_string(index=False))
    
    return df

def main():
    print(f"{'='*80}")
    print("Dataset112_Perovskite 测试集对比评估 (优化版-单通道)")
    print(f"{'='*80}")
    
    if not os.path.exists(RAW_TEST_IMAGES) or not os.path.exists(NNUNET_RESULTS):
        print(f"❌ 关键目录不存在，请检查路径。")
        return
        
    # Dataset112 为单通道，直接使用 RAW_TEST_IMAGES
    current_test_images = RAW_TEST_IMAGES
    
    # 1. 获取所有trainer
    print("\n[1/3] 正在发现所有trainer模型...")
    trainers = get_all_trainers()
    if not trainers:
        print("❌ 未找到trainer模型")
        return
    
    print(f"发现 {len(trainers)} 个trainer模型。")
    
    # 2. 对每个trainer进行评估
    print(f"\n[2/3] 开始评估每个trainer...")
    all_results = []
    
    for trainer in tqdm(trainers, desc="评估进度"):
        pred_folder = f"{OUTPUT_DIR}/{trainer['name']}_predictions"
        
        # 传递 current_test_images
        success = predict_with_trainer(trainer, pred_folder, current_test_images)
        
        if success:
            result = evaluate_predictions(pred_folder, trainer)
            if result:
                all_results.append(result)
                # 可视化
                visualize_errors_for_trainer(trainer['name'], pred_folder, current_test_images)
            else:
                all_results.append(None)
        else:
            all_results.append(None)
    
    # 3. 保存对比结果
    save_comparison_results(all_results)
    print(f"\n✅ 全部完成!")

if __name__ == "__main__":
    main()
