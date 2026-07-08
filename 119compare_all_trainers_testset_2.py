#!/usr/bin/env python3
"""
对Dataset119_Perovskite数据集下的所有trainer进行测试集对比评估
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

# ========== 配置 ==========
DATASET_NAME = "Dataset119_Perovskite"
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
            return False
        
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
                return True
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
                    return False
    except Exception as e:
        if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
            print(f"  警告: 模型权重加载时出现键不匹配，跳过此模型...")
            print(f"  错误详情: {str(e)[:500]}...")
            return False
        print(f"  ❌ 预测失败: {e}")
        traceback.print_exc()
        return False

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

def evaluate_predictions(pred_folder, trainer_info):
    """评估预测结果"""
    try:
        if not [f for f in os.listdir(pred_folder) if f.endswith('.png')]:
            return None
        
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
        dice_sum = sum(
            (mean_metrics.get(cid) or mean_metrics.get(str(cid), {})).get("Dice", 0)
            for cid in VALID_CLASS_IDS
        )
        iou_sum = sum(
            (mean_metrics.get(cid) or mean_metrics.get(str(cid), {})).get("IoU", 0)
            for cid in VALID_CLASS_IDS
        )
        
        avg_dice = dice_sum / len(VALID_CLASS_IDS)
        avg_iou = iou_sum / len(VALID_CLASS_IDS)
        
        results = {
            "trainer": trainer_info['name'],
            "fold": trainer_info['fold_name'],
            "checkpoint": trainer_info['checkpoint_name'],
            "timestamp": datetime.now().isoformat(),
            "model_type": os.path.basename(trainer_info['path']),
            "overall": {"dice": avg_dice, "iou": avg_iou},
            "per_class": {}
        }
        
        for cid in VALID_CLASS_IDS:
            cm = mean_metrics.get(cid, {}) or mean_metrics.get(str(cid), {})
            class_name = CLASS_NAMES.get(cid, f"class_{cid}")
            results["per_class"][f"{class_name} ({cid})"] = {
                "dice": cm.get("Dice", 0),
                "iou": cm.get("IoU", 0)
            }
        
        print(f"  平均 Dice: {avg_dice:.4f}, 平均 IoU: {avg_iou:.4f}")
        return results
    except Exception as e:
        print(f"  ❌ 评估失败: {e}")
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
        row = {
            "Trainer": r["trainer"], "Model Type": r.get("model_type", "Unknown"),
            "Fold": r["fold"], "Checkpoint": r["checkpoint"],
            "Avg Dice": r["overall"]["dice"], "Avg IoU": r["overall"]["iou"]
        }
        for ck, cv in r["per_class"].items():
            class_name = ck.split(" (")[0]
            row[f"{class_name} Dice"] = cv["dice"]
            row[f"{class_name} IoU"] = cv["iou"]
        df_data.append(row)
    
    df = pd.DataFrame(df_data).sort_values("Avg Dice", ascending=False)
    df.to_csv(f"{OUTPUT_DIR}/comparison_results.csv", index=False)
    
    print(f"\n{'='*100}")
    print("模型性能排名 (按平均Dice排序):")
    print(f"{'='*100}")
    print(df.to_string(index=False))
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
        
        if predict_with_trainer(trainer, pred_folder, input_dir):
            result = evaluate_predictions(pred_folder, trainer)
            if result:
                all_results.append(result)
                visualize_errors(trainer['name'], pred_folder, input_dir)
    
    save_comparison_results(all_results)
    print(f"\n✅ 全部完成!")

if __name__ == "__main__":
    main()
