#!/usr/bin/env python3
"""
改进版YOLO-UMamba融合策略
1. 使用YOLO分类概率分布而非单一标签
2. 添加置信度阈值和空间权重
3. 支持多种融合方式对比
"""

import os
import sys
sys.path.append("/home/chen/seg6/U-Mamba/umamba")

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
import argparse

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO

# Monkey patch for multiprocessing
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

# Apply monkey patch
data_iterators.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous
import nnunetv2.inference.predict_from_raw_data
nnunetv2.inference.predict_from_raw_data.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous

# ========== 配置 ==========
DATASET_NAME = "Dataset114_Perovskite"  # 可修改
NNUNET_RAW = "/home/chen/seg6/U-Mamba/data/nnUNet_raw"
NNUNET_RESULTS = "/home/chen/seg6/U-Mamba/data/nnUNet_results"

# YOLO配置
YOLO_WEIGHTS = '/home/chen/seg6/perovskite_grains_opt/train29/weights/best.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.6  # 置信度阈值

# 类别映射
CLASS_MAPPING = {
    0: 2,  # ABO3 -> 2
    1: 1,  # PbI2 -> 1  
    2: 3   # defect -> 3
}

# 可视化颜色
CLASS_COLORS = {
    1: (0, 255, 0),    # PbI2 - 绿色
    2: (0, 165, 255),  # ABO3 - 橙色
    3: (0, 0, 255)     # defect - 红色
}

CLASS_NAMES = {
    1: "PbI2",
    2: "ABO3", 
    3: "defect"
}

# ========== 融合策略类 ==========
class YOLOFusionStrategy:
    """YOLO融合策略基类"""
    
    def __init__(self, yolo_model, confidence_threshold=0.6):
        self.model = yolo_model
        self.conf_threshold = confidence_threshold
        self.name = "base"
    
    def generate_auxiliary_channels(self, image_path):
        """生成辅助通道，返回numpy数组 (C, H, W)"""
        raise NotImplementedError

class Top1MaskFusion(YOLOFusionStrategy):
    """原始策略：使用top1标签重复填充"""
    
    def __init__(self, yolo_model, confidence_threshold=0.6):
        super().__init__(yolo_model, confidence_threshold)
        self.name = "top1_mask"
    
    def generate_auxiliary_channels(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        h, w = img.shape
        
        # YOLO预测
        results = self.model(image_path, verbose=False)
        probs = results[0].probs
        
        if probs.top1conf < self.conf_threshold:
            # 置信度不足，返回全0
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            # 使用top1标签
            yolo_class = probs.top1
            mask_value = CLASS_MAPPING.get(yolo_class, 0)
            mask = np.full((h, w), mask_value, dtype=np.float32)
        
        return mask[np.newaxis, :, :]  # (1, H, W)

class ProbabilityMapFusion(YOLOFusionStrategy):
    """改进策略1：使用概率分布图"""
    
    def __init__(self, yolo_model, confidence_threshold=0.6):
        super().__init__(yolo_model, confidence_threshold)
        self.name = "probability_map"
    
    def generate_auxiliary_channels(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        h, w = img.shape
        
        # YOLO预测
        results = self.model(image_path, verbose=False)
        probs = results[0].probs
        
        # 获取所有类别的概率
        probabilities = probs.data.cpu().numpy()  # (num_classes,)
        
        # 创建多通道概率图
        channels = []
        
        # 通道0: 背景概率
        bg_prob = 1.0 - np.max(probabilities)
        bg_map = np.full((h, w), bg_prob, dtype=np.float32)
        channels.append(bg_map)
        
        # 通道1-3: 各类别概率
        for yolo_class in [0, 1, 2]:  # ABO3, PbI2, defect
            class_prob = probabilities[yolo_class]
            class_map = np.full((h, w), class_prob, dtype=np.float32)
            channels.append(class_map)
        
        return np.stack(channels, axis=0)  # (4, H, W)

class ConfidenceWeightedFusion(YOLOFusionStrategy):
    """改进策略2：置信度加权的空间权重图"""
    
    def __init__(self, yolo_model, confidence_threshold=0.6):
        super().__init__(yolo_model, confidence_threshold)
        self.name = "confidence_weighted"
    
    def generate_auxiliary_channels(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        h, w = img.shape
        
        # YOLO预测
        results = self.model(image_path, verbose=False)
        probs = results[0].probs
        
        # 获取top1信息
        top1_class = probs.top1
        top1_conf = probs.top1conf
        
        channels = []
        
        # 通道0: 空间权重（基于置信度）
        if top1_conf >= self.conf_threshold:
            weight_map = np.full((h, w), top1_conf, dtype=np.float32)
        else:
            weight_map = np.zeros((h, w), dtype=np.float32)
        channels.append(weight_map)
        
        # 通道1: 类别标签
        if top1_conf >= self.conf_threshold:
            label_value = CLASS_MAPPING.get(top1_class, 0)
        else:
            label_value = 0
        label_map = np.full((h, w), label_value, dtype=np.float32)
        channels.append(label_map)
        
        # 通道2: 置信度熵（不确定性度量）
        probabilities = probs.data.cpu().numpy()
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        entropy_map = np.full((h, w), entropy, dtype=np.float32)
        channels.append(entropy_map)
        
        return np.stack(channels, axis=0)  # (3, H, W)

class MultiChannelFusion(YOLOFusionStrategy):
    """改进策略3：多通道综合信息"""
    
    def __init__(self, yolo_model, confidence_threshold=0.6):
        super().__init__(yolo_model, confidence_threshold)
        self.name = "multichannel"
    
    def generate_auxiliary_channels(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        h, w = img.shape
        
        # YOLO预测
        results = self.model(image_path, verbose=False)
        probs = results[0].probs
        
        # 获取完整概率分布
        probabilities = probs.data.cpu().numpy()
        top1_class = probs.top1
        top1_conf = probs.top1conf
        
        channels = []
        
        # 通道0: 最大置信度
        max_conf_map = np.full((h, w), top1_conf, dtype=np.float32)
        channels.append(max_conf_map)
        
        # 通道1-3: 各类别原始概率
        for i in range(3):  # 3个YOLO类别
            prob_map = np.full((h, w), probabilities[i], dtype=np.float32)
            channels.append(prob_map)
        
        # 通道4: 预测类别标签（离散值）
        if top1_conf >= self.conf_threshold:
            pred_label = CLASS_MAPPING.get(top1_class, 0)
        else:
            pred_label = 0
        label_map = np.full((h, w), pred_label, dtype=np.float32)
        channels.append(label_map)
        
        # 通道5: 置信度比率（top1 vs top2）
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) > 1:
            conf_ratio = sorted_probs[0] / (sorted_probs[1] + 1e-8)
        else:
            conf_ratio = 1.0
        ratio_map = np.full((h, w), conf_ratio, dtype=np.float32)
        channels.append(ratio_map)
        
        return np.stack(channels, axis=0)  # (6, H, W)

# ========== 辅助函数 ==========
def get_all_trainers():
    """获取所有trainer模型"""
    dataset_dir = Path(f"{NNUNET_RESULTS}/{DATASET_NAME}")
    trainers = []
    
    checkpoint_priority = ["checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"]
    fold_priority = [f"fold_0", "fold_all"]
    
    if not dataset_dir.exists():
        print(f"警告: 数据集目录不存在: {dataset_dir}")
        return trainers
    
    for item in dataset_dir.iterdir():
        if item.is_dir() and item.name.startswith("nnUNetTrainer"):
            trainer_path = item
            
            best_fold = None
            best_checkpoint = None
            best_score = -1
            
            for fold_idx, fold_name in enumerate(fold_priority):
                fold_dir = trainer_path / fold_name
                if fold_dir.exists():
                    for ckpt_idx, ckpt_name in enumerate(checkpoint_priority):
                        checkpoint_file = fold_dir / ckpt_name
                        if checkpoint_file.exists():
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

def prepare_yolo_input_enhanced(src_dir, dst_dir, fusion_strategy, num_channels=1):
    """
    使用增强策略准备YOLO辅助通道
    
    Args:
        src_dir: 原始图像目录
        dst_dir: 输出目录
        fusion_strategy: YOLOFusionStrategy实例
        num_channels: 辅助通道数量
    """
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        print(f"  使用融合策略: {fusion_strategy.name}")
        print(f"  生成 {num_channels} 个辅助通道")
        
        # 获取所有_0000.png文件
        img_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_0000.png')])
        
        for img_file in tqdm(img_files, desc=f"生成YOLO通道 ({fusion_strategy.name})"):
            case_id = img_file.replace('_0000.png', '')
            src_img_path = os.path.join(src_dir, img_file)
            
            # 1. 复制原始通道 _0000.png
            dst_img_path_0 = os.path.join(dst_dir, img_file)
            if not os.path.exists(dst_img_path_0):
                shutil.copy2(src_img_path, dst_img_path_0)
            
            # 2. 生成辅助通道
            auxiliary_data = fusion_strategy.generate_auxiliary_channels(src_img_path)
            if auxiliary_data is None:
                print(f"  警告: 无法生成 {case_id} 的辅助通道")
                continue
            
            # 确保通道数匹配
            if auxiliary_data.shape[0] != num_channels:
                print(f"  警告: 通道数不匹配，期望 {num_channels}, 实际 {auxiliary_data.shape[0]}")
                continue
            
            # 保存每个通道
            for ch in range(num_channels):
                dst_img_path_ch = os.path.join(dst_dir, f"{case_id}_{ch+1:04d}.png")
                # 归一化到0-255范围
                channel_data = auxiliary_data[ch]
                if channel_data.max() <= 1.0:
                    # 概率值，乘以255
                    vis_data = (channel_data * 255).astype(np.uint8)
                else:
                    # 标签值，直接转换
                    vis_data = channel_data.astype(np.uint8)
                cv2.imwrite(dst_img_path_ch, vis_data)
        
        return True
        
    except Exception as e:
        print(f"  准备YOLO输入时出错: {e}")
        traceback.print_exc()
        return False

def predict_with_trainer(trainer_info, output_folder, test_images_dir):
    """使用指定trainer进行预测"""
    print(f"\n{'='*60}")
    print(f"评估: {trainer_info['name']}")
    print(f"Fold: {trainer_info['fold_name']}")
    print(f"Checkpoint: {trainer_info['checkpoint_name']}")
    print(f"{'='*60}")
    
    try:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        fold_str = trainer_info['fold_name']
        use_folds = ['all'] if fold_str == 'fold_all' else [int(re.match(r'fold_(\d+)', fold_str).group(1))]
        
        print(f"使用folds: {use_folds}")
        
        predictor.initialize_from_trained_model_folder(
            trainer_info["path"],
            use_folds=use_folds,
            checkpoint_name=trainer_info["checkpoint_name"]
        )
        
        os.makedirs(output_folder, exist_ok=True)
        
        predictor.predict_from_files(
            test_images_dir,
            output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
        
        return True
        
    except Exception as e:
        print(f"预测失败: {str(e)}")
        traceback.print_exc()
        return False

def evaluate_predictions(pred_folder, trainer_info):
    """评估预测结果"""
    test_labels = f"{NNUNET_RAW}/{DATASET_NAME}/labelsTs"
    
    try:
        metrics = compute_metrics_on_folder(
            folder_ref=test_labels,
            folder_pred=pred_folder,
            output_file=f"{pred_folder}/summary.json",
            image_reader_writer=NaturalImage2DIO(),
            file_ending='.png',
            regions_or_labels=[1, 2, 3],
            ignore_label=None,
            num_processes=3,
            chill=True
        )
        
        results = {
            "trainer": trainer_info['name'],
            "fold": trainer_info['fold_name'],
            "checkpoint": trainer_info['checkpoint_name'],
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "dice": metrics.get("foreground_mean", {}).get("Dice", 0),
                "iou": metrics.get("foreground_mean", {}).get("IoU", 0)
            },
            "per_class": {}
        }
        
        mean_metrics = metrics.get("mean", {})
        
        for class_id in [1, 2, 3]:
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            class_metrics = mean_metrics.get(class_id, {})
            results["per_class"][class_name] = {
                "dice": class_metrics.get("Dice", 0),
                "iou": class_metrics.get("IoU", 0)
            }
        
        return results
        
    except Exception as e:
        print(f"评估失败: {str(e)}")
        return None

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description='YOLO-UMamba融合策略对比')
    parser.add_argument('--fusion', type=str, default='all', 
                        choices=['top1_mask', 'probability_map', 'confidence_weighted', 'multichannel', 'all'],
                        help='融合策略选择')
    parser.add_argument('--dataset', type=str, default='Dataset114_Perovskite',
                        help='数据集名称')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='输出目录后缀')
    
    args = parser.parse_args()
    
    global DATASET_NAME
    DATASET_NAME = args.dataset
    
    print(f"{'='*80}")
    print(f"YOLO-UMamba增强融合策略评估")
    print(f"数据集: {DATASET_NAME}")
    print(f"{'='*80}")
    
    # 加载YOLO模型
    print(f"加载YOLO模型: {YOLO_WEIGHTS}")
    try:
        yolo_model = YOLO(YOLO_WEIGHTS)
    except Exception as e:
        print(f"无法加载YOLO模型: {e}")
        return
    
    # 定义融合策略
    strategies = {
        'top1_mask': Top1MaskFusion(yolo_model, YOLO_CONFIDENCE_THRESHOLD),
        'probability_map': ProbabilityMapFusion(yolo_model, YOLO_CONFIDENCE_THRESHOLD),
        'confidence_weighted': ConfidenceWeightedFusion(yolo_model, YOLO_CONFIDENCE_THRESHOLD),
        'multichannel': MultiChannelFusion(yolo_model, YOLO_CONFIDENCE_THRESHOLD)
    }
    
    # 选择要运行的策略
    if args.fusion == 'all':
        selected_strategies = list(strategies.keys())
    else:
        selected_strategies = [args.fusion]
    
    print(f"选择的融合策略: {selected_strategies}")
    
    # 获取所有trainer
    trainers = get_all_trainers()
    if not trainers:
        print("没有找到可用的trainer模型")
        return
    
    print(f"发现 {len(trainers)} 个trainer模型")
    
    # 对每个融合策略进行评估
    all_results = {}
    
    for strategy_name in selected_strategies:
        print(f"\n{'='*80}")
        print(f"运行融合策略: {strategy_name}")
        print(f"{'='*80}")
        
        strategy = strategies[strategy_name]
        
        # 准备测试数据
        test_images = f"{NNUNET_RAW}/{DATASET_NAME}/imagesTs"
        temp_test_dir = f"{NNUNET_RESULTS}/{DATASET_NAME}/temp_test_{strategy_name}{args.output_suffix}"
        
        print(f"\n[1/3] 准备YOLO辅助通道...")
        if not prepare_yolo_input_enhanced(test_images, temp_test_dir, strategy, 
                                         num_channels=strategy.generate_auxiliary_channels(test_images + "/image_0000_0000.png").shape[0] if os.path.exists(test_images + "/image_0000_0000.png") else 1):
            print("数据准备失败，跳过此策略")
            continue
        
        print(f"✅ 数据准备完成: {temp_test_dir}")
        
        # 评估每个trainer
        print(f"\n[2/3] 评估trainer模型...")
        strategy_results = []
        
        for trainer in trainers:
            pred_folder = f"{NNUNET_RESULTS}/{DATASET_NAME}/{strategy_name}_{trainer['name']}_predictions{args.output_suffix}"
            
            # 预测
            if predict_with_trainer(trainer, pred_folder, temp_test_dir):
                # 评估
                result = evaluate_predictions(pred_folder, trainer)
                if result:
                    result["fusion_strategy"] = strategy_name
                    strategy_results.append(result)
        
        all_results[strategy_name] = strategy_results
        
        # 清理临时目录
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir)
    
    # 保存对比结果
    print(f"\n{'='*80}")
    print("保存对比结果...")
    print(f"{'='*80}")
    
    output_dir = f"{NNUNET_RESULTS}/{DATASET_NAME}/fusion_comparison{args.output_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON
    json_file = f"{output_dir}/fusion_comparison_results.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON结果已保存: {json_file}")
    
    # 生成对比表格
    comparison_data = []
    
    for strategy_name, results in all_results.items():
        if not results:
            continue
        
        # 计算平均指标
        avg_dice = np.mean([r["overall"]["dice"] for r in results])
        avg_iou = np.mean([r["overall"]["iou"] for r in results])
        
        # 计算每个类别的平均Dice
        per_class_dice = {}
        for class_name in CLASS_NAMES.values():
            class_dices = [r["per_class"].get(class_name, {}).get("dice", 0) for r in results]
            per_class_dice[class_name] = np.mean(class_dices)
        
        row = {
            "Fusion Strategy": strategy_name,
            "Avg Dice": avg_dice,
            "Avg IoU": avg_iou,
            "Num Trainers": len(results)
        }
        
        # 添加每个类别的Dice
        for class_name, dice_val in per_class_dice.items():
            row[f"{class_name} Dice"] = dice_val
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values("Avg Dice", ascending=False)
        
        csv_file = f"{output_dir}/fusion_comparison.csv"
        df.to_csv(csv_file, index=False)
        print(f"CSV对比结果已保存: {csv_file}")
        
        print(f"\n{'='*100}")
        print("融合策略性能排名 (按平均Dice排序):")
        print(f"{'='*100}")
        print(df.to_string(index=False))
        
        # 显示最佳策略
        best_strategy = df.iloc[0]
        print(f"\n🏆 最佳融合策略: {best_strategy['Fusion Strategy']}")
        print(f"   平均 Dice: {best_strategy['Avg Dice']:.4f}")
        print(f"   平均 IoU: {best_strategy['Avg IoU']:.4f}")
    
    print(f"\n✅ 所有评估完成!")
    print(f"结果保存目录: {output_dir}")

if __name__ == "__main__":
    main()