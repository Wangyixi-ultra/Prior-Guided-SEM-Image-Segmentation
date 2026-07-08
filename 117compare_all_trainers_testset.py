#!/usr/bin/env python3
"""
对Dataset117_Perovskite数据集下的所有trainer进行测试集对比评估
自动发现所有trainer模型，对测试集进行预测并计算指标
"""

import os
import sys
# 添加U-Mamba源码路径到sys.path
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

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

# ========== 配置 ==========
DATASET_NAME = "Dataset117_Perovskite"
NNUNET_RAW = "/home/chen/seg6/U-Mamba/data/nnUNet_raw"
NNUNET_RESULTS = "/home/chen/seg6/U-Mamba/data/nnUNet_results"

# 测试集路径
TEST_IMAGES = f"{NNUNET_RAW}/{DATASET_NAME}/imagesTs"
TEST_LABELS = f"{NNUNET_RAW}/{DATASET_NAME}/labelsTs"

# 输出结果目录
OUTPUT_DIR = f"{NNUNET_RESULTS}/{DATASET_NAME}/testset_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预测参数
FOLD = 0  # 默认使用哪个fold的模型，但也会检查fold_all
CHECKPOINT = "checkpoint_final.pth"  # 默认使用最终checkpoint，但也会检查其他checkpoint

# ========== 辅助函数 ==========
def get_all_trainers():
    """获取Dataset117_Perovskite下所有的trainer模型"""
    dataset_dir = Path(f"{NNUNET_RESULTS}/{DATASET_NAME}")
    trainers = []
    
    # 可能的checkpoint文件名列表（按优先级排序）
    checkpoint_priority = [
        "checkpoint_final.pth",
        "checkpoint_best.pth",
        "checkpoint_latest.pth"
    ]
    
    # 可能的fold目录名列表（按优先级排序）
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
                            # 计算评分：fold优先级 + checkpoint优先级
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
    """
    从trainer类信息中检测模型类型（更可靠的方法）
    """
    try:
        # 读取debug.json文件获取trainer类信息
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
        
        # 从目录名推断
        trainer_name = os.path.basename(trainer_path)
        return trainer_name
        
    except Exception as e:
        print(f"  检测模型类型时出错: {str(e)}")
        return os.path.basename(trainer_path)

def load_trainer_class(trainer_name):
    """
    动态加载trainer类
    """
    try:
        # 首先在自定义trainer目录中查找
        custom_trainer_path = "/home/chen/seg6/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer"
        if os.path.exists(custom_trainer_path):
            trainer_class = recursive_find_python_class(
                [custom_trainer_path],
                trainer_name,
                "nnunetv2.training.nnUNetTrainer"
            )
            if trainer_class:
                return trainer_class
        
        # 然后在标准nnU-Net trainer目录中查找
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

def predict_with_trainer(trainer_info, output_folder):
    """使用指定trainer对测试集进行预测"""
    print(f"\n{'='*60}")
    print(f"正在评估: {trainer_info['name']}")
    print(f"  Fold: {trainer_info['fold_name']}")
    print(f"  Checkpoint: {trainer_info['checkpoint_name']}")
    print(f"{'='*60}")
    
    try:
        # 初始化预测器
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
        
        # 确定使用的fold
        fold_str = trainer_info['fold_name']
        if fold_str == 'fold_all':
            use_folds = ['all']
        else:
            # 提取fold数字
            fold_match = re.match(r'fold_(\d+)', fold_str)
            if fold_match:
                use_folds = [int(fold_match.group(1))]
            else:
                use_folds = [FOLD]
        
        print(f"  使用folds: {use_folds}")
        
        # 加载训练好的模型
        predictor.initialize_from_trained_model_folder(
            trainer_info["path"],
            use_folds=use_folds,
            checkpoint_name=trainer_info["checkpoint_name"]
        )
        
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        # 进行预测
        print(f"  开始对测试集进行预测...")
        predictor.predict_from_files(
            TEST_IMAGES,
            output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=3,
            num_processes_segmentation_export=3,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
        
        print(f"  预测完成，结果保存在: {output_folder}")
        return True
        
    except RuntimeError as e:
        error_str = str(e)
        if "Missing key(s) in state_dict" in error_str or "Unexpected key(s) in state_dict" in error_str:
            print(f"  ⚠️  检测到模型结构不匹配错误")
            print(f"  尝试使用trainer类信息重新加载...")
            
            try:
                # 使用trainer类重新加载模型
                trainer_name = trainer_info['name'].split('__')[0]  # 提取trainer类名
                trainer_class = load_trainer_class(trainer_name)
                
                if trainer_class:
                    print(f"  成功加载trainer类: {trainer_name}")
                    
                    # 重新初始化预测器，使用正确的trainer类
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
                    
                    # 使用trainer_class_name参数指定正确的trainer类
                    predictor.initialize_from_trained_model_folder(
                        trainer_info["path"],
                        use_folds=use_folds,
                        checkpoint_name=trainer_info["checkpoint_name"],
                        nnunet_trainer_class=trainer_class
                    )
                    
                    # 进行预测
                    print(f"  使用正确的trainer类重新预测...")
                    predictor.predict_from_files(
                        TEST_IMAGES,
                        output_folder,
                        save_probabilities=False,
                        overwrite=True,
                        num_processes_preprocessing=3,
                        num_processes_segmentation_export=3,
                        folder_with_segs_from_prev_stage=None,
                        num_parts=1,
                        part_id=0
                    )
                    
                    print(f"  预测完成，结果保存在: {output_folder}")
                    return True
                else:
                    print(f"  ❌ 无法加载trainer类: {trainer_name}")
                    return False
                    
            except Exception as e2:
                print(f"  ❌ 使用trainer类重新加载失败: {str(e2)}")
                print(f"  错误详情: {traceback.format_exc()}")
                return False
        else:
            print(f"  ❌ 预测失败: {str(e)}")
            print(f"  错误详情: {traceback.format_exc()}")
            return False
    except Exception as e:
        print(f"  ❌ 预测失败: {str(e)}")
        print(f"  错误详情: {traceback.format_exc()}")
        return False

def visualize_errors_for_trainer(trainer_name, pred_folder):
    """为指定训练器的预测结果生成错误可视化图"""
    print(f"正在生成 {trainer_name} 的错误可视化图...")
    
    # 创建可视化输出目录
    vis_output_dir = f"{OUTPUT_DIR}/{trainer_name}_error_visualizations"
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 遍历测试集中的所有标签文件
    for filename in os.listdir(TEST_LABELS):
        if filename.endswith('.png'):
            case_name = filename[:-4]  # 去掉.png后缀
            
            # 构建文件路径
            gt_path = os.path.join(TEST_LABELS, filename)
            pred_path = os.path.join(pred_folder, filename)
            # 修正: imagesTs中的文件通常有_0000后缀
            raw_img_path = os.path.join(TEST_IMAGES, f"{case_name}_0000.png")
            
            # 检查文件是否存在
            if not os.path.exists(pred_path):
                print(f"  跳过 {filename}: 预测结果不存在")
                continue
                
            if not os.path.exists(raw_img_path):
                print(f"  跳过 {filename}: 原始图像不存在")
                continue
            
            # 读取真实标签和预测结果
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            
            if gt is None or pred is None:
                print(f"  跳过 {filename}: 无法读取图像")
                continue
            
            # 读取原始图像并转换为BGR
            overlay = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
            if overlay is None:
                print(f"  跳过 {filename}: 无法读取原始图像")
                continue
                
            # 如果原始图像是单通道灰度图，转换为三通道
            if len(overlay.shape) == 2:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            elif overlay.shape[2] == 1:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            
            # 标记错误区域（预测与真实标签不一致，只关注前景区域）
            errors = (gt != pred) & (gt > 0)  # 只关注前景区域
            overlay[errors] = [0, 0, 255]  # 错误区域标记为红色
            
            # 为真实标签和预测结果绘制轮廓
            # 为每个类别绘制不同的轮廓
            for class_id in [1, 2, 3]:  # 三个类别: PbI2, ABO3, defect
                # 创建掩码用于寻找轮廓
                gt_mask = (gt == class_id).astype(np.uint8) * 255
                pred_mask = (pred == class_id).astype(np.uint8) * 255
                
                # 查找真实标签的轮廓（绿色）
                gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 1)  # 绿色轮廓表示真实标签
                
                # 查找预测结果的轮廓（蓝色）
                pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 1)  # 蓝色轮廓表示预测结果
            
            # 保存可视化结果
            output_path = os.path.join(vis_output_dir, f"{case_name}_error_map.png")
            cv2.imwrite(output_path, overlay)
    
    print(f"错误可视化图已保存到: {vis_output_dir}")

def visualize_borders_for_trainer(trainer_name, pred_folder):
    """生成预测边界可视化图（仅预测轮廓）"""
    print(f"正在生成 {trainer_name} 的边界可视化图...")

    vis_output_dir = f"{OUTPUT_DIR}/{trainer_name}_border_visualizations"
    os.makedirs(vis_output_dir, exist_ok=True)

    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.png')]
    for pred_name in pred_files:
        if pred_name.endswith("_0000.png"):
            case_id = pred_name.replace("_0000.png", "")
        elif pred_name.endswith("_0001.png"):
            case_id = pred_name.replace("_0001.png", "")
        else:
            case_id = pred_name.replace('.png', '')

        pred_path = os.path.join(pred_folder, pred_name)
        raw_img_path = os.path.join(TEST_IMAGES, f"{case_id}_0000.png")
        if not os.path.exists(raw_img_path):
            # fallback: 找任意通道
            candidates = glob.glob(os.path.join(TEST_IMAGES, f"{case_id}_*.png"))
            if candidates:
                raw_img_path = candidates[0]
            else:
                continue

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or raw is None:
            continue

        overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        class_colors = {
            1: (0, 255, 0),    # PbI2 - 绿色
            2: (0, 165, 255),  # ABO3 - 橙色 (BGR)
            3: (0, 0, 255)     # defect - 红色
        }

        for class_id, color in class_colors.items():
            mask = (pred == class_id).astype(np.uint8) * 255
            if mask.sum() == 0:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(overlay, contours, -1, color, 2)

        output_path = os.path.join(vis_output_dir, f"{case_id}_border.png")
        cv2.imwrite(output_path, overlay)

    print(f"边界可视化图已保存到: {vis_output_dir}")

def evaluate_predictions(pred_folder, trainer_info):
    """评估预测结果"""
    print(f"  正在计算评估指标...")
    
    try:
        # 检查预测文件是否存在
        pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.png')]
        if not pred_files:
            print(f"  ⚠️  警告: 预测目录中没有找到预测结果: {pred_folder}")
            return None
        
        # 检查标签文件是否存在
        if not os.path.exists(TEST_LABELS):
            print(f"  ⚠️  警告: 测试标签目录不存在: {TEST_LABELS}")
            return None
        
        # 计算指标
        metrics = compute_metrics_on_folder(
            folder_ref=TEST_LABELS,
            folder_pred=pred_folder,
            output_file=f"{pred_folder}/summary.json",
            image_reader_writer=NaturalImage2DIO(),
            file_ending='.png',
            regions_or_labels=[1, 2, 3],
            ignore_label=None,
            num_processes=3,
            chill=True
        )
        
        # 提取关键指标
        results = {
            "trainer": trainer_info['name'],
            "fold": trainer_info['fold_name'],
            "checkpoint": trainer_info['checkpoint_name'],
            "timestamp": datetime.now().isoformat(),
            "model_type": detect_model_type_from_trainer(trainer_info['path']),
            "overall": {
                "dice": metrics.get("foreground_mean", {}).get("Dice", 0),
                "iou": metrics.get("foreground_mean", {}).get("IoU", 0)
            },
            "per_class": {}
        }
        
        # 提取每个类别的指标
        mean_metrics = metrics.get("mean", {})
        class_names = {1: "PbI2", 2: "ABO3", 3: "defect"}
        
        for class_id in [1, 2, 3]:
            class_name = class_names.get(class_id, f"class_{class_id}")
            class_metrics = mean_metrics.get(class_id, {})
            results["per_class"][f"{class_name} ({class_id})"] = {
                "dice": class_metrics.get("Dice", 0),
                "iou": class_metrics.get("IoU", 0)
            }
        
        print(f"  评估完成!")
        print(f"    平均 Dice: {results['overall']['dice']:.4f}")
        print(f"    平均 IoU: {results['overall']['iou']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ 评估失败: {str(e)}")
        return None

def save_comparison_results(all_results):
    """保存对比结果"""
    if not all_results:
        print("\n❌ 没有成功评估任何模型")
        return None
    
    # 过滤掉None结果
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        print("\n❌ 没有有效的评估结果")
        return None
    
    print(f"\n[3/3] 保存对比结果...")
    
    # 保存为JSON
    json_file = f"{OUTPUT_DIR}/comparison_results.json"
    with open(json_file, 'w') as f:
        json.dump(valid_results, f, indent=2)
    print(f"  JSON结果已保存: {json_file}")
    
    # 转换为DataFrame并保存为CSV
    df_data = []
    for result in valid_results:
        row = {
            "Trainer": result["trainer"],
            "Model Type": result.get("model_type", "Unknown"),
            "Fold": result["fold"],
            "Checkpoint": result["checkpoint"],
            "Avg Dice": result["overall"]["dice"],
            "Avg IoU": result["overall"]["iou"]
        }
        
        # 添加每个类别的Dice
        for class_key, class_data in result["per_class"].items():
            class_name = class_key.split(" (")[0]  # 提取类别名称
            row[f"{class_name} Dice"] = class_data["dice"]
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # 按平均Dice排序
    df = df.sort_values("Avg Dice", ascending=False)
    
    csv_file = f"{OUTPUT_DIR}/comparison_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"  CSV结果已保存: {csv_file}")
    
    # 打印排名
    print(f"\n{'='*100}")
    print("模型性能排名 (按平均Dice排序):")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    
    return df

# ========== 主流程 ==========
def main():
    print(f"{'='*80}")
    print("Dataset117_Perovskite 测试集对比评估 (优化版)")
    print(f"{'='*80}")
    print(f"测试集路径: {TEST_IMAGES}")
    print(f"真实标签路径: {TEST_LABELS}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"默认 Fold: {FOLD}")
    print(f"{'='*80}")
    
    # 检查必要目录是否存在
    if not os.path.exists(TEST_IMAGES):
        print(f"❌ 错误: 测试图像目录不存在: {TEST_IMAGES}")
        return
    
    if not os.path.exists(NNUNET_RESULTS):
        print(f"❌ 错误: nnU-Net结果目录不存在: {NNUNET_RESULTS}")
        return
    
    # 1. 获取所有trainer
    print("\n[1/3] 正在发现所有trainer模型...")
    trainers = get_all_trainers()
    
    if not trainers:
        print("❌ 错误: 没有找到任何可用的trainer模型!")
        print(f"请检查目录: {NNUNET_RESULTS}/{DATASET_NAME}")
        return
    
    print(f"发现 {len(trainers)} 个trainer模型:")
    for i, trainer in enumerate(trainers, 1):
        print(f"  {i}. {trainer['name']}")
        print(f"     - Fold: {trainer['fold_name']}")
        print(f"     - Checkpoint: {trainer['checkpoint_name']}")
        
        # 检测模型类型
        model_type = detect_model_type_from_trainer(trainer['path'])
        print(f"     - 模型类型: {model_type}")
    
    # 2. 对每个trainer进行评估
    print(f"\n[2/3] 开始评估每个trainer...")
    all_results = []
    successful_count = 0
    failed_count = 0
    
    for trainer in tqdm(trainers, desc="评估进度"):
        # 创建该trainer的输出目录
        pred_folder = f"{OUTPUT_DIR}/{trainer['name']}_predictions"
        
        # 预测
        success = predict_with_trainer(trainer, pred_folder)
        
        if success:
            # 评估
            result = evaluate_predictions(pred_folder, trainer)
            if result:
                all_results.append(result)
                successful_count += 1
                
                # 生成错误可视化图
                visualize_errors_for_trainer(trainer['name'], pred_folder)
                # 生成预测边界可视化图
                visualize_borders_for_trainer(trainer['name'], pred_folder)
            else:
                all_results.append(None)
                failed_count += 1
        else:
            print(f"  跳过 {trainer['name']} 的评估")
            all_results.append(None)
            failed_count += 1
    
    # 统计结果
    print(f"\n评估统计:")
    print(f"  成功: {successful_count} 个")
    print(f"  失败: {failed_count} 个")
    
    # 3. 保存对比结果
    df = save_comparison_results(all_results)
    
    if df is not None:
        print(f"\n✅ 评估完成!")
        print(f"结果已保存到: {OUTPUT_DIR}")
        print(f"  - comparison_results.json: 详细结果")
        print(f"  - comparison_results.csv: CSV表格")
        
        # 显示最佳模型
        best_trainer = df.iloc[0]
        print(f"\n🏆 最佳模型: {best_trainer['Trainer']}")
        print(f"   模型类型: {best_trainer['Model Type']}")
        print(f"   平均 Dice: {best_trainer['Avg Dice']:.4f}")
        print(f"   平均 IoU: {best_trainer['Avg IoU']:.4f}")
    else:
        print("\n❌ 没有成功评估任何模型")

if __name__ == "__main__":
    main()