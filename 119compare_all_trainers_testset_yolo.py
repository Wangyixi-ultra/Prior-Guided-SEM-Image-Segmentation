#!/usr/bin/env python3
"""
对Dataset122_Perovskite_YOLO数据集下的YoloInstance训练器进行测试集对比评估
适配双通道输入 (SEM + YOLO) - 简化版（无OOM处理）
"""

import os
import sys
# 添加U-Mamba源码路径到sys.path
sys.path.insert(0, "/home/chen/seg6/U-Mamba/umamba")

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

# 导入argparse模块来处理命令行参数
import argparse

# 导入nnunet相关模块
try:
    from nnunetv2.paths import nnUNet_raw, nnUNet_results
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
except ImportError as e:
    print(f"nnunetv2模块导入错误: {e}")
    nnUNet_raw = "/not/available"
    nnUNet_results = "/not/available"
    nnUNetPredictor = None
    recursive_find_python_class = None
    PlansManager = None
    ConfigurationManager = None
    compute_metrics_on_folder = None

# 尝试导入图像读写类
try:
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    class NaturalImage2DIO(SimpleITKIO):
        pass
except ImportError:
    try:
        from nnunetv2.imageio.natural_image2d_reader_writer import NaturalImage2DIO
    except ImportError:
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

# MONKEY PATCH: 避免多进程错误
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

if PlansManager is not None:
    data_iterators.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous
    import nnunetv2.inference.predict_from_raw_data
    nnunetv2.inference.predict_from_raw_data.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous
else:
    print("DEBUG: Skipping monkey patch due to missing nnunetv2 modules")

# ========== 配置 ==========
DATASET_NAME = "Dataset122_Perovskite"  # 修正数据集名称
NNUNET_RAW = "/home/chen/seg6/U-Mamba/data/nnUNet_raw"
NNUNET_RESULTS = "/home/chen/seg6/U-Mamba/data/nnUNet_results"

# 测试集路径
RAW_TEST_IMAGES = f"{NNUNET_RAW}/{DATASET_NAME}/imagesTs"
TEST_LABELS = f"{NNUNET_RAW}/{DATASET_NAME}/labelsTs"

# 输出结果目录
OUTPUT_DIR = f"{NNUNET_RESULTS}/{DATASET_NAME}/testset_comparison_yolo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预测参数
FOLD = 0
checkpoint_priority = [
    "checkpoint_final.pth",
    "checkpoint_best.pth",
    "checkpoint_latest.pth"
]

# 类别配置
CLASS_COLORS = {
    1: (0, 255, 0),    # PbI2 - 绿色
    2: (0, 165, 255)   # ABO3 - 橙色 (BGR格式)
}

CLASS_NAMES = {
    1: "PbI2",
    2: "ABO3"
}

VALID_CLASS_IDS = [1, 2]

# 要对比的训练器列表
TARGET_TRAINERS = [
    "nnUNetTrainerUMambaBotDualChannelCBAM",
   # "nnUNetTrainerUMambaBotYoloInstanceSpatial", 
    "nnUNetTrainerUMambaBotActiveContourDualChannelCBAM"
   # "nnUNetTrainerUMambaBotYoloInstance",
   # "nnUNetTrainerUMambaBotDualChannelopt", #best
   # "nnUNetTrainerUMambaBotEdgeAttentionDualChannel"
]

# ========== 辅助函数 ==========
def get_yolo_trainers():
    """获取Dataset122_Perovskite下所有的YoloInstance trainer模型"""  # 修正注释
    dataset_dir = Path(f"{NNUNET_RESULTS}/{DATASET_NAME}")
    trainers = []
    
    if not dataset_dir.exists():
        print(f"警告: 数据集目录不存在: {dataset_dir}")
        return trainers
    
    for item in dataset_dir.iterdir():
        if item.is_dir():
            trainer_name = item.name
            is_target = any(target in trainer_name for target in TARGET_TRAINERS)
            if not is_target:
                continue
                
            trainer_path = item
            
            # 查找最优的fold和checkpoint组合
            best_fold = None
            best_checkpoint = None
            best_score = -1
            
            fold_priority = [f"fold_{FOLD}", "fold_all"]
            
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
                    "checkpoint_name": best_checkpoint.name,
                    "fusion_type": extract_fusion_type(trainer_name)
                })
    
    return sorted(trainers, key=lambda x: x["name"])

def extract_fusion_type(trainer_name):
    """从训练器名称中提取融合类型"""
    if "Weighted" in trainer_name:
        return "weighted"
    elif "Spatial" in trainer_name:
        return "spatial"
    elif "Adaptive" in trainer_name:
        return "yolo_adaptive"
    else:
        return "simple"

def load_trainer_class(trainer_name):
    """动态加载trainer类"""
    try:
        custom_trainer_path = "/home/chen/seg6/U-Mamba/umamba/nnunetv2/training/nnUNetTrainer"
        if os.path.exists(custom_trainer_path) and recursive_find_python_class is not None:
            trainer_class = recursive_find_python_class(
                [custom_trainer_path],
                trainer_name,
                "nnunetv2.training.nnUNetTrainer"
            )
            if trainer_class:
                return trainer_class
        
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

def predict_with_trainer(trainer_info, output_folder, input_folder, gpu_id=0):
    """使用指定trainer对测试集进行预测（简化版）"""
    print(f"\n{'='*60}")
    print(f"正在评估: {trainer_info['name']}")
    print(f"  融合类型: {trainer_info['fusion_type']}")
    print(f"  Fold: {trainer_info['fold_name']}")
    print(f"  Checkpoint: {trainer_info['checkpoint_name']}")
    print(f"  GPU ID: {gpu_id}")
    print(f"{'='*60}")
    
    if nnUNetPredictor is None:
        print("  ❌ nnUNetPredictor 不可用，跳过预测")
        return False
    
    try:
        # 设置GPU设备
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
        print(f"  使用设备: {device}")
        
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
        
        # 2. 加载Trainer Class
        trainer_class = None
        potential_class_name = trainer_info['name'].split('__')[0]
        if "nnUNetTrainer" in potential_class_name:
             trainer_class = load_trainer_class(potential_class_name)

        # 3. 初始化预测器（使用指定GPU）
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        # 4. 加载模型权重和配置
        predictor.initialize_from_trained_model_folder(
            trainer_info["path"],
            use_folds=use_folds,
            checkpoint_name=trainer_info["checkpoint_name"]
        )

        # 5. 检测模型期望的输入通道数（双通道）
        if hasattr(predictor.network, 'encoder') and hasattr(predictor.network.encoder, 'stem'):
             num_input_channels = predictor.network.encoder.stem[0].conv1.in_channels
        elif hasattr(predictor.network, 'input_channels'):
             num_input_channels = predictor.network.input_channels
        else:
             try:
                 params = list(predictor.network.parameters())
                 if len(params) > 0:
                      num_input_channels = params[0].shape[1]
                 else:
                      num_input_channels = 2  # 双通道默认
             except:
                  num_input_channels = 2  # 双通道默认
        
        print(f"  ℹ️ 模型期望输入通道数: {num_input_channels}")
        assert num_input_channels == 2, f"期望双通道输入，但模型需要 {num_input_channels} 通道"

        # 6. 构建双通道输入文件列表
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

        # 7. 执行预测（无OOM重试逻辑）
        os.makedirs(output_folder, exist_ok=True)
        print(f"  开始对 {len(list_of_lists)} 个样本进行预测...")

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
    """评估预测结果"""
    if compute_metrics_on_folder is None:
        return None
    
    try:
        pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.png')]
        if not pred_files:
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
        dice_sum = 0
        iou_sum = 0
        
        valid_classes = VALID_CLASS_IDS
        
        for class_id in valid_classes:
            class_metrics = mean_metrics.get(class_id) or mean_metrics.get(str(class_id), {})
            dice = class_metrics.get("Dice", 0)
            iou = class_metrics.get("IoU", 0)
            
            dice_sum += dice
            iou_sum += iou
        
        avg_dice = dice_sum / len(valid_classes) if valid_classes else 0.0
        avg_iou = iou_sum / len(valid_classes) if valid_classes else 0.0

        results = {
            "trainer": trainer_info['name'],
            "fusion_type": trainer_info['fusion_type'],
            "fold": trainer_info['fold_name'],
            "checkpoint": trainer_info['checkpoint_name'],
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "dice": avg_dice,
                "iou": avg_iou
            },
            "per_class": {}
        }
        
        for class_id in valid_classes:
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            class_metrics = mean_metrics.get(class_id, {})
            if not class_metrics:
                class_metrics = mean_metrics.get(str(class_id), {})
                
            dice_val = class_metrics.get("Dice", 0)
            iou_val = class_metrics.get("IoU", 0)
            
            results["per_class"][f"{class_name} ({class_id})"] = {
                "dice": dice_val,
                "iou": iou_val
            }
        
        print(f"  平均 Dice: {results['overall']['dice']:.4f}")
        print(f"  平均 IoU: {results['overall']['iou']:.4f}")
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
    
    with open(f"{OUTPUT_DIR}/comparison_results_yolo.json", 'w') as f:
        json.dump(valid_results, f, indent=2)
    
    df_data = []
    for result in valid_results:
        row = {
            "Trainer": result["trainer"],
            "Fusion Type": result["fusion_type"],
            "Fold": result["fold"],
            "Checkpoint": result["checkpoint"],
            "Avg Dice": result["overall"]["dice"],
            "Avg IoU": result["overall"]["iou"]
        }
        for class_key, class_data in result["per_class"].items():
            class_name = class_key.split(" (")[0]
            row[f"{class_name} Dice"] = class_data["dice"]
            row[f"{class_name} IoU"] = class_data["iou"]
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df = df.sort_values("Avg Dice", ascending=False)
    
    csv_file = f"{OUTPUT_DIR}/comparison_results_yolo.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*100}")
    print("模型性能排名 (按平均Dice排序):")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    return df

def main(gpu_id=0):
    print(f"{'='*80}")
    print("Dataset122_Perovskite 测试集对比评估 (双通道) - 简化版")
    print(f"{'='*80}")
    
    if not os.path.exists(RAW_TEST_IMAGES) or not os.path.exists(NNUNET_RESULTS):
        print(f"❌ 关键目录不存在，请检查路径。")
        print(f"   nnUNet_raw: {NNUNET_RAW}")
        print(f"   nnUNet_results: {NNUNET_RESULTS}")
        return
        
    current_test_images = RAW_TEST_IMAGES
    
    print(f"\n测试集路径: {current_test_images}")
    print(f"结果输出路径: {OUTPUT_DIR}")
    print(f"使用GPU ID: {gpu_id}")
    
    # 1. 获取所有YoloInstance训练器
    print("\n[1/3] 正在发现所有YoloInstance trainer模型...")
    trainers = get_yolo_trainers()
    if not trainers:
        print("❌ 未找到YoloInstance trainer模型")
        print("请确保训练已完成，并且模型保存在:")
        print(f"   {NNUNET_RESULTS}/{DATASET_NAME}/")
        return
    
    print(f"发现 {len(trainers)} 个YoloInstance trainer模型:")
    for i, t in enumerate(trainers, 1):
        print(f"  {i}. {t['name']} ({t['fusion_type']})")
    
    # 2. 对每个trainer进行评估
    print(f"\n[2/3] 开始评估每个trainer...")
    all_results = []
    
    for trainer in tqdm(trainers, desc="评估进度"):
        pred_folder = f"{OUTPUT_DIR}/{trainer['name']}_predictions"
        
        success = predict_with_trainer(trainer, pred_folder, current_test_images, gpu_id)
        
        if success:
            result = evaluate_predictions(pred_folder, trainer)
            if result:
                all_results.append(result)
                visualize_errors_for_trainer(trainer['name'], pred_folder, current_test_images)
            else:
                all_results.append(None)
        else:
            all_results.append(None)
    
    # 3. 保存对比结果
    df = save_comparison_results(all_results)
    
    if df is not None:
        print(f"\n✅ 全部完成!")
        print(f"结果已保存到: {OUTPUT_DIR}")
        print(f"  - comparison_results_yolo.json")
        print(f"  - comparison_results_yolo.csv")
        print(f"  - 各trainer的预测结果和可视化图")
        
        best_trainer = df.iloc[0]
        print(f"\n🏆 最佳模型: {best_trainer['Trainer']}")
        print(f"   融合类型: {best_trainer['Fusion Type']}")
        print(f"   平均 Dice: {best_trainer['Avg Dice']:.4f}")
        print(f"   平均 IoU: {best_trainer['Avg IoU']:.4f}")
    else:
        print("\n❌ 没有有效的评估结果")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset122 Perovskite YOLO Testset Comparison')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    args = parser.parse_args()
    
    main(gpu_id=args.gpu)
