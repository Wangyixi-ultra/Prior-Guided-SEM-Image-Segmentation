#!/usr/bin/env python3
"""
对Dataset114(umamba_pero) chen@chen-Rack-Server:~/seg6$ /home/chen/anaconda3/envs/umamba_pero/bin/python /home/chen/seg6/113compare_all_trainers_testset_3.py
  File "/home/chen/seg6/113compare_all_trainers_testset_3.py", line 680
    global TEST_IMAGES
    ^^^^^^^^^^^^^^^^^^
SyntaxError: name 'TEST_IMAGES' is used prior to global declaration_Perovskite数据集下的所有trainer进行测试集对比评估
自动发现所有trainer模型，对测试集进行预测并计算指标
添加可视化功能：橙色表示ABO3，绿色表示PbI2，红色表示defect,以及对比
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
import shutil
from ultralytics import YOLO

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

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

# Apply the monkey patch
data_iterators.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous
# Also patch the function in predict_from_raw_data module where it is imported
import nnunetv2.inference.predict_from_raw_data

# FORCE patch the module attribute
nnunetv2.inference.predict_from_raw_data.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous

# Verify patch
print(f"DEBUG: Monkey patch applied to data_iterators.preprocessing_iterator_fromfiles: {data_iterators.preprocessing_iterator_fromfiles}")
print(f"DEBUG: Monkey patch applied to predict_from_raw_data.preprocessing_iterator_fromfiles: {nnunetv2.inference.predict_from_raw_data.preprocessing_iterator_fromfiles}")
# ==================================================================================================

# ========== 配置 ==========
DATASET_NAME = "Dataset114_Perovskite"
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

# 类别配置
CLASS_COLORS = {
    1: (0, 255, 0),    # PbI2 - 绿色
    2: (0, 165, 255),  # ABO3 - 橙色 (BGR格式)
    3: (0, 0, 255)     # defect - 红色
}

CLASS_NAMES = {
    1: "PbI2",
    2: "ABO3", 
    3: "defect"
}

# ========== 辅助函数 ==========
def get_all_trainers():
    """获取Dataset114_Perovskite下所有的trainer模型"""
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

def prepare_yolo_input(src_dir, dst_dir):
    """
    读取原始测试集图像，添加YOLO预测通道，保存到目标目录
    """
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        # YOLO模型路径
        yolo_weights = '/home/chen/seg6/perovskite_grains_opt/train29/weights/best.pt'
        # 映射关系: YOLO class -> Mask value
        # 0: ABO3 -> 2
        # 1: PbI2 -> 1
        # 2: defect -> 3
        yolo_map = {0: 2, 1: 1, 2: 3}
        
        print(f"  加载YOLO模型: {yolo_weights}")
        try:
            model = YOLO(yolo_weights)
        except Exception as e:
            print(f"  ⚠️ 无法加载YOLO模型: {e}")
            return False
        
        # 获取所有_0000.png文件
        img_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_0000.png')])
        
        for img_file in tqdm(img_files, desc="生成YOLO通道"):
            case_id = img_file.replace('_0000.png', '')
            src_img_path = os.path.join(src_dir, img_file)
            
            # 1. 复制 _0000.png
            dst_img_path_0 = os.path.join(dst_dir, img_file)
            if not os.path.exists(dst_img_path_0):
                shutil.copy2(src_img_path, dst_img_path_0)
                
            # 2. 生成 _0001.png
            dst_img_path_1 = os.path.join(dst_dir, f"{case_id}_0001.png")
            
            # 运行YOLO预测
            try:
                results = model(src_img_path, verbose=False)
                top1 = results[0].probs.top1
                fill_val = yolo_map.get(top1, 0)
            except Exception as e:
                print(f"  推断失败 {case_id}: {e}")
                fill_val = 0
            
            # 创建mask
            img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            mask = np.full((h, w), fill_val, dtype=np.uint8)
            cv2.imwrite(dst_img_path_1, mask)
            
        return True
    
    except Exception as e:
        print(f"  准备YOLO输入时出错: {e}")
        # traceback.print_exc()
        return False

# ADDED FUNCTION to detect channel name format in dataset
def get_channel_format(dataset_json):
    """
    Check dataset.json to see if we should use _0000 suffix.
    """
    # Assuming standard nnU-Net, it expects _0000 if multiple channels or configured
    # The image reader handles this, but file existence check is what matters.
    # In compare logic, we create _0000 and _0001.
    return True

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
            num_processes_preprocessing=1,  # 减少预处理进程数以避免资源竞争
            num_processes_segmentation_export=1,  # 减少分割导出进程数
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
                        num_processes_preprocessing=1,
                        num_processes_segmentation_export=1,
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
        elif "expected input" in error_str and "channels" in error_str:
            print(f"  ⚠️  检测到输入通道数不匹配错误: {error_str}")
            print(f"  尝试调整输入数据的通道数...")
            
            try:
                # 尝试重新初始化预测器，这次使用更保守的设置
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
                
                # 加载模型，但这次使用更多进程控制
                predictor.initialize_from_trained_model_folder(
                    trainer_info["path"],
                    use_folds=use_folds,
                    checkpoint_name=trainer_info["checkpoint_name"]
                )
                
                # 创建输出目录
                os.makedirs(output_folder, exist_ok=True)
                
                # 尝试单线程处理
                print(f"  使用单线程模式进行预测...")
                predictor.predict_from_files(
                    TEST_IMAGES,
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
            except Exception as e3:
                print(f"  ❌ 单线程模式仍然失败: {str(e3)}")
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

def visualize_prediction_mask(pred_mask, raw_image_path, output_path):
    """
    将预测mask转换为边界轮廓可视化图像
    橙色表示ABO3，绿色表示PbI2，红色表示defect
    只绘制边界轮廓，不显示填充区域
    """
    # 读取原始图像
    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
    if raw_image is None:
        print(f"  警告: 无法读取原始图像: {raw_image_path}")
        return False
    
    # 将灰度图转换为三通道BGR图像
    if len(raw_image.shape) == 2:
        overlay = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = raw_image.copy()
    
    # 为每个类别绘制边界轮廓
    for class_id, color in CLASS_COLORS.items():
        # 创建该类别的mask
        class_mask = (pred_mask == class_id).astype(np.uint8)
        
        if np.any(class_mask):
            # 找到轮廓
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 只绘制轮廓，不绘制填充区域
            if contours:
                cv2.drawContours(overlay, contours, -1, color, 2)  # 线宽为2
    
    # 保存可视化结果
    cv2.imwrite(output_path, overlay)
    return True

def visualize_errors_for_trainer(trainer_name, pred_folder):
    """为指定训练器的预测结果生成错误可视化图"""
    print(f"正在生成 {trainer_name} 的错误可视化图...")
    
    # 创建可视化输出目录
    vis_output_dir = f"{OUTPUT_DIR}/{trainer_name}_error_visualizations"
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 创建预测可视化输出目录
    pred_vis_output_dir = f"{OUTPUT_DIR}/{trainer_name}_prediction_visualizations"
    os.makedirs(pred_vis_output_dir, exist_ok=True)
    
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
            
            # 保存错误可视化结果
            error_output_path = os.path.join(vis_output_dir, f"{case_name}_error_map.png")
            cv2.imwrite(error_output_path, overlay)
            
            # 生成预测结果的可视化（边界轮廓图）
            pred_vis_path = os.path.join(pred_vis_output_dir, f"{case_name}_prediction_vis.png")
            visualize_prediction_mask(pred, raw_img_path, pred_vis_path)
    
    print(f"错误可视化图已保存到: {vis_output_dir}")
    print(f"预测可视化图已保存到: {pred_vis_output_dir}")

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
        
        for class_id in [1, 2, 3]:
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
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
    global TEST_IMAGES
    print(f"{'='*80}")
    print("Dataset105_Perovskite 测试集对比评估 (优化版)")
    print("可视化: 橙色=ABO3, 绿色=PbI2, 红色=defect")
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
        
    # 0. 准备测试数据 (添加YOLO通道)
    print("\n[0/3] 正在准备YOLO辅助通道数据...")
    temp_test_dir = os.path.join(OUTPUT_DIR, "temp_test_yolo")
    if prepare_yolo_input(TEST_IMAGES, temp_test_dir):
        TEST_IMAGES = temp_test_dir
        print(f"✅ 数据准备完成，使用临时目录: {TEST_IMAGES}")
    else:
        print("⚠️ 数据准备失败，将尝试使用原始数据目录")
    
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
                
                # 生成错误可视化图和预测可视化图
                visualize_errors_for_trainer(trainer['name'], pred_folder)
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
        print(f"  - *_prediction_visualizations/: 预测可视化图 (彩色mask)")
        print(f"  - *_error_visualizations/: 错误可视化图")
        
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
