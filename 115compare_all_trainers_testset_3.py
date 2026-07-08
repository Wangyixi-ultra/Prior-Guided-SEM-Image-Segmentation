#!/usr/bin/env python3
"""
对Dataset115_Perovskite数据集下的所有trainer进行测试集对比评估
自动发现所有trainer模型，对测试集进行预测并计算指标
优化逻辑：支持自动检测模型输入通道数，正确处理单通道/双通道(YOLO辅助)预测
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
from ultralytics import YOLO
import ultralytics.nn.tasks
from ultralytics.nn.modules import CBAM

# 注册 CBAM 模块 (确保全局生效)
ultralytics.nn.tasks.CBAM = CBAM

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
DATASET_NAME = "Dataset115_Perovskite"
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
    """获取Dataset115_Perovskite下所有的trainer模型"""
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

def prepare_yolo_input(src_dir, dst_dir):
    """
    读取原始测试集图像，添加YOLO预测通道，保存到目标目录
    """
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        # YOLO模型路径
        detector_path = '/home/chen/runs/detect/train18/weights/best.pt'
        classifier_path = '/home/chen/seg6/perovskite_grains_opt/yolo_cbam_s_128/weights/best.pt'
        
        # 标签映射
        LABEL_MAP = {
            'background': 0, 'pbi2': 1, 'abo3': 2, 'defect': 3
        }

        # 检查是否已经处理过 (简单检查)
        src_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_0000.png')])
        dst_files_0 = sorted([f for f in os.listdir(dst_dir) if f.endswith('_0000.png')])
        dst_files_1 = sorted([f for f in os.listdir(dst_dir) if f.endswith('_0001.png')])
        
        if len(src_files) > 0 and len(dst_files_0) == len(src_files) and len(dst_files_1) == len(src_files):
             print(f"  检测到目标目录已包含完整数据，跳过YOLO生成步骤。")
             return True

        print(f"  加载检测器: {detector_path}")
        try:
            detector = YOLO(detector_path)
        except Exception as e:
            print(f"  ⚠️ 无法加载检测器: {e}")
            return False
            
        print(f"  加载分类器: {classifier_path}")
        try:
            classifier = YOLO(classifier_path)
        except Exception as e:
            print(f"  ⚠️ 无法加载分类器: {e}")
            return False
        
        for img_file in tqdm(src_files, desc="生成YOLO通道"):
            case_id = img_file.replace('_0000.png', '')
            src_img_path = os.path.join(src_dir, img_file)
            
            # 1. 复制 _0000.png
            dst_img_path_0 = os.path.join(dst_dir, img_file)
            if not os.path.exists(dst_img_path_0):
                shutil.copy2(src_img_path, dst_img_path_0)
                
            # 2. 生成 _0001.png (mask)
            dst_img_path_1 = os.path.join(dst_dir, f"{case_id}_0001.png")
            
            # 如果已存在，可以选择跳过
            if os.path.exists(dst_img_path_1):
                continue

            # 读取原始图像
            img = cv2.imread(src_img_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # 创建 mask 通道
            mask = np.zeros((h, w), dtype=np.uint8)

            # 1. 检测
            det_results = detector(img, verbose=False, conf=0.25)
            
            boxes = []
            if len(det_results) > 0:
                boxes = det_results[0].boxes

            # 2. 分类并填充
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                cls_results = classifier(crop, verbose=False)
                
                if len(cls_results) > 0:
                    top1_idx = cls_results[0].probs.top1
                    class_name = cls_results[0].names[top1_idx]
                    
                    def normalize_name(name):
                        name = name.lower()
                        if 'pbi' in name: return 'pbi2'
                        if 'abo' in name: return 'abo3'
                        return name
                        
                    norm_name = normalize_name(class_name)
                    label_id = LABEL_MAP.get(norm_name, 0)
                    
                    if label_id > 0:
                        mask[y1:y2, x1:x2] = label_id

            cv2.imwrite(dst_img_path_1, mask)
            
        return True
    
    except Exception as e:
        print(f"  准备YOLO输入时出错: {e}")
        traceback.print_exc()
        return False

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
        # 这里会自动加载 plans.json 和 dataset.json
        # 注意: initialize_from_trained_model_folder 不接受 nnunet_trainer_class 参数
        predictor.initialize_from_trained_model_folder(
            trainer_info["path"],
            use_folds=use_folds,
            checkpoint_name=trainer_info["checkpoint_name"]
            # nnunet_trainer_class=trainer_class # 这个参数在新版/此版本nnUNetPredictor中可能不支持，移除
        )

        # 5. 【关键优化】检测模型期望的输入通道数
        # num_input_channels = predictor.configuration_manager.num_input_channels # Attribute Error 
        # 可以尝试从 plans_manager 获取，或者查看 list_of_parameters
        # 简单起见，我们假设 115 数据集是单通道的，但是因为是双通道实验，这里我们需要根据 trainer 类型判断
        
        # 尝试从 trainer_info 推断，或者从 plans.json 读取
        # 实际上，我们可以查看 predictor.list_of_parameters[0][0].shape[1] 
        # 但是权重可能已经加载到了模型中，模型本身有 input_channels 属性吗？
        # 我们这里做一个安全的 fallback
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
        # 无论模型是一通道还是两通道，我们都从input_folder中提取正确的文件组合
        # input_folder (即temp_test_yolo) 包含了 _0000.png, _0001.png 等
        
        # 找到所有case id (基于 _0000.png)
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
            if num_input_channels > 1:
                print("     提示: 模型需要多通道输入，请确保数据准备步骤已生成辅助通道。")
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
    """可视化: 橙色=ABO3, 绿色=PbI2, 红色=defect"""
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
            
            for class_id in [1, 2, 3]:
                gt_mask = (gt == class_id).astype(np.uint8) * 255
                pred_mask = (pred == class_id).astype(np.uint8) * 255
                
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
        
        # 定义Reader（尝试修复评估时的读写器问题）
        try:
            from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
            image_reader_writer = SimpleITKIO()
        except ImportError:
            image_reader_writer = None # nnU-Net会尝试自动推断

        metrics = compute_metrics_on_folder(
            folder_ref=TEST_LABELS,
            folder_pred=pred_folder,
            output_file=f"{pred_folder}/summary.json",
            image_reader_writer=image_reader_writer,
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
            "model_type": detect_model_type_from_trainer(trainer_info['path']),
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
            results["per_class"][f"{class_name} ({class_id})"] = {
                "dice": class_metrics.get("Dice", 0),
                "iou": class_metrics.get("IoU", 0)
            }
        
        print(f"  平均 Dice: {results['overall']['dice']:.4f}")
        return results
        
    except Exception as e:
        print(f"  ❌ 评估失败: {str(e)}")
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
        row = {
            "Trainer": result["trainer"],
            "Model Type": result.get("model_type", "Unknown"),
            "Fold": result["fold"],
            "Checkpoint": result["checkpoint"],
            "Avg Dice": result["overall"]["dice"],
            "Avg IoU": result["overall"]["iou"]
        }
        for class_key, class_data in result["per_class"].items():
            class_name = class_key.split(" (")[0]
            row[f"{class_name} Dice"] = class_data["dice"]
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df = df.sort_values("Avg Dice", ascending=False)
    
    csv_file = f"{OUTPUT_DIR}/comparison_results.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*100}")
    print("模型性能排名 (按平均Dice排序):")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    return df

def main():
    print(f"{'='*80}")
    print("Dataset115_Perovskite 测试集对比评估 (优化版)")
    print(f"{'='*80}")
    
    if not os.path.exists(RAW_TEST_IMAGES) or not os.path.exists(NNUNET_RESULTS):
        print(f"❌ 关键目录不存在，请检查路径。")
        return
        
    # 0. 准备测试数据 (添加YOLO通道)
    # 统一定义一个数据准备目录，包含所有输入通道
    prepared_data_dir = os.path.join(OUTPUT_DIR, "test_input_prepared")
    
    print("\n[0/3] 正在准备YOLO辅助通道数据...")
    if prepare_yolo_input(RAW_TEST_IMAGES, prepared_data_dir):
        print(f"✅ 数据准备完成: {prepared_data_dir}")
        current_test_images = prepared_data_dir
    else:
        print("⚠️ 数据准备遇到问题，将尝试使用原始数据（可能导致双通道模型报错）")
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
        
        # 传递 current_test_images，不再依赖全局变量
        success = predict_with_trainer(trainer, pred_folder, current_test_images)
        
        if success:
            result = evaluate_predictions(pred_folder, trainer)
            if result:
                all_results.append(result)
                # 可视化需要原始图像，使用 current_test_images 确保能找到 _0000.png
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
