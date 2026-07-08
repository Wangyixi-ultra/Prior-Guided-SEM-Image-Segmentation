#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简 nnUNet 预测流水线（无 YOLO 单通道版本）
在文件内部即可改路径，无需命令行参数
"""

import cv2, subprocess, shutil, json, os
from pathlib import Path
import numpy as np
from imageio.v2 import imread

# Custom Imports for bypassing multiprocessing
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference import data_iterators
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from typing import List, Union
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

# Define synchronous iterator to bypass multiprocessing issues
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

# ========== 1. 用户只改这里 ==========
INPUT_DIR   = Path('/home/chen/seg6/predict_no_label/experiment/in/experiment')  # 原始图像
OUTPUT_DIR  = Path('/home/chen/seg6/predict_no_label/experiment/in/2')      # nnUNet 预测结果
BORDER_DIR  = Path('/home/chen/seg6/predict_no_label/experiment/in/2')   # 轮廓叠加图
JSON_DIR    = Path('/home/chen/seg6/predict_no_label/experiment/in/2')                 # JSON标注文件输出目录
# nnUNet 参数
DATASET_ID  = 123
CONFIG      = '2d'
FOLD        = 0
TRAINER     = 'nnUNetTrainerUMambaBotActiveContourSemBoost' # 单通道训练器
CHECKPOINT  = 'checkpoint_best.pth'
# ========== 2. 以下内容勿动 ========== 

CLASS_COLOR = {1: (0, 140, 255), 2: (0, 255, 0), 3: (255, 0, 255)}
CLASS_LABELS = {1: "PbI₂", 2: "ABO₃", 3: "defect"}  # 类别标签映射

def find_imgs(p):
    return sorted([i for i in Path(p).iterdir()
                   if i.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}])

# ---------- 替换这两个函数 ----------
def to1ch_uint8(img):
    """保证输出单通道 uint8"""
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def prep_input(in_dir, tmp_dir):
    """保存临时输入，并返回 case 名到原图路径的映射"""
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    name_map = {}
    for idx, f in enumerate(find_imgs(in_dir)):
        # Read original
        img_original = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img_original is None:
            try:
                img_original = imread(str(f))  # 兼容.tif等特殊格式
            except:
                pass
                
        if img_original is None:
            print(f"无法读取文件: {f}")
            continue
            
        case_name = f'case{idx:03d}'
        
        # Single channel: Grayscale Image only (no YOLO Channel 1)
        img_0000 = to1ch_uint8(img_original)
        cv2.imwrite(str(tmp_dir / f'{case_name}_0000.png'), img_0000)

        name_map[case_name] = f
    return tmp_dir, name_map

def predict(tmp_dir, out_dir, model_folder):
    """
    运行nnUNet预测 (使用Python API，单线程预处理)
    """
    print(f"Predicting using model in: {model_folder}")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Threading settings moved to main/global to avoid RuntimeError on multiple calls

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False, # Disable TTA
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(FOLD,),
        checkpoint_name=CHECKPOINT
    )
    
    predictor.predict_from_files(
        str(tmp_dir),
        str(out_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

def draw_contour(tmp_dir, out_dir, border_dir):
    border_dir.mkdir(exist_ok=True,parents=True)
    for m in sorted(out_dir.glob('*.png')):
        name = m.stem
        img = cv2.imread(str(tmp_dir/f'{name}_0000.png'), cv2.IMREAD_COLOR)
        mask= imread(m).astype(np.uint8)
        base= img.copy()
        for cls in np.unique(mask):
            if cls==0:continue
            bin = ((mask==cls)*255).astype(np.uint8)
            cnt,_=cv2.findContours(bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base,cnt,-1,CLASS_COLOR.get(cls,(255,255,255)),2)
        cv2.imwrite(str(border_dir/f'{name}_contour.png'), base)

def create_labelme_json(original_img_path, mask, output_dir):
    """根据预测mask生成LabelMe格式的JSON文件"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 读取原始图像获取尺寸信息
    img = cv2.imread(str(original_img_path))
    if img is None:
        try:
            # 兼容.tif等特殊格式
            img = imread(str(original_img_path))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            # imageio 读取彩色为 RGB，需转为 BGR 以兼容后续处理
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"无法读取原始图像: {original_img_path}, 错误: {e}")
            return False
            
    if img is None:
        print(f"无法读取原始图像: {original_img_path}")
        return False
    
    # 转换图像为RGB（OpenCV使用BGR）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 将图像编码为PNG格式的字节流
    success, encoded_img = cv2.imencode('.png', img_rgb)
    if not success:
        print(f"图像编码失败: {original_img_path}")
        return False
    
    # 将字节流转换为base64字符串
    import base64
    imageData = base64.b64encode(encoded_img).decode('utf-8')
    
    height, width = img.shape[:2]
    
    # 获取原始图像文件名（不含路径）
    original_name = original_img_path.stem
    
    # 构造JSON文件路径
    json_path = output_dir / f"{original_name}.json"
    
    # 初始化shapes列表
    shapes = []
    
    # 遍历每个类别（跳过背景0）
    for cls in np.unique(mask):
        if cls == 0:
            continue
            
        # 获取类别标签
        label = CLASS_LABELS.get(cls, f"class_{cls}")
        
        # 二值化mask
        bin_mask = ((mask == cls) * 255).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 为每个轮廓创建shape
        for contour in contours:
            # 将轮廓点转换为列表格式
            points = []
            for point in contour:
                x, y = point[0]
                points.append([float(x), float(y)])
            
            # 如果点数足够多，添加到shapes
            if len(points) >= 3:  # 多边形至少需要3个点
                shape = {
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None
                }
                shapes.append(shape)
    
    # 构造LabelMe格式的JSON数据
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": original_img_path.name,
        "imageData": imageData,  # 嵌入图像数据
        "imageHeight": height,
        "imageWidth": width
    }
    
    # 写入JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)
    
    print(f"已生成JSON标注文件: {json_path}")
    return True

def generate_json_annotations(input_dir, output_dir, json_output_dir, name_map=None):
    """批量生成JSON标注文件；优先使用预处理时记录的映射"""
    json_output_dir.mkdir(exist_ok=True, parents=True)
    
    # 获取所有预测结果文件
    mask_files = [f for f in sorted(output_dir.glob('*.png')) if not f.stem.endswith('_contour')]
    
    if not mask_files:
        print("未找到预测结果文件")
        return
    
    print(f"找到 {len(mask_files)} 个预测结果，开始生成JSON标注文件...")
    
    success_count = 0
    for mask_file in mask_files:
        name = mask_file.stem

        if name.endswith('_contour'):
            continue

        # 优先使用预处理阶段的映射，避免猜测
        if name_map and name in name_map:
            original_img_path = name_map[name]
            debug_files = [original_img_path]
        else:
            original_files = []
            original_files.extend(list(input_dir.glob(f"{name}.*")))
            original_files.extend(list(input_dir.glob(f"{name}_0000.*")))

            if name.startswith('case'):
                try:
                    num = int(name[4:])
                    actual_num = num + 4
                    original_files.extend(list(input_dir.glob(f"case{actual_num:03d}.*")))
                    original_files.extend(list(input_dir.glob(f"case{actual_num:03d}_0000.*")))
                except ValueError:
                    pass

            original_files = list(set(original_files))
            debug_files = original_files

            if not original_files:
                print(f"调试: 为 {mask_file.name} 找到 0 个可能的原始图像: []")
                print(f"警告: 未找到与 {mask_file.name} 对应的原始图像")
                continue

            original_img_path = original_files[0]

        print(f"调试: 为 {mask_file.name} 找到 {len(debug_files)} 个可能的原始图像: {[f.name for f in debug_files]}")
        
        # 读取mask
        mask = imread(mask_file).astype(np.uint8)
        
        # 生成JSON文件
        if create_labelme_json(original_img_path, mask, json_output_dir):
            success_count += 1
    
    print(f"完成！成功生成 {success_count}/{len(mask_files)} 个JSON标注文件")

def main():
    # 查找数据集目录
    nnunet_results = Path('/home/chen/seg6/U-Mamba/data/nnUNet_results')
    dataset_dirs = list(nnunet_results.glob(f'Dataset{DATASET_ID}_*'))
    if not dataset_dirs:
        print(f"在 {nnunet_results} 未找到 Dataset{DATASET_ID}")
        return
    dataset_dir = dataset_dirs[0]
    print(f"Using dataset: {dataset_dir}")
    
    tmp = INPUT_DIR.parent/'temp_nnUNet'
    try:
        # 预处理输入图像
        tmp, name_map = prep_input(INPUT_DIR, tmp)
        
        # 遍历所有Trainer
        # 目录结构: DatasetXXX/TrainerName__Plans__Config/...
        trainers_found = []
        for trainer_dir in sorted(dataset_dir.iterdir()):
            if not trainer_dir.is_dir():
                continue
            if 'nnUNetTrainer' not in trainer_dir.name:
                continue
            
            # 检查是否有checkpoint
            ckpt_path = trainer_dir / f'fold_{FOLD}' / CHECKPOINT
            if not ckpt_path.exists():
                print(f"跳过 {trainer_dir.name}: 未找到 {CHECKPOINT}")
                continue
                
            # 解析trainer名称
            # 文件夹名通常是: TrainerName__PlansName__Configuration
            # 兼容性处理：取第一个部分作为TrainerName
            parts = trainer_dir.name.split('__')
            trainer_name = parts[0]
            
            print(f"\n======== 开始处理 Trainer: {trainer_name} ========")
            trainers_found.append(trainer_name)
            
            # 设置该Trainer的输出目录
            current_out = OUTPUT_DIR / trainer_name
            current_border = BORDER_DIR / trainer_name
            current_json = JSON_DIR / trainer_name
            
            # 预测
            try:
                predict(tmp, current_out, trainer_dir)
                draw_contour(tmp, current_out, current_border)
                generate_json_annotations(INPUT_DIR, current_out, current_json, name_map)
            except Exception as e:
                print(f"Trainer {trainer_name} 处理出错: {e}")
                
        if not trainers_found:
            print("未找到任何包含有效checkpoint的Trainer目录")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)      # 再删临时目录
    print('全部完成！')

if __name__ == '__main__':
    # Initialize threading early to avoid runtime errors during repeated inference
    try:
        if torch.cuda.is_available():
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    main()