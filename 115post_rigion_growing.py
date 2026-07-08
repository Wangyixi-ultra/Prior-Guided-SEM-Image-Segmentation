#!/usr/bin/env python3
"""
后处理脚本：使用区域生长算法对nnU-Net模型预测结果进行精细化处理
结合了深度学习网络的语义理解能力和传统图像处理算法（区域生长）的精确定位优势
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
import glob

# 添加U-Mamba源码路径到sys.path
sys.path.insert(0, "/home/chen/seg6/U-Mamba/umamba")

# nnU-Net v2 相关
try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
except Exception:
    nnUNetPredictor = None
    determine_num_input_channels = None

# ============================
# 内置配置（直接在这里修改）
# ============================
CONFIG = {
    # 模式: "batch" or "single"
    "mode": "batch",

    # 批量预测参数
    "dataset_raw": "/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset115_Perovskite",
    "trained_model": "/home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset115_Perovskite/nnUNetTrainer",
    # 可选：指定trainer名称关键词（多trainer时用于匹配）
    "trainer_name": "nnUNetTrainerRegionGrowingDualChannel__nnUNetPlans__2d",
    "pred_output": "/home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset115_Perovskite/test_predictions",
    "fold": "0",
    "checkpoint": "checkpoint_final.pth",
    # 输入通道顺序: yolo在前用 [1, 0]；默认 [0, 1]
    "channel_order": [1, 0],
    # 是否生成可视化叠加图
    "enable_visualization": True,
    # 可视化输出目录（为空则使用 pred_output 下的 visualizations）
    "vis_output": "",
    # 可视化使用的原图通道（例如 0 或 1）
    "vis_raw_channel": 0,

    # 单图后处理参数
    "image": "/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset115_Perovskite/imagesTs",
    "model": "/home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset115_Perovskite/nnUNetTrainerRegionGrowingDualChannel__nnUNetPlans__2d/fold_0/checkpoint_best.pth",
    "output": "/home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset115_Perovskite/post_test_predictions",

    # 设备
    "device": "cuda",
}


def region_growing_refinement(image: torch.Tensor, 
                             output_logits: torch.Tensor, 
                             seed_threshold: float = 0.9,
                             intensity_threshold: float = 0.1,
                             max_iterations: int = 50) -> torch.Tensor:
    """
    应用区域生长算法细化分割结果
    Args:
        image: [B, C, H, W] or [B, C, D, H, W] - 原始输入图像
        output_logits: [B, NumClasses, ...] - 网络输出的logits
        seed_threshold: 种子点阈值
        intensity_threshold: 强度差异阈值
        max_iterations: 最大迭代次数
    Returns:
        refined_mask: 细化后的分割掩码
    """
    with torch.no_grad():
        # 1. 获取概率图
        probs = torch.softmax(output_logits, dim=1)
        
        # 获取类别数
        num_classes = probs.shape[1]
        
        # 初始化最终的分割结果掩码
        refined_mask = torch.zeros_like(probs[:, 0]).long()
        
        # 针对每个前景类别分别进行区域生长
        # 假设 class 0 是背景，从 1 开始
        for c in range(1, num_classes):
            # 2. 选择种子点: 概率大于阈值的点
            class_probs = probs[:, c]
            seeds = (class_probs > seed_threshold)
            
            if seeds.sum() == 0:
                continue
            
            # 当前类别的区域掩码
            current_region = seeds.clone()
            
            # Image通常是多通道，这里取第一个通道作为参考强度
            # 注意：数据已经标准化
            img_channel0 = image[:, 0] 
            
            # 迭代生长
            for i in range(max_iterations):
                # 膨胀操作获取邻域 (Candidates)
                # 使用 MaxPool 作为膨胀操作
                kernel_size = 3
                padding = 1
                stride = 1
                
                if current_region.dim() == 3:  # 2D: [B, H, W] -> unsqueeze -> [B, 1, H, W]
                    dilated = torch.nn.functional.max_pool2d(
                        current_region.float().unsqueeze(1), 
                        kernel_size=kernel_size, stride=stride, padding=padding
                    ).squeeze(1).bool()
                elif current_region.dim() == 4:  # 3D: [B, D, H, W]
                    dilated = torch.nn.functional.max_pool3d(
                        current_region.float().unsqueeze(1), 
                        kernel_size=kernel_size, stride=stride, padding=padding
                    ).squeeze(1).bool()
                else:
                    break  # Should not happen
                
                # 候选点：在膨胀区域内 但 不在当前区域内
                candidates = dilated & (~current_region)
                
                if candidates.sum() == 0:
                    break
                
                # 3. 相似性判断 (Merge rule)
                # 计算每个样本种子区域的平均强度，并广播
                batch_size = image.shape[0]
                seed_means = []
                for b in range(batch_size):
                    b_seeds = seeds[b]
                    if b_seeds.sum() > 0:
                        mean_val = img_channel0[b][b_seeds].mean()
                    else:
                        mean_val = img_channel0[b].mean()  # Fallback
                    seed_means.append(mean_val)
                
                seed_means = torch.stack(seed_means).to(image.device)
                
                # Broadcast seed_means to match image dimensions
                view_shape = [batch_size] + [1] * (image.dim() - 2)  # [B, 1, 1] or [B, 1, 1, 1]
                seed_means_expanded = seed_means.view(*view_shape)
                
                diff = torch.abs(img_channel0 - seed_means_expanded)
                
                # 新增像素：是候选点 且 强度差异小
                new_pixels = candidates & (diff < intensity_threshold)
                
                if new_pixels.sum() == 0:
                    break
                    
                current_region = current_region | new_pixels
            
            # 将生长后的区域赋值给该类别
            refined_mask[current_region] = c
            
        return refined_mask


def load_model(model_path: str):
    """
    加载训练好的模型
    这里需要根据实际情况修改加载模型的逻辑
    """
    # 注意：这里的实现取决于你实际使用的模型类型
    # 下面是一个占位符实现，你需要根据实际情况修改
    print(f"Loading model from {model_path}")
    # 示例：如果模型是以PyTorch方式保存的
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def post_process_prediction(image_path: str, model_path: str, output_path: str, device: str = 'cuda'):
    """
    对单张图像进行后处理预测
    """
    # 加载模型
    model = load_model(model_path)
    if model is None:
        print("Could not load model, exiting...")
        return

    # 读取图像
    if image_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        # 对于2D图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read image from {image_path}")
            return
        # 转换为tensor并添加批次维度和通道维度
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:
        # 其他格式需要根据实际情况处理
        print(f"Unsupported image format: {image_path}")
        return

    # 移动到设备
    img_tensor = img_tensor.to(device)

    # 模型预测
    with torch.no_grad():
        model = model.to(device)
        prediction = model(img_tensor)

    # 应用区域生长后处理
    refined_result = region_growing_refinement(img_tensor, prediction)

    # 保存结果
    result_np = refined_result.cpu().numpy()[0]  # 移除批次维度
    cv2.imwrite(output_path, result_np.astype(np.uint8))
    print(f"Post-processed result saved to {output_path}")


def build_case_file_list(images_dir: str, num_input_channels: int, channel_order: Optional[List[int]] = None) -> List[List[str]]:
    """构建 nnU-Net 预测所需的 List[List[str]] 文件列表"""
    case_files_0000 = sorted([f for f in os.listdir(images_dir) if f.endswith('_0000.png')])
    case_ids = [f.replace('_0000.png', '') for f in case_files_0000]
    if not case_ids:
        raise RuntimeError(f"未在 {images_dir} 中找到 _0000.png 文件")

    if channel_order is None:
        channel_order = list(range(num_input_channels))
    if len(channel_order) != num_input_channels:
        raise RuntimeError(f"channel_order 长度 {len(channel_order)} 与 num_input_channels {num_input_channels} 不一致")

    list_of_lists = []
    missing_files = []
    for case_id in case_ids:
        case_file_list = []
        for i in channel_order:
            fname = f"{case_id}_{i:04d}.png"
            fpath = os.path.join(images_dir, fname)
            if not os.path.exists(fpath):
                missing_files.append(fpath)
            case_file_list.append(fpath)
        list_of_lists.append(case_file_list)

    if missing_files:
        raise RuntimeError(
            f"缺少必要的通道文件（共 {len(missing_files)} 个）。示例: {missing_files[0]}"
        )

    return list_of_lists


def resolve_trained_model_dir(trained_model_dir: str, trainer_name: str = "") -> str:
    """解析训练结果目录，必要时在子目录中自动选择trainer"""
    if os.path.isfile(os.path.join(trained_model_dir, "dataset.json")):
        return trained_model_dir

    if not os.path.isdir(trained_model_dir):
        # 兼容传入 .../nnUNetTrainer 但实际不存在的情况，回退到父目录
        parent_dir = os.path.dirname(trained_model_dir)
        if os.path.isdir(parent_dir):
            trained_model_dir = parent_dir
        else:
            raise RuntimeError(f"trained_model_dir 不存在: {trained_model_dir}")

    candidates = []
    for item in os.listdir(trained_model_dir):
        sub = os.path.join(trained_model_dir, item)
        if os.path.isdir(sub) and item.startswith("nnUNetTrainer"):
            if os.path.isfile(os.path.join(sub, "dataset.json")):
                candidates.append(sub)

    if not candidates:
        raise RuntimeError(
            f"在 {trained_model_dir} 下未找到包含 dataset.json 的 trainer 目录"
        )

    if trainer_name:
        for c in candidates:
            if trainer_name in os.path.basename(c):
                return c
        raise RuntimeError(
            f"未找到匹配 trainer_name='{trainer_name}' 的目录。可选项: {', '.join(os.path.basename(c) for c in candidates)}"
        )

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"检测到多个 trainer 目录，请设置 CONFIG['trainer_name'] 指定其一。可选项: {', '.join(os.path.basename(c) for c in candidates)}"
    )


def predict_testset_with_trained_model(
    dataset_raw_dir: str,
    trained_model_dir: str,
    output_dir: str,
    fold: str = "0",
    checkpoint: str = "checkpoint_final.pth",
    device: str = "cuda",
    channel_order: Optional[List[int]] = None,
    trainer_name: str = "",
    enable_visualization: bool = False,
    vis_output: str = "",
    vis_raw_channel: int = 0
):
    """使用训练结果对 Dataset115_Perovskite 的测试集进行预测"""
    if nnUNetPredictor is None:
        raise RuntimeError("nnUNetPredictor 不可用，请检查 U-Mamba/umamba 是否可导入")

    images_ts = os.path.join(dataset_raw_dir, "imagesTs")
    if not os.path.exists(images_ts):
        raise RuntimeError(f"imagesTs 不存在: {images_ts}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    use_folds = [int(fold)] if fold != "all" else ["all"]
    resolved_model_dir = resolve_trained_model_dir(trained_model_dir, trainer_name=trainer_name)

    predictor.initialize_from_trained_model_folder(
        resolved_model_dir,
        use_folds=use_folds,
        checkpoint_name=checkpoint
    )

    if determine_num_input_channels is not None:
        num_input_channels = determine_num_input_channels(
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json
        )
    else:
        # 兜底：从网络权重推断
        params = list(predictor.network.parameters())
        num_input_channels = params[0].shape[1] if params else 1

    list_of_lists = build_case_file_list(images_ts, num_input_channels, channel_order=channel_order)
    os.makedirs(output_dir, exist_ok=True)

    predictor.predict_from_files(
        list_of_lists,
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

    print(f"预测完成，结果保存到: {output_dir}")

    if enable_visualization:
        if not vis_output:
            vis_output = os.path.join(output_dir, "visualizations")
        visualize_predictions(images_ts, output_dir, vis_output, raw_channel=vis_raw_channel)


def visualize_predictions(images_dir: str, pred_dir: str, vis_dir: str, raw_channel: int = 0):
    """生成预测可视化叠加图（基于 _0000.png 和预测mask）"""
    os.makedirs(vis_dir, exist_ok=True)
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    for pred_name in pred_files:
        if pred_name.endswith("_0000.png"):
            case_id = pred_name.replace("_0000.png", "")
        elif pred_name.endswith("_0001.png"):
            case_id = pred_name.replace("_0001.png", "")
        else:
            case_id = pred_name.replace('.png', '')

        raw_path = os.path.join(images_dir, f"{case_id}_{raw_channel:04d}.png")
        if not os.path.exists(raw_path):
            candidates = glob.glob(os.path.join(images_dir, f"{case_id}_*.png"))
            if candidates:
                raw_path = candidates[0]
        pred_path = os.path.join(pred_dir, pred_name)
        if not os.path.exists(raw_path):
            continue

        raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if raw is None or pred is None:
            continue

        overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        # 颜色映射（BGR）
        class_colors = {
            1: (0, 255, 0),
            2: (0, 165, 255),
            3: (0, 0, 255)
        }

        for class_id, color in class_colors.items():
            mask = (pred == class_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(overlay, contours, -1, color, 2)

        out_path = os.path.join(vis_dir, f"{case_id}_vis.png")
        cv2.imwrite(out_path, overlay)


def main():
    # 默认路径 - 直接读取 CONFIG
    DEFAULT_IMAGE_PATH = CONFIG["image"]
    DEFAULT_MODEL_PATH = CONFIG["model"]
    DEFAULT_OUTPUT_PATH = CONFIG["output"]
    DEFAULT_DEVICE = CONFIG["device"]

    DEFAULT_DATASET_RAW = CONFIG["dataset_raw"]
    DEFAULT_TRAINED_MODEL = CONFIG["trained_model"]
    DEFAULT_PRED_OUTPUT = CONFIG["pred_output"]
    DEFAULT_CHANNEL_ORDER = ",".join(str(x) for x in CONFIG["channel_order"])
    DEFAULT_TRAINER_NAME = CONFIG.get("trainer_name", "")
    DEFAULT_ENABLE_VIS = CONFIG.get("enable_visualization", False)
    DEFAULT_VIS_OUTPUT = CONFIG.get("vis_output", "")
    DEFAULT_VIS_RAW_CHANNEL = CONFIG.get("vis_raw_channel", 0)

    parser = argparse.ArgumentParser(description="Post-process or batch predict with nnU-Net")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default=CONFIG["mode"],
                        help="single: 单图后处理；batch: 预测Dataset测试集")

    # 单图模式参数
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH, help="Path to input image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path for output result")

    # 批量预测参数
    parser.add_argument("--dataset_raw", type=str, default=DEFAULT_DATASET_RAW,
                        help="Path to nnUNet_raw/DatasetXXX")
    parser.add_argument("--trained_model", type=str, default=DEFAULT_TRAINED_MODEL,
                        help="Path to trained model folder (nnUNetTrainer*)")
    parser.add_argument("--pred_output", type=str, default=DEFAULT_PRED_OUTPUT,
                        help="Output folder for predictions")
    parser.add_argument("--trainer_name", type=str, default=DEFAULT_TRAINER_NAME,
                        help="trainer目录名称关键词（可选）")
    parser.add_argument("--fold", type=str, default="0", help="Fold index or 'all'")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth",
                        help="Checkpoint name")
    parser.add_argument("--channel_order", type=str, default=DEFAULT_CHANNEL_ORDER,
                        help="输入通道顺序，例如 yolo在前用 1,0；默认 0,1")
    parser.add_argument("--enable_visualization", action="store_true", default=DEFAULT_ENABLE_VIS,
                        help="是否生成可视化叠加图")
    parser.add_argument("--vis_output", type=str, default=DEFAULT_VIS_OUTPUT,
                        help="可视化输出目录（空则用 pred_output/visualizations）")
    parser.add_argument("--vis_raw_channel", type=int, default=DEFAULT_VIS_RAW_CHANNEL,
                        help="可视化使用的原图通道索引")

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to run inference on")

    args = parser.parse_args()

    if args.mode == "single":
        post_process_prediction(args.image, args.model, args.output, args.device)
    else:
        channel_order = [int(x) for x in args.channel_order.split(',') if x.strip() != ""]
        predict_testset_with_trained_model(
            dataset_raw_dir=args.dataset_raw,
            trained_model_dir=args.trained_model,
            output_dir=args.pred_output,
            fold=args.fold,
            checkpoint=args.checkpoint,
            device=args.device,
            channel_order=channel_order,
            trainer_name=args.trainer_name,
            enable_visualization=args.enable_visualization,
            vis_output=args.vis_output,
            vis_raw_channel=args.vis_raw_channel
        )


if __name__ == "__main__":
    # 使用说明
    print("="*60)
    print("后处理脚本：支持单图区域生长 + nnU-Net 测试集批量预测")
    print("="*60)
    print("使用方法：")
    print("1) 批量预测测试集（默认）：")
    print("   python 115post_rigion_growing.py --mode batch --dataset_raw <nnUNet_raw/Dataset115_Perovskite> ")
    print("       --trained_model <nnUNet_results/Dataset115_Perovskite> ")
    print("       --pred_output <output_dir> --fold 0 --checkpoint checkpoint_final.pth --device cuda")
    print("       --channel_order 1,0  # yolo通道在前")
    print("       --trainer_name nnUNetTrainerUMambaBot  # 可选，多个trainer时指定")
    print("       --enable_visualization --vis_output <vis_dir>  # 生成可视化")
    print("")
    print("2) 单图后处理：")
    print("   python 115post_rigion_growing.py --mode single --image IMAGE_PATH --model MODEL_PATH ")
    print("       --output OUTPUT_PATH --device cuda")
    print("="*60)
    
    # 如果没有命令行参数，使用默认值
    if len(sys.argv) == 1:
        print("使用默认路径，如需修改，请编辑脚本中的 CONFIG 或传参")
    
    main()