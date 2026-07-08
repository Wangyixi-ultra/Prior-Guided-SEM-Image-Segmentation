#!/usr/bin/env python3
"""
多尺度YOLO-UMamba融合策略
通过图像金字塔生成多尺度YOLO特征，实现更丰富的上下文信息融合
"""

import os
import sys
sys.path.append("/home/chen/seg6/U-Mamba/umamba")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import traceback

class MultiScaleYOLORetriever:
    """
    多尺度YOLO特征提取器
    通过图像金字塔提取不同尺度的YOLO特征
    """
    
    def __init__(self, yolo_model, scales=[1.0, 0.5, 0.25], confidence_threshold=0.6):
        """
        Args:
            yolo_model: YOLO模型实例
            scales: 多尺度列表，如[1.0, 0.5, 0.25]表示原图、半图、四分之一图
            confidence_threshold: 置信度阈值
        """
        self.model = yolo_model
        self.scales = scales
        self.conf_threshold = confidence_threshold
        
        # 类别映射
        self.class_mapping = {0: 2, 1: 1, 2: 3}  # YOLO类别 -> 分割类别
        self.num_yolo_classes = 3
        
    def extract_multiscale_features(self, image_path):
        """
        提取多尺度YOLO特征
        
        Returns:
            dict: 包含多尺度特征的字典
            {
                'probabilities': [scale1_probs, scale2_probs, ...],  # 每个尺度的概率分布
                'confidences': [scale1_conf, scale2_conf, ...],      # 每个尺度的置信度
                'weighted_features': numpy array,  # 加权融合后的特征图
                'uncertainty_map': numpy array     # 不确定性图
            }
        """
        # 读取原始图像
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            return None
            
        h, w = original_img.shape
        
        all_probabilities = []
        all_confidences = []
        scale_weights = []
        
        # 对每个尺度进行预测
        for scale in self.scales:
            if scale == 1.0:
                # 原图尺度
                img_to_predict = original_img
                scale_h, scale_w = h, w
            else:
                # 缩放图像
                scale_h, scale_w = int(h * scale), int(w * scale)
                img_to_predict = cv2.resize(original_img, (scale_w, scale_h))
            
            # 临时保存缩放后的图像（YOLO需要文件路径）
            temp_path = f"/tmp/temp_scale_{scale}.png"
            cv2.imwrite(temp_path, img_to_predict)
            
            try:
                # YOLO预测
                results = self.model(temp_path, verbose=False)
                probs = results[0].probs
                
                # 获取概率和置信度
                probabilities = probs.data.cpu().numpy()  # (num_classes,)
                top1_conf = probs.top1conf.item()
                
                all_probabilities.append(probabilities)
                all_confidences.append(top1_conf)
                
                # 尺度权重（更大的尺度权重更高，因为细节更丰富）
                scale_weight = scale  # 或者使用 scale ** 2 来增强大尺度权重
                scale_weights.append(scale_weight)
                
            except Exception as e:
                print(f"尺度 {scale} 预测失败: {e}")
                all_probabilities.append(np.ones(self.num_yolo_classes) / self.num_yolo_classes)
                all_confidences.append(0.0)
                scale_weights.append(scale)
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # 归一化尺度权重
        scale_weights = np.array(scale_weights)
        scale_weights = scale_weights / np.sum(scale_weights)
        
        # 计算加权平均概率
        weighted_probs = np.zeros(self.num_yolo_classes)
        for i, (probs, weight) in enumerate(zip(all_probabilities, scale_weights)):
            weighted_probs += probs * weight
        
        # 生成多尺度融合特征图
        weighted_features = self._create_multiscale_feature_maps(
            weighted_probs, all_probabilities, all_confidences, h, w
        )
        
        # 计算不确定性图（基于不同尺度预测的一致性）
        uncertainty_map = self._compute_uncertainty_map(all_probabilities, h, w)
        
        return {
            'probabilities': all_probabilities,
            'confidences': all_confidences,
            'weighted_features': weighted_features,
            'uncertainty_map': uncertainty_map,
            'scale_weights': scale_weights
        }
    
    def _create_multiscale_feature_maps(self, weighted_probs, all_probs, all_confs, h, w):
        """
        创建多尺度融合特征图
        
        Returns:
            numpy array: (num_channels, H, W)
        """
        channels = []
        
        # 通道0: 加权融合后的概率分布（3个类别）
        for i in range(self.num_yolo_classes):
            prob_map = np.full((h, w), weighted_probs[i], dtype=np.float32)
            channels.append(prob_map)
        
        # 通道3: 最大置信度（跨尺度）
        max_conf = max(all_confs) if all_confs else 0.0
        max_conf_map = np.full((h, w), max_conf, dtype=np.float32)
        channels.append(max_conf_map)
        
        # 通道4: 平均置信度
        avg_conf = np.mean(all_confs) if all_confs else 0.0
        avg_conf_map = np.full((h, w), avg_conf, dtype=np.float32)
        channels.append(avg_conf_map)
        
        # 通道5: 置信度方差（反映尺度间一致性）
        if len(all_confs) > 1:
            conf_variance = np.var(all_confs)
        else:
            conf_variance = 0.0
        variance_map = np.full((h, w), conf_variance, dtype=np.float32)
        channels.append(variance_map)
        
        # 通道6: 预测类别标签（基于加权概率）
        pred_class = np.argmax(weighted_probs)
        pred_label = self.class_mapping.get(pred_class, 0)
        label_map = np.full((h, w), pred_label, dtype=np.float32)
        channels.append(label_map)
        
        # 通道7-9: 各尺度最高置信度（如果有3个尺度）
        for i, conf in enumerate(all_confs[:3]):
            conf_map = np.full((h, w), conf, dtype=np.float32)
            channels.append(conf_map)
        
        return np.stack(channels, axis=0)
    
    def _compute_uncertainty_map(self, all_probabilities, h, w):
        """
        计算不确定性图
        基于不同尺度预测的概率分布差异
        """
        if len(all_probabilities) <= 1:
            return np.zeros((h, w), dtype=np.float32)
        
        # 计算概率分布的熵（每个尺度）
        scale_entropies = []
        for probs in all_probabilities:
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            scale_entropies.append(entropy)
        
        # 计算尺度间概率分布的方差
        prob_variance = np.var(all_probabilities, axis=0)  # (num_classes,)
        total_variance = np.sum(prob_variance)
        
        # 综合不确定性（熵 + 方差）
        avg_entropy = np.mean(scale_entropies)
        uncertainty = (avg_entropy + total_variance) / 2.0
        
        # 归一化到[0, 1]
        uncertainty = min(uncertainty / 2.0, 1.0)  # 假设最大熵约为2.0
        
        return np.full((h, w), uncertainty, dtype=np.float32)

class SpatialPyramidFusion:
    """
    空间金字塔融合
    将图像分成不同区域，在每个区域上应用YOLO预测
    """
    
    def __init__(self, yolo_model, grid_sizes=[1, 2, 4], confidence_threshold=0.6):
        """
        Args:
            yolo_model: YOLO模型实例
            grid_sizes: 网格划分大小，如[1, 2, 4]表示1x1, 2x2, 4x4网格
            confidence_threshold: 置信度阈值
        """
        self.model = yolo_model
        self.grid_sizes = grid_sizes
        self.conf_threshold = confidence_threshold
        self.class_mapping = {0: 2, 1: 1, 2: 3}
    
    def generate_spatial_pyramid_features(self, image_path):
        """
        生成空间金字塔特征
        
        Returns:
            dict: 空间金字塔特征
        """
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            return None
        
        h, w = original_img.shape
        
        # 存储每个网格的预测结果
        grid_predictions = {}
        
        for grid_size in self.grid_sizes:
            grid_predictions[grid_size] = {
                'probs': np.zeros((grid_size, grid_size, 3)),  # 3个YOLO类别
                'confs': np.zeros((grid_size, grid_size)),
                'labels': np.zeros((grid_size, grid_size), dtype=int)
            }
            
            cell_h = h // grid_size
            cell_w = w // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # 提取网格区域
                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w
                    
                    if y2 > h: y2 = h
                    if x2 > w: x2 = w
                    
                    grid_region = original_img[y1:y2, x1:x2]
                    
                    # 临时保存网格图像
                    temp_path = f"/tmp/temp_grid_{grid_size}_{i}_{j}.png"
                    cv2.imwrite(temp_path, grid_region)
                    
                    try:
                        # YOLO预测
                        results = self.model(temp_path, verbose=False)
                        probs = results[0].probs
                        
                        # 获取概率和置信度
                        probabilities = probs.data.cpu().numpy()
                        top1_conf = probs.top1conf.item()
                        top1_class = probs.top1
                        
                        grid_predictions[grid_size]['probs'][i, j] = probabilities
                        grid_predictions[grid_size]['confs'][i, j] = top1_conf
                        grid_predictions[grid_size]['labels'][i, j] = top1_class
                        
                    except Exception as e:
                        print(f"网格预测失败 ({grid_size}x{grid_size}, cell {i},{j}): {e}")
                        grid_predictions[grid_size]['probs'][i, j] = np.ones(3) / 3
                        grid_predictions[grid_size]['confs'][i, j] = 0.0
                        grid_predictions[grid_size]['labels'][i, j] = 0
                    
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        # 生成特征图
        feature_maps = self._create_spatial_pyramid_maps(grid_predictions, h, w)
        
        return {
            'grid_predictions': grid_predictions,
            'feature_maps': feature_maps
        }
    
    def _create_spatial_pyramid_maps(self, grid_predictions, h, w):
        """
        创建空间金字塔特征图
        """
        channels = []
        
        # 对每个网格尺度
        for grid_size in self.grid_sizes:
            pred_data = grid_predictions[grid_size]
            cell_h = h // grid_size
            cell_w = w // grid_size
            
            # 通道: 该尺度的置信度图
            conf_map = np.zeros((h, w), dtype=np.float32)
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * cell_h, min((i + 1) * cell_h, h)
                    x1, x2 = j * cell_w, min((j + 1) * cell_w, w)
                    conf_map[y1:y2, x1:x2] = pred_data['confs'][i, j]
            channels.append(conf_map)
            
            # 通道: 该尺度的标签图
            label_map = np.zeros((h, w), dtype=np.float32)
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * cell_h, min((i + 1) * cell_h, h)
                    x1, x2 = j * cell_w, min((j + 1) * cell_w, w)
                    yolo_class = pred_data['labels'][i, j]
                    label_value = self.class_mapping.get(yolo_class, 0)
                    label_map[y1:y2, x1:x2] = label_value
            channels.append(label_map)
        
        # 添加全局预测（1x1网格）
        global_probs = grid_predictions[1]['probs'][0, 0]
        for i in range(3):  # 3个类别
            prob_map = np.full((h, w), global_probs[i], dtype=np.float32)
            channels.append(prob_map)
        
        return np.stack(channels, axis=0)

# ========== 使用示例和测试 ==========
def test_multiscale_fusion():
    """测试多尺度融合功能"""
    
    # 加载YOLO模型
    yolo_weights = '/home/chen/seg6/perovskite_grains_opt/train29/weights/best.pt'
    yolo_model = YOLO(yolo_weights)
    
    # 测试多尺度融合
    print("测试多尺度YOLO融合...")
    multiscale_retriever = MultiScaleYOLORetriever(
        yolo_model, 
        scales=[1.0, 0.5, 0.25],
        confidence_threshold=0.6
    )
    
    # 测试图像路径
    test_image = "/home/chen/seg6/U-Mamba/data/nnUNet_raw/Dataset114_Perovskite/imagesTs/image_0000_0000.png"
    
    if os.path.exists(test_image):
        features = multiscale_retriever.extract_multiscale_features(test_image)
        
        if features:
            print(f"多尺度特征提取成功!")
            print(f"尺度数量: {len(features['probabilities'])}")
            print(f"融合特征图形状: {features['weighted_features'].shape}")
            print(f"不确定性图形状: {features['uncertainty_map'].shape}")
            print(f"各尺度权重: {features['scale_weights']}")
            
            # 保存可视化
            output_dir = "/home/chen/seg6/multiscale_test"
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存不确定性图
            uncertainty_vis = (features['uncertainty_map'][0] * 255).astype(np.uint8)
            cv2.imwrite(f"{output_dir}/uncertainty_map.png", uncertainty_vis)
            
            # 保存各通道
            for i in range(features['weighted_features'].shape[0]):
                channel = features['weighted_features'][i]
                if channel.max() <= 1.0:
                    vis = (channel * 255).astype(np.uint8)
                else:
                    vis = channel.astype(np.uint8)
                cv2.imwrite(f"{output_dir}/channel_{i:02d}.png", vis)
            
            print(f"可视化结果保存到: {output_dir}")
        else:
            print("特征提取失败!")
    else:
        print(f"测试图像不存在: {test_image}")
    
    # 测试空间金字塔融合
    print("\n测试空间金字塔融合...")
    spatial_fusion = SpatialPyramidFusion(
        yolo_model,
        grid_sizes=[1, 2, 4],
        confidence_threshold=0.6
    )
    
    if os.path.exists(test_image):
        spatial_features = spatial_fusion.generate_spatial_pyramid_features(test_image)
        
        if spatial_features:
            print(f"空间金字塔特征提取成功!")
            print(f"网格尺度: {list(spatial_features['grid_predictions'].keys())}")
            print(f"特征图形状: {spatial_features['feature_maps'].shape}")
            
            # 保存空间金字塔特征图
            output_dir = "/home/chen/seg6/spatial_pyramid_test"
            os.makedirs(output_dir, exist_ok=True)
            
            for i in range(spatial_features['feature_maps'].shape[0]):
                channel = spatial_features['feature_maps'][i]
                if channel.max() <= 1.0:
                    vis = (channel * 255).astype(np.uint8)
                else:
                    vis = channel.astype(np.uint8)
                cv2.imwrite(f"{output_dir}/spatial_channel_{i:02d}.png", vis)
            
            print(f"空间金字塔可视化保存到: {output_dir}")
        else:
            print("空间金字塔特征提取失败!")

if __name__ == "__main__":
    test_multiscale_fusion()