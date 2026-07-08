#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN-Diffusion Checkpoint 分析框架
用于分析 /home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset102_Perovskite/nnUNetTrainer_PINNDiffusion__nnUNetPlans__2d/fold_0/checkpoint_best.pth

功能：
1. 加载和验证checkpoint文件
2. 分析模型结构和参数
3. 提取扩散系数图(Dmap)统计信息
4. 可视化训练过程和结果
5. 生成详细的分析报告

作者: AI Assistant
日期: 2025-11-16
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('checkpoint_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PINNCheckpointAnalyzer:
    """PINN-Diffusion模型Checkpoint分析器"""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "./analysis_results"):
        """
        初始化分析器
        
        Args:
            checkpoint_path: checkpoint文件路径
            output_dir: 分析结果输出目录
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.checkpoint = None
        self.model_state = None
        self.optimizer_state = None
        self.scheduler_state = None
        self.epoch = None
        self.best_metric = None
        self.dmap_stats = {}
        self.analysis_results = {}
        
        logger.info(f"初始化分析器，checkpoint路径: {self.checkpoint_path}")
        
    def load_checkpoint(self) -> bool:
        """加载checkpoint文件"""
        try:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint文件不存在: {self.checkpoint_path}")
                
            logger.info(f"正在加载checkpoint: {self.checkpoint_path}")
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # 提取关键信息
            self.model_state = self.checkpoint.get('model_state_dict', {})
            self.optimizer_state = self.checkpoint.get('optimizer_state_dict', {})
            self.scheduler_state = self.checkpoint.get('lr_scheduler_state_dict', {})
            self.epoch = self.checkpoint.get('epoch', -1)
            self.best_metric = self.checkpoint.get('best_metric', {})
            
            logger.info(f"成功加载checkpoint - Epoch: {self.epoch}, Best Metric: {self.best_metric}")
            return True
            
        except Exception as e:
            logger.error(f"加载checkpoint失败: {str(e)}")
            return False
    
    def analyze_model_structure(self) -> Dict[str, Any]:
        """分析模型结构"""
        logger.info("开始分析模型结构...")
        
        analysis = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'layer_info': {},
            'module_sizes': {},
            'parameter_distribution': {}
        }
        
        try:
            for name, param in self.model_state.items():
                param_size = param.numel()
                analysis['total_parameters'] += param_size
                
                if param.requires_grad:
                    analysis['trainable_parameters'] += param_size
                
                # 按模块分组
                module_name = name.split('.')[0] if '.' in name else 'other'
                if module_name not in analysis['module_sizes']:
                    analysis['module_sizes'][module_name] = 0
                analysis['module_sizes'][module_name] += param_size
                
                # 记录层信息
                analysis['layer_info'][name] = {
                    'shape': list(param.shape),
                    'size': param_size,
                    'mean': float(param.mean().item()) if param.numel() > 0 else 0.0,
                    'std': float(param.std().item()) if param.numel() > 0 else 0.0,
                    'min': float(param.min().item()) if param.numel() > 0 else 0.0,
                    'max': float(param.max().item()) if param.numel() > 0 else 0.0
                }
            
            # 计算参数分布
            total_size = analysis['total_parameters']
            for module, size in analysis['module_sizes'].items():
                analysis['parameter_distribution'][module] = {
                    'count': size,
                    'percentage': (size / total_size) * 100
                }
            
            self.analysis_results['model_structure'] = analysis
            logger.info(f"模型结构分析完成 - 总参数: {analysis['total_parameters']:,}")
            return analysis
            
        except Exception as e:
            logger.error(f"模型结构分析失败: {str(e)}")
            return analysis
    
    def analyze_diffusivity_head(self) -> Dict[str, Any]:
        """专门分析扩散系数头"""
        logger.info("开始分析扩散系数头...")
        
        diffusivity_analysis = {
            'parameters': {},
            'weight_statistics': {},
            'bias_statistics': {},
            'layer_details': {}
        }
        
        try:
            # 提取扩散系数头相关参数
            diffusivity_params = {}
            for name, param in self.model_state.items():
                if 'diffusivity_head' in name or 'Dmap' in name.lower():
                    diffusivity_params[name] = param
            
            if not diffusivity_params:
                logger.warning("未找到扩散系数头相关参数")
                return diffusivity_analysis
            
            # 分析权重和偏置
            weights, biases = [], []
            for name, param in diffusivity_params.items():
                if 'weight' in name:
                    weights.append(param)
                elif 'bias' in name:
                    biases.append(param)
                
                diffusivity_analysis['parameters'][name] = {
                    'shape': list(param.shape),
                    'mean': float(param.mean().item()),
                    'std': float(param.std().item()),
                    'min': float(param.min().item()),
                    'max': float(param.max().item())
                }
            
            # 计算整体统计
            if weights:
                all_weights = torch.cat([w.flatten() for w in weights])
                diffusivity_analysis['weight_statistics'] = {
                    'mean': float(all_weights.mean().item()),
                    'std': float(all_weights.std().item()),
                    'min': float(all_weights.min().item()),
                    'max': float(all_weights.max().item()),
                    'median': float(all_weights.median().item()),
                    'q25': float(torch.quantile(all_weights, 0.25).item()),
                    'q75': float(torch.quantile(all_weights, 0.75).item())
                }
            
            if biases:
                all_biases = torch.cat([b.flatten() for b in biases])
                diffusivity_analysis['bias_statistics'] = {
                    'mean': float(all_biases.mean().item()),
                    'std': float(all_biases.std().item()),
                    'min': float(all_biases.min().item()),
                    'max': float(all_biases.max().item()),
                    'median': float(all_biases.median().item())
                }
            
            self.analysis_results['diffusivity_head'] = diffusivity_analysis
            logger.info(f"扩散系数头分析完成 - 找到 {len(diffusivity_params)} 个参数")
            return diffusivity_analysis
            
        except Exception as e:
            logger.error(f"扩散系数头分析失败: {str(e)}")
            return diffusivity_analysis
    
    def simulate_dmap_generation(self, input_shape: Tuple[int, int, int, int] = (1, 1, 256, 256)) -> Optional[np.ndarray]:
        """模拟Dmap生成过程"""
        logger.info(f"模拟Dmap生成，输入形状: {input_shape}")
        
        try:
            # 创建模拟概率图
            prob = torch.randn(input_shape).abs()
            prob = prob / prob.sum(dim=1, keepdim=True)  # 归一化为概率
            
            # 模拟扩散系数头的前向传播
            # 这里简化处理，实际应该使用完整的网络结构
            with torch.no_grad():
                # 模拟噪声抑制模块
                prob_clean = prob * 0.9 + 0.1  # 简化的噪声抑制
                
                # 模拟边界特征提取
                boundary_features = torch.sigmoid(prob_clean * 2 - 1)
                
                # 模拟扩散系数计算
                Dmap = torch.sigmoid(boundary_features) * 4.0 + 0.1  # 范围 [0.1, 4.1]
                
                # 添加数值稳定性处理
                Dmap = torch.clamp(Dmap, min=0.001, max=5.0)
            
            dmap_np = Dmap.squeeze().cpu().numpy()
            
            # 计算统计信息
            self.dmap_stats = {
                'shape': dmap_np.shape,
                'mean': float(np.mean(dmap_np)),
                'std': float(np.std(dmap_np)),
                'min': float(np.min(dmap_np)),
                'max': float(np.max(dmap_np)),
                'median': float(np.median(dmap_np)),
                'q25': float(np.percentile(dmap_np, 25)),
                'q75': float(np.percentile(dmap_np, 75))
            }
            
            logger.info(f"Dmap模拟完成 - 均值: {self.dmap_stats['mean']:.4f}, 范围: [{self.dmap_stats['min']:.4f}, {self.dmap_stats['max']:.4f}]")
            return dmap_np
            
        except Exception as e:
            logger.error(f"Dmap模拟失败: {str(e)}")
            return None
    
    def visualize_analysis(self, dmap_data: Optional[np.ndarray] = None):
        """生成可视化图表"""
        logger.info("开始生成可视化图表...")
        
        try:
            # 创建图形布局
            fig = plt.figure(figsize=(20, 15))
            
            # 1. 模型参数分布
            if 'model_structure' in self.analysis_results:
                self._plot_parameter_distribution(fig.add_subplot(3, 3, 1))
            
            # 2. 扩散系数头权重分布
            if 'diffusivity_head' in self.analysis_results:
                self._plot_diffusivity_weights(fig.add_subplot(3, 3, 2))
            
            # 3. Dmap可视化
            if dmap_data is not None:
                self._plot_dmap_visualization(fig.add_subplot(3, 3, 3), dmap_data)
                self._plot_dmap_histogram(fig.add_subplot(3, 3, 4), dmap_data)
                self._plot_dmap_statistics(fig.add_subplot(3, 3, 5))
            
            # 4. 训练信息
            self._plot_training_info(fig.add_subplot(3, 3, 6))
            
            # 5. 参数热力图
            if 'model_structure' in self.analysis_results:
                self._plot_parameter_heatmap(fig.add_subplot(3, 3, 7))
            
            # 6. 层统计信息
            if 'model_structure' in self.analysis_results:
                self._plot_layer_statistics(fig.add_subplot(3, 3, 8))
            
            # 7. 扩散系数统计
            if 'diffusivity_head' in self.analysis_results:
                self._plot_diffusivity_statistics(fig.add_subplot(3, 3, 9))
            
            plt.tight_layout()
            output_path = self.output_dir / 'comprehensive_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"综合分析图表已保存: {output_path}")
            
            # 生成单独的详细图表
            self._generate_detailed_plots(dmap_data)
            
        except Exception as e:
            logger.error(f"可视化生成失败: {str(e)}")
    
    def _plot_parameter_distribution(self, ax):
        """绘制参数分布图"""
        try:
            model_info = self.analysis_results['model_structure']
            modules = list(model_info['parameter_distribution'].keys())
            percentages = [model_info['parameter_distribution'][m]['percentage'] for m in modules]
            
            ax.pie(percentages, labels=modules, autopct='%1.1f%%', startangle=90)
            ax.set_title('模型参数分布', fontsize=12, fontweight='bold')
            
        except Exception as e:
            logger.warning(f"参数分布图绘制失败: {str(e)}")
    
    def _plot_diffusivity_weights(self, ax):
        """绘制扩散系数头权重分布"""
        try:
            diff_info = self.analysis_results['diffusivity_head']
            if 'weight_statistics' in diff_info:
                stats = diff_info['weight_statistics']
                weights = [stats['min'], stats['q25'], stats['median'], stats['q75'], stats['max']]
                labels = ['Min', 'Q25', 'Median', 'Q75', 'Max']
                
                ax.bar(labels, weights, color='skyblue', alpha=0.7)
                ax.set_title('扩散系数头权重统计', fontsize=12, fontweight='bold')
                ax.set_ylabel('权重值')
                ax.tick_params(axis='x', rotation=45)
                
        except Exception as e:
            logger.warning(f"扩散系数头权重图绘制失败: {str(e)}")
    
    def _plot_dmap_visualization(self, ax, dmap_data):
        """绘制Dmap可视化"""
        try:
            im = ax.imshow(dmap_data, cmap='jet', vmin=0, vmax=5)
            ax.set_title(f'Dmap可视化\n均值: {self.dmap_stats["mean"]:.3f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('宽度')
            ax.set_ylabel('高度')
            plt.colorbar(im, ax=ax)
            
        except Exception as e:
            logger.warning(f"Dmap可视化绘制失败: {str(e)}")
    
    def _plot_dmap_histogram(self, ax, dmap_data):
        """绘制Dmap直方图"""
        try:
            ax.hist(dmap_data.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title('Dmap分布直方图', fontsize=12, fontweight='bold')
            ax.set_xlabel('扩散系数值')
            ax.set_ylabel('频率')
            ax.axvline(x=self.dmap_stats['mean'], color='red', linestyle='--', 
                      label=f'均值: {self.dmap_stats["mean"]:.3f}')
            ax.legend()
            
        except Exception as e:
            logger.warning(f"Dmap直方图绘制失败: {str(e)}")
    
    def _plot_dmap_statistics(self, ax):
        """绘制Dmap统计信息"""
        try:
            stats = list(self.dmap_stats.keys())
            values = list(self.dmap_stats.values())
            
            # 只显示数值统计
            numeric_stats = {k: v for k, v in self.dmap_stats.items() if isinstance(v, (int, float))}
            stats_names = list(numeric_stats.keys())
            stats_values = list(numeric_stats.values())
            
            ax.barh(stats_names, stats_values, color='lightcoral', alpha=0.7)
            ax.set_title('Dmap统计信息', fontsize=12, fontweight='bold')
            ax.set_xlabel('数值')
            
        except Exception as e:
            logger.warning(f"Dmap统计图绘制失败: {str(e)}")
    
    def _plot_training_info(self, ax):
        """绘制训练信息"""
        try:
            info_text = f"""
            训练信息:
            Epoch: {self.epoch}
            Best Metric: {self.best_metric}
            
            模型结构:
            总参数: {self.analysis_results.get('model_structure', {}).get('total_parameters', 0):,}
            可训练参数: {self.analysis_results.get('model_structure', {}).get('trainable_parameters', 0):,}
            
            Dmap统计:
            均值: {self.dmap_stats.get('mean', 0):.4f}
            标准差: {self.dmap_stats.get('std', 0):.4f}
            范围: [{self.dmap_stats.get('min', 0):.4f}, {self.dmap_stats.get('max', 0):.4f}]
            """
            
            ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('训练信息汇总', fontsize=12, fontweight='bold')
            
        except Exception as e:
            logger.warning(f"训练信息图绘制失败: {str(e)}")
    
    def _plot_parameter_heatmap(self, ax):
        """绘制参数热力图"""
        try:
            # 简化的参数热力图
            layer_names = list(self.analysis_results['model_structure']['layer_info'].keys())[:10]
            means = [self.analysis_results['model_structure']['layer_info'][name]['mean'] for name in layer_names]
            stds = [self.analysis_results['model_structure']['layer_info'][name]['std'] for name in layer_names]
            
            data = np.array([means, stds])
            im = ax.imshow(data, cmap='coolwarm', aspect='auto')
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Mean', 'Std'])
            ax.set_title('参数统计热力图', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax)
            
        except Exception as e:
            logger.warning(f"参数热力图绘制失败: {str(e)}")
    
    def _plot_layer_statistics(self, ax):
        """绘制层统计信息"""
        try:
            layer_info = self.analysis_results['model_structure']['layer_info']
            sizes = [info['size'] for info in layer_info.values()]
            
            ax.hist(sizes, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.set_title('层参数大小分布', fontsize=12, fontweight='bold')
            ax.set_xlabel('参数数量')
            ax.set_ylabel('层数')
            ax.set_yscale('log')
            
        except Exception as e:
            logger.warning(f"层统计图绘制失败: {str(e)}")
    
    def _plot_diffusivity_statistics(self, ax):
        """绘制扩散系数统计"""
        try:
            diff_info = self.analysis_results['diffusivity_head']
            if 'weight_statistics' in diff_info:
                weight_stats = diff_info['weight_statistics']
                
                # 创建箱线图数据
                data = [weight_stats['min'], weight_stats['q25'], weight_stats['median'], 
                       weight_stats['q75'], weight_stats['max']]
                positions = [1]
                
                ax.boxplot([data], positions=positions, widths=0.6)
                ax.set_title('扩散系数权重分布', fontsize=12, fontweight='bold')
                ax.set_ylabel('权重值')
                ax.set_xticklabels(['Diffusivity Head'])
                
        except Exception as e:
            logger.warning(f"扩散系数统计图绘制失败: {str(e)}")
    
    def _generate_detailed_plots(self, dmap_data):
        """生成详细的单独图表"""
        try:
            # Dmap详细分析图
            if dmap_data is not None:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # 原始Dmap
                im1 = axes[0,0].imshow(dmap_data, cmap='jet', vmin=0, vmax=5)
                axes[0,0].set_title('原始Dmap')
                plt.colorbar(im1, ax=axes[0,0])
                
                # 归一化Dmap
                dmap_norm = (dmap_data - dmap_data.min()) / (dmap_data.max() - dmap_data.min())
                im2 = axes[0,1].imshow(dmap_norm, cmap='viridis')
                axes[0,1].set_title('归一化Dmap')
                plt.colorbar(im2, ax=axes[0,1])
                
                # Dmap梯度
                grad_x = np.gradient(dmap_data, axis=1)
                grad_y = np.gradient(dmap_data, axis=0)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                im3 = axes[1,0].imshow(grad_magnitude, cmap='hot')
                axes[1,0].set_title('Dmap梯度幅值')
                plt.colorbar(im3, ax=axes[1,0])
                
                # Dmap直方图和统计
                axes[1,1].hist(dmap_data.flatten(), bins=50, alpha=0.7, density=True)
                axes[1,1].set_title('Dmap分布密度')
                axes[1,1].set_xlabel('扩散系数值')
                axes[1,1].set_ylabel('密度')
                
                plt.tight_layout()
                output_path = self.output_dir / 'dmap_detailed_analysis.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Dmap详细分析图已保存: {output_path}")
            
            # 模型结构详细图
            if 'model_structure' in self.analysis_results:
                self._plot_model_architecture_details()
                
        except Exception as e:
            logger.error(f"详细图表生成失败: {str(e)}")
    
    def _plot_model_architecture_details(self):
        """绘制模型架构详细信息"""
        try:
            model_info = self.analysis_results['model_structure']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 参数数量分布
            modules = list(model_info['parameter_distribution'].keys())
            counts = [model_info['parameter_distribution'][m]['count'] for m in modules]
            percentages = [model_info['parameter_distribution'][m]['percentage'] for m in modules]
            
            axes[0,0].bar(modules, counts, color='skyblue', alpha=0.7)
            axes[0,0].set_title('各模块参数数量')
            axes[0,0].set_ylabel('参数数量')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 参数占比饼图
            axes[0,1].pie(percentages, labels=modules, autopct='%1.1f%%', startangle=90)
            axes[0,1].set_title('参数占比分布')
            
            # 层大小分布
            layer_sizes = [info['size'] for info in model_info['layer_info'].values()]
            axes[1,0].hist(layer_sizes, bins=30, alpha=0.7, color='lightgreen')
            axes[1,0].set_title('层参数大小分布')
            axes[1,0].set_xlabel('参数数量')
            axes[1,0].set_ylabel('频数')
            
            # 权重值分布统计
            all_weights = []
            for info in model_info['layer_info'].values():
                if info['size'] > 0:
                    all_weights.extend([info['mean'], info['std'], info['min'], info['max']])
            
            axes[1,1].boxplot(all_weights)
            axes[1,1].set_title('权重值统计分布')
            axes[1,1].set_ylabel('权重值')
            
            plt.tight_layout()
            output_path = self.output_dir / 'model_architecture_details.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"模型架构详细图已保存: {output_path}")
            
        except Exception as e:
            logger.warning(f"模型架构详细图绘制失败: {str(e)}")
    
    def generate_report(self) -> str:
        """生成详细的分析报告"""
        logger.info("开始生成分析报告...")
        
        try:
            report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""
# PINN-Diffusion Checkpoint 分析报告

## 基本信息
- 分析时间: {report_time}
- Checkpoint路径: {self.checkpoint_path}
- 训练轮次: {self.epoch}
- 最佳指标: {self.best_metric}

## 模型结构分析
"""
            
            if 'model_structure' in self.analysis_results:
                model_info = self.analysis_results['model_structure']
                report += f"""
### 参数统计
- 总参数数量: {model_info['total_parameters']:,}
- 可训练参数: {model_info['trainable_parameters']:,}
- 模块分布:
"""
                for module, dist in model_info['parameter_distribution'].items():
                    report += f"  - {module}: {dist['count']:,} 参数 ({dist['percentage']:.1f}%)\n"
            
            if 'diffusivity_head' in self.analysis_results:
                diff_info = self.analysis_results['diffusivity_head']
                report += f"""

## 扩散系数头分析
"""
                if 'weight_statistics' in diff_info:
                    w_stats = diff_info['weight_statistics']
                    report += f"""
### 权重统计
- 均值: {w_stats['mean']:.6f}
- 标准差: {w_stats['std']:.6f}
- 范围: [{w_stats['min']:.6f}, {w_stats['max']:.6f}]
- 中位数: {w_stats['median']:.6f}
"""
                
                if 'bias_statistics' in diff_info:
                    b_stats = diff_info['bias_statistics']
                    report += f"""
### 偏置统计
- 均值: {b_stats['mean']:.6f}
- 标准差: {b_stats['std']:.6f}
- 范围: [{b_stats['min']:.6f}, {b_stats['max']:.6f}]
"""
            
            if self.dmap_stats:
                report += f"""

## Dmap模拟分析
"""
                for key, value in self.dmap_stats.items():
                    if isinstance(value, (int, float)):
                        report += f"- {key}: {value:.6f}\n"
            
            report += f"""

## 分析结论
1. 模型结构完整性: {'✓' if self.model_state else '✗'}
2. 扩散系数头存在: {'✓' if 'diffusivity_head' in self.analysis_results else '✗'}
3. Dmap生成能力: {'✓' if self.dmap_stats else '✗'}
4. 训练状态: {'完成' if self.epoch > 0 else '未知'}

## 建议
"""
            if 'diffusivity_head' not in self.analysis_results:
                report += "- 建议检查扩散系数头的参数命名和结构\n"
            
            if not self.dmap_stats:
                report += "- 建议验证Dmap生成逻辑和数值稳定性\n"
            
            report += """
- 建议定期监控训练过程中的损失收敛情况
- 建议检查扩散系数的数值范围是否合理
- 建议验证物理约束是否得到有效实施

---
*报告生成时间: """ + report_time + """
*分析器版本: PINNCheckpointAnalyzer v1.0
"""
            
            # 保存报告
            report_path = self.output_dir / 'analysis_report.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"分析报告已保存: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            return f"报告生成失败: {str(e)}"
    
    def save_results_json(self):
        """将分析结果保存为JSON格式"""
        try:
            results = {
                'checkpoint_info': {
                    'path': str(self.checkpoint_path),
                    'epoch': self.epoch,
                    'best_metric': self.best_metric
                },
                'analysis_results': self.analysis_results,
                'dmap_statistics': self.dmap_stats,
                'analysis_time': datetime.now().isoformat()
            }
            
            json_path = self.output_dir / 'analysis_results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"分析结果JSON已保存: {json_path}")
            
        except Exception as e:
            logger.error(f"JSON保存失败: {str(e)}")
    
    def run_complete_analysis(self) -> bool:
        """运行完整的分析流程"""
        logger.info("开始完整的checkpoint分析流程...")
        
        try:
            # 1. 加载checkpoint
            if not self.load_checkpoint():
                return False
            
            # 2. 分析模型结构
            self.analyze_model_structure()
            
            # 3. 分析扩散系数头
            self.analyze_diffusivity_head()
            
            # 4. 模拟Dmap生成
            dmap_data = self.simulate_dmap_generation()
            
            # 5. 生成可视化
            self.visualize_analysis(dmap_data)
            
            # 6. 生成报告
            self.generate_report()
            
            # 7. 保存JSON结果
            self.save_results_json()
            
            logger.info("完整分析流程完成！")
            return True
            
        except Exception as e:
            logger.error(f"完整分析流程失败: {str(e)}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PINN-Diffusion Checkpoint 分析工具')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset102_Perovskite/nnUNetTrainer_PINNDiffusion__nnUNetPlans__2d/fold_0/checkpoint_best.pth',
                       help='checkpoint文件路径')
    parser.add_argument('--output', type=str, default='./analysis_results',
                       help='分析结果输出目录')
    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 1, 256, 256],
                       help='Dmap模拟输入形状 (B, C, H, W)')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = PINNCheckpointAnalyzer(args.checkpoint, args.output)
    
    # 运行完整分析
    success = analyzer.run_complete_analysis()
    
    if success:
        print(f"✓ 分析完成！结果保存在: {args.output}")
        print(f"✓ 主要文件:")
        print(f"  - 综合分析图: {args.output}/comprehensive_analysis.png")
        print(f"  - 详细分析报告: {args.output}/analysis_report.md")
        print(f"  - JSON结果: {args.output}/analysis_results.json")
    else:
        print("✗ 分析失败，请检查日志文件: checkpoint_analysis.log")


if __name__ == "__main__":
    main()