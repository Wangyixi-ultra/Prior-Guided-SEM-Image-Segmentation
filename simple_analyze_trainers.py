#!/usr/bin/env python3
"""
简化版U-Mamba训练器分析脚本
直接分析网络架构的参数量、层数和理论计算量
"""
import torch
import numpy as np
from prettytable import PrettyTable

# 导入网络构建函数
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_layers_detailed(model):
    """详细计算模型层数"""
    layer_stats = {
        'conv2d': 0,
        'linear': 0,
        'batchnorm': 0,
        'layernorm': 0,
        'activation': 0,
        'mamba': 0,
        'other': 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            layer_stats['conv2d'] += 1
        elif isinstance(module, torch.nn.Linear):
            layer_stats['linear'] += 1
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            layer_stats['batchnorm'] += 1
        elif isinstance(module, torch.nn.LayerNorm):
            layer_stats['layernorm'] += 1
        elif isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.GELU)):
            layer_stats['activation'] += 1
        elif 'mamba' in str(type(module)).lower():
            layer_stats['mamba'] += 1
        elif hasattr(module, '_modules') and len(module._modules) > 0:
            continue  # 跳过容器模块
        else:
            layer_stats['other'] += 1
    
    return layer_stats

def estimate_flops(model, input_shape):
    """估算FLOPs（理论计算量）"""
    # 这是一个简化的估算，实际FLOPs会更复杂
    total_flops = 0
    
    # 获取模型信息
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Conv2d FLOPs = output_elements * (kernel_size * in_channels + bias)
            kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            if module.bias is not None:
                kernel_flops += 1
            # 估算输出尺寸（简化版）
            output_elements = np.prod(input_shape[2:]) // (module.stride[0] * module.stride[1])
            total_flops += output_elements * kernel_flops * module.out_channels
            
        elif isinstance(module, torch.nn.Linear):
            # Linear FLOPs = in_features * out_features + bias
            linear_flops = module.in_features * module.out_features
            if module.bias is not None:
                linear_flops += module.out_features
            total_flops += linear_flops
            
        elif 'mamba' in str(type(module)).lower():
            # Mamba层的FLOPs估算（简化版）
            # 主要包含：线性变换 + 卷积 + SSM计算
            try:
                # 假设Mamba层的维度
                d_model = getattr(module, 'd_model', 256)  # 默认值
                d_state = getattr(module, 'd_state', 16)
                d_conv = getattr(module, 'd_conv', 4)
                expand = getattr(module, 'expand', 2)
                
                # 简化的Mamba FLOPs估算
                mamba_flops = d_model * d_model * 4  # 线性变换
                mamba_flops += d_model * d_conv * d_conv  # 卷积
                mamba_flops += d_model * d_state * 2  # SSM核心计算
                
                total_flops += mamba_flops * np.prod(input_shape[2:])  # 乘以特征图大小
            except:
                pass  # 如果无法获取参数，跳过估算
    
    return total_flops

def create_test_plans():
    """创建测试用的plans"""
    return {
        'dataset_name': 'TestDataset',
        'plans_name': 'nnUNetPlans',
        'configurations': {
            '2d': {
                'data_identifier': 'nnUNetPlans_2d',
                'preprocessor_name': 'DefaultPreprocessor',
                'batch_size': 2,
                'patch_size': [256, 256],
                'median_image_size_in_voxels': [512, 512],
                'spacing': [1.0, 1.0],
                'normalization_schemes': ['ZScoreNormalization'],
                'use_mask_for_norm': [False],
                'UNet_class_name': 'PlainConvUNet',
                'UNet_base_num_features': 32,
                'n_conv_per_stage_encoder': [2, 2, 2, 2, 2],
                'n_conv_per_stage_decoder': [2, 2, 2, 2],
                'num_pool_per_axis': [4, 4],
                'pool_op_kernel_sizes': [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
                'conv_kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                'unet_max_num_features': 512,
                'resampling_fn_data': 'resample_data_or_seg_to_shape',
                'resampling_fn_seg': 'resample_data_or_seg_to_shape',
                'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None},
                'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None},
                'batch_dice': True
            }
        }
    }

def create_test_dataset_json():
    """创建测试用的dataset_json"""
    return {
        'name': 'TestDataset',
        'description': 'Test dataset for analysis',
        'reference': '',
        'licence': '',
        'relase': '0.0',
        'tensorImageSize': '4D',
        'modality': {'0': 'CT'},
        'labels': {'background': 0, 'class1': 1, 'class2': 2, 'class3': 3},
        'numTraining': 100,
        'numTest': 20,
        'training': [],
        'test': []
    }

def analyze_network(network, network_name, input_shape=(1, 1, 256, 256)):
    """分析单个网络"""
    print(f"\n{'='*60}")
    print(f"分析 {network_name}")
    print(f"{'='*60}")
    
    try:
        # 计算参数量
        total_params = count_parameters(network)
        print(f"总参数量: {total_params:,}")
        
        # 计算详细层数
        layer_stats = count_layers_detailed(network)
        total_layers = sum(layer_stats.values())
        print(f"总层数: {total_layers}")
        print("层数分布:")
        for layer_type, count in layer_stats.items():
            if count > 0:
                print(f"  {layer_type}: {count}")
        
        # 估算FLOPs
        flops = estimate_flops(network, input_shape)
        print(f"理论FLOPs估算: {flops:,}")
        
        # 计算模型大小（MB）
        param_size = total_params * 4  # 假设float32，每个参数4字节
        model_size_mb = param_size / (1024 * 1024)
        print(f"模型大小估算: {model_size_mb:.2f} MB")
        
        return {
            'network_name': network_name,
            'total_params': total_params,
            'total_layers': total_layers,
            'layer_stats': layer_stats,
            'flops': flops,
            'model_size_mb': model_size_mb
        }
        
    except Exception as e:
        print(f"分析 {network_name} 时出错: {str(e)}")
        return None

def main():
    """主函数"""
    print("U-Mamba 网络架构性能分析")
    print("="*60)
    
    # 创建测试配置
    plans = create_test_plans()
    dataset_json = create_test_dataset_json()
    
    # 创建PlansManager和ConfigurationManager
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('2d')
    
    print(f"配置: 2D分割，patch_size={configuration_manager.patch_size}")
    print(f"类别数: {len(dataset_json['labels'])}")
    
    # 分析的网络列表
    networks = [
        (get_network_from_plans, "nnUNet基准网络 (PlainConvUNet)"),
        (get_umamba_enc_2d_from_plans, "UMambaEnc_2d"),
        (get_umamba_bot_2d_from_plans, "UMambaBot_2d")
    ]
    
    results = []
    
    for network_func, network_name in networks:
        try:
            print(f"\n正在构建 {network_name}...")
            network = network_func(
                plans_manager=plans_manager,
                dataset_json=dataset_json,
                configuration_manager=configuration_manager,
                num_input_channels=1,
                deep_supervision=True
            )
            
            result = analyze_network(network, network_name)
            if result:
                results.append(result)
                
        except Exception as e:
            print(f"构建 {network_name} 时出错: {str(e)}")
            continue
    
    # 创建对比表格
    if results:
        print(f"\n{'='*60}")
        print("对比总结")
        print(f"{'='*60}")
        
        table = PrettyTable()
        table.field_names = ["网络类型", "参数量", "总层数", "模型大小(MB)", "理论FLOPs"]
        
        for result in results:
            table.add_row([
                result['network_name'],
                f"{result['total_params']:,}",
                result['total_layers'],
                f"{result['model_size_mb']:.2f}",
                f"{result['flops']:,.0f}"
            ])
        
        print(table)
        
        # 性能分析
        print(f"\n{'='*60}")
        print("性能分析")
        print(f"{'='*60}")
        
        baseline = results[0]  # 基准是PlainConvUNet
        for result in results[1:]:
            param_ratio = result['total_params'] / baseline['total_params']
            layer_ratio = result['total_layers'] / baseline['total_layers']
            flops_ratio = result['flops'] / baseline['flops']
            size_ratio = result['model_size_mb'] / baseline['model_size_mb']
            
            print(f"\n{result['network_name']} vs 基准:")
            print(f"  参数量比例: {param_ratio:.2f}x")
            print(f"  层数比例: {layer_ratio:.2f}x")
            print(f"  FLOPs比例: {flops_ratio:.2f}x")
            print(f"  模型大小比例: {size_ratio:.2f}x")
            
            if param_ratio > 1:
                print(f"  ⚠️  参数量增加 {((param_ratio-1)*100):.1f}%")
            else:
                print(f"  ✅ 参数量减少 {((1-param_ratio)*100):.1f}%")
                
            if flops_ratio > 1:
                print(f"  ⚠️  计算量增加 {((flops_ratio-1)*100):.1f}%")
            else:
                print(f"  ✅ 计算量减少 {((1-flops_ratio)*100):.1f}%")
        
        # 详细层数对比
        print(f"\n{'='*60}")
        print("详细层数分布对比")
        print(f"{'='*60}")
        
        detail_table = PrettyTable()
        detail_table.field_names = ["网络类型", "Conv2D", "Linear", "BatchNorm", "LayerNorm", "激活层", "Mamba层", "其他"]
        
        for result in results:
            stats = result['layer_stats']
            detail_table.add_row([
                result['network_name'],
                stats['conv2d'],
                stats['linear'],
                stats['batchnorm'],
                stats['layernorm'],
                stats['activation'],
                stats['mamba'],
                stats['other']
            ])
        
        print(detail_table)

if __name__ == "__main__":
    main()