#!/usr/bin/env python3
"""
处理JSON标注文件，删除所有标签为"defect"的标注信息
"""

import json
import os
from pathlib import Path

def process_json_file(input_path, output_path):
    """
    读取JSON文件，删除所有label为"defect"的标注，保存到新位置
    """
    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 过滤掉label为"defect"的标注
        if 'shapes' in data:
            original_count = len(data['shapes'])
            data['shapes'] = [shape for shape in data['shapes'] if shape.get('label') != 'defect']
            filtered_count = len(data['shapes'])
            removed_count = original_count - filtered_count
            
            print(f"  处理 {os.path.basename(input_path)}: 原始 {original_count} 个标注，删除 {removed_count} 个defect标注，保留 {filtered_count} 个标注")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存处理后的JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return True
        
    except Exception as e:
        print(f"  处理 {os.path.basename(input_path)} 时出错: {e}")
        return False

def main():
    # 设置输入和输出目录
    input_dir = Path('/home/chen/seg6/raw/json_dir')
    output_dir = Path('/home/chen/seg6/raw/json_dir_nodefect')
    
    # 确保输入目录存在
    if not input_dir.exists():
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = sorted(input_dir.glob('*.json'))
    
    if not json_files:
        print(f"警告: 在 {input_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件需要处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 60)
    
    # 处理每个JSON文件
    success_count = 0
    for json_file in json_files:
        output_file = output_dir / json_file.name
        if process_json_file(json_file, output_file):
            success_count += 1
    
    print("-" * 60)
    print(f"处理完成: 成功处理 {success_count}/{len(json_files)} 个文件")
    
    # 验证结果 - 检查输出文件是否确实没有defect标签
    print("\n验证结果...")
    defect_found = False
    for json_file in output_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'shapes' in data:
                for shape in data['shapes']:
                    if shape.get('label') == 'defect':
                        print(f"  警告: {json_file.name} 中仍然包含defect标签")
                        defect_found = True
                        break
        except Exception as e:
            print(f"  验证 {json_file.name} 时出错: {e}")
    
    if not defect_found:
        print("  验证通过: 所有输出文件都不包含defect标签")

if __name__ == '__main__':
    main()
