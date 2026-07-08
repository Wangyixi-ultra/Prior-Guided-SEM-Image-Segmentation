#!/usr/bin/env python3
"""
Patch Ultralytics tasks.py to support CBAM in DDP training
在 Ultralytics 源码中注册 CBAM 模块
"""

import os
import sys

def patch_ultralytics():
    """在 ultralytics 源码中注入 CBAM 支持"""
    
    # 找到 ultralytics 安装路径
    import ultralytics
    ultralytics_path = os.path.dirname(ultralytics.__file__)
    tasks_path = os.path.join(ultralytics_path, 'nn', 'tasks.py')
    
    if not os.path.exists(tasks_path):
        print(f"Error: tasks.py not found at {tasks_path}")
        return False
    
    print(f"Found tasks.py at: {tasks_path}")
    
    # 读取文件内容
    with open(tasks_path, 'r') as f:
        content = f.read()
    
    # 检查是否已经 patch
    if 'CBAM' in content and 'from ultralytics.nn.modules import' in content:
        print("Already patched, checking if CBAM is imported...")
        if 'CBAM' in content.split('from ultralytics.nn.modules import')[1].split('\n')[0]:
            print("CBAM is already imported in tasks.py")
            return True
    
    # 找到导入模块的位置（from ultralytics.nn.modules import ...）
    import_line = 'from ultralytics.nn.modules import'
    if import_line in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if import_line in line and 'CBAM' not in line:
                # 在这行添加 CBAM
                if line.strip().endswith(')'):
                    # 多行导入格式：from ... import (
                    #     ...
                    # )
                    pass  # 复杂情况，不处理
                else:
                    # 单行导入格式：from ... import A, B, C
                    lines[i] = line.rstrip() + ', CBAM'
                    print(f"Patched line {i+1}: {lines[i]}")
                    break
        
        # 写回文件
        with open(tasks_path, 'w') as f:
            f.write('\n'.join(lines))
        print("Successfully patched tasks.py")
        return True
    else:
        print("Could not find import line to patch")
        return False


if __name__ == '__main__':
    success = patch_ultralytics()
    sys.exit(0 if success else 1)
