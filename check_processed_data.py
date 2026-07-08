import numpy as np
import os

def check_processed_data():
    """检查预处理后的数据"""
    preprocessed_folder = '/home/chen/seg6/U-Mamba/data/nnUNet_preprocessed/Dataset114_Perovskite/nnUNetPlans_2d'
    
    print('预处理文件夹内容:')
    files = os.listdir(preprocessed_folder)
    npz_files = [f for f in files if f.endswith('.npz')]
    print(f'Found .npz files: {len(npz_files)}')
    
    if npz_files:
        print(f'Sample .npz files: {npz_files[:5]}')  # 打印前5个npz文件

        # 检查几个预处理数据文件
        for i, npz_file in enumerate(npz_files[:3]):  # 检查前3个文件
            data_path = os.path.join(preprocessed_folder, npz_file)
            print(f'\n检查数据文件 {i+1}: {data_path}')
            
            try:
                data = np.load(data_path)
                print(f'Data keys: {list(data.keys())}')

                if 'data' in data:
                    d = data['data']
                    print(f'Data shape: {d.shape}')
                    print(f'Data channels: {d.shape[0]}')
                    
                    # 检查每个通道的形状
                    for ch_idx in range(min(d.shape[0], 5)):  # 最多检查5个通道
                        print(f'  Channel {ch_idx} shape: {d[ch_idx].shape}')
                        
                    break  # 只检查第一个文件的详细信息
            except Exception as e:
                print(f"Error loading {data_path}: {e}")

if __name__ == "__main__":
    check_processed_data()