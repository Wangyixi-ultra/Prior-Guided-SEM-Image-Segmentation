import os
from PIL import Image

def resize_images_in_folder(input_dir, output_dir, width, height):
    """
    将输入文件夹中的所有图片调整为指定大小并保存到输出文件夹。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 获取重采样滤镜，兼容不同版本的 Pillow
    if hasattr(Image, 'Resampling'):
        resample_filter = Image.Resampling.LANCZOS
    elif hasattr(Image, 'ANTIALIAS'):
        resample_filter = Image.ANTIALIAS
    else:
        resample_filter = Image.BILINEAR

    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    files = os.listdir(input_dir)
    print(f"Found {len(files)} files in {input_dir}")

    count = 0
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            file_path = os.path.join(input_dir, filename)
            try:
                with Image.open(file_path) as img:
                    # 调整大小
                    # 注意: 如果要保持纵横比，需要额外的逻辑。这里按要求强制设置为指定尺寸。
                    new_img = img.resize((width, height), resample=resample_filter)
                    
                    # 保存
                    out_path = os.path.join(output_dir, filename)
                    new_img.save(out_path)
                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} images...")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    print(f"Done. Successfully processed {count} images.")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    # --- 配置区域 ---
    # 请根据实际情况修改输入文件夹路径
    # 假设当前工作目录下有一个名为 'in' 或具体的日期文件夹
    input_folder = r'/home/chen/seg6/predict_no_label/experiment/anneal/image' 
    
    # 输出图片保存的文件夹路径
    output_folder = r'/home/chen/seg6/predict_no_label/experiment/anneal/image/resized'
    
    target_width = 1024
    target_height = 768
    # ----------------
    
    print(f"Configured to resize images to {target_width}x{target_height}")
    print(f"Input: {input_folder}")
    
    # 如果默认路径不存在，尝试提示用户
    if not os.path.exists(input_folder):
        print(f"\n Warning: 默认输入路径 '{input_folder}' 不存在。")
        print(" 请打开 4.py 修改 'input_folder' 变量为您的实际图片路径。")
    else:
        resize_images_in_folder(input_folder, output_folder, target_width, target_height)
