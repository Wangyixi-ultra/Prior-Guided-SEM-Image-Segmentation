from PIL import Image
import os

def convert_tiff_to_png(tiff_path, png_path=None):
    """
    将TIFF文件转换为PNG格式
    
    Args:
        tiff_path (str): TIFF文件路径
        png_path (str, optional): 输出PNG文件路径，如果不指定则使用相同文件名
    
    Returns:
        bool: 转换成功返回True，否则返回False
    """
    try:
        # 如果未指定输出路径，则使用原文件名但扩展名改为.png
        if png_path is None:
            base_name = os.path.splitext(tiff_path)[0]
            png_path = base_name + '.png'
        
        # 打开TIFF文件
        with Image.open(tiff_path) as img:
            # 如果图像是多页TIFF，只转换第一页
            img.save(png_path, 'PNG')
        
        print(f"成功转换: {tiff_path} -> {png_path}")
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False

def batch_convert_tiff_to_png(input_folder, output_folder=None):
    """
    批量将文件夹中的TIFF文件转换为PNG格式
    
    Args:
        input_folder (str): 包含TIFF文件的文件夹路径
        output_folder (str, optional): 输出文件夹路径，如果不指定则与输入文件夹相同
    """
    if output_folder is None:
        output_folder = input_folder
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历文件夹中的所有TIFF文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tiff', '.tif')):
            tiff_path = os.path.join(input_folder, filename)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(output_folder, png_filename)
            
            convert_tiff_to_png(tiff_path, png_path)

# 使用示例
if __name__ == "__main__":
    # 单个文件转换
    # convert_tiff_to_png("example.tiff", "example.png")
    
    # 批量转换
    batch_convert_tiff_to_png("/home/chen/seg6/predict_no_label/experiment/in/experiment", "/home/chen/seg6/predict_no_label/experiment/in")