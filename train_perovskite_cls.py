from ultralytics import YOLO

def train_augmented():
    # 1. 使用 Small 模型 (比 Nano 强，适合纹理特征)
    #    加载 .pt 权重以利用 ImageNet 的边缘/形状先验
    model = YOLO('yolo11s-cls.pt') 

    # 2. 针对晶粒优化的训练配置
    results = model.train(
        data='raw/cropped_yolo_dataset_all',
        
        # --- 基础配置 ---
        imgsz=128,              # 128x128 足够覆盖你的小尺寸晶粒，且比224更快
        epochs=300,             # 训练轮数
        batch=128,              # 小 Batch (128/256) 能带来更好的泛化噪声
        device=0,
        patience=40,            # 早停耐性

        # --- 针对钙钛矿晶粒的关键增强 ---
        degrees=180,            # 开启 ±180度 全旋转 (晶粒朝向无关)
        fliplr=0.5,             # 水平翻转
        flipud=0.5,             # 垂直翻转 (晶粒同样也可能上下颠倒)
        scale=0.3,              # 适度的尺度缩放 (模拟不同切片大小)
        
        # --- 关键：保护物理特征 ---
        # 电镜图中，亮度差异(Z-contrast)是区分 PbI2 和 ABO3 的核心特征
        # 必须降低 HSV 增强，防止模型把"暗的PbI2"误认为"ABO3"
        hsv_h=0.0,              # 关闭色相变换 (SEM通常是灰度)
        hsv_s=0.0,              # 关闭饱和度变换
        hsv_v=0.1,              # 仅允许极轻微的亮度变化 (模拟曝光波动)，不要太大
        
        project='perovskite_grains_opt',
        name='yolo11s_128_grain_specific',
        exist_ok=True
    )

if __name__ == '__main__':
    train_augmented()
