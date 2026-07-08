from ultralytics import YOLO

def train():
    # 1. 加载模型
    # 使用 .pt 后缀表示加载预训练权重（推荐），而不是 .yaml (从头训练)
    # 即使你的类别不同，加载预训练权重也能提取到底层通用特征
    model = YOLO('yolo11n-cls.pt') 

    # 2. 开始训练
    results = model.train(
        data='raw/cropped_yolo_dataset_all', # 数据集路径
        epochs=300,                          # 训练轮数 (早停机制会自动停止，设大点没关系)
        imgsz=224,                           # 图像尺寸：建议 224 (标准) 或 128，96 可能太小
        device=4,                            # 指定 GPU
        batch=256,                           # Batch Size: 4096 太大了，建议改小以提高泛化能力
        
        # --- 优化参数 ---
        patience=50,                         # Early Stopping: 50轮没有提升则停止
        lr0=0.01,                            # 初始学习率 (SGD默认0.01)
        optimizer='auto',                    # 优化器: auto, SGD, AdamW
        
        # --- 数据增强 (针对显微/纹理图像通常很有用) ---
        fliplr=0.5,                          # 水平翻转概率
        flipud=0.5,                          # 垂直翻转概率 (对于材料结构通常是旋转不变的)
        scale=0.5,                           # 缩放增强
        degrees=10,                          # 旋转增强 (+/- 10度)
        
        # --- 其他设置 ---
        project='yolo_cls_optimization',     # 项目名称 (结果保存在此文件夹)
        name='yolo11n_224_pretrained',       # 实验名称
        exist_ok=True                        # 覆盖同名实验
    )

if __name__ == '__main__':
    train()
