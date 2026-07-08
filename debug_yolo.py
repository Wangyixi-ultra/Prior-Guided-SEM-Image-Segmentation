from ultralytics import YOLO
import sys

weights_122 = '/home/chen/seg6/yolo_cls_optimization/train21_l_aug5/weights/best.pt'
weights_119_cls = '/home/chen/seg6/perovskite_grains_opt/yolo_cbam_s_128/weights/best.pt'

print(f"Checking YOLO weights...")

try:
    model_122 = YOLO(weights_122)
    print(f"Model 122 (used in 122.py) names: {model_122.names}")
except Exception as e:
    print(f"Error loading 122: {e}")

try:
    model_119_cls = YOLO(weights_119_cls)
    print(f"Model 119 cls (used in 119.py) names: {model_119_cls.names}")
except Exception as e:
    print(f"Error loading 119 cls: {e}")
