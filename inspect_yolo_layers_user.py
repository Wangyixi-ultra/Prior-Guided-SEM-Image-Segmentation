from ultralytics import YOLO
import torch

yolo_weights = '/home/chen/seg6/yolo_cls_optimization/train21_l_aug5/weights/best.pt'
try:
    model = YOLO(yolo_weights)
    print(f"Task: {model.task}")
    print(f"Names: {model.names}")
    
    # Print the model architecture
    print("\nModel Architecture:")
    print(model.model)
    
except Exception as e:
    print(f"Error loading model: {e}")
