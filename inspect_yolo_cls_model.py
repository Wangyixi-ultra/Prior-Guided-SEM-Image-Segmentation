from ultralytics import YOLO
import torch

yolo_weights = '/home/chen/seg6/yolo_cls_optimization/train21_l_aug5/weights/best.pt'
model = YOLO(yolo_weights)

# Print model structure
print(model.model)

# Also iterate over named_modules to find the head
print("\n--- Named Modules ---")
for name, module in model.model.named_modules():
    print(name, module.__class__.__name__)
