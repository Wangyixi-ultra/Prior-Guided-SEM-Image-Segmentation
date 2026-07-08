from ultralytics import YOLO
import torch

# Load a model
model = YOLO('/home/chen/seg6/yolo_cls_optimization/train21_l_aug5/weights/best.pt')

# Access the Detect Head
head = model.model.model[-1]
print("Head Type:", type(head))
print(head)

# Check for cv2 and cv3
if hasattr(head, 'cv2'):
    print("\nhas cv2 (box): Yes")
    print("cv2 length:", len(head.cv2))
    print(head.cv2[0])

if hasattr(head, 'cv3'):
    print("\nhas cv3 (cls): Yes")
    print("cv3 length:", len(head.cv3))
    print(head.cv3[0])

# Check if there are other layers like dfl or cl
print("\nSubmodules:")
for name, mod in head.named_children():
    print(f"{name}: {type(mod)}")
