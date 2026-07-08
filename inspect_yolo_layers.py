from ultralytics import YOLO
import sys

try:
    model = YOLO('/home/chen/seg6/yolo11n-cls.pt')
    print("Model detected.")
    print(model.model)
except Exception as e:
    print(f"Error: {e}")
