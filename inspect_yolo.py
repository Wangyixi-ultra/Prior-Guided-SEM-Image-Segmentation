from ultralytics import YOLO
import yaml

try:
    model = YOLO('yolo11s-cls.pt')
    if hasattr(model.model, 'yaml'):
        print(yaml.dump(model.model.yaml))
    else:
        print("Model has no yaml attribute directly accessible, accessing cfg")
        print(model.cfg)
except Exception as e:
    print(f"Error: {e}")

# Check available modules
try:
    from ultralytics.nn.modules import CBAM
    print("CBAM is available.")
except ImportError:
    print("CBAM is NOT available in ultralytics.nn.modules directly.")

try:
    from ultralytics.nn.modules import CA
    print("CA (Coordinate Attention) is available.")
except ImportError:
    print("CA is NOT available.")
