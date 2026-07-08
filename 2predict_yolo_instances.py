import os
import glob
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm

def main():
    # Paths
    weights_path = '/home/chen/runs/classify/train20/weights/best.pt'
    source_dir = '/home/chen/seg6/processed_instances_output/instance_crops'
    output_dir = '/home/chen/seg6/processed_instances_output/yolo_predictions'
    
    # Check if weights exist (using try/except block during load as we can't check explicitly if outside workspace restricted area, but Python can)
    print(f"Loading model from {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try looking in the root of the train dir if weights dir doesn't exist? 
        # But standard structure is weights/best.pt
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, ext)))
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    if not image_files:
        print("No images found.")
        return

    # Run inference
    # We can pass the list of images or the directory to model.predict
    # Running on directory is usually faster/optimized
    
    # Results storage
    results_data = []

    print("Running inference...")
    # predict method can verify the source automatically
    # stream=True to handle large datasets effectively
    results = model.predict(source=source_dir, stream=True)
    
    for result in results:
        path = result.path
        filename = os.path.basename(path)
        
        # Classification result
        if result.probs is not None:
            top1 = result.probs.top1
            top1conf = result.probs.top1conf.item()
            cls_name = result.names[top1]
            
            results_data.append({
                'filename': filename,
                'predicted_class': cls_name,
                'confidence': top1conf,
                'probs': result.probs.data.tolist()
            })
            
            # Save plotted result
            # result.save(filename=os.path.join(output_dir, filename)) # This saves to default runs/ detect/
            # To save manually:
            result.save(filename=os.path.join(output_dir, "pred_" + filename))

        # If it happens to be detection/segmentation (just in case), probs is None
        elif result.boxes is not None:
             # It's a detection model
            json_result = result.tojson()
            results_data.append({
                'filename': filename,
                'detection': json_result
            })
            result.save(filename=os.path.join(output_dir, "pred_" + filename))

    # Save summary validity to CSV
    if results_data:
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")
        print(f"Annotated images saved to {output_dir}")

if __name__ == '__main__':
    main()
