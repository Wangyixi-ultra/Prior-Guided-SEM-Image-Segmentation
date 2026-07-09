#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnUNet single-channel prediction pipeline.
Paths are set inside the file; no command-line arguments are required.
"""

import cv2, subprocess, shutil, json, os
from pathlib import Path
import numpy as np
from imageio.v2 import imread

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference import data_iterators
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from typing import List, Union
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

# Synchronous preprocessing iterator (avoids multiprocessing issues)
def preprocessing_iterator_fromfiles_synchronous(list_of_lists: List[List[str]],
                                     list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    
    label_manager = plans_manager.get_label_manager(dataset_json)
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)

    if list_of_segs_from_prev_stage_files is None:
        list_of_segs_from_prev_stage_files = [None] * len(list_of_lists)
    if output_filenames_truncated is None:
        output_filenames_truncated = [None] * len(list_of_lists)

    for idx, (data_files, seg_prev, ofile) in enumerate(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)):
        data, seg, data_properties = preprocessor.run_case(data_files,
                                                           seg_prev,
                                                           plans_manager,
                                                           configuration_manager,
                                                           dataset_json)
        if seg_prev is not None:
             seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
             data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data).contiguous().float()
        
        item = {'data': data, 'data_properties': data_properties,
                'ofile': ofile}
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item

data_iterators.preprocessing_iterator_fromfiles = preprocessing_iterator_fromfiles_synchronous

# ========== 1. User-editable section ==========
INPUT_DIR   = Path('/path/to/test_images')
OUTPUT_DIR  = Path('/path/to/predictions')
BORDER_DIR  = Path('/path/to/predictions')
JSON_DIR    = Path('/path/to/predictions')
# nnUNet parameters
DATASET_ID  = 123
CONFIG      = '2d'
FOLD        = 0
TRAINER     = 'nnUNetTrainerUMambaBotActiveContourSemBoost' # single-channel trainer
CHECKPOINT  = 'checkpoint_best.pth'
# ========== 2. Do not modify below ========== 

CLASS_COLOR = {1: (0, 140, 255), 2: (0, 255, 0), 3: (255, 0, 255)}
CLASS_LABELS = {1: "PbI₂", 2: "ABO₃", 3: "defect"}  # class label mapping

def find_imgs(p):
    return sorted([i for i in Path(p).iterdir()
                   if i.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}])

# ---------- Input utilities ----------
def to1ch_uint8(img):
    """Ensure output is single-channel uint8."""
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def prep_input(in_dir, tmp_dir):
    """Save temporary inputs and return a mapping from case names to original image paths."""
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    name_map = {}
    for idx, f in enumerate(find_imgs(in_dir)):
            img_original = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img_original is None:
            try:
                img_original = imread(str(f))
            except:
                pass
                
        if img_original is None:
            print(f"Unable to read file: {f}")
            continue
            
        case_name = f'case{idx:03d}'
        
        # Single channel: Grayscale Image only (no YOLO Channel 1)
        img_0000 = to1ch_uint8(img_original)
        cv2.imwrite(str(tmp_dir / f'{case_name}_0000.png'), img_0000)

        name_map[case_name] = f
    return tmp_dir, name_map

def predict(tmp_dir, out_dir, model_folder):
    """Run nnUNet prediction."""
    print(f"Predicting using model in: {model_folder}")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(FOLD,),
        checkpoint_name=CHECKPOINT
    )
    
    predictor.predict_from_files(
        str(tmp_dir),
        str(out_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

def draw_contour(tmp_dir, out_dir, border_dir):
    border_dir.mkdir(exist_ok=True,parents=True)
    for m in sorted(out_dir.glob('*.png')):
        name = m.stem
        img = cv2.imread(str(tmp_dir/f'{name}_0000.png'), cv2.IMREAD_COLOR)
        mask= imread(m).astype(np.uint8)
        base= img.copy()
        for cls in np.unique(mask):
            if cls==0:continue
            bin = ((mask==cls)*255).astype(np.uint8)
            cnt,_=cv2.findContours(bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base,cnt,-1,CLASS_COLOR.get(cls,(255,255,255)),2)
        cv2.imwrite(str(border_dir/f'{name}_contour.png'), base)

def create_labelme_json(original_img_path, mask, output_dir):
    """Generate a LabelMe-format JSON file from the predicted mask."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    img = cv2.imread(str(original_img_path))
    if img is None:
        try:
            img = imread(str(original_img_path))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Unable to read original image: {original_img_path}, error: {e}")
            return False
            
    if img is None:
        print(f"Unable to read original image: {original_img_path}")
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    success, encoded_img = cv2.imencode('.png', img_rgb)
    if not success:
        print(f"Image encoding failed: {original_img_path}")
        return False
    
    import base64
    imageData = base64.b64encode(encoded_img).decode('utf-8')
    
    height, width = img.shape[:2]
    
    original_name = original_img_path.stem
    
    json_path = output_dir / f"{original_name}.json"
    
    shapes = []
    
    for cls in np.unique(mask):
        if cls == 0:
            continue
            
        label = CLASS_LABELS.get(cls, f"class_{cls}")
        
        bin_mask = ((mask == cls) * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            points = []
            for point in contour:
                x, y = point[0]
                points.append([float(x), float(y)])
            
            if len(points) >= 3:
                shape = {
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None
                }
                shapes.append(shape)
    
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": original_img_path.name,
        "imageData": imageData,
        "imageHeight": height,
        "imageWidth": width
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated JSON annotation file: {json_path}")
    return True

def generate_json_annotations(input_dir, output_dir, json_output_dir, name_map=None):
    """Generate LabelMe JSON annotations in batch."""
    json_output_dir.mkdir(exist_ok=True, parents=True)
    
    mask_files = [f for f in sorted(output_dir.glob('*.png')) if not f.stem.endswith('_contour')]
    
    if not mask_files:
        print("No prediction files found")
        return
    
    print(f"Found {len(mask_files)} predictions, starting JSON annotation generation...")
    
    success_count = 0
    for mask_file in mask_files:
        name = mask_file.stem

        if name.endswith('_contour'):
            continue

        if name_map and name in name_map:
            original_img_path = name_map[name]
            debug_files = [original_img_path]
        else:
            original_files = list(input_dir.glob(f"{name}.*"))
            original_files.extend(list(input_dir.glob(f"{name}_0000.*")))

            if name.startswith('case'):
                try:
                    num = int(name[4:])
                    actual_num = num + 4
                    original_files.extend(list(input_dir.glob(f"case{actual_num:03d}.*")))
                    original_files.extend(list(input_dir.glob(f"case{actual_num:03d}_0000.*")))
                except ValueError:
                    pass

            original_files = list(set(original_files))
            debug_files = original_files

            if not original_files:
                print(f"Debug: found 0 possible original images for {mask_file.name}: []")
                print(f"Warning: no original image corresponding to {mask_file.name} found")
                continue

            original_img_path = original_files[0]

        print(f"Debug: found {len(debug_files)} possible original images for {mask_file.name}: {[f.name for f in debug_files]}")
        
        mask = imread(mask_file).astype(np.uint8)
        
        if create_labelme_json(original_img_path, mask, json_output_dir):
            success_count += 1
    
    print(f"Done! Successfully generated {success_count}/{len(mask_files)} JSON annotation files")

def main():
    # Find dataset directory
    nnunet_results = Path('/path/to/U-Mamba/data/nnUNet_results')
    dataset_dirs = list(nnunet_results.glob(f'Dataset{DATASET_ID}_*'))
    if not dataset_dirs:
        print(f"Dataset{DATASET_ID} not found in {nnunet_results}")
        return
    dataset_dir = dataset_dirs[0]
    print(f"Using dataset: {dataset_dir}")
    
    tmp = INPUT_DIR.parent/'temp_nnUNet'
    try:
        tmp, name_map = prep_input(INPUT_DIR, tmp)
        
        trainers_found = []
        for trainer_dir in sorted(dataset_dir.iterdir()):
            if not trainer_dir.is_dir():
                continue
            if 'nnUNetTrainer' not in trainer_dir.name:
                continue
            
            ckpt_path = trainer_dir / f'fold_{FOLD}' / CHECKPOINT
            if not ckpt_path.exists():
                print(f"Skipping {trainer_dir.name}: {CHECKPOINT} not found")
                continue
                
            parts = trainer_dir.name.split('__')
            trainer_name = parts[0]
            
            print(f"\n======== Starting Trainer: {trainer_name} ========")
            trainers_found.append(trainer_name)

            current_out = OUTPUT_DIR / trainer_name
            current_border = BORDER_DIR / trainer_name
            current_json = JSON_DIR / trainer_name
            
            try:
                predict(tmp, current_out, trainer_dir)
                draw_contour(tmp, current_out, current_border)
                generate_json_annotations(INPUT_DIR, current_out, current_json, name_map)
            except Exception as e:
                print(f"Trainer {trainer_name} processing error: {e}")
                
        if not trainers_found:
            print("No trainer directories with a valid checkpoint found")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('All done!')

if __name__ == '__main__':
    # Set threading limits early to avoid runtime errors during repeated inference
    try:
        if torch.cuda.is_available():
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    main()