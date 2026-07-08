# -*- coding: utf-8 -*-
"""
Batch analysis script for perovskite thin-film SEM images (lightweight version).
Outputs only 9 core parameters, ordered ABX3 -> PbI2.
"""
import os
import re
import json
import glob
import logging
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage import measure, morphology
from scipy.ndimage import distance_transform_edt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# =============================================================================
# 1. Parameter configuration
# =============================================================================
IMAGE_DIR = "D:/article_sem/image"
JSON_DIR = "D:/article_sem/image"
EXCEL_PATH = "D:/article_sem/article_data3.30.xlsx"
OUTPUT_DIR = "D:/article_sem/output"
VIS_DIR = os.path.join(OUTPUT_DIR, "vis")
SAVE_VISUALIZATION = True

SUPPORTED_IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
MIN_GRAIN_AREA_PIXEL = 20

# Distance threshold (pixels) for external PbI2 judgement
GRAIN_BOUNDARY_BUFFER_PIXEL = 5

LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "processing_log.txt")

# =============================================================================
# 2. Helper functions
# =============================================================================

def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Script started.")

def parse_scale_text(text):
    if not isinstance(text, str):
        return np.nan
    cleaned_text = re.sub(r'[^\d.-]', '', text)
    try:
        return float(cleaned_text)
    except (ValueError, TypeError):
        return np.nan

def robust_read_excel(path):
    try:
        df = pd.read_excel(path, engine='openpyxl')
        logging.info(f"Successfully read Excel file: {path}")
    except FileNotFoundError:
        logging.error(f"Excel file not found: {path}")
        return None
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        return None

    original_columns = df.columns.tolist()
    normalized_columns = {}
    column_mapping = {
        'name': r'^name$',
        'num': r'^num$',
        'scale_text_um': r'scale[\s_]*text.*',
        'scale_bar_pixel_length': r'scale[\s_]*bar[\s_]*pixel[\s_]*length.*',
        'um_per_pixel': r'μm[\s_]*per[\s_]*pixel.*|um[\s_]*per[\s_]*pixel.*'
    }
    for standard_name, pattern in column_mapping.items():
        for col in original_columns:
            if re.search(pattern, col, re.IGNORECASE):
                normalized_columns[col] = standard_name
                logging.info(f"Excel column mapping: '{col}' -> '{standard_name}'")
                break
    df.rename(columns=normalized_columns, inplace=True)

    required_cols = ['name', 'scale_text_um', 'scale_bar_pixel_length']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"Excel file is missing the following key columns: {missing_cols}. Subsequent calculations may fail.")
    return df

def get_image_num(filename):
    # Supports two naming formats:
    #   - "1-1.jpg", "60-1.png"  -> returns "1-1", "60-1"
    #   - "60.png", "61.jpg"     -> returns "60", "61"
    match = re.search(r'^(\d+(?:-\d+)?)', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def parse_labelme_json(json_path, image_shape):
    def normalize_label(raw_label):
        if not isinstance(raw_label, str):
            return None
        subscript_map = str.maketrans({
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        })
        normalized = raw_label.translate(subscript_map)
        normalized = re.sub(r'\s+', '', normalized)
        normalized = normalized.upper()
        if normalized == 'ABO3':
            normalized = 'ABX3'
        return normalized

    masks = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Unable to read or parse JSON file {json_path}: {e}")
        return None

    if 'shapes' not in data:
        logging.warning(f"JSON file {json_path} has no 'shapes' field.")
        return {}

    for shape in data['shapes']:
        label = shape.get('label')
        points = shape.get('points')
        shape_type = shape.get('shape_type')
        if not label or not points or shape_type != 'polygon':
            continue
        label = normalize_label(label)
        if not label:
            continue
        if label not in masks:
            masks[label] = np.zeros(image_shape[:2], dtype=np.uint8)
        try:
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(masks[label], [polygon], 1)
        except Exception as e:
            logging.warning(f"Unable to create polygon for label '{shape.get('label')}' in {json_path}: {e}")
            continue
    return masks

def calculate_pbi2_spatial_uniformity_score(pbi2_grains, image_shape):
    if len(pbi2_grains) < 2:
        return 0
    h, w = image_shape[:2]
    if h <= 0 or w <= 0:
        return 0
    centroids = np.array([
        [g.centroid[1] / w, g.centroid[0] / h] for g in pbi2_grains
    ], dtype=np.float64)
    dist_matrix = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    nn_dist = np.min(dist_matrix, axis=1)
    nn_mean = np.mean(nn_dist)
    spacing_score = 1 / (1 + np.std(nn_dist) / nn_mean) if nn_mean > 0 else 0
    x_idx = np.clip((centroids[:, 0] * 3).astype(int), 0, 2)
    y_idx = np.clip((centroids[:, 1] * 3).astype(int), 0, 2)
    grid_counts = np.zeros((3, 3), dtype=np.float64)
    for gx, gy in zip(x_idx, y_idx):
        grid_counts[gy, gx] += 1
    mean_count = np.mean(grid_counts)
    coverage_score = 1 / (1 + np.std(grid_counts) / mean_count) if mean_count > 0 else 0
    return float((0.5 * spacing_score + 0.5 * coverage_score) * 100)

def natural_sort_key(s):
    s = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# =============================================================================
# 3. Core analysis function (only 9 parameters are computed)
# =============================================================================

def analyze_image(img_path, json_path, scale_info):
    logging.info(f"--- Started processing image: {os.path.basename(img_path)} ---")

    try:
        img_bytes = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("Unable to decode image")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logging.error(f"Failed to read or convert image {img_path}: {e}")
        return None

    masks = parse_labelme_json(json_path, gray_image.shape)
    if masks is None:
        return None

    abx3_mask = masks.get('ABX3', np.zeros(gray_image.shape, dtype=np.uint8))
    pbi2_mask = masks.get('PBI2', np.zeros(gray_image.shape, dtype=np.uint8))

    um_per_pixel = scale_info.get('um_per_pixel')
    if pd.isna(um_per_pixel) or um_per_pixel == 0:
        scale_text = parse_scale_text(str(scale_info.get('scale_text_um')))
        pixel_len = scale_info.get('scale_bar_pixel_length')
        if pd.notna(scale_text) and pd.notna(pixel_len) and pixel_len > 0:
            um_per_pixel = scale_text / pixel_len
        else:
            logging.error(f"Unable to calculate um_per_pixel for {os.path.basename(img_path)}")
            return None
    pixel_area_to_um2 = um_per_pixel ** 2

    pbi2_total_area_pixel = np.sum(pbi2_mask)
    abx3_total_area_pixel = np.sum(abx3_mask)

    abx3_labels, num_abx3 = measure.label(abx3_mask, connectivity=2, return_num=True)
    pbi2_labels, num_pbi2 = measure.label(pbi2_mask, connectivity=2, return_num=True)

    abx3_props = measure.regionprops(abx3_labels, intensity_image=gray_image)
    pbi2_props = measure.regionprops(pbi2_labels)

    abx3_grains_all = [p for p in abx3_props if p.area > 0]
    abx3_grains = [p for p in abx3_props if p.area >= MIN_GRAIN_AREA_PIXEL]
    pbi2_grains = [p for p in pbi2_props if p.area >= MIN_GRAIN_AREA_PIXEL]

    summary_results = {
        'image_name': os.path.basename(img_path),
        'name': scale_info.get('name', ''),
        'num': scale_info.get('num', ''),
    }

    # 1. ABX3_grain_size_mean_actual_area_um2
    if abx3_grains_all:
        summary_results['ABX3_grain_size_mean_actual_area_um2'] = float(
            np.mean([p.area * pixel_area_to_um2 for p in abx3_grains_all])
        )
    else:
        summary_results['ABX3_grain_size_mean_actual_area_um2'] = 0

    # 2. ABX3_gray_cv
    if abx3_total_area_pixel > 0:
        abx3_pixels_gray = gray_image[abx3_mask == 1]
        gray_mean = np.mean(abx3_pixels_gray)
        summary_results['ABX3_gray_cv'] = float(
            np.std(abx3_pixels_gray) / gray_mean
        ) if gray_mean > 0 else 0
    else:
        summary_results['ABX3_gray_cv'] = 0

    # 4. PbI2_to_ABX3_area_ratio
    summary_results['PbI2_to_ABX3_area_ratio'] = (
        pbi2_total_area_pixel / abx3_total_area_pixel if abx3_total_area_pixel > 0 else 0
    )

    # 4.5 PbI2_to_ABX3_count_ratio
    summary_results['PbI2_to_ABX3_count_ratio'] = (
        len(pbi2_grains) / len(abx3_grains) if abx3_grains else 0
    )

    # 5. PbI2_ABX3_associated_fraction (Grain-Boundary-Associated PbI2 Fraction)
    # A(PbI2 grains with GRAIN_BOUNDARY_BUFFER_PIXEL < dist <= 50 nm and overlap < 5%) / A(total PbI2)
    # Note: 5 px excludes annotation noise at mask edges; 50 nm is the physical upper limit for grain-boundary association.
    PbI2_ABX3_associated_fraction = 0
    if pbi2_grains and num_abx3 > 0:
        dist_to_abx3_pixel = distance_transform_edt(abx3_mask == 0)
        # Pixel distance corresponding to 50 nm (um_per_pixel is in um/px)
        d_gb_px = 0.05 / um_per_pixel
        for pbi2_grain in pbi2_grains:
            grain_coords = pbi2_grain.coords
            overlap_pixel_count = np.count_nonzero(abx3_mask[grain_coords[:, 0], grain_coords[:, 1]])
            overlap_fraction = overlap_pixel_count / pbi2_grain.area if pbi2_grain.area > 0 else 0
            grain_mask = (pbi2_labels == pbi2_grain.label)
            min_dist_pixel = float(np.min(dist_to_abx3_pixel[grain_mask]))
            if overlap_fraction < 0.05 and GRAIN_BOUNDARY_BUFFER_PIXEL < min_dist_pixel <= d_gb_px:
                PbI2_ABX3_associated_fraction += pbi2_grain.area
    summary_results['PbI2_ABX3_associated_fraction'] = (
        PbI2_ABX3_associated_fraction / pbi2_total_area_pixel if pbi2_total_area_pixel > 0 else 0
    )

    # 6. PbI2_spatial_uniformity_score
    summary_results['PbI2_spatial_uniformity_score'] = calculate_pbi2_spatial_uniformity_score(
        pbi2_grains, gray_image.shape
    )

    # 6.5 PbI2 particle statistics (count and mean area)
    if pbi2_grains:
        summary_results['PbI2_particle_count'] = len(pbi2_grains)
        summary_results['PbI2_mean_particle_area_um2'] = float(
            np.mean([g.area * pixel_area_to_um2 for g in pbi2_grains])
        )
    else:
        summary_results['PbI2_particle_count'] = 0
        summary_results['PbI2_mean_particle_area_um2'] = 0

    # 7. PbI2_large_particle_area_um2 is defined as 1/4 of ABX3_grain_size_mean_actual_area_um2
    summary_results['PbI2_large_particle_area_um2'] = (
        summary_results.get('ABX3_grain_size_mean_actual_area_um2', 0.0) / 4.0
    )

    # Visualization
    if SAVE_VISUALIZATION:
        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)
            if num_abx3 > 0:
                ax.contour(abx3_mask, levels=[0.5], colors='#5AAAE6', linewidths=1)
            if num_pbi2 > 0:
                ax.contour(pbi2_mask, levels=[0.5], colors='#EB784B', linewidths=1)
            boundary_buffer = morphology.dilation(abx3_mask, morphology.disk(GRAIN_BOUNDARY_BUFFER_PIXEL)) - abx3_mask if num_abx3 > 0 else np.zeros(gray_image.shape, dtype=np.uint8)
            if num_abx3 > 0:
                ax.contourf(boundary_buffer, levels=[0.5, 1], colors='yellow', alpha=0.3)
            ax.set_title(f"{os.path.basename(img_path)}\nABX3 Grains: {len(abx3_grains)}, PbI2 Grains: {len(pbi2_grains)}")
            ax.axis('off')
            vis_path = os.path.join(VIS_DIR, os.path.basename(img_path).rsplit('.', 1)[0] + '.png')
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Visualization saved to: {vis_path}")
        except Exception as e:
            logging.error(f"Error creating visualization for {os.path.basename(img_path)}: {e}")

    logging.info(f"--- Finished processing: {os.path.basename(img_path)} ---")
    return summary_results

# =============================================================================
# 4. Main program
# =============================================================================

def main():
    setup_logging()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_VISUALIZATION:
        os.makedirs(VIS_DIR, exist_ok=True)

    scale_df = robust_read_excel(EXCEL_PATH)
    if scale_df is None:
        logging.critical("Unable to read Excel file; program terminated.")
        return
    if 'name' in scale_df.columns:
        scale_df['name'] = scale_df['name'].astype(str)
    else:
        logging.critical("'name' column not found in Excel; program terminated.")
        return

    json_files = glob.glob(os.path.join(JSON_DIR, '*.json'))
    logging.info(f"Found {len(json_files)} JSON files.")

    all_summary_results = []
    for json_path in json_files:
        base_name = os.path.basename(json_path).rsplit('.', 1)[0]
        img_path = None
        for ext in SUPPORTED_IMG_EXTENSIONS:
            potential_path = os.path.join(IMAGE_DIR, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        if not img_path:
            logging.warning(f"Image file for {base_name} not found; skipping.")
            continue

        name_key = get_image_num(base_name)
        if not name_key:
            logging.warning(f"Unable to extract identifier from filename {base_name}; skipping.")
            continue

        scale_info_row = scale_df[scale_df['name'] == name_key]
        if scale_info_row.empty:
            logging.warning(f"No record with name='{name_key}' found in Excel; skipping {base_name}.")
            continue

        scale_info = scale_info_row.iloc[0].to_dict()
        summary = analyze_image(img_path, json_path, scale_info)
        if summary:
            all_summary_results.append(summary)

    if all_summary_results:
        summary_df = pd.DataFrame(all_summary_results)
        summary_df = summary_df.sort_values(by='num', key=lambda col: col.map(natural_sort_key)).reset_index(drop=True)
        summary_output_path = os.path.join(OUTPUT_DIR, 'sem_morphology_summary.xlsx')
        try:
            summary_df.to_excel(summary_output_path, index=False, engine='openpyxl')
            logging.info(f"Image-level summary saved to: {summary_output_path}")
        except PermissionError:
            logging.error(f"Failed to save file: {summary_output_path}. Please check if the file is open in another program (e.g., Excel).")
        except Exception as e:
            logging.error(f"Unknown error while saving image-level summary: {e}")

    logging.info("All tasks completed.")

if __name__ == '__main__':
    main()
