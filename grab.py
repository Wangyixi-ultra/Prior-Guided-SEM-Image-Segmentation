#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis script for grain segmentation results.
Target directory: predict_no_label/experiment/border

Features:
- Count grains for Class 1 and Class 2.
- Compute Class 2 geometry: Area, Perimeter (implied), Centroids.
- Sort Class 2 grains (Top-Left to Bottom-Right) and label them on images.
- Compute inter-grain distances (centroids) for Class 2.
- Compute grain angularity (polygon inner angles) for Class 2.
- Compare Area Ratio (Class 2 / Class 1).
- NEW: Ratio of Perovskite (Class 2) grain count to Lead Iodide (Class 1) grain count.
- NEW: Average size of Perovskite grains.
- NEW: Perovskite grain flatness (roughness) based on grayscale intensity StdDev.
- NEW: Average number of edges of Perovskite grains.
- Generate summary CSV and Visualization plots.
"""

import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio.v2 import imread

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# ================= Configuration =================
CONFIG = {
    'dirs': {
        'border': '/home/chen/seg6/predict_no_label/experiment/border122/nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost',  # Input contours
        'mask':   '/home/chen/seg6/predict_no_label/experiment/out122/nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost',     # Input masks
        'output': '/home/chen/seg6/predict_no_label/experiment/json122/nnUNetTrainerUMambaBotActiveContourDualChannelSemBoost',   # Output results
        'grayscale': '/home/chen/seg6/predict_no_label/experiment/image_for_grab' # Path to original grayscale images
    },
    'plot_colors': {
        'c1': (149/255, 167/255, 126/255), # PbI2 (Class 1) Area
        'c2': (245/255, 166/255, 115/255), # ABX3 (Class 2) Area
        'ratio': (224/255, 210/255, 229/255), # Ratio
        'dist': (115/255, 107/255, 157/255), # (Unused in plots)
        'angle': (62/255, 134/255, 181/255), # (Unused in plots)
        'size': (139/255, 194/255, 126/255), # Color for avg size
    },
    'target_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12],  # IDs to include in detailed bar charts
    'pbi2_abx3_buffer_pixel': 1,  # Treat PbI2 touching this ABX3 buffer as inside/related
    # Optional: custom plotting order, accepts file names (without extension) or IDs.
    # Example: ['12', '3', '7'] or ['img12', 'img03', 'img07']
    'plot_image_order': [],
    # Optional: x tick labels aligned with the final plotting order.
    # If provided, length should match plotted sample count.
    'plot_xtick_labels': [],
    # Optional: map file name or ID to x tick label. Used when plot_xtick_labels is empty.
    # Example: {'12': 'A', 'img03': 'B'}
    'plot_xtick_label_map': {'case001': 'A', 'case005': 'B', 'case009': 'C','case011': 'D', 'case003': 'E', 'case002': 'F'},
}
# =================================================

class GeometryUtils:
    """Utilities for geometric calculations on contours."""
    
    @staticmethod
    def get_centroid(contour):
        """Calculate centroid (cx, cy) of a contour."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            return (M["m10"] / M["m00"], M["m01"] / M["m00"])
        # Fallback to bounding box center
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w / 2, y + h / 2)

    @staticmethod
    def sort_contours(contours):
        """Sort contours from top-left to bottom-right (Primary: Y, Secondary: X)."""
        boxes = []
        for cnt in contours:
            c = GeometryUtils.get_centroid(cnt)
            # Store as (y, x, contour) to sort by y then x
            boxes.append((c[1], c[0], cnt))
        
        boxes.sort(key=lambda b: (b[0], b[1]))
        return [b[2] for b in boxes]

    @staticmethod
    def calculate_inter_grain_distances(centroids):
        """Calculate average pairwise distance between all centroids."""
        if len(centroids) < 2:
            return 0
        
        points = np.array(centroids)
        # Compute pairwise distance matrix efficiently
        # dist[i, j] = sqrt((x1-x2)^2 + (y1-y2)^2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        dist_matrix = np.sqrt(sq_dist)
        
        # Get upper triangle indices (excluding diagonal) to avoid duplicates and self-distance
        upper_indices = np.triu_indices(len(points), k=1)
        distances = dist_matrix[upper_indices]
        
        return np.mean(distances) if len(distances) > 0 else 0

    @staticmethod
    def calculate_polygon_avg_angle(contour):
        """Calculate the average internal angle of the polygon approximation."""
        if len(contour) < 3:
            return 0
        
        pts = contour.reshape(-1, 2)
        n = len(pts)
        angles = []
        
        for i in range(n):
            p_prev = pts[i-1]
            p_curr = pts[i]
            p_next = pts[(i+1)%n]
            
            # Vectors pointing away from current point
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_theta))
                angles.append(angle_deg)
        
        return np.mean(angles) if angles else 0

    @staticmethod
    def calculate_edges_count(contour):
        """Calculate number of edges using polygon approximation."""
        # epsilon is the maximum distance from contour to approximated contour
        # 0.02 * arcLength is a common starting point
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return len(approx)

    @staticmethod
    def _calculate_pbi2_spatial_uniformity_components(pbi2_grains, image_shape):
        """
        Return normalized component scores in [0, 1]:
        1) nearest-neighbor spacing uniformity
        2) 3x3 grid occupancy uniformity
        """
        if len(pbi2_grains) < 2:
            return 0.0, 0.0

        h, w = image_shape[:2]
        if h <= 0 or w <= 0:
            return 0.0, 0.0

        centroids = np.asarray(pbi2_grains, dtype=np.float64)
        if centroids.ndim != 2 or centroids.shape[1] != 2:
            return 0.0, 0.0

        # Use normalized centroid coordinates to remove image-size dependency.
        centroids = np.array([[c[0] / w, c[1] / h] for c in centroids], dtype=np.float64)

        # A) Nearest-neighbor spacing uniformity
        dist_matrix = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dist = np.min(dist_matrix, axis=1)

        nn_mean = np.mean(nn_dist)
        if nn_mean > 0:
            nn_cv = np.std(nn_dist) / nn_mean
            spacing_score = 1 / (1 + nn_cv)
        else:
            spacing_score = 0.0

        # B) 3x3 grid occupancy uniformity
        x_idx = np.clip((centroids[:, 0] * 3).astype(int), 0, 2)
        y_idx = np.clip((centroids[:, 1] * 3).astype(int), 0, 2)
        grid_counts = np.zeros((3, 3), dtype=np.float64)
        for gx, gy in zip(x_idx, y_idx):
            grid_counts[gy, gx] += 1

        mean_count = np.mean(grid_counts)
        if mean_count > 0:
            grid_cv = np.std(grid_counts) / mean_count
            coverage_score = 1 / (1 + grid_cv)
        else:
            coverage_score = 0.0

        return float(spacing_score), float(coverage_score)

class ImageProcessor:
    """Handles image processing, statistic extraction, and result image generation."""
    
    def __init__(self, config):
        self.config = config
        os.makedirs(config['dirs']['output'], exist_ok=True)

    @staticmethod
    def _find_existing_file(base_dir, base_name, extensions):
        for ext in extensions:
            candidate = os.path.join(base_dir, f'{base_name}{ext}')
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _extract_file_id(name_core):
        match = re.search(r'\d+', name_core)
        return int(match.group(0)) if match else -1

    def _load_grayscale_image(self, name_core, file_id):
        if file_id == -1:
            return None

        # Keep search order compatible with previous behavior.
        candidate_bases = [f'{file_id:02d}', str(file_id), name_core]
        seen = set()
        for base in candidate_bases:
            for ext in IMAGE_EXTENSIONS:
                key = (base, ext)
                if key in seen:
                    continue
                seen.add(key)
                path = os.path.join(self.config['dirs']['grayscale'], f'{base}{ext}')
                if os.path.exists(path):
                    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        example_path = os.path.join(self.config['dirs']['grayscale'], f'{candidate_bases[0]}{IMAGE_EXTENSIONS[0]}')
        print(f"[Info] Grayscale image not found for ID {file_id}. Example search path: {example_path}")
        return None

    def process_all(self):
        files = []
        for ext in IMAGE_EXTENSIONS:
            pattern = os.path.join(self.config['dirs']['border'], f'*_contour{ext}')
            files.extend(glob.glob(pattern))
        files = sorted(files)
        
        if not files:
            print(f"No *_contour.* files found in {self.config['dirs']['border']}")
            return [], [], [], [], [], []

        file_stats = []
        all_cls2_areas = []
        all_cls2_distances = []
        all_cls2_angles = []
        all_cls2_flatness = []
        all_cls2_edges = []

        print(f"Found {len(files)} files. Starting processing...")

        for contour_path in files:
            result, areas, dist, angle, flatness_list, edges_list = self.process_single_file(contour_path)
            if result:
                file_stats.append(result)
                all_cls2_areas.extend(areas)
                if areas: # Only add global stats if class 2 exists
                    all_cls2_distances.append(dist)
                    all_cls2_angles.append(angle)
                    all_cls2_flatness.extend(flatness_list)
                    all_cls2_edges.extend(edges_list)

        return file_stats, all_cls2_areas, all_cls2_distances, all_cls2_angles, all_cls2_flatness, all_cls2_edges

    def process_single_file(self, contour_path):
        # 1. Path Parsing
        filename = os.path.basename(contour_path)
        name_core = os.path.splitext(filename)[0].replace('_contour', '')
        
        # Extract numeric ID
        file_id = self._extract_file_id(name_core)
        
        mask_path = self._find_existing_file(self.config['dirs']['mask'], name_core, IMAGE_EXTENSIONS)
                
        if not mask_path:
            print(f"[Warn] Mask not found for {name_core}")
            return None, [], 0, 0, [], []

        gray_img = self._load_grayscale_image(name_core, file_id)

        # 2. Basic Processing
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0] # Assume grayscale/index is in the first channel
        mask = mask.astype(np.uint8)
        
        # Pixel-based Area Calculation (More accurate for total area/ratio)
        c1_pixel_area = np.count_nonzero(mask == 1)
        c2_pixel_area = np.count_nonzero(mask == 2)
        
        # Get contours for Grain Stats
        bin_mask1 = (mask == 1).astype(np.uint8) * 255
        cnts1, _ = cv2.findContours(bin_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        bin_mask2 = (mask == 2).astype(np.uint8) * 255
        cnts2, _ = cv2.findContours(bin_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 9. PbI2 与 ABX3 空间关系（二分类）
        # - outside: 与 ABX3 既不重叠也不接触其缓冲区
        # - inside: 与 ABX3 有任意重叠，或接触 ABX3 缓冲区
        # Note: In single-label masks, class 1 and class 2 are usually mutually exclusive,
        # so overlap-only logic can overestimate outside ratio.
        pbi2_inside_abx3_count = 0
        pbi2_outside_abx3_count = 0
        abx3_mask_bool = (mask == 2)

        buffer_px = int(self.config.get('pbi2_abx3_buffer_pixel', 1))
        if buffer_px > 0:
            kernel_size = buffer_px * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            abx3_buffer_bool = cv2.dilate(abx3_mask_bool.astype(np.uint8), kernel, iterations=1) > 0
        else:
            abx3_buffer_bool = abx3_mask_bool

        if cnts1:
            for pbi2_cnt in cnts1:
                pbi2_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                cv2.drawContours(pbi2_mask, [pbi2_cnt], -1, 1, -1)
                pbi2_region = (pbi2_mask == 1)

                overlap_pixel_count = np.count_nonzero(abx3_mask_bool & pbi2_region)
                contact_pixel_count = np.count_nonzero(abx3_buffer_bool & pbi2_region)

                if overlap_pixel_count > 0 or contact_pixel_count > 0:
                    pbi2_inside_abx3_count += 1
                else:
                    pbi2_outside_abx3_count += 1

        num_pbi2_grains = len(cnts1)
        pbi2_inside_abx3_ratio = (
            pbi2_inside_abx3_count / num_pbi2_grains if num_pbi2_grains > 0 else 0
        )
        pbi2_outside_abx3_ratio = (
            pbi2_outside_abx3_count / num_pbi2_grains if num_pbi2_grains > 0 else 0
        )

        # 3. Class 1 Stats
        c1_count = len(cnts1)
        # For average grain area, use contours. For total area, use pixels.
        c1_avg_area = (sum(cv2.contourArea(c) for c in cnts1) / c1_count) if c1_count > 0 else 0
        c1_sizes = [2 * np.sqrt(cv2.contourArea(c) / np.pi) for c in cnts1 if cv2.contourArea(c) > 0]

        c1_centroids = [GeometryUtils.get_centroid(c) for c in cnts1]
        pbi2_spacing_uniformity, pbi2_grid_uniformity = GeometryUtils._calculate_pbi2_spatial_uniformity_components(
            c1_centroids, mask.shape
        )
        # Avoid recomputing component metrics a second time.
        pbi2_spatial_uniformity = (0.5 * pbi2_spacing_uniformity + 0.5 * pbi2_grid_uniformity) * 100
        
        # 4. Class 2 Stats (Detailed)
        c2_areas = []
        c2_centroids = []
        c2_angles = []
        c2_labeled_contours = []
        c2_sizes = [] # Equivalent Circular Diameter = 2 * sqrt(Area / pi)
        c2_flatness = []
        c2_edges = []

        if cnts2:
            sorted_cnts2 = GeometryUtils.sort_contours(cnts2)
            for idx, cnt in enumerate(sorted_cnts2):
                area = cv2.contourArea(cnt)
                centroid = GeometryUtils.get_centroid(cnt)
                angle = GeometryUtils.calculate_polygon_avg_angle(cnt)
                
                size = 2 * np.sqrt(area / np.pi)
                edges = GeometryUtils.calculate_edges_count(cnt)

                # Flatness calculation
                flatness = 0
                if gray_img is not None:
                     # Create a mask for just this contour
                    c_mask = np.zeros_like(gray_img)
                    cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                    # Extract pixels within the contour
                    pixels = gray_img[c_mask == 255]
                    if len(pixels) > 0:
                        flatness = np.std(pixels)
                
                c2_areas.append(area)
                c2_centroids.append(centroid)
                c2_angles.append(angle)
                c2_sizes.append(size)
                c2_labeled_contours.append((cnt, idx + 1)) # 1-based index
                c2_edges.append(edges)
                c2_flatness.append(flatness)

        # Aggregates for this file
        avg_dist = GeometryUtils.calculate_inter_grain_distances(c2_centroids)
        avg_angle = np.mean(c2_angles) if c2_angles else 0
        avg_flatness = np.mean(c2_flatness) if c2_flatness else 0
        avg_edges = np.mean(c2_edges) if c2_edges else 0

        # Use Pixel Areas for Ratio and Total
        c1_total_area = c1_pixel_area
        c2_total_area = c2_pixel_area
        
        # Ratio
        ratio_area = (c2_total_area / c1_total_area) if c1_total_area > 0 else 0
        ratio_count = (len(cnts2) / c1_count) if c1_count > 0 else 0

        stats = {
            'file': name_core,
            'id': file_id,
            'class1_count': c1_count,
            'class2_count': len(cnts2),
            'class2_to_cls1_count_ratio': ratio_count, # Metric 2
            'class1_total_area': c1_total_area,
            'class1_avg_area': c1_avg_area,
            'class1_avg_size': np.mean(c1_sizes) if c1_sizes else 0,
            'class2_total_area': c2_total_area,
            'class2_to_cls1_area_ratio': ratio_area, # Metric 1
            'class2_avg_area': np.mean(c2_areas) if c2_areas else 0, # Metric 3
            'class2_avg_size': np.mean(c2_sizes) if c2_sizes else 0,
            'class2_avg_flatness': avg_flatness, # Metric 4
            'class2_avg_edges': avg_edges, # Metric 5
            'PbI2_inside_ABX3_count': pbi2_inside_abx3_count,
            'PbI2_outside_ABX3_count': pbi2_outside_abx3_count,
            'PbI2_inside_ABX3_ratio': pbi2_inside_abx3_ratio,
            'PbI2_outside_ABX3_ratio': pbi2_outside_abx3_ratio,
            'PbI2_overlap_or_inside_count': pbi2_inside_abx3_count,
            'PbI2_overlap_or_inside_ratio': pbi2_inside_abx3_ratio,
            'pbi2_outside_abx3_ratio': pbi2_outside_abx3_ratio,
            'pbi2_spacing_uniformity': pbi2_spacing_uniformity * 100,
            'pbi2_grid_uniformity': pbi2_grid_uniformity * 100,
            'pbi2_spatial_uniformity': pbi2_spatial_uniformity,
            'class2_area_variance': np.var(c2_areas) if c2_areas else 0,
            'class2_avg_distance': avg_dist,
            'class2_avg_angle': avg_angle
        }

        # 5. Draw Result Image
        if c2_labeled_contours:
            self.draw_labels(contour_path, c2_labeled_contours)

        return stats, c2_areas, avg_dist, avg_angle, c2_flatness, c2_edges

    def draw_labels(self, contour_path, indexed_contours):
        img = cv2.imread(contour_path)
        if img is None: return

        for cnt, idx in indexed_contours:
            c = GeometryUtils.get_centroid(cnt)
            # Draw ID at centroid
            cv2.putText(img, str(idx), (int(c[0]), int(c[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        filename = os.path.basename(contour_path)
        name_ext = os.path.splitext(filename)[1]
        outfile = filename.replace(f'_contour{name_ext}', f'_labeled{name_ext}')
        outpath = os.path.join(self.config['dirs']['output'], outfile)
        cv2.imwrite(outpath, img)
        # print(f"Saved labeled image: {outpath}")

class Visualizer:
    """Handles plotting and reporting of statistical data."""
    
    def __init__(self, config, df, all_areas, all_dists, all_angles, all_flatness, all_edges):
        self.config = config
        self.df = df
        self.all_areas = all_areas
        self.all_dists = all_dists
        self.all_angles = all_angles
        self.all_flatness = all_flatness
        self.all_edges = all_edges
        self.colors = config['plot_colors']

    def print_summary(self):
        print('\n' + '='*10 + ' Dataset Summary ' + '='*10)
        c1_total = self.df['class1_count'].sum()
        c2_total = self.df['class2_count'].sum()
        
        print(f"Class 1 (PbI2) grain count Total: {c1_total}")
        print(f"Class 2 (ABX3) grain count Total: {c2_total}")
        
        # Metric 2
        overall_count_ratio = c2_total / c1_total if c1_total > 0 else 0
        print(f"Overall Grain Count Ratio (ABX3 / PbI2): {overall_count_ratio:.4f}")
        
        print(f"Total Class 2 grains processed for stats: {len(self.all_areas)}")

        if self.all_areas:
            print(f'\n-------- Class 2 Statistics (Per Grain) --------')
            print(f'Average Area (Metric 3): {np.mean(self.all_areas):.2f} px')
            print(f'Area Variance:           {np.var(self.all_areas):.2f} px²')
            print(f'Avg Inter-Dist:          {np.mean(self.all_dists):.2f} px')
            print(f'Avg Angle:               {np.mean(self.all_angles):.2f} degrees')
            
            # Metric 4
            if self.all_flatness:
                print(f'Avg Flatness (Metric 4): {np.mean(self.all_flatness):.2f} (StdDev of Intensity)')
            else:
                print('Avg Flatness (Metric 4): N/A (Grayscale images not found)')

            # Metric 5
            if self.all_edges:
                print(f'Avg Edges (Metric 5):    {np.mean(self.all_edges):.2f}')

        c1_area_total = self.df['class1_total_area'].sum()
        c2_area_total = self.df['class2_total_area'].sum()
        
        # Metric 1
        overall_area_ratio = c2_area_total / c1_area_total if c1_area_total > 0 else 0
        
        print(f'\n-------- Global Area Ratio (Metric 1) --------')
        print(f'Total Class 1 Area: {c1_area_total:.2f}')
        print(f'Total Class 2 Area: {c2_area_total:.2f}')
        print(f'Overall Area Ratio (Class 2 / Class 1): {overall_area_ratio:.4f}')

        if 'PbI2_outside_ABX3_count' in self.df.columns and 'class1_count' in self.df.columns:
            total_pbi2 = self.df['class1_count'].sum()
            total_outside = self.df['PbI2_outside_ABX3_count'].sum()
            outside_ratio = total_outside / total_pbi2 if total_pbi2 > 0 else 0
            print(f'Overall PbI2 Outside ABX3 Ratio: {outside_ratio:.4f}')

        csv_path = os.path.join(self.config['dirs']['output'], 'class2_stats.csv')
        self.df.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"\nStats saved to: {csv_path}")

    def _reorder_df_for_plot(self, df_target):
        """Reorder plot rows by CONFIG['plot_image_order'] while keeping unmatched rows."""
        order_cfg = self.config.get('plot_image_order', [])
        if not order_cfg:
            return self._reorder_df_by_label_map(df_target)

        ordered_tokens = [str(x).strip() for x in order_cfg if str(x).strip()]
        if not ordered_tokens:
            return self._reorder_df_by_label_map(df_target)

        work = df_target.copy()
        work['_id_str'] = work['id'].astype(str)
        work['_file_str'] = work['file'].astype(str)

        used_idx = set()
        ordered_parts = []
        for token in ordered_tokens:
            matched = work[(work['_id_str'] == token) | (work['_file_str'] == token)]
            if not matched.empty:
                matched = matched.loc[~matched.index.isin(used_idx)]
                if not matched.empty:
                    ordered_parts.append(matched)
                    used_idx.update(matched.index.tolist())

        remaining = work.loc[~work.index.isin(used_idx)].sort_values('id')
        if ordered_parts:
            ordered = pd.concat(ordered_parts + [remaining], axis=0)
        else:
            ordered = remaining

        return ordered.drop(columns=['_id_str', '_file_str'])

    def _reorder_df_by_label_map(self, df_target):
        """Fallback reorder: follow plot_xtick_label_map value order (for example A->B->C)."""
        label_map_cfg = self.config.get('plot_xtick_label_map', {})
        if not isinstance(label_map_cfg, dict) or not label_map_cfg:
            return df_target.sort_values('id').copy()

        # Python dict keeps insertion order, so this respects user-defined label order.
        label_order = list(dict.fromkeys(str(v) for v in label_map_cfg.values()))
        label_rank = {label: idx for idx, label in enumerate(label_order)}

        work = df_target.copy()
        mapped_labels = []
        for _, row in work.iterrows():
            file_key = str(row['file'])
            id_key = str(row['id'])
            mapped_labels.append(str(label_map_cfg.get(file_key, label_map_cfg.get(id_key, id_key))))

        work['_mapped_label'] = mapped_labels
        work['_mapped_rank'] = work['_mapped_label'].map(label_rank).fillna(len(label_rank)).astype(int)

        ordered = work.sort_values(['_mapped_rank', 'id'])
        return ordered.drop(columns=['_mapped_label', '_mapped_rank'])

    def _build_x_labels(self, df_target, custom_labels=None):
        """Build x-axis labels according to user input and CONFIG mapping."""
        if custom_labels is not None and len(custom_labels) >= len(df_target):
            return custom_labels[:len(df_target)]

        label_list_cfg = self.config.get('plot_xtick_labels', [])
        if label_list_cfg and len(label_list_cfg) >= len(df_target):
            return [str(x) for x in label_list_cfg[:len(df_target)]]

        label_map_cfg = self.config.get('plot_xtick_label_map', {})
        if isinstance(label_map_cfg, dict) and label_map_cfg:
            labels = []
            for _, row in df_target.iterrows():
                file_key = str(row['file'])
                id_key = str(row['id'])
                labels.append(str(label_map_cfg.get(file_key, label_map_cfg.get(id_key, id_key))))
            return labels

        return df_target['id'].astype(str).tolist()

    def plot_charts(self, custom_labels=None):
        if self.df.empty: return

        # Prepare filtered data for bar charts (target IDs only)
        df_target = self.df[self.df['id'].isin(self.config['target_ids'])].copy()
        df_target = self._reorder_df_for_plot(df_target)
        
        if df_target.empty:
            print("No data found for target IDs.")
            return

        # Build x-axis labels according to custom input/configured order.
        ids = self._build_x_labels(df_target, custom_labels)

        style_rc = {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#4a4a4a',
            'axes.linewidth': 0.9,
            'axes.titlesize': 12,
            'axes.titleweight': 'semibold',
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.color': '#3a3a3a',
            'ytick.color': '#3a3a3a',
            'grid.color': '#d9d9d9',
            'grid.linestyle': '--',
            'grid.linewidth': 0.6,
            'grid.alpha': 0.65,
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'white',
        }

        # Muted, low-saturation palette for publication style.
        palette = ['#8FA5A0', '#B8A18A', '#8EA1B5', '#A8A6B8']

        with plt.rc_context(style_rc):
            fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), facecolor='white')
        
            # Flatten axes for easier indexing
            axes = axes.flatten()

            # ---------------- Chart 1: Ratio (ABX3/PbI2) Area ----------------
            self._plot_metric(axes[0], df_target, ids, 'class2_to_cls1_area_ratio',
                             'ABX3-to-PbI2 Area Ratio', 'Area Ratio (a.u.)',
                             palette[0], '(a)')
            axes[0].set_ylim(0, 110)

            # ---------------- Chart 2: ABX3 Avg Grain Size ----------------
            self._plot_metric(axes[1], df_target, ids, 'class2_avg_size',
                             'Mean Equivalent Diameter of ABX3 Grains', 'Equivalent Diameter (px)',
                             palette[1], '(b)')
            axes[1].set_ylim(top=80)

            # ---------------- Chart 3: ABX3 Grain Flatness ----------------
            self._plot_metric(axes[2], df_target, ids, 'class2_avg_flatness',
                             'ABX3 Grain Flatness', 'Intensity Std. Dev. (a.u.)',
                             palette[2], '(c)')

            # ---------------- Chart 4: PbI2 Outside ABX3 Ratio ----------------
            self._plot_metric(axes[3], df_target, ids, 'PbI2_outside_ABX3_ratio',
                             'PbI2 Outside ABX3 Ratio', 'Ratio',
                             palette[3], '(d)')
            axes[3].set_ylim(top=1)

            # tighter than default tight_layout while preserving readable whitespace
            fig.tight_layout(pad=1.1)
            fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.965, wspace=0.20, hspace=0.28)

            plot_path = os.path.join(self.config['dirs']['output'], 'analysis_charts.png')
            fig.savefig(plot_path, dpi=400, bbox_inches='tight', facecolor='white')
            plt.show()
            print(f"Saved analysis plot: {plot_path}")


    def _plot_metric(self, ax, data, x_labels, col, title, ylabel, color, panel_tag):
        if data.empty or col not in data.columns: return
        
        values = data[col].astype(float).to_numpy()
        x = np.arange(len(values))

        # Lollipop plot: thin stem + filled marker.
        ax.vlines(x, 0, values, color=color, linewidth=1.8, alpha=0.95)
        ax.scatter(x, values, s=42, color=color, edgecolors='white', linewidths=0.8, zorder=3)

        ax.set_title(title)
        ax.set_xlabel('Sample')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis='y')

        # Panel labels for manuscript-style multi-panel figures.
        ax.text(0.01, 0.98, panel_tag, transform=ax.transAxes,
                ha='left', va='top', fontsize=12, fontweight='semibold', color='#2f2f2f')

        # Keep annotations readable when values span different scales.
        y_min = float(np.min(values)) if len(values) else 0.0
        y_max = float(np.max(values)) if len(values) else 0.0
        y_span = max(y_max - y_min, 1e-8)
        top_pad = 0.16 * y_span if y_span > 0 else max(abs(y_max) * 0.16, 1.0)
        bottom_pad = 0.08 * y_span if y_span > 0 else max(abs(y_max) * 0.08, 0.5)
        ax.set_ylim(min(0, y_min - bottom_pad), y_max + top_pad)

        offset = max(y_span * 0.022, (y_max if y_max != 0 else 1.0) * 0.012, 0.02)
        for xi, val in zip(x, values):
            if abs(val) >= 1000:
                label = f'{val:.0f}'
            elif abs(val) >= 100:
                label = f'{val:.1f}'
            else:
                label = f'{val:.2f}'
            ax.text(xi, val + offset, label,
                    ha='center', va='bottom', fontsize=9.2, color='#2f2f2f')
        
        # Rotate x-axis labels if there are many items to prevent overlap
        if len(x_labels) > 5:
            ax.tick_params(axis='x', rotation=45)

        ax.tick_params(axis='both', width=0.9, length=4)

def main():
    processor = ImageProcessor(CONFIG)
    
    # Run processing
    file_stats, all_areas, all_dists, all_angles, all_flatness, all_edges = processor.process_all()
    
    if not file_stats:
        print("No statistics generated.")
        return

    # Create DataFrame
    df = pd.DataFrame(file_stats)
    
    # Ask user for custom labels
    print("\nCustomize X-axis labels:")
    try:
        use_custom_labels = input("Do you want to customize the X-axis labels? (y/n): ").lower().strip()
    except EOFError:
        use_custom_labels = 'n'
    
    custom_labels = None
    if use_custom_labels == 'y':
        print(f"Detected {len(df)} samples: {[row['file'] for _, row in df.iterrows()]}")
        print("Enter custom labels separated by commas (e.g., SampleA,SampleB,SampleC): ")
        try:
            label_input = input().strip()
            if label_input:
                custom_labels = [label.strip() for label in label_input.split(',')]
        except EOFError:
            pass
    
    # Visualization & Reporting
    viz = Visualizer(CONFIG, df, all_areas, all_dists, all_angles, all_flatness, all_edges)
    viz.print_summary()
    viz.plot_charts(custom_labels)

if __name__ == '__main__':
    main()
