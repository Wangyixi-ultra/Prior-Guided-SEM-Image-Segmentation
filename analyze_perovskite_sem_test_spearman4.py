# -*- coding: utf-8 -*-
"""
Extract SEM features from test images and score with the trained Spearman-4 MQS model.

Usage: python -B analyze_perovskite_sem_test_spearman4.py [dataset_name]
Default dataset name is "dataset1".
"""

import glob
import logging
import os
import sys

import pandas as pd

import analyze_perovskite_sem_test as _sem_test
import mqs_spearman4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Dataset configurations (edit paths to match your local layout)
DATASETS: dict[str, dict[str, str]] = {
    "dataset1": {
        "image_dir": "D:/your_project/dataset1/raw",
        "json_dir": "D:/your_project/dataset1/json",
        "excel_path": "D:/your_project/dataset1/test_metadata.xlsx",
        "output_dir": "D:/your_project/dataset1/output",
    },
    "dataset2": {
        "image_dir": "D:/your_project/dataset2/raw",
        "json_dir": "D:/your_project/dataset2/json",
        "excel_path": "D:/your_project/dataset2/test_metadata.xlsx",
        "output_dir": "D:/your_project/dataset2/output",
    },
}

# Training model paths
TRAIN_EXCEL_PATH = "D:/your_project/train_features.xlsx"
MODEL_DIR = "D:/your_project/mqs_results"
MODEL_PATH = os.path.join(MODEL_DIR, "spearman4_model.pkl")

SAVE_VISUALIZATION = True


def get_dataset_config(dataset_name: str | None = None) -> dict[str, str]:
    """Return dataset config."""
    if dataset_name is None:
        dataset_name = "dataset1"

    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available options: {available}\n"
            f"Usage: python -B analyze_perovskite_sem_test_spearman4.py [dataset_name]"
        )

    return DATASETS[dataset_name]


# Main program
def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "dataset1"
    config = get_dataset_config(dataset_name)

    image_dir = config["image_dir"]
    json_dir = config["json_dir"]
    excel_path = config["excel_path"]
    output_dir = config["output_dir"]
    vis_dir = os.path.join(output_dir, "vis")
    log_file_path = os.path.join(output_dir, "processing_spearman4_log.txt")

    _sem_test.LOG_FILE_PATH = log_file_path
    _sem_test.setup_logging()
    logging.info(f"Spearman-4 MQS test script started, dataset: {dataset_name}")

    os.makedirs(output_dir, exist_ok=True)
    if SAVE_VISUALIZATION:
        os.makedirs(vis_dir, exist_ok=True)

    _sem_test.VIS_DIR = vis_dir
    _sem_test.SAVE_VISUALIZATION = SAVE_VISUALIZATION

    logging.info("Loading Spearman-4 MQS model...")
    if os.path.exists(MODEL_PATH):
        model = mqs_spearman4.load_model(MODEL_DIR)
        logging.info(f"Loaded pre-trained model: {MODEL_PATH}")
    else:
        logging.warning(f"Model file not found at {MODEL_PATH}; retraining on training data.")
        df_train = pd.read_excel(TRAIN_EXCEL_PATH)
        model = mqs_spearman4.build_mqs_spearman4_model(df_train)
        mqs_spearman4.save_model(model, MODEL_DIR)

    logging.info("Spearman-4 MQS model formula:")
    formula_parts = []
    for col in model.features:
        r, _p = model.rho_dict[col]
        w = model.normalised_weight(col)
        d = "+" if model.directions[col] > 0 else "-"
        logging.info(f"  -> {col}: |ρ|={abs(r):.4f}, weight={w:.4f}, direction={d}")
        formula_parts.append(f"{d}{w:.4f}·z({col})")
    formula_str = "MQS = " + " ".join(formula_parts).replace("+", " + ").replace("-", " - ")
    logging.info(f"[Formula] {formula_str}")

    # Read test-set scale info
    scale_df = _sem_test.robust_read_excel(excel_path)
    if scale_df is None:
        logging.critical("Unable to read Excel file; program terminated.")
        return
    if "num" in scale_df.columns:
        scale_df["num"] = scale_df["num"].astype(str)
    else:
        logging.critical("'num' column not found in Excel; program terminated.")
        return

    # Image feature extraction
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    logging.info(f"Found {len(json_files)} JSON files.")

    all_summary_results = []
    for json_path in json_files:
        base_name = os.path.basename(json_path).rsplit(".", 1)[0]
        img_path = None
        for ext in _sem_test.SUPPORTED_IMG_EXTENSIONS:
            potential_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        if not img_path:
            logging.warning(f"Image file for {base_name} not found; skipping.")
            continue

        num_key = _sem_test.get_image_num(base_name)
        if not num_key:
            logging.warning(f"Unable to extract identifier from filename {base_name}; skipping.")
            continue

        scale_info_row = scale_df[scale_df["num"] == num_key]
        if scale_info_row.empty:
            logging.warning(f"No record with num='{num_key}' found in Excel; skipping {base_name}.")
            continue

        scale_info = scale_info_row.iloc[0].to_dict()
        summary = _sem_test.analyze_image(img_path, json_path, scale_info)
        if summary:
            all_summary_results.append(summary)

    if not all_summary_results:
        logging.warning("No images were successfully processed; program ended.")
        return

    summary_df = pd.DataFrame(all_summary_results)
    summary_df = summary_df.sort_values(
        by="num", key=lambda col: col.map(_sem_test.natural_sort_key)
    ).reset_index(drop=True)

    summary_output_path = os.path.join(output_dir, "image_features.xlsx")
    try:
        summary_df.to_excel(summary_output_path, index=False, engine="openpyxl")
        logging.info(f"Image-level summary saved to: {summary_output_path}")
    except Exception as e:
        logging.error(f"Error saving image-level summary: {e}")
        return

    # Spearman-4 MQS scoring
    logging.info("Starting Spearman-4 MQS calculation...")
    try:
        mqs_test = mqs_spearman4.apply_mqs_spearman4(summary_df, model)
        summary_df["MQS_raw"] = mqs_test
        summary_df["MQS"] = mqs_spearman4.normalize_mqs_spearman4(mqs_test, model)

        mqs_min_test = float(summary_df["MQS"].min())
        mqs_max_test = float(summary_df["MQS"].max())
        logging.info(f"Test-set MQS range (forced to 0-100): [{mqs_min_test:.3f}, {mqs_max_test:.3f}]")

        # Save results
        mqs_output_path = os.path.join(output_dir, "mqs_results.xlsx")
        display_cols = ["image_name", "num"] + list(model.features) + ["MQS_raw", "MQS"]
        available_cols = [c for c in display_cols if c in summary_df.columns]
        summary_df[available_cols].to_excel(mqs_output_path, index=False, engine="openpyxl")
        logging.info(f"Spearman-4 MQS results saved to: {mqs_output_path}")

        # Print results to console
        print("\n" + "=" * 70)
        print(f"Spearman-4 MQS test results ({dataset_name})")
        print("=" * 70)
        print(summary_df[available_cols].to_string(index=False))
        print("=" * 70)

    except Exception as e:
        logging.error(f"Error during MQS calculation: {e}", exc_info=True)

    logging.info("All tasks completed.")


if __name__ == "__main__":
    main()
