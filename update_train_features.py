# -*- coding: utf-8 -*-
"""
Update the latest extracted SEM features (from analyze_perovskite_sem.py output)
into the training data file sem_summary_new.xlsx.
"""
import pandas as pd

# Paths
NEW_FEATURES_PATH = r"D:\article_sem\output\sem_morphology_summary.xlsx"
TRAIN_PATH = r"D:\article_sem\sem_summary_new.xlsx"
OUTPUT_PATH = r"D:\article_sem\sem_summary_new.xlsx"

# Read data
new_df = pd.read_excel(NEW_FEATURES_PATH)
train_df = pd.read_excel(TRAIN_PATH)

# SEM feature columns to update (identifier columns excluded)
id_cols = ["image_name", "name", "num"]
feature_cols = [c for c in new_df.columns if c not in id_cols]

print(f"Found {len(feature_cols)} feature columns to update:")
for c in feature_cols:
    print(f"  - {c}")

# Determine matching key
merge_col = None
for col in ["num", "name", "image_name"]:
    if col in new_df.columns and col in train_df.columns:
        merge_col = col
        break

if merge_col is None:
    raise ValueError("No matching identifier column (num/name/image_name) found. Please check both files.")

print(f"Using '{merge_col}' as the matching key.")

# Ensure matching column types are consistent (convert to string)
train_df[merge_col] = train_df[merge_col].astype(str)
new_df[merge_col] = new_df[merge_col].astype(str)

# Keep only samples present in the training set
new_df_subset = new_df[[merge_col] + feature_cols].copy()

# Remove existing feature columns from the training set to avoid conflicts
for c in feature_cols:
    if c in train_df.columns:
        del train_df[c]

# Merge
updated = train_df.merge(new_df_subset, on=merge_col, how="left")

# Check match rate
matched = updated[feature_cols[0]].notna().sum()
total = len(train_df)
print(f"\nMatched successfully: {matched} / {total} samples")

# Save
updated.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
print(f"\nUpdated and saved to: {OUTPUT_PATH}")
