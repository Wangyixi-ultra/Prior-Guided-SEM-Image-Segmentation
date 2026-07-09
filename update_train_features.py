# -*- coding: utf-8 -*-
"""
Merge newly extracted SEM morphology features into the training data file.
"""
import pandas as pd

# Paths
NEW_FEATURES_PATH = r"D:\your_project\output\image_features.xlsx"
TRAIN_PATH = r"D:\your_project\train_features.xlsx"
OUTPUT_PATH = r"D:\your_project\train_features.xlsx"

# Read data
new_df = pd.read_excel(NEW_FEATURES_PATH)
train_df = pd.read_excel(TRAIN_PATH)

id_cols = ["image_name", "name", "num"]
feature_cols = [c for c in new_df.columns if c not in id_cols]

print(f"Found {len(feature_cols)} feature columns to update:")
for c in feature_cols:
    print(f"  - {c}")

merge_col = None
for col in ["num", "name", "image_name"]:
    if col in new_df.columns and col in train_df.columns:
        merge_col = col
        break

if merge_col is None:
    raise ValueError("No matching identifier column (num/name/image_name) found.")

print(f"Using '{merge_col}' as the matching key.")

train_df[merge_col] = train_df[merge_col].astype(str)
new_df[merge_col] = new_df[merge_col].astype(str)

new_df_subset = new_df[[merge_col] + feature_cols].copy()

for c in feature_cols:
    if c in train_df.columns:
        del train_df[c]

updated = train_df.merge(new_df_subset, on=merge_col, how="left")

matched = updated[feature_cols[0]].notna().sum()
total = len(train_df)
print(f"\nMatched successfully: {matched} / {total} samples")

updated.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")
print(f"\nUpdated and saved to: {OUTPUT_PATH}")
