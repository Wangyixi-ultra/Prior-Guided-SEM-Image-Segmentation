# -*- coding: utf-8 -*-
"""
Evaluate Spearman-4 MQS generalisation against measured PCE on independent test sets.

Usage: python -B evaluate_spearman4_test.py [dataset_name]
Default dataset name is "dataset1".
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# Dataset configuration (edit paths to match your local layout)
DATASETS: dict[str, dict[str, str]] = {
    "dataset1": {
        "excel_path": "D:/your_project/dataset1/test_metadata.xlsx",
        "output_dir": "D:/your_project/dataset1/output",
    },
    "dataset2": {
        "excel_path": "D:/your_project/dataset2/test_metadata.xlsx",
        "output_dir": "D:/your_project/dataset2/output",
    },
}

PCE_COL = "JV_reverse_scan_PCE"
MQS_RAW_COL = "MQS_raw"
MQS_SCORE_COL = "MQS"


def get_dataset_config(dataset_name: str | None = None) -> tuple[str, str]:
    """Return paths for the requested dataset."""
    if dataset_name is None:
        dataset_name = "dataset1"

    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available options: {available}\n"
            f"Usage: python -B evaluate_spearman4_test.py [dataset_name]"
        )

    cfg = DATASETS[dataset_name]
    mqs_path = os.path.join(cfg["output_dir"], "mqs_results.xlsx")
    return mqs_path, cfg["excel_path"], cfg["output_dir"]


# Helper functions
def load_and_merge(mqs_path: str, test_path: str) -> pd.DataFrame:
    """Load MQS results and PCE, merge by 'num'."""
    mqs_df = pd.read_excel(mqs_path)
    test_df = pd.read_excel(test_path)

    if "num" not in mqs_df.columns:
        raise KeyError(f"{mqs_path} is missing the 'num' column")
    if "num" not in test_df.columns:
        raise KeyError(f"{test_path} is missing the 'num' column")
    if PCE_COL not in test_df.columns:
        raise KeyError(f"{test_path} is missing the '{PCE_COL}' column.")

    mqs_df["num"] = mqs_df["num"].astype(str)
    test_df["num"] = test_df["num"].astype(str)

    merged = mqs_df.merge(
        test_df[["num", PCE_COL]],
        on="num",
        how="inner",
    )

    # Drop rows with missing PCE or MQS
    merged = merged.dropna(subset=[PCE_COL, MQS_SCORE_COL])

    if len(merged) == 0:
        raise ValueError("No valid samples after merging; check 'num' values.")

    return merged


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute correlation and regression metrics."""
    mqs = df[MQS_SCORE_COL].values
    pce = df[PCE_COL].values

    spearman_rho, spearman_p = spearmanr(mqs, pce)
    pearson_r, pearson_p = pearsonr(mqs, pce)
    kendall_tau, kendall_p = kendalltau(mqs, pce)

    # R^2 from linear regression of PCE ~ MQS
    z = np.polyfit(mqs, pce, 1)
    p = np.poly1d(z)
    y_pred = p(mqs)
    ss_res = np.sum((pce - y_pred) ** 2)
    ss_tot = np.sum((pce - np.mean(pce)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Mean Absolute Error
    mae = np.mean(np.abs(pce - y_pred))

    return {
        "N": len(df),
        "Spearman_rho": float(spearman_rho),
        "Spearman_p": float(spearman_p),
        "Pearson_r": float(pearson_r),
        "Pearson_p": float(pearson_p),
        "Kendall_tau": float(kendall_tau),
        "Kendall_p": float(kendall_p),
        "R2": float(r2),
        "MAE": float(mae),
        "PCE_mean": float(np.mean(pce)),
        "PCE_std": float(np.std(pce)),
    }


def plot_mqs_vs_pce(df: pd.DataFrame, output_dir: str, dataset_name: str) -> None:
    """Scatter plot of MQS vs PCE with regression line."""
    mqs = df[MQS_SCORE_COL].values
    pce = df[PCE_COL].values

    rho, pval = spearmanr(mqs, pce)
    r, _ = pearsonr(mqs, pce)

    plt.figure(figsize=(7, 6))
    plt.scatter(mqs, pce, edgecolors="k", alpha=0.7, s=80)

    z = np.polyfit(mqs, pce, 1)
    p = np.poly1d(z)
    x_line = np.linspace(mqs.min(), mqs.max(), 100)
    plt.plot(x_line, p(x_line), "r--", lw=1.5, label="Linear fit")

    plt.xlabel("Spearman-4 MQS (0-100)")
    plt.ylabel("Measured PCE (%)")
    plt.title(
        f"{dataset_name} test set: MQS vs PCE\n"
        f"Spearman rho = {rho:.3f} (p = {pval:.3g}); Pearson r = {r:.3f}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "fig_mqs_vs_pce_test.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig_mqs_vs_pce_test.pdf"))
    plt.close()


def write_report(
    df: pd.DataFrame, metrics: dict, output_dir: str, dataset_name: str
) -> None:
    """Write evaluation report."""
    report_path = os.path.join(output_dir, "mqs_spearman4_test_evaluation.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Spearman-4 MQS {dataset_name} Test Set Evaluation\n\n")
        f.write(f"**Samples:** {metrics['N']}\n\n")
        f.write("## Correlation metrics\n\n")
        f.write("| Metric | Value | p-value |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Spearman rho | {metrics['Spearman_rho']:.4f} | {metrics['Spearman_p']:.4g} |\n")
        f.write(f"| Pearson r | {metrics['Pearson_r']:.4f} | {metrics['Pearson_p']:.4g} |\n")
        f.write(f"| Kendall tau | {metrics['Kendall_tau']:.4f} | {metrics['Kendall_p']:.4g} |\n")
        f.write(f"| R2 | {metrics['R2']:.4f} | - |\n")
        f.write(f"| MAE | {metrics['MAE']:.4f} | - |\n\n")

        f.write("## Sample ranking (1 = highest)\n\n")
        f.write("| num | MQS | MQS_rank | PCE | PCE_rank |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['num']} | {row[MQS_SCORE_COL]:.3f} | {int(row['MQS_rank'])} | "
                f"{row[PCE_COL]:.2f} | {int(row['PCE_rank'])} |\n"
            )
        f.write("\n")

        f.write("## PCE statistics\n\n")
        f.write(f"- mean = {metrics['PCE_mean']:.2f}%\n")
        f.write(f"- std = {metrics['PCE_std']:.2f}%\n")
    print(f"Report saved: {report_path}")


# Main entry point
def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "dataset1"
    mqs_path, test_path, output_dir = get_dataset_config(dataset_name)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Spearman-4 MQS {dataset_name} test set generalisation evaluation")
    print("=" * 70)

    df = load_and_merge(mqs_path, test_path)
    metrics = compute_metrics(df)

    # Add ranks
    df["MQS_rank"] = df[MQS_SCORE_COL].rank(ascending=False, method="min").astype(int)
    df["PCE_rank"] = df[PCE_COL].rank(ascending=False, method="min").astype(int)

    print(f"\nValid test samples: {metrics['N']}")
    print(f"Spearman rho = {metrics['Spearman_rho']:.4f}  (p = {metrics['Spearman_p']:.4g})")
    print(f"Pearson  r   = {metrics['Pearson_r']:.4f}  (p = {metrics['Pearson_p']:.4g})")
    print(f"Kendall tau  = {metrics['Kendall_tau']:.4f}  (p = {metrics['Kendall_p']:.4g})")
    print(f"R^2          = {metrics['R2']:.4f}")
    print(f"MAE          = {metrics['MAE']:.4f}%")
    print(f"PCE mean = {metrics['PCE_mean']:.2f}%, std = {metrics['PCE_std']:.2f}%")

    print("\nSample-level ranking comparison (1 = highest):")
    rank_cols = ["num", MQS_SCORE_COL, "MQS_rank", PCE_COL, "PCE_rank"]
    print(df[rank_cols].to_string(index=False))

    if metrics["Spearman_p"] < 0.05 and metrics["Spearman_rho"] > 0:
        print("\n[OK] MQS and PCE are significantly correlated on the test set.")
    else:
        print("\n[NOTE] MQS-PCE correlation not significant.")

    # Save merged data
    out_excel = os.path.join(output_dir, "mqs_spearman4_test_evaluation.xlsx")
    df.to_excel(out_excel, index=False, engine="openpyxl")
    print(f"\nMerged results saved: {out_excel}")

    # Plot and report
    plot_mqs_vs_pce(df, output_dir, dataset_name)
    write_report(df, metrics, output_dir, dataset_name)

    print("=" * 70)


if __name__ == "__main__":
    main()
