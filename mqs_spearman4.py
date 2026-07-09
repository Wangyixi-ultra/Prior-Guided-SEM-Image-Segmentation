# -*- coding: utf-8 -*-
"""
Train and validate the Spearman-4 Morphology Quality Score (MQS).

The composite score is a weighted sum of z-score-standardised features:

    MQS = Σ (|ρ_i| / Σ|ρ_j|) * sign(ρ_i) * z_i
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
TRAIN_PATH: str = r"D:/your_project/train_features.xlsx"
PCE_COL: str = "JV_reverse_scan_PCE"
OUTPUT_DIR: str = r"D:/your_project/mqs_results"

# Four Spearman-selected descriptors
SPEARMAN4_FEATURES: list[str] = [
    "ABX3_grain_size_mean_actual_area_um2",
    "ABX3_gray_cv",
    "PbI2_large_particle_area_um2",
    "PbI2_ABX3_associated_fraction",
]

N_SPLITS: int = 5
N_REPEATS: int = 5
RANDOM_STATE: int = 42

# Group-aware CV settings
GROUP_COL: str | None = None
N_GROUP_SPLITS: int = 5


# Model dataclass
@dataclass
class Spearman4MQSModel:
    """Trained Spearman-4 MQS model state."""

    features: list[str]
    weights: dict[str, float]          # raw absolute Spearman ρ
    directions: dict[str, int]         # +1 or -1
    scaler: StandardScaler
    train_min: float
    train_max: float
    rho_dict: dict[str, tuple[float, float]]

    def normalised_weight(self, feature: str) -> float:
        """Normalised absolute Spearman ρ weight."""
        total = sum(abs(w) for w in self.weights.values())
        if total == 0:
            return 0.0
        return abs(self.weights[feature]) / total


# Core functions
def _get_spearman_weights(
    df: pd.DataFrame,
    features: list[str],
    pce_col: str = PCE_COL,
) -> tuple[dict[str, float], dict[str, int], dict[str, tuple[float, float]]]:
    """Compute Spearman weights, directions and raw (ρ, p)."""
    weights: dict[str, float] = {}
    directions: dict[str, int] = {}
    rho_dict: dict[str, tuple[float, float]] = {}

    for f in features:
        r, p = spearmanr(df[f].values, df[pce_col].values)
        if np.isnan(r):
            r, p = 0.0, 1.0
        weights[f] = abs(r)
        directions[f] = 1 if r > 0 else -1
        rho_dict[f] = (float(r), float(p))

    return weights, directions, rho_dict


def build_mqs_spearman4_model(
    df_train: pd.DataFrame,
    features: list[str] | None = None,
    pce_col: str = PCE_COL,
) -> Spearman4MQSModel:
    """Build a Spearman-4 MQS model from training data."""
    features = features if features is not None else SPEARMAN4_FEATURES
    missing = [c for c in features + [pce_col] if c not in df_train.columns]
    if missing:
        raise KeyError(f"Training data missing columns: {missing}")

    weights, directions, rho_dict = _get_spearman_weights(df_train, features, pce_col)

    scaler = StandardScaler()
    scaler.fit(df_train[features].values)

    scores = _compute_mqs(df_train, features, weights, directions, scaler)

    return Spearman4MQSModel(
        features=features,
        weights=weights,
        directions=directions,
        scaler=scaler,
        train_min=float(scores.min()),
        train_max=float(scores.max()),
        rho_dict=rho_dict,
    )


def _compute_mqs(
    df: pd.DataFrame,
    features: list[str],
    weights: dict[str, float],
    directions: dict[str, int],
    scaler: StandardScaler,
) -> np.ndarray:
    """Compute raw MQS scores."""
    Xs = scaler.transform(df[features].values)
    total_w = sum(abs(weights[f]) for f in features)
    if total_w == 0:
        return np.zeros(len(df))

    score = np.zeros(len(df))
    for i, f in enumerate(features):
        w = abs(weights[f]) / total_w
        score += w * directions[f] * Xs[:, i]
    return score


def apply_mqs_spearman4(
    df: pd.DataFrame,
    model: Spearman4MQSModel,
) -> pd.Series:
    """Apply a trained Spearman-4 MQS model."""
    scores = _compute_mqs(
        df,
        model.features,
        model.weights,
        model.directions,
        model.scaler,
    )
    return pd.Series(scores, index=df.index)


def normalize_mqs_spearman4(
    scores: pd.Series,
    model: Spearman4MQSModel,
) -> pd.Series:
    """Scale raw MQS scores to 0-100 using the training range."""
    import mqs

    return mqs.normalize_mqs(scores, model.train_min, model.train_max)


def evaluate_repeated_kfold_cv(
    df_train: pd.DataFrame,
    features: list[str] | None = None,
    pce_col: str = PCE_COL,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, np.ndarray]:
    """5x5 repeated K-fold CV."""
    features = features if features is not None else SPEARMAN4_FEATURES
    y = df_train[pce_col].values
    rkf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    oof_scores = np.zeros(len(df_train))

    for train_idx, val_idx in rkf.split(df_train):
        df_tr = df_train.iloc[train_idx]
        df_val = df_train.iloc[val_idx]

        model_fold = build_mqs_spearman4_model(df_tr, features=features, pce_col=pce_col)
        oof_scores[val_idx] += apply_mqs_spearman4(df_val, model_fold).values

    oof_scores /= n_repeats
    rho, pval = spearmanr(oof_scores, y)
    return float(rho), float(pval), oof_scores


# Group-aware CV helpers

def _derive_groups(df_train: pd.DataFrame, group_col: str | None = GROUP_COL) -> np.ndarray | None:
    """Derive group labels for GroupKFold / LOGO validation."""
    if group_col is not None and group_col in df_train.columns:
        groups = df_train[group_col].astype(str).values
        n_groups = len(np.unique(groups))
        logger.info(f"Using specified group column '{group_col}': {n_groups} groups")
        return groups

    if "name" in df_train.columns:
        raw = df_train["name"].astype(str)
        # take leading numeric token before the first '-'
        groups = raw.str.split("-", n=1).str[0].str.extract(r"(\d+)", expand=False).values
        if np.all(groups == groups):  # no NaN
            n_groups = len(np.unique(groups))
            logger.info(f"Auto-derived study/group from 'name' prefix: {n_groups} groups")
            return groups

    logger.warning("No usable group information found; skipping GroupKFold / Leave-One-Group-Out CV.")
    return None


def evaluate_group_kfold_cv(
    df_train: pd.DataFrame,
    groups: np.ndarray,
    features: list[str] | None = None,
    pce_col: str = PCE_COL,
    n_splits: int = N_GROUP_SPLITS,
) -> tuple[float, float, np.ndarray, pd.DataFrame]:
    """GroupKFold CV: keep samples from the same study together."""
    features = features if features is not None else SPEARMAN4_FEATURES
    y = df_train[pce_col].values
    n_groups = len(np.unique(groups))
    effective_splits = min(n_splits, n_groups)
    if effective_splits < n_splits:
        logger.warning(
            f"GroupKFold n_splits reduced from {n_splits} to {effective_splits} "
            f"(only {n_groups} groups available)."
        )
    gkf = GroupKFold(n_splits=effective_splits)
    oof_scores = np.full(len(df_train), np.nan)
    fold_records: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, y, groups), 1):
        df_tr = df_train.iloc[train_idx]
        df_val = df_train.iloc[val_idx]
        model_fold = build_mqs_spearman4_model(df_tr, features=features, pce_col=pce_col)
        scores = apply_mqs_spearman4(df_val, model_fold).values
        oof_scores[val_idx] = scores

        n_val = len(val_idx)
        if n_val > 2:
            rho_f, p_f = spearmanr(scores, y[val_idx])
        else:
            rho_f, p_f = np.nan, np.nan

        fold_records.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_val": n_val,
                "groups_val": ", ".join(str(g) for g in np.unique(groups[val_idx])),
                "spearman_rho": float(rho_f) if not np.isnan(rho_f) else None,
                "p_value": float(p_f) if not np.isnan(p_f) else None,
            }
        )

    valid = ~np.isnan(oof_scores)
    rho, pval = spearmanr(oof_scores[valid], y[valid])
    return float(rho), float(pval), oof_scores, pd.DataFrame(fold_records)


def evaluate_leave_one_group_out_cv(
    df_train: pd.DataFrame,
    groups: np.ndarray,
    features: list[str] | None = None,
    pce_col: str = PCE_COL,
) -> tuple[float, float, np.ndarray, pd.DataFrame]:
    """Leave-one-group-out CV."""
    features = features if features is not None else SPEARMAN4_FEATURES
    y = df_train[pce_col].values
    logo = LeaveOneGroupOut()
    oof_scores = np.full(len(df_train), np.nan)
    fold_records: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(logo.split(df_train, y, groups), 1):
        df_tr = df_train.iloc[train_idx]
        df_val = df_train.iloc[val_idx]
        model_fold = build_mqs_spearman4_model(df_tr, features=features, pce_col=pce_col)
        scores = apply_mqs_spearman4(df_val, model_fold).values
        oof_scores[val_idx] = scores

        n_val = len(val_idx)
        if n_val > 2:
            rho_f, p_f = spearmanr(scores, y[val_idx])
        else:
            rho_f, p_f = np.nan, np.nan

        fold_records.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_val": n_val,
                "group_left_out": str(np.unique(groups[val_idx])[0]),
                "spearman_rho": float(rho_f) if not np.isnan(rho_f) else None,
                "p_value": float(p_f) if not np.isnan(p_f) else None,
            }
        )

    valid = ~np.isnan(oof_scores)
    rho, pval = spearmanr(oof_scores[valid], y[valid])
    return float(rho), float(pval), oof_scores, pd.DataFrame(fold_records)


# I/O
def save_model(model: Spearman4MQSModel, output_dir: str) -> None:
    """Save model state and CSV summary."""
    os.makedirs(output_dir, exist_ok=True)

    pkl_path = os.path.join(output_dir, "spearman4_model.pkl")
    state = {
        "features": model.features,
        "weights": model.weights,
        "directions": model.directions,
        "scaler": model.scaler,
        "train_min": model.train_min,
        "train_max": model.train_max,
        "rho_dict": model.rho_dict,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(state, f)
    logger.info(f"Model saved: {pkl_path}")

    summary_rows = []
    for f in model.features:
        r, p = model.rho_dict[f]
        summary_rows.append(
            {
                "feature": f,
                "spearman_rho": r,
                "p_value": p,
                "abs_rho": abs(r),
                "normalised_weight": model.normalised_weight(f),
                "direction": "+" if model.directions[f] > 0 else "-",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "spearman4_model_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Model summary saved: {csv_path}")


def load_model(output_dir: str) -> Spearman4MQSModel:
    """Load a pickled Spearman-4 MQS model."""
    pkl_path = os.path.join(output_dir, "spearman4_model.pkl")
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)
    model = Spearman4MQSModel(**state)
    logger.info(f"Model loaded: {pkl_path}")
    return model


# Figures and report
def _plot_results(
    result_df: pd.DataFrame,
    model: Spearman4MQSModel,
    fitted_rho: float,
    cv_rho: float,
    output_dir: str,
) -> None:
    """Generate validation figures."""
    sns.set_style("whitegrid")

    # 1. Fitted vs CV scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(
        result_df["Spearman4_MQS_raw"],
        result_df["PCE"],
        edgecolor="k",
        alpha=0.7,
        s=60,
    )
    ax.set_xlabel("Spearman-4 MQS (raw, fitted)")
    ax.set_ylabel("PCE (%)")
    ax.set_title(f"Fitted Spearman-4 MQS vs PCE: ρ = {fitted_rho:.3f}")
    z = np.polyfit(result_df["Spearman4_MQS_raw"], result_df["PCE"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(
        result_df["Spearman4_MQS_raw"].min(),
        result_df["Spearman4_MQS_raw"].max(),
        100,
    )
    ax.plot(x_line, p(x_line), "r--", lw=1)

    ax = axes[1]
    valid = result_df["Spearman4_MQS_cv_raw"].notna()
    ax.scatter(
        result_df.loc[valid, "Spearman4_MQS_cv_raw"],
        result_df.loc[valid, "PCE"],
        edgecolor="k",
        alpha=0.7,
        s=60,
    )
    ax.set_xlabel("Spearman-4 MQS (raw, 5x5 repeated CV)")
    ax.set_ylabel("PCE (%)")
    ax.set_title(f"CV Spearman-4 MQS vs PCE: ρ = {cv_rho:.3f}")
    z = np.polyfit(
        result_df.loc[valid, "Spearman4_MQS_cv_raw"],
        result_df.loc[valid, "PCE"],
        1,
    )
    p = np.poly1d(z)
    x_line = np.linspace(
        result_df.loc[valid, "Spearman4_MQS_cv_raw"].min(),
        result_df.loc[valid, "Spearman4_MQS_cv_raw"].max(),
        100,
    )
    ax.plot(x_line, p(x_line), "r--", lw=1)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_spearman4_mqs_scatter.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig_spearman4_mqs_scatter.pdf"))
    plt.close(fig)

    # 2. Normalised weight bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    weight_df = pd.DataFrame(
        {
            "feature": model.features,
            "weight": [model.normalised_weight(f) for f in model.features],
            "direction": ["+" if model.directions[f] > 0 else "-" for f in model.features],
        }
    )
    weight_df = weight_df.sort_values("weight", ascending=True)
    colors = ["#EB784B" if d == "-" else "#5AAAE6" for d in weight_df["direction"]]
    weight_df.set_index("feature")["weight"].plot.barh(
        ax=ax, color=colors, edgecolor="black"
    )
    ax.set_xlabel("Normalised weight in Spearman-4 MQS")
    ax.set_title("Spearman-4 MQS feature weights")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_spearman4_weights.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig_spearman4_weights.pdf"))
    plt.close(fig)


def _write_report(
    output_dir: str,
    model: Spearman4MQSModel,
    fitted_rho: float,
    fitted_pval: float,
    cv_rho: float,
    cv_pval: float,
) -> None:
    """Write markdown validation report."""
    report_path = os.path.join(output_dir, "spearman4_mqs_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Spearman-4 Morphology Quality Index (MQS) Report\n\n")
        f.write("## Selected features and weights\n\n")
        f.write("| Feature | Spearman ρ | p-value | Normalised weight | Direction |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for feat in model.features:
            r, p = model.rho_dict[feat]
            w = model.normalised_weight(feat)
            d = "+" if model.directions[feat] > 0 else "-"
            f.write(f"| {feat} | {r:+.4f} | {p:.4g} | {w:.4f} | {d} |\n")
        f.write("\n")

        f.write("## Model formula\n\n")
        f.write("Raw MQS is computed as a weighted sum of z-score-standardised features:\n\n")
        f.write("```\nMQS = ")
        terms = []
        for feat in model.features:
            w = model.normalised_weight(feat)
            d = "+" if model.directions[feat] > 0 else "-"
            terms.append(f"{d}{w:.4f} · z({feat})")
        formula = " ".join(terms).replace("+", " + ").replace("-", " - ")
        f.write(formula + "\n")
        f.write("where z(x) is the StandardScaler output (zero mean, unit variance)\n")
        f.write("```\n\n")

        f.write("## Validation results\n\n")
        f.write(f"- Fitted ρ = {fitted_rho:.4f} (p = {fitted_pval:.4g})\n")
        f.write(f"- 5×5 repeated K-fold CV ρ = {cv_rho:.4f} (p = {cv_pval:.4g})\n")
    logger.info(f"Report saved: {report_path}")


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Convert a DataFrame to a Markdown table."""
    lines: list[str] = []
    headers = [str(c) for c in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:{floatfmt}}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _plot_group_cv_results(
    result_df: pd.DataFrame,
    fitted_rho: float,
    cv_rho: float,
    group_rho: float | None,
    logo_rho: float | None,
    output_dir: str,
) -> None:
    """Scatter plots comparing CV protocols."""
    sns.set_style("whitegrid")
    y = result_df["PCE"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("Spearman4_MQS_raw", f"Fitted (ρ={fitted_rho:.3f})"),
        ("Spearman4_MQS_cv_raw", f"5×5 repeated K-fold (ρ={cv_rho:.3f})"),
        ("Spearman4_MQS_groupcv_raw", f"{N_GROUP_SPLITS}-fold GroupKFold (ρ={group_rho:.3f})"),
        ("Spearman4_MQS_logo_raw", f"Leave-one-group-out (ρ={logo_rho:.3f})"),
    ]

    for ax, (col, title) in zip(axes.flat, panels):
        if col not in result_df.columns:
            ax.axis("off")
            ax.set_title(f"{title}\n(not available)")
            continue
        x = result_df[col].values
        valid = ~np.isnan(x)
        ax.scatter(x[valid], y[valid], edgecolor="k", alpha=0.7, s=60)
        ax.set_xlabel("MQS (raw)")
        ax.set_ylabel("PCE (%)")
        ax.set_title(title)
        if valid.sum() > 1:
            z = np.polyfit(x[valid], y[valid], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
            ax.plot(x_line, p(x_line), "r--", lw=1)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_spearman4_group_cv_scatter.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "fig_spearman4_group_cv_scatter.pdf"))
    plt.close(fig)
    logger.info("Group-aware CV comparison figure saved")


def _write_group_cv_report(
    output_dir: str,
    model: Spearman4MQSModel,
    fitted_rho: float,
    fitted_pval: float,
    cv_rho: float,
    cv_pval: float,
    group_rho: float | None,
    group_pval: float | None,
    logo_rho: float | None,
    logo_pval: float | None,
    group_fold_df: pd.DataFrame | None,
    logo_fold_df: pd.DataFrame | None,
    groups: np.ndarray,
) -> None:
    """Write group-aware CV report."""
    report_path = os.path.join(output_dir, "spearman4_group_cv_report.md")
    unique_groups, group_counts = np.unique(groups, return_counts=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Spearman-4 MQS Group-Aware Cross-Validation Report\n\n")
        f.write(f"- Total samples: {len(groups)}\n")
        f.write(f"- Groups: {len(unique_groups)}\n")
        f.write(f"- Group sizes: min={group_counts.min()}, median={int(np.median(group_counts))}, max={group_counts.max()}\n\n")

        f.write("## Validation summary\n\n")
        f.write("| Protocol | Spearman ρ | p-value |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Fitted | {fitted_rho:+.4f} | {fitted_pval:.4g} |\n")
        f.write(f"| 5×5 repeated K-fold | {cv_rho:+.4f} | {cv_pval:.4g} |\n")
        if group_rho is not None:
            f.write(f"| {N_GROUP_SPLITS}-fold GroupKFold | {group_rho:+.4f} | {group_pval:.4g} |\n")
        if logo_rho is not None:
            f.write(f"| Leave-one-group-out | {logo_rho:+.4f} | {logo_pval:.4g} |\n")
        f.write("\n")

        if group_fold_df is not None:
            f.write(f"## {N_GROUP_SPLITS}-fold GroupKFold fold details\n\n")
            f.write(_df_to_markdown(group_fold_df, floatfmt=".4f"))
            f.write("\n\n")

        if logo_fold_df is not None:
            f.write("## Leave-one-group-out fold details\n\n")
            f.write(_df_to_markdown(logo_fold_df, floatfmt=".4f"))
            f.write("\n\n")

    logger.info(f"Group-aware CV report saved: {report_path}")


# Main entry point
def run_spearman4_validation(
    train_path: str = TRAIN_PATH,
    output_dir: str = OUTPUT_DIR,
    pce_col: str = PCE_COL,
    features: list[str] | None = None,
) -> dict:
    """Run full Spearman-4 MQS validation."""
    os.makedirs(output_dir, exist_ok=True)
    features = features if features is not None else SPEARMAN4_FEATURES

    df_train = pd.read_excel(train_path)
    logger.info(
        f"Spearman-4 MQS training data: n={len(df_train)}, features={features}"
    )

    model = build_mqs_spearman4_model(df_train, features=features, pce_col=pce_col)

    scores_raw = apply_mqs_spearman4(df_train, model)
    scores_norm = normalize_mqs_spearman4(scores_raw, model)
    fitted_rho, fitted_pval = spearmanr(scores_raw.values, df_train[pce_col].values)
    logger.info(
        f"Fitted Spearman-4 MQS-PCE: rho={fitted_rho:.3f}, p={fitted_pval:.4g}"
    )

    logger.info("Running 5x5 repeated K-fold CV...")
    cv_rho, cv_pval, cv_raw_arr = evaluate_repeated_kfold_cv(
        df_train,
        features=features,
        pce_col=pce_col,
    )
    cv_raw = pd.Series(cv_raw_arr, index=df_train.index)
    cv_norm = normalize_mqs_spearman4(cv_raw, model)
    logger.info(
        f"5x5 repeated K-fold CV Spearman-4 MQS-PCE: rho={cv_rho:.3f}, p={cv_pval:.4g}"
    )

    groups = _derive_groups(df_train)
    group_rho = group_pval = logo_rho = logo_pval = None
    group_oof = logo_oof = None
    group_fold_df = logo_fold_df = None
    if groups is not None:
        logger.info(f"Running {N_GROUP_SPLITS}-fold GroupKFold CV...")
        group_rho, group_pval, group_oof, group_fold_df = evaluate_group_kfold_cv(
            df_train,
            groups,
            features=features,
            pce_col=pce_col,
            n_splits=N_GROUP_SPLITS,
        )
        logger.info(
            f"GroupKFold CV Spearman-4 MQS-PCE: rho={group_rho:.3f}, p={group_pval:.4g}"
        )

        logger.info("Running leave-one-group-out CV...")
        logo_rho, logo_pval, logo_oof, logo_fold_df = evaluate_leave_one_group_out_cv(
            df_train,
            groups,
            features=features,
            pce_col=pce_col,
        )
        logger.info(
            f"Leave-one-group-out CV Spearman-4 MQS-PCE: rho={logo_rho:.3f}, p={logo_pval:.4g}"
        )

    logger.info("Spearman-4 MQS weights and directions:")
    for f in model.features:
        r, _p = model.rho_dict[f]
        w = model.normalised_weight(f)
        d = "+" if model.directions[f] > 0 else "-"
        logger.info(f"  -> {f}: |ρ|={abs(r):.4f}, normalised_weight={w:.4f}, direction={d}")

    result_df = df_train.copy()
    result_df["Spearman4_MQS_raw"] = scores_raw
    result_df["Spearman4_MQS_score"] = scores_norm
    result_df["Spearman4_MQS_cv_raw"] = cv_raw
    result_df["Spearman4_MQS_cv_score"] = cv_norm
    if groups is not None:
        result_df["Spearman4_MQS_groupcv_raw"] = pd.Series(group_oof, index=df_train.index)
        result_df["Spearman4_MQS_groupcv_score"] = normalize_mqs_spearman4(
            pd.Series(group_oof, index=df_train.index), model
        )
        result_df["Spearman4_MQS_logo_raw"] = pd.Series(logo_oof, index=df_train.index)
        result_df["Spearman4_MQS_logo_score"] = normalize_mqs_spearman4(
            pd.Series(logo_oof, index=df_train.index), model
        )
    result_df["PCE"] = df_train[pce_col]

    save_model(model, output_dir)
    result_df.to_excel(
        os.path.join(output_dir, "spearman4_mqs_results.xlsx"),
        index=False,
        engine="openpyxl",
    )

    _plot_results(result_df, model, fitted_rho, cv_rho, output_dir)
    _write_report(
        output_dir,
        model,
        fitted_rho,
        fitted_pval,
        cv_rho,
        cv_pval,
    )

    if groups is not None:
        group_fold_df.to_excel(
            os.path.join(output_dir, "groupkfold_cv_fold_summary.xlsx"),
            index=False,
            engine="openpyxl",
        )
        logo_fold_df.to_excel(
            os.path.join(output_dir, "logo_cv_fold_summary.xlsx"),
            index=False,
            engine="openpyxl",
        )

        _plot_group_cv_results(
            result_df,
            fitted_rho,
            cv_rho,
            group_rho,
            logo_rho,
            output_dir,
        )
        _write_group_cv_report(
            output_dir,
            model,
            fitted_rho,
            fitted_pval,
            cv_rho,
            cv_pval,
            group_rho,
            group_pval,
            logo_rho,
            logo_pval,
            group_fold_df,
            logo_fold_df,
            groups,
        )

    return {
        "df": result_df,
        "model": model,
        "fitted_rho": fitted_rho,
        "fitted_pval": fitted_pval,
        "cv_rho": cv_rho,
        "cv_pval": cv_pval,
        "group_rho": group_rho,
        "group_pval": group_pval,
        "logo_rho": logo_rho,
        "logo_pval": logo_pval,
    }


if __name__ == "__main__":
    run_spearman4_validation()
