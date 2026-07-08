# -*- coding: utf-8 -*-
"""
MQI (Morphology Quality Index) computation module.

Provides a single source of truth for building and applying the MQI,
which is a weighted z-score composite index based on Spearman correlations
between SEM morphology features and device PCE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Default column definitions (shared across scripts)
# ---------------------------------------------------------------------------
DEFAULT_SEM_COLS: list[str] = [
    "ABX3_grain_size_mean_actual_area_um2",
    "ABX3_gray_cv",
    "PbI2_to_ABX3_area_ratio",
    "PbI2_ABX3_associated_fraction",
    "PbI2_spatial_uniformity_score",
    "PbI2_large_particle_area_um2",
]

DEFAULT_PERF_COLS: list[str] = [
    "JV_reverse_scan_Voc",
    "JV_reverse_scan_Jsc",
    "JV_reverse_scan_FF",
    "JV_reverse_scan_PCE",
]

DEFAULT_PCE_COL: str = "JV_reverse_scan_PCE"
DEFAULT_MIN_RHO: float = 0.15


# ---------------------------------------------------------------------------
# Data class for trained MQI model
# ---------------------------------------------------------------------------
@dataclass
class MQIModel:
    """Trained MQI model parameters derived from a training set."""

    selected_cols: list[str]
    weights: np.ndarray
    directions: np.ndarray
    train_mean: pd.Series
    train_std: pd.Series
    feature_min: pd.Series
    feature_max: pd.Series
    train_min: float
    train_max: float
    rho_dict: dict[str, tuple[float, float]]

    def summary(self) -> pd.DataFrame:
        """Return a human-readable summary of selected features and weights."""
        rows = []
        for idx, col in enumerate(self.selected_cols):
            r, p = self.rho_dict[col]
            w = self.weights[idx]
            d = "+" if self.directions[idx] > 0 else "-"
            rows.append(
                {
                    "feature": col,
                    "spearman_rho": r,
                    "p_value": p,
                    "weight": w,
                    "direction": d,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def _fill_na(
    df: pd.DataFrame,
    strategy: Literal["mean", "median", "drop"] = "mean",
    reference: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    """Fill missing values with a given strategy.

    If *reference* is provided, its statistics are used for imputation
    (required when applying a model trained on another dataset).
    """
    if strategy == "drop":
        return df.dropna()

    ref = reference if reference is not None else df

    if strategy == "mean":
        return df.fillna(ref.mean())
    if strategy == "median":
        return df.fillna(ref.median())

    raise ValueError(f"Unknown fillna strategy: {strategy!r}")


def compute_spearman(
    df: pd.DataFrame,
    sem_cols: list[str] | None = None,
    pce_col: str = DEFAULT_PCE_COL,
) -> dict[str, tuple[float, float]]:
    """Compute Spearman correlation of each SEM feature against *pce_col*.

    Returns
    -------
    dict
        Mapping ``feature_name -> (rho, p_value)``.
    """
    cols = sem_cols if sem_cols is not None else DEFAULT_SEM_COLS
    pce_vals = df[pce_col].values
    rho_dict: dict[str, tuple[float, float]] = {}
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not found in DataFrame")
        r, p = spearmanr(df[col].values, pce_vals)
        # 常数列（标准差为 0）无法计算相关性，标记为 0
        if np.isnan(r):
            r, p = 0.0, 1.0
        rho_dict[col] = (float(r), float(p))
    return rho_dict


def build_mqi_model(
    df_train: pd.DataFrame,
    sem_cols: list[str] | None = None,
    pce_col: str = DEFAULT_PCE_COL,
    min_rho: float = DEFAULT_MIN_RHO,
    fillna: Literal["mean", "median", "drop"] = "mean",
) -> MQIModel:
    """Build an MQI model from a training DataFrame.

    Steps
    -----
    1. Impute missing values in *sem_cols*.
    2. Compute Spearman ρ between each feature and *pce_col*.
    3. Select features with ``|ρ| >= min_rho``.
    4. Store training mean / std / min / max for later standardisation.
    5. Return an :class:`MQIModel` containing weights (|ρ|) and directions (sign(ρ)).

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data containing both SEM features and the PCE column.
    sem_cols : list[str] | None
        Morphology feature names. Defaults to :data:`DEFAULT_SEM_COLS`.
    pce_col : str
        Target performance column. Defaults to ``"JV_reverse_scan_PCE"``.
    min_rho : float
        Minimum absolute Spearman correlation for a feature to be included.
    fillna : {"mean", "median", "drop"}
        Missing-value strategy applied *before* any statistics are computed.

    Returns
    -------
    MQIModel
        The fitted model ready to be applied with :func:`apply_mqi`.
    """
    cols = sem_cols if sem_cols is not None else DEFAULT_SEM_COLS
    X = df_train[cols].copy()
    X = _fill_na(X, strategy=fillna)

    # Spearman correlations
    rho_dict = compute_spearman(df_train, sem_cols=cols, pce_col=pce_col)

    # Feature selection
    selected: list[str] = []
    weights: list[float] = []
    directions: list[float] = []
    for col in cols:
        r, _p = rho_dict[col]
        if abs(r) < min_rho:
            continue
        selected.append(col)
        weights.append(abs(r))
        directions.append(1.0 if r > 0 else -1.0)

    # Normalise weights so that sum(|w|) = 1 → cleaner formula for publication
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    if not selected:
        raise ValueError(
            f"No features passed the |ρ| >= {min_rho} threshold. "
            "Consider lowering min_rho or checking your data."
        )

    # Training statistics for standardisation
    train_mean = X[selected].mean()
    train_std = X[selected].std()

    # Guard against zero std (e.g. constant column)
    zero_std_mask = train_std == 0
    if zero_std_mask.any():
        bad_cols = train_std[zero_std_mask].index.tolist()
        raise ValueError(
            f"Zero standard deviation in training data for columns: {bad_cols}. "
            "These columns cannot be z-score standardised."
        )

    # Feature min/max for Min-Max normalisation
    feature_min = X[selected].min()
    feature_max = X[selected].max()

    # Guard against zero range (e.g. constant column)
    zero_range_mask = (feature_max - feature_min) == 0
    if zero_range_mask.any():
        bad_cols = feature_max[zero_range_mask].index.tolist()
        raise ValueError(
            f"Zero range (max == min) in training data for columns: {bad_cols}. "
            "These columns cannot be Min-Max normalised."
        )

    # Compute training MQI using Min-Max normalisation
    z_train = (X[selected] - feature_min) / (feature_max - feature_min)
    mqi_train = pd.Series(
        np.sum(
            [w * d * z_train[c].values for w, d, c in zip(weights, directions, selected)],
            axis=0,
        ),
        index=X.index,
    )

    return MQIModel(
        selected_cols=selected,
        weights=np.array(weights, dtype=float),
        directions=np.array(directions, dtype=float),
        train_mean=train_mean,
        train_std=train_std,
        feature_min=feature_min,
        feature_max=feature_max,
        train_min=float(mqi_train.min()),
        train_max=float(mqi_train.max()),
        rho_dict=rho_dict,
    )


def apply_mqi(
    df: pd.DataFrame,
    model: MQIModel,
    fillna: Literal["mean", "median", "drop"] = "mean",
) -> pd.Series:
    """Apply a trained :class:`MQIModel` to new data.

    The new data is z-score standardised using the *training* mean and std
    stored in *model*, then the weighted sum is computed.

    Parameters
    ----------
    df : pd.DataFrame
        Data to score. Must contain all :attr:`model.selected_cols`.
    model : MQIModel
        Model returned by :func:`build_mqi_model`.
    fillna : {"mean", "median", "drop"}
        Missing-value strategy. When ``"mean"`` or ``"median"``,
        the *training* statistics from *model* are used for imputation.

    Returns
    -------
    pd.Series
        Raw MQI values (not normalised).
    """
    X = df[model.selected_cols].copy()

    if fillna in ("mean", "median"):
        X = _fill_na(X, strategy=fillna, reference=model.train_mean.to_frame().T)
    elif fillna == "drop":
        X = X.dropna()

    z = (X - model.feature_min) / (model.feature_max - model.feature_min)
    mqi = pd.Series(
        np.sum(
            [
                w * d * z[c].values
                for w, d, c in zip(model.weights, model.directions, model.selected_cols)
            ],
            axis=0,
        ),
        index=X.index,
    )
    return mqi


def normalize_mqi(
    mqi: pd.Series,
    min_val: float | None = None,
    max_val: float | None = None,
) -> pd.Series:
    """Normalise MQI to a 0–100 scale.

    If *min_val* and *max_val* are omitted, the min/max of *mqi* itself are used.
    A guard against ``max_val == min_val`` is included.
    """
    lo = min_val if min_val is not None else float(mqi.min())
    hi = max_val if max_val is not None else float(mqi.max())
    denom = hi - lo
    if denom == 0:
        return pd.Series(50.0, index=mqi.index)
    return (mqi - lo) / denom * 100
