# data_processing.py
"""
Handles all data manipulation: raw CSV loading, initial rounding transforms,
feature engineering, train/val/test splitting, StandardScaler fitting,
scaler persistence, and PyTorch DataLoader creation.

Column reference (main_data.csv)
─────────────────────────────────
Pre-transformer  (CAPITAL letters): V_A, V_B, V_C, I_A, I_B, I_C,
                                    phase_B, phase_C   (no phase_A in data)
Post-transformer (small letters)  : V_a, V_b, V_c, I_a, I_b, I_c,
                                    phase_a, phase_b, phase_c
Other                             : No (index), vuf (measured), faulted (target)
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config

# ── Every column that engineer_features() can produce ────────────────────────
# Use these names in config.TRAINING_RUNS[*]["features"].
ALL_ENGINEERED_FEATURES: list[str] = [
    # ── Post-transformer raw measurements ────────────────────────────────────
    "Va", "Vb", "Vc",                   # post-transformer voltages
    "Ia", "Ib", "Ic",                   # post-transformer currents  (I_a/b/c)
    "phase_a", "phase_b", "phase_c",    # post-transformer phase angles
    # ── Post-transformer derived (voltage asymmetry) ──────────────────────────
    "diff_ab", "diff_bc", "diff_ca",    # pairwise voltage differences
    "dev_a", "dev_b", "dev_c",          # normalised deviations from mean V
    "min_max_ratio",                    # min/max voltage ratio  (≈1 balanced)
    # NEMA voltage unbalance factor (derived)
    "vuf",
    "neg_seq_ratio",                    # negative-seq / positive-seq ratio
    # ── Pre-transformer raw measurements ─────────────────────────────────────
    "VA", "VB", "VC",                   # pre-transformer voltages
    "IA", "IB", "IC",                   # pre-transformer currents  (I_A/B/C)
    "phase_B", "phase_C",              # pre-transformer phase angles (no A)
    # ── Pre-transformer derived (voltage asymmetry) ───────────────────────────
    "diff_AB", "diff_BC", "diff_CA",    # pairwise voltage differences
    "dev_A", "dev_B", "dev_C",          # normalised deviations from mean V
    # min/max voltage ratio (pre-transformer)
    "min_max_ratio_pre",
    "vuf_pre",                          # NEMA VUF (pre-transformer, derived)
    "neg_seq_ratio_pre",                # negative-seq ratio (pre-transformer)
    # ── Other raw columns ─────────────────────────────────────────────────────
    "vuf_raw",                          # VUF as recorded in the CSV
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _voltage_asymmetry_features(
    Va: np.ndarray,
    Vb: np.ndarray,
    Vc: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute the full set of voltage-asymmetry features for one measurement side.

    All inputs are taken as absolute values internally, so signed measurements
    are handled correctly.

    Parameters
    ----------
    Va, Vb, Vc : np.ndarray
        Phase voltage arrays (any numeric dtype).

    Returns
    -------
    dict
        Keys: ``diff_xy``, ``dev_x``, ``min_max_ratio``, ``vuf``,
        ``neg_seq_ratio`` (un-prefixed; caller adds prefix as needed).
    """
    Va = np.abs(Va).astype(np.float64)
    Vb = np.abs(Vb).astype(np.float64)
    Vc = np.abs(Vc).astype(np.float64)

    mean_V = (Va + Vb + Vc) / 3.0 + 1e-9  # guard div-by-zero

    diff_ab = np.abs(Va - Vb)
    diff_bc = np.abs(Vb - Vc)
    diff_ca = np.abs(Vc - Va)

    dev_a = (Va - mean_V) / mean_V
    dev_b = (Vb - mean_V) / mean_V
    dev_c = (Vc - mean_V) / mean_V

    min_V = np.minimum(np.minimum(Va, Vb), Vc)
    max_V = np.maximum(np.maximum(Va, Vb), Vc)
    min_max_ratio = min_V / (max_V + 1e-9)

    max_dev = np.maximum(
        np.maximum(np.abs(Va - mean_V), np.abs(Vb - mean_V)),
        np.abs(Vc - mean_V),
    )
    vuf = max_dev / mean_V

    # Negative-sequence ratio via symmetrical components
    # V2 = (Va + a²·Vb + a·Vc) / 3,   a = e^(j·2π/3)
    a = np.exp(1j * 2 * np.pi / 3)
    a2 = np.exp(1j * 4 * np.pi / 3)
    V2 = np.abs((Va.astype(complex) + a2 * Vb.astype(complex) +
                a * Vc.astype(complex)) / 3.0)
    V1 = np.abs((Va.astype(complex) + a * Vb.astype(complex) +
                a2 * Vc.astype(complex)) / 3.0) + 1e-9
    neg_seq_ratio = V2 / V1

    return {
        "diff_ab": diff_ab, "diff_bc": diff_bc, "diff_ca": diff_ca,
        "dev_a": dev_a, "dev_b": dev_b, "dev_c": dev_c,
        "min_max_ratio": min_max_ratio,
        "vuf": vuf,
        "neg_seq_ratio": neg_seq_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loading & initial preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the CSV and apply mandatory rounding to voltages and currents.

    Rounding rules (per specification):
    - Post-transformer voltages (V_a, V_b, V_c)       → 1 decimal place.
    - Post-transformer currents (I_a, I_b, I_c)        → nearest integer.
    - Pre-transformer voltages  (V_A, V_B, V_C)        → 1 decimal place.
    - Pre-transformer currents  (I_A, I_B, I_C)        → nearest integer.

    Parameters
    ----------
    filepath : str
        Path to ``main_data.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with rounded voltage and current columns.
    """
    if config.DATA_COUNT == 0:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath).head(config.DATA_COUNT)

    # Post-transformer voltages → 1 decimal place
    for col in ("V_a", "V_b", "V_c"):
        df[col] = df[col].round(1)

    # Post-transformer currents → nearest integer
    for col in ("I_a", "I_b", "I_c"):
        df[col] = df[col].round(0).astype(np.int32)

    # Pre-transformer voltages → 1 decimal place
    for col in ("V_A", "V_B", "V_C"):
        df[col] = df[col].round(1)

    # Pre-transformer currents → nearest integer
    for col in ("I_A", "I_B", "I_C"):
        df[col] = df[col].round(0).astype(np.int32)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all engineered features from the pre-processed DataFrame.

    Produces *every* feature listed in ``ALL_ENGINEERED_FEATURES`` for both
    the post-transformer and pre-transformer measurement sides.  Each training
    run then selects a subset of these columns via ``config.TRAINING_RUNS``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_samples, len(ALL_ENGINEERED_FEATURES))``, float32.
        Column order matches ``ALL_ENGINEERED_FEATURES``.
    """
    # ── Post-transformer raw measurements ────────────────────────────────────
    Va = df["V_a"].values
    Vb = df["V_b"].values
    Vc = df["V_c"].values
    Ia = df["I_a"].values.astype(np.float32)
    Ib = df["I_b"].values.astype(np.float32)
    Ic = df["I_c"].values.astype(np.float32)

    # ── Pre-transformer raw measurements ─────────────────────────────────────
    VA = df["V_A"].values
    VB = df["V_B"].values
    VC = df["V_C"].values
    IA = df["I_A"].values.astype(np.float32)
    IB = df["I_B"].values.astype(np.float32)
    IC = df["I_C"].values.astype(np.float32)

    # ── Derived voltage-asymmetry features (post) ─────────────────────────────
    post = _voltage_asymmetry_features(Va, Vb, Vc)

    # ── Derived voltage-asymmetry features (pre) ──────────────────────────────
    pre = _voltage_asymmetry_features(VA, VB, VC)

    feature_arrays: Dict[str, np.ndarray] = {
        # Post-transformer raw
        "Va": Va.astype(np.float32),
        "Vb": Vb.astype(np.float32),
        "Vc": Vc.astype(np.float32),
        "Ia": Ia, "Ib": Ib, "Ic": Ic,
        "phase_a": df["phase_a"].values.astype(np.float32),
        "phase_b": df["phase_b"].values.astype(np.float32),
        "phase_c": df["phase_c"].values.astype(np.float32),
        # Post-transformer derived
        "diff_ab": post["diff_ab"].astype(np.float32),
        "diff_bc": post["diff_bc"].astype(np.float32),
        "diff_ca": post["diff_ca"].astype(np.float32),
        "dev_a":   post["dev_a"].astype(np.float32),
        "dev_b":   post["dev_b"].astype(np.float32),
        "dev_c":   post["dev_c"].astype(np.float32),
        "min_max_ratio":  post["min_max_ratio"].astype(np.float32),
        "vuf":            post["vuf"].astype(np.float32),
        "neg_seq_ratio":  post["neg_seq_ratio"].astype(np.float32),
        # Pre-transformer raw
        "VA": VA.astype(np.float32),
        "VB": VB.astype(np.float32),
        "VC": VC.astype(np.float32),
        "IA": IA, "IB": IB, "IC": IC,
        "phase_B": df["phase_B"].values.astype(np.float32),
        "phase_C": df["phase_C"].values.astype(np.float32),
        # Pre-transformer derived
        "diff_AB":        pre["diff_ab"].astype(np.float32),
        "diff_BC":        pre["diff_bc"].astype(np.float32),
        "diff_CA":        pre["diff_ca"].astype(np.float32),
        "dev_A":          pre["dev_a"].astype(np.float32),
        "dev_B":          pre["dev_b"].astype(np.float32),
        "dev_C":          pre["dev_c"].astype(np.float32),
        "min_max_ratio_pre": pre["min_max_ratio"].astype(np.float32),
        "vuf_pre":           pre["vuf"].astype(np.float32),
        "neg_seq_ratio_pre": pre["neg_seq_ratio"].astype(np.float32),
        # Other raw columns
        "vuf_raw": df["vuf"].values.astype(np.float32),
    }

    return pd.DataFrame(
        {col: feature_arrays[col] for col in ALL_ENGINEERED_FEATURES}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Splitting, scaling, and DataLoader preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataloaders(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    selected_features: list[str],
    run_dir: str,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, StandardScaler]:
    """Select features, split data, fit+save a scaler, and return DataLoaders.

    Parameters
    ----------
    features_df : pd.DataFrame
        Full engineered-feature DataFrame (all columns from ALL_ENGINEERED_FEATURES).
    labels : np.ndarray
        Binary target array (float32), shape ``(n_samples,)``.
    selected_features : list[str]
        Column names to use for this experiment run.
    run_dir : str
        Directory where ``scaler.pkl`` will be saved.

    Returns
    -------
    train_loader  : DataLoader
    val_loader    : DataLoader
    X_test_tensor : torch.Tensor   – scaled test features, shape (n, input_dim)
    y_test_tensor : torch.Tensor   – test labels, shape (n,)
    scaler        : StandardScaler – fitted scaler (already persisted to disk)

    Raises
    ------
    ValueError
        If any name in *selected_features* is not a column in *features_df*.
    """
    missing = [f for f in selected_features if f not in features_df.columns]
    if missing:
        raise ValueError(
            f"Unknown feature(s): {missing}. "
            f"Valid names are: {list(features_df.columns)}"
        )

    X_all = features_df[selected_features].values.astype(np.float32)

    # ── Stratified 70 / 15 / 15 split ────────────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.VAL_FRACTION,
        random_state=config.RANDOM_STATE,
        stratify=y_temp,
    )

    # ── Fit scaler on training data only, then transform all splits ───────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── Persist fitted scaler for inference use ───────────────────────────────
    scaler_path = os.path.join(run_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # ── Convert to tensors ────────────────────────────────────────────────────
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr)

    train_loader = DataLoader(
        TensorDataset(_t(X_train), _t(y_train)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(_t(X_val), _t(y_val)),
        batch_size=config.BATCH_SIZE,
    )

    return train_loader, val_loader, _t(X_test), _t(y_test), scaler
