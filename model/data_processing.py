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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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
    ang_a: np.ndarray | None = None,
    ang_b: np.ndarray | None = None,
    ang_c: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Compute the full set of voltage-asymmetry features for one measurement side.

    Magnitudes are taken as absolute values internally, so signed measurements
    are handled correctly.

    Parameters
    ----------
    Va, Vb, Vc : np.ndarray
        Phase voltage magnitude arrays (any numeric dtype).
    ang_a, ang_b, ang_c : np.ndarray, optional
        Measured phase angles in DEGREES. Required for a correct
        symmetrical-component decomposition. If omitted, the phases are assumed
        to be the ideal 0 / -120 / +120 degrees, which makes ``neg_seq_ratio``
        depend on magnitude imbalance only.

    Returns
    -------
    dict
        Keys: ``diff_xy``, ``dev_x``, ``min_max_ratio``, ``vuf``,
        ``neg_seq_ratio`` (un-prefixed; caller adds prefix as needed).

    Notes
    -----
    An earlier version formed V1 and V2 from the magnitudes alone, i.e. it fed
    real numbers into the ``a``-operator sum. That is not a symmetrical-component
    transform: the negative-sequence component is defined on phasors, and
    discarding the angles both loses the dominant part of the open-phase
    signature and yields a quantity no relay actually computes. Since the
    classical-detection comparison rests entirely on this quantity, the angles
    are now used wherever the dataset provides them.
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

    # ── Negative-sequence ratio via true symmetrical components ──────────────
    if ang_a is None:
        ang_a = np.zeros_like(Va)
    if ang_b is None:
        ang_b = np.full_like(Va, -120.0)
    if ang_c is None:
        ang_c = np.full_like(Va, 120.0)

    pa = Va * np.exp(1j * np.deg2rad(np.asarray(ang_a, dtype=np.float64)))
    pb = Vb * np.exp(1j * np.deg2rad(np.asarray(ang_b, dtype=np.float64)))
    pc = Vc * np.exp(1j * np.deg2rad(np.asarray(ang_c, dtype=np.float64)))

    # Phase convention of this dataset: a healthy set is logged as
    # phase_B = +120 deg and phase_C = -120 deg, i.e. the reverse of the usual
    # textbook A-B-C ordering (which has B at -120). The sequence operators are
    # assigned to match, so that V1 is the large component and V2 the small one
    # on a balanced system. Using the textbook assignment here drives V1 to
    # nearly zero and inflates V2/V1 to ~10^4, which silently inverts the
    # relay decision and produces an AUC below 0.5.
    a = np.exp(1j * 2 * np.pi / 3)
    a2 = np.exp(1j * 4 * np.pi / 3)
    V1 = np.abs((pa + a2 * pb + a * pc) / 3.0) + 1e-9
    V2 = np.abs((pa + a * pb + a2 * pc) / 3.0)
    neg_seq_ratio = V2 / V1

    return {
        "diff_ab": diff_ab, "diff_bc": diff_bc, "diff_ca": diff_ca,
        "dev_a": dev_a, "dev_b": dev_b, "dev_c": dev_c,
        "min_max_ratio": min_max_ratio,
        "vuf": vuf,
        "neg_seq_ratio": neg_seq_ratio,
        "V1": V1,
        "V2": V2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loading & initial preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def rated_currents() -> tuple[float, float]:
    """Return (pre-side, post-side) rated currents in amperes.

    Derived from the transformer nameplate in Table 3.1: S / (sqrt(3) * V).
    A CT's accuracy class is referred to these values, not to the reading.
    """
    pre = config.S_RATED_VA / (np.sqrt(3) * config.V_RATED_PRE)
    post = config.S_RATED_VA / (np.sqrt(3) * config.V_RATED_POST)
    return float(pre), float(post)


def load_level(df: pd.DataFrame) -> np.ndarray:
    """Return a per-row load index, computed on the CLEAN measurements.

    Uses the mean post-side current magnitude, which tracks the load drawn by
    the RLC bank and is unaffected by which primary phase is open. Must be
    computed before the instrument model is applied so that binning is not
    itself corrupted by measurement noise.
    """
    return df[["I_a", "I_b", "I_c"]].abs().mean(axis=1).values


def load_bins(levels: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    """Bin load levels into equal-count quantile bins (0 = lightest)."""
    n_bins = n_bins or config.LOAD_BINS
    ranks = pd.Series(levels).rank(method="first", pct=True).values
    return np.clip((ranks * n_bins).astype(int), 0, n_bins - 1)


def apply_instrument_model(
    df: pd.DataFrame,
    ct_class: float | None = None,
    vt_class: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Degrade ideal measurements to what real instrumentation would report.

    Two error mechanisms are applied to currents, following IEC 61869-2:

    * a proportional (ratio/gain) error on the reading, and
    * an additive error referred to the CT's RATED current.

    The second term is the important one. The exported dataset distinguishes a
    faulted pre-side current of 0.0 A from a healthy 0.7 A, but rated pre-side
    current is 251 A, so that distinction lives at ~0.28% of rating — below what
    any protection-class CT resolves. Modelling noise as a percentage of the
    *reading* would preserve the distinction and miss the point entirely.

    Voltages (IEC 61869-3) take a proportional error only, since VT error is
    referred to the reading and the bus stays near nominal throughout.

    Parameters
    ----------
    df        : Clean dataframe as loaded from ``main_data.csv``.
    ct_class  : CT accuracy class as a fraction (0.005 = class 0.5).
                ``0.0`` returns the ideal measurements unchanged.
    vt_class  : VT accuracy class as a fraction.
    seed      : RNG seed; defaults to ``config.INSTRUMENT_RNG_SEED`` so that a
                given accuracy class always produces the same dataset.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with noisy V_* and I_* columns. Phase angles are left
        untouched; ``faulted``, ``No`` and derived columns are preserved.
    """
    ct_class = config.CT_ACCURACY_CLASS if ct_class is None else ct_class
    vt_class = config.VT_ACCURACY_CLASS if vt_class is None else vt_class

    if ct_class <= 0 and vt_class <= 0:
        return df.copy()

    rng = np.random.default_rng(
        config.INSTRUMENT_RNG_SEED if seed is None else seed
    )
    out = df.copy()
    n = len(out)
    i_rated_pre, i_rated_post = rated_currents()

    for cols, i_rated in ((("I_A", "I_B", "I_C"), i_rated_pre),
                          (("I_a", "I_b", "I_c"), i_rated_post)):
        for col in cols:
            x = out[col].values.astype(np.float64)
            x = x * (1.0 + rng.normal(0.0, config.CT_RATIO_ERROR, n))
            x = x + rng.normal(0.0, ct_class * i_rated, n)
            out[col] = np.abs(x)          # a magnitude cannot be negative

    if vt_class > 0:
        for col in ("V_A", "V_B", "V_C", "V_a", "V_b", "V_c"):
            x = out[col].values.astype(np.float64)
            out[col] = np.abs(x * (1.0 + rng.normal(0.0, vt_class, n)))

    return out


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the CSV and apply rounding consistent with the MATLAB export.

    ``main.m`` writes ``main_data.csv`` having already applied::

        round(V)        -> voltages at integer precision
        round(I, 1)     -> currents at one decimal place

    Re-applying the *inverse* convention here (voltages to 1 dp, currents to
    integer) silently re-quantised the currents and destroyed precision that
    the simulation had deliberately retained. The rounding below mirrors
    ``main.m`` and is therefore a no-op on a correctly exported file; it is
    kept so the pipeline is robust to a CSV exported at higher precision.

    Precision is controlled by ``config.ROUND_VOLTAGE_DECIMALS`` and
    ``config.ROUND_CURRENT_DECIMALS``.

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

    v_dec = config.ROUND_VOLTAGE_DECIMALS
    i_dec = config.ROUND_CURRENT_DECIMALS

    # Voltages (both sides) → config.ROUND_VOLTAGE_DECIMALS
    for col in ("V_a", "V_b", "V_c", "V_A", "V_B", "V_C"):
        df[col] = df[col].round(v_dec)
        if v_dec == 0:
            df[col] = df[col].astype(np.int64)

    # Currents (both sides) → config.ROUND_CURRENT_DECIMALS
    for col in ("I_a", "I_b", "I_c", "I_A", "I_B", "I_C"):
        df[col] = df[col].round(i_dec)
        if i_dec == 0:
            df[col] = df[col].astype(np.int32)

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
    # Delta secondary: all three phase angles are measured.
    post = _voltage_asymmetry_features(
        Va, Vb, Vc,
        df["phase_a"].values, df["phase_b"].values, df["phase_c"].values,
    )

    # ── Derived voltage-asymmetry features (pre) ──────────────────────────────
    # Phase A is the angle reference on the primary side, hence 0 degrees.
    pre = _voltage_asymmetry_features(
        VA, VB, VC,
        np.zeros(len(df)), df["phase_B"].values, df["phase_C"].values,
    )

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
    groups: np.ndarray | None = None,
    bins: np.ndarray | None = None,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, StandardScaler,
           np.ndarray | None]:
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
    groups : np.ndarray, optional
        Per-row simulation-run id. When ``config.GROUPED_SPLIT`` is set, a run's
        healthy and faulted rows are kept on the same side of every split.
    bins : np.ndarray, optional
        Per-row load bin, carried through so test-set metrics can be reported
        per loading level rather than averaged over a 1000x sweep.

    Returns
    -------
    train_loader  : DataLoader
    val_loader    : DataLoader
    X_test_tensor : torch.Tensor   – scaled test features, shape (n, input_dim)
    y_test_tensor : torch.Tensor   – test labels, shape (n,)
    scaler        : StandardScaler – fitted scaler (already persisted to disk)
    test_bins     : np.ndarray | None – load bin for each test row

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

    X_all = features_df[selected_features].values.astype(np.float64)

    # ── Condition currents: log-compress the heavy tail ───────────────────────
    # Stateless, so applying before the split introduces no leakage.
    if config.LOG_TRANSFORM:
        log_cols = [
            i for i, f in enumerate(selected_features)
            if f in config.LOG_TRANSFORM_FEATURES
        ]
        if log_cols:
            block = X_all[:, log_cols]
            if (block < 0).any():
                raise ValueError(
                    "LOG_TRANSFORM_FEATURES contains a column with negative "
                    "values; log1p requires non-negative input."
                )
            X_all[:, log_cols] = np.log1p(block)

    # ── Condition voltages: express as per-unit deviation from nominal ────────
    # Voltages are a narrow band on a ~133 kV offset, so log1p barely changes
    # them and StandardScaler leaves the signal at ~0.01 std units. Dividing by
    # nominal and subtracting 1 puts the deviation at O(1).
    if config.VOLTAGE_CONDITIONING == "per_unit_deviation":
        for i, f in enumerate(selected_features):
            if f not in config.VOLTAGE_FEATURES:
                continue
            nominal = (config.V_NOMINAL_PRE if f in ("VA", "VB", "VC")
                       else config.V_NOMINAL_POST)
            X_all[:, i] = X_all[:, i] / nominal - 1.0
    elif config.VOLTAGE_CONDITIONING == "log1p":
        for i, f in enumerate(selected_features):
            if f in config.VOLTAGE_FEATURES:
                X_all[:, i] = np.log1p(np.abs(X_all[:, i]))

    X_all = X_all.astype(np.float32)

    # ── 70 / 15 / 15 split ───────────────────────────────────────────────────
    idx = np.arange(len(X_all))
    if config.GROUPED_SPLIT and groups is not None:
        # Keep each simulation run's healthy/faulted pair on the same side.
        gss = GroupShuffleSplit(
            n_splits=1, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )
        train_idx, temp_idx = next(gss.split(idx, labels, groups))
        gss2 = GroupShuffleSplit(
            n_splits=1, test_size=config.VAL_FRACTION,
            random_state=config.RANDOM_STATE,
        )
        rel_val, rel_test = next(
            gss2.split(temp_idx, labels[temp_idx], groups[temp_idx])
        )
        val_idx, test_idx = temp_idx[rel_val], temp_idx[rel_test]
    else:
        train_idx, temp_idx = train_test_split(
            idx, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=labels,
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=config.VAL_FRACTION,
            random_state=config.RANDOM_STATE, stratify=labels[temp_idx],
        )

    X_train, y_train = X_all[train_idx], labels[train_idx]
    X_val,   y_val   = X_all[val_idx],   labels[val_idx]
    X_test,  y_test  = X_all[test_idx],  labels[test_idx]

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

    test_bins = None if bins is None else bins[test_idx]
    return (train_loader, val_loader, _t(X_test), _t(y_test), scaler, test_bins)
