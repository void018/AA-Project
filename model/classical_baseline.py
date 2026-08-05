"""Classical negative-sequence relay baseline.

Chapter 1 claims a direct, like-for-like comparison between the machine-learning
classifier and classical derived-quantity detection. Feeding derived features to
the same neural network does not achieve that: it compares two feature sets, not
two *detection methods*. A protection relay does something much simpler — it
compares a single derived quantity against a fixed, site-configured threshold
(device 46 for negative-sequence overcurrent, device 47 or 60 for the voltage
equivalent, per Section 2.6.2).

This module implements that decision rule directly:

    alarm  if  V2 / V1  >  threshold

and sweeps the threshold to produce a full ROC curve, so the relay can be
compared against the classifier at matched false-alarm rates rather than at one
arbitrary operating point. Because the relay has no trained parameters, it is
evaluated on the same test split as the network and needs no training data.

Everything here operates on the SAME instrument-degraded measurements the
classifier sees, so the comparison is genuinely like-for-like.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config
import data_processing as dp


def negative_sequence_ratio(df: pd.DataFrame, side: str = "pre") -> np.ndarray:
    """Return V2/V1 computed from true phasors for one measurement side.

    Parameters
    ----------
    df   : Dataframe with magnitude and phase-angle columns.
    side : ``"pre"`` (primary, phase A is the 0-degree reference) or
           ``"post"`` (secondary, all three angles measured).
    """
    if side == "pre":
        mags = (df["V_A"].values, df["V_B"].values, df["V_C"].values)
        angs = (np.zeros(len(df)), df["phase_B"].values, df["phase_C"].values)
    elif side == "post":
        mags = (df["V_a"].values, df["V_b"].values, df["V_c"].values)
        angs = (df["phase_a"].values, df["phase_b"].values, df["phase_c"].values)
    else:
        raise ValueError("side must be 'pre' or 'post'")

    out = dp._voltage_asymmetry_features(*mags, *angs)
    return out["neg_seq_ratio"]


def sweep_threshold(
    ratio: np.ndarray,
    labels: np.ndarray,
    n_points: int = 400,
) -> Dict[str, Any]:
    """Sweep the relay pickup threshold and return the operating characteristic.

    Returns
    -------
    dict with:
        thresholds      : the swept pickup values
        tpr, fpr        : true / false positive rate at each threshold
        accuracy        : accuracy at each threshold
        best_threshold  : threshold maximising accuracy
        best_accuracy   : accuracy there
        auc             : area under the ROC curve (trapezoidal)
        thresh_at_1pct_fa : pickup giving <=1% false-alarm rate, and its recall
    """
    finite = ratio[np.isfinite(ratio)]
    lo, hi = np.percentile(finite, 0.1), np.percentile(finite, 99.9)
    thresholds = np.unique(np.concatenate([
        np.linspace(lo, hi, n_points),
        np.geomspace(max(lo, 1e-9), max(hi, 1e-8), n_points),
    ]))

    pos = labels == 1
    neg = ~pos
    n_pos, n_neg = pos.sum(), neg.sum()

    tpr, fpr, acc = [], [], []
    for t in thresholds:
        alarm = ratio > t
        tp = np.count_nonzero(alarm & pos)
        fp = np.count_nonzero(alarm & neg)
        tpr.append(tp / max(n_pos, 1))
        fpr.append(fp / max(n_neg, 1))
        acc.append((tp + (n_neg - fp)) / len(labels))

    tpr = np.asarray(tpr)
    fpr = np.asarray(fpr)
    acc = np.asarray(acc)

    order = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[order], fpr[order]))

    best = int(np.argmax(acc))

    # Relays are configured to a false-alarm budget, not to peak accuracy: a
    # spurious trip of a nuclear auxiliary bus is not a cheap event.
    ok = np.where(fpr <= 0.01)[0]
    if len(ok):
        pick = ok[np.argmax(tpr[ok])]
        constrained = {
            "threshold": float(thresholds[pick]),
            "recall": float(tpr[pick]),
            "false_alarm_rate": float(fpr[pick]),
            "accuracy": float(acc[pick]),
        }
    else:
        constrained = None

    return {
        "thresholds": thresholds.tolist(),
        "tpr": tpr.tolist(),
        "fpr": fpr.tolist(),
        "accuracy": acc.tolist(),
        "best_threshold": float(thresholds[best]),
        "best_accuracy": float(acc[best]),
        "best_recall": float(tpr[best]),
        "best_false_alarm_rate": float(fpr[best]),
        "auc": auc,
        "at_1pct_false_alarm": constrained,
    }


def evaluate_relay(
    df: pd.DataFrame,
    labels: np.ndarray,
    test_idx: np.ndarray,
    bins: Optional[np.ndarray] = None,
    side: str = "pre",
) -> Dict[str, Any]:
    """Evaluate the classical relay on the test split, overall and per load bin.

    The threshold is selected on the NON-test rows and then applied unchanged to
    the test rows, mirroring how a relay is commissioned at one operating point
    and then left alone. Selecting it on the test set would flatter the relay.
    """
    ratio = negative_sequence_ratio(df, side=side)

    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    fit_sweep = sweep_threshold(ratio[~mask], labels[~mask])
    threshold = fit_sweep["best_threshold"]

    r_test, y_test = ratio[test_idx], labels[test_idx]
    alarm = r_test > threshold
    pos = y_test == 1

    result: Dict[str, Any] = {
        "side": side,
        "commissioned_threshold": threshold,
        "test_accuracy": float((alarm == pos).mean()),
        "test_recall": float(alarm[pos].mean()) if pos.any() else None,
        "test_false_alarm_rate": float(alarm[~pos].mean()) if (~pos).any() else None,
        "test_sweep": sweep_threshold(r_test, y_test),
    }

    if bins is not None:
        per_bin = []
        for b in sorted(np.unique(bins[test_idx])):
            sel = bins[test_idx] == b
            if not sel.any():
                continue
            per_bin.append({
                "load_bin": int(b),
                "n": int(sel.sum()),
                "accuracy": float((alarm[sel] == pos[sel]).mean()),
                "recall": float(alarm[sel & pos].mean()) if (sel & pos).any() else None,
            })
        result["per_load_bin"] = per_bin

    return result


def run_sweep(
    input_file: str | None = None,
    output_file: str = "classical_baseline.json",
) -> List[Dict[str, Any]]:
    """Evaluate the relay across every CT accuracy class in the sweep.

    Produces the counterpart to the ML sweep, so the two can be plotted on the
    same axes.
    """
    log = logging.getLogger("classical_baseline")
    input_file = input_file or config.INPUT_FILE

    clean = dp.load_raw_data(input_file)
    labels = clean["faulted"].values.astype(np.float32)
    groups = ((clean["No"].values + 1) // 2)
    bins = dp.load_bins(dp.load_level(clean))

    # Reproduce exactly the test split the classifier uses.
    from sklearn.model_selection import GroupShuffleSplit
    idx = np.arange(len(clean))
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE)
    _, temp_idx = next(gss.split(idx, labels, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=config.VAL_FRACTION,
                             random_state=config.RANDOM_STATE)
    _, rel_test = next(gss2.split(temp_idx, labels[temp_idx], groups[temp_idx]))
    test_idx = temp_idx[rel_test]

    results = []
    for ct_class in config.CT_CLASS_SWEEP:
        noisy = dp.apply_instrument_model(
            clean, ct_class=ct_class, vt_class=ct_class
        )
        for side in ("pre", "post"):
            r = evaluate_relay(noisy, labels, test_idx, bins, side=side)
            r["ct_accuracy_class"] = ct_class
            results.append(r)
            log.info(
                "relay %-4s | CT class %.3f | acc %.4f | recall %.4f | FA %.4f | AUC %.4f",
                side, ct_class, r["test_accuracy"], r["test_recall"] or 0.0,
                r["test_false_alarm_rate"] or 0.0, r["test_sweep"]["auc"],
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Wrote %s (%d entries).", output_file, len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_sweep()
