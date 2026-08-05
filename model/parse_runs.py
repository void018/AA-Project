"""Parse run_log.txt files into structured JSON summaries.

Emits two files:

``runs_summary.json``
    One record per run directory. Each record now also carries the seed, the
    canonical configuration name (timestamp and seed suffix stripped), and the
    stability block introduced when the training loop began reporting best /
    final / last-50 validation accuracy.

``runs_aggregate.json``
    One record per configuration, aggregating across seeds: mean and standard
    deviation of test accuracy, plus the worst-case stability spread. This is
    the view that should be quoted in write-ups, because a single run's
    final-epoch number can differ from a rerun of the identical configuration
    by a large margin when training is unstable.

Both old-format and new-format logs parse; fields absent from an older log are
returned as ``None`` rather than causing the run to be skipped.
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── Regex patterns ───────────────────────────────────────────────────────────

# Run codename: 0508_093332_derivations_post_s1
RE_CODENAME = re.compile(r"Run codename:\s*(.+)")

# Seed: 1
RE_SEED = re.compile(r"^\s*.*?Seed:\s*(\d+)\s*$", re.MULTILINE)

# Features (6): ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
RE_FEATURES = re.compile(r"Features \(\d+\): (.+)")

# Log-transform: True | dropout: 0.10 | monitor: val_acc | best-ckpt: True
RE_SETTINGS = re.compile(
    r"Log-transform:\s*(\w+)\s*\|\s*dropout:\s*([0-9.]+)\s*\|\s*"
    r"monitor:\s*(\w+)\s*\|\s*best-ckpt:\s*(\w+)"
)

# Epoch 400 | Train acc 0.7955  loss 0.3672 | Val acc 0.7597  loss 0.3457 | lr 1.00e-03
# The trailing "| lr ..." is optional so pre-change logs still match.
RE_EPOCH = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Train acc\s+([0-9.]+)\s+loss\s+([0-9.]+)\s+\|\s+"
    r"Val acc\s+([0-9.]+)\s+loss\s+([0-9.]+)"
    r"(?:\s+\|\s+lr\s+([0-9.eE+-]+))?"
)

# Val accuracy — final epoch 1.0000 | best 1.0000 (epoch 1) | last-50 mean 1.0000  min 1.0000  max 1.0000
# The dash may be rendered as an em dash or a plain hyphen depending on console encoding.
RE_VAL_SUMMARY = re.compile(
    r"Val accuracy\s*[—\-]+\s*final epoch\s+([0-9.]+)\s*\|\s*best\s+([0-9.]+)\s*"
    r"\(epoch\s+(\d+)\)\s*\|\s*last-50 mean\s+([0-9.]+)\s+min\s+([0-9.]+)\s+max\s+([0-9.]+)"
)

# Early stopping at epoch 137 (no val_acc improvement for 40 epochs).
RE_EARLY_STOP = re.compile(r"Early stopping at epoch\s+(\d+)")

# Split sizes — train: 14000 | val: 3000 | test: 3000
RE_SPLIT = re.compile(
    r"Split sizes\s*[—\-]+\s*train:\s*(\d+)\s*\|\s*val:\s*(\d+)\s*\|\s*test:\s*(\d+)"
)

RE_CLASS_REPORT_LINE = re.compile(
    r"^\s*(\w+|\w+\s\w+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)"
)
RE_TEST_ACCURACY = re.compile(r"^\s*accuracy\s+([0-9.]+)")
RE_CM_LINE = re.compile(r"^\s+True\s(Healthy|Faulted)\s+([0-9]+)\s+([0-9]+)")

# Trailing "_s<digits>" seed marker, and leading "MMDD_HHMMSS_" timestamp.
RE_SEED_SUFFIX = re.compile(r"_s(\d+)$")
RE_TIMESTAMP_PREFIX = re.compile(r"^\d{4}_\d{6}_")


# ── Helpers ──────────────────────────────────────────────────────────────────

def config_name_from_codename(codename: str) -> str:
    """Strip the timestamp prefix and seed suffix to get the config name.

    ``0508_093332_derivations_post_s1`` -> ``derivations_post``

    This is what groups repeated runs of the same feature set together.
    """
    name = RE_TIMESTAMP_PREFIX.sub("", codename)
    return RE_SEED_SUFFIX.sub("", name)


def seed_from_codename(codename: str) -> Optional[int]:
    """Recover the seed from the codename suffix, if present."""
    match = RE_SEED_SUFFIX.search(codename)
    return int(match.group(1)) if match else None


def parse_classification_report(report_str: str) -> Dict[str, Any]:
    """Parse sklearn's classification_report text into a dictionary."""
    report_data: Dict[str, Any] = {}
    for line in report_str.strip().split("\n")[2:]:
        if not line.strip():
            continue

        accuracy_match = RE_TEST_ACCURACY.search(line)
        if accuracy_match:
            report_data["accuracy"] = float(accuracy_match.group(1))
            continue

        match = RE_CLASS_REPORT_LINE.search(line)
        if match:
            key_name, precision, recall, f1_score, support = match.groups()
            report_data[key_name.strip().replace(" ", "_")] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1_score),
                "support": int(support),
            }
    return report_data


def parse_log_file(log_path: str) -> Optional[Dict[str, Any]]:
    """Parse a single ``run_log.txt`` into a structured record."""
    if not os.path.exists(log_path):
        logging.warning("Log file not found: %s", log_path)
        return None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    run_dir = os.path.dirname(log_path)

    run: Dict[str, Any] = {
        "run_codename": None,
        "config_name": None,
        "seed": None,
        "features_used": [],
        "num_features": 0,
        # final-epoch metrics (kept for backward compatibility)
        "last_train_acc": None,
        "last_train_loss": None,
        "last_val_acc": None,
        "last_val_loss": None,
        "last_lr": None,
        "epochs_completed": None,
        "early_stopped_at": None,
        # stability block
        "best_val_acc": None,
        "best_epoch": None,
        "val_acc_last50_mean": None,
        "val_acc_last50_min": None,
        "val_acc_last50_max": None,
        "val_acc_last50_spread": None,
        "final_vs_best_gap": None,
        # settings
        "log_transform": None,
        "dropout": None,
        "lr_monitor": None,
        "use_best_checkpoint": None,
        "split_sizes": None,
        # results
        "test_results": {},
        "confusion_matrix": None,
        "training_curves_path": None,
        "best_model_path": None,
        "last_model_path": None,
    }

    # ── Metadata ─────────────────────────────────────────────────────────────
    codename_match = RE_CODENAME.search(content)
    if codename_match:
        codename = codename_match.group(1).strip()
        run["run_codename"] = codename
        run["config_name"] = config_name_from_codename(codename)
        run["seed"] = seed_from_codename(codename)

    seed_match = RE_SEED.search(content)
    if seed_match:
        run["seed"] = int(seed_match.group(1))

    features_match = RE_FEATURES.search(content)
    if features_match:
        try:
            run["features_used"] = eval(features_match.group(1))
            run["num_features"] = len(run["features_used"])
        except Exception:
            logging.warning("Could not parse feature list in %s", log_path)

    settings_match = RE_SETTINGS.search(content)
    if settings_match:
        log_t, dropout, monitor, best_ckpt = settings_match.groups()
        run["log_transform"] = log_t == "True"
        run["dropout"] = float(dropout)
        run["lr_monitor"] = monitor
        run["use_best_checkpoint"] = best_ckpt == "True"

    split_match = RE_SPLIT.search(content)
    if split_match:
        run["split_sizes"] = {
            "train": int(split_match.group(1)),
            "val": int(split_match.group(2)),
            "test": int(split_match.group(3)),
        }

    # ── Epoch metrics ────────────────────────────────────────────────────────
    epoch_matches = RE_EPOCH.findall(content)
    if epoch_matches:
        epoch_no, tr_acc, tr_loss, vl_acc, vl_loss, lr = epoch_matches[-1]
        run["epochs_completed"] = int(epoch_no)
        run["last_train_acc"] = float(tr_acc)
        run["last_train_loss"] = float(tr_loss)
        run["last_val_acc"] = float(vl_acc)
        run["last_val_loss"] = float(vl_loss)
        run["last_lr"] = float(lr) if lr else None

    early_match = RE_EARLY_STOP.search(content)
    if early_match:
        run["early_stopped_at"] = int(early_match.group(1))

    # ── Stability block ──────────────────────────────────────────────────────
    val_summary = RE_VAL_SUMMARY.search(content)
    if val_summary:
        final_acc, best_acc, best_ep, mean50, min50, max50 = val_summary.groups()
        run["best_val_acc"] = float(best_acc)
        run["best_epoch"] = int(best_ep)
        run["val_acc_last50_mean"] = float(mean50)
        run["val_acc_last50_min"] = float(min50)
        run["val_acc_last50_max"] = float(max50)
        run["val_acc_last50_spread"] = round(float(max50) - float(min50), 6)
        run["final_vs_best_gap"] = round(float(best_acc) - float(final_acc), 6)
        # Prefer the summary line's final value; it is authoritative because the
        # per-epoch lines are only emitted every 20 epochs.
        run["last_val_acc"] = float(final_acc)

    # ── Test results ─────────────────────────────────────────────────────────
    try:
        start = content.index("TEST SET RESULTS")
        end = content.index("Confusion Matrix:", start)
        run["test_results"] = parse_classification_report(content[start:end])

        cm_data: Dict[str, Dict[str, int]] = {"Healthy": {}, "Faulted": {}}
        for line in content[end:].split("\n"):
            cm_match = RE_CM_LINE.search(line)
            if cm_match:
                true_label, pred_healthy, pred_faulted = cm_match.groups()
                cm_data[true_label]["pred_healthy"] = int(pred_healthy)
                cm_data[true_label]["pred_faulted"] = int(pred_faulted)

        run["confusion_matrix"] = [
            [cm_data["Healthy"].get("pred_healthy", 0),
             cm_data["Healthy"].get("pred_faulted", 0)],
            [cm_data["Faulted"].get("pred_healthy", 0),
             cm_data["Faulted"].get("pred_faulted", 0)],
        ]
    except ValueError:
        logging.warning(
            "Could not find test results in %s. Run may be incomplete.", log_path
        )

    # ── Artefact paths ───────────────────────────────────────────────────────
    def _path(filename: str) -> Optional[str]:
        p = os.path.join(run_dir, filename).replace("\\", "/")
        return p if os.path.exists(p) else None

    run["training_curves_path"] = os.path.join(
        run_dir, "training_curves.png"
    ).replace("\\", "/")
    run["best_model_path"] = _path("model_best.pt")
    run["last_model_path"] = _path("model_last.pt")

    return run


# ── Aggregation across seeds ─────────────────────────────────────────────────

def aggregate_by_config(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group runs by configuration name and summarise across seeds.

    Reporting mean +- std across seeds is what makes two configurations
    genuinely comparable. A single run's test accuracy carries the variance of
    weight initialisation and batch ordering, which for an unstable
    configuration can exceed the difference being claimed between feature sets.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        key = run.get("config_name") or run.get("run_codename")
        grouped.setdefault(key, []).append(run)

    def _mean(values: List[float]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    def _std(values: List[float]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        if len(vals) < 2:
            return 0.0 if vals else None
        return round(statistics.stdev(vals), 4)

    aggregates: List[Dict[str, Any]] = []
    for config_name, group in grouped.items():
        test_accs = [r["test_results"].get("accuracy") for r in group]
        spreads = [r.get("val_acc_last50_spread") for r in group]
        gaps = [r.get("final_vs_best_gap") for r in group]

        healthy_recall = [
            r["test_results"].get("Healthy", {}).get("recall") for r in group
        ]
        faulted_recall = [
            r["test_results"].get("Faulted", {}).get("recall") for r in group
        ]

        valid_spreads = [s for s in spreads if s is not None]
        aggregates.append({
            "config_name": config_name,
            "n_runs": len(group),
            "seeds": sorted(r["seed"] for r in group if r["seed"] is not None),
            "num_features": group[0].get("num_features", 0),
            "features_used": group[0].get("features_used", []),
            "test_acc_mean": _mean(test_accs),
            "test_acc_std": _std(test_accs),
            "test_acc_min": round(min([a for a in test_accs if a is not None]), 4)
                            if any(a is not None for a in test_accs) else None,
            "test_acc_max": round(max([a for a in test_accs if a is not None]), 4)
                            if any(a is not None for a in test_accs) else None,
            "best_val_acc_mean": _mean([r.get("best_val_acc") for r in group]),
            "healthy_recall_mean": _mean(healthy_recall),
            "faulted_recall_mean": _mean(faulted_recall),
            "val_spread_mean": _mean(spreads),
            "val_spread_worst": round(max(valid_spreads), 4) if valid_spreads else None,
            "final_vs_best_gap_mean": _mean(gaps),
            "log_transform": group[0].get("log_transform"),
            "dropout": group[0].get("dropout"),
            "run_codenames": [r["run_codename"] for r in group],
        })

    aggregates.sort(
        key=lambda a: (a["test_acc_mean"] is not None, a["test_acc_mean"]),
        reverse=True,
    )
    return aggregates


# ── Entry point ──────────────────────────────────────────────────────────────

def main(
    runs_directory: str = "runs",
    output_file: str = "runs_summary.json",
    aggregate_file: str = "runs_aggregate.json",
) -> None:
    """Parse every run directory and write the per-run and aggregated summaries."""
    if not os.path.isdir(runs_directory):
        logging.error(
            "Directory not found: '%s'. Please run your experiments first.",
            runs_directory,
        )
        return

    run_dirs = sorted(
        (d for d in os.listdir(runs_directory)
         if os.path.isdir(os.path.join(runs_directory, d))),
        reverse=True,
    )
    logging.info(
        "Found %d experiment directories in '%s'. Parsing...",
        len(run_dirs), runs_directory,
    )

    all_runs: List[Dict[str, Any]] = []
    for run_name in run_dirs:
        parsed = parse_log_file(
            os.path.join(runs_directory, run_name, "run_log.txt")
        )
        if parsed and parsed["run_codename"]:
            all_runs.append(parsed)
        else:
            logging.warning(
                "Skipping directory '%s' due to parsing errors or missing codename.",
                run_name,
            )

    aggregates = aggregate_by_config(all_runs)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2)
        with open(aggregate_file, "w", encoding="utf-8") as f:
            json.dump(aggregates, f, indent=2)
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to write summary files: %s", exc)
        return

    logging.info(
        "Parsed %d runs across %d configuration(s). Wrote '%s' and '%s'.",
        len(all_runs), len(aggregates), output_file, aggregate_file,
    )

    n_missing = sum(1 for r in all_runs if r["best_val_acc"] is None)
    if n_missing:
        logging.warning(
            "%d run(s) predate the stability logging and have no best/last-50 "
            "fields. Re-run those experiments for comparable numbers.", n_missing
        )


if __name__ == "__main__":
    main()