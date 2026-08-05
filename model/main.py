# main.py
"""
Entry point for the modular ML experimentation pipeline.

Running this file executes every experiment defined in config.TRAINING_RUNS
and writes all artefacts (model, scaler, curves, log) to runs/<codename>/.

To export a consistent hold-out test CSV, uncomment the export_test_data()
call at the bottom of the __main__ block.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from time import time

import numpy as np
import torch

import config
import data_processing as dp
import training as tr
import utils
from model import FaultDetector

# Module-level logger (used only for startup/teardown messages in main).
_log = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Test-data export utility  (standalone – not part of the training loop)
# ─────────────────────────────────────────────────────────────────────────────

def export_test_data(output_path: str = "test_data_export.csv") -> None:
    """Export only the test split of the raw dataset to a CSV file.

    Mirrors the **exact two-step stratified split** used by
    :func:`data_processing.prepare_dataloaders` so the exported rows are
    identical to the held-out test set seen during training:

    - Step 1: 70 % train  |  30 % temp   (``TEST_SIZE = 0.30``)
    - Step 2: 50 % val    |  50 % test   (``VAL_FRACTION = 0.50``)
      → final test set is **15 %** of the full dataset.

    The validation split is discarded; only the test rows are saved.
    All original CSV columns are preserved in the output file.

    This function is intentionally separate from the training loop so that
    a stable holdout set can be created on demand without affecting any run.

    Parameters
    ----------
    output_path : str
        Destination CSV file path (default: ``test_data_export.csv``).
    """
    from sklearn.model_selection import train_test_split

    print(f"[export_test_data] Loading {config.INPUT_FILE} …")
    df = dp.load_raw_data(config.INPUT_FILE)
    labels = df["faulted"].values

    # ── Step 1: carve out the same 30 % temp block used in training ───────────
    _, df_temp = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels,
    )

    # ── Step 2: split temp in half — discard val, keep only test ─────────────
    labels_temp = df_temp["faulted"].values
    _, df_test = train_test_split(
        df_temp,
        test_size=config.VAL_FRACTION,
        random_state=config.RANDOM_STATE,
        stratify=labels_temp,
    )

    df_test.to_csv(output_path, index=False)
    print(f"[export_test_data] Saved {len(df_test)} rows ({
          len(df_test)/len(df):.1%} of dataset) → '{output_path}'")
    print(f"[export_test_data] Class balance — faulted: {
          df_test['faulted'].mean():.2%}")
    print(f"[export_test_data] Columns: {list(df_test.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    experiment: dict,
    features_df,
    labels: np.ndarray,
    device: torch.device,
    seed: int = 0,
) -> None:
    """Execute one full training run for a single experiment definition.

    Steps
    -----
    1. Create a timestamped run directory under ``runs/``.
    2. Set up dual-sink logging (console + ``run_log.txt``).
    3. Prepare DataLoaders and persist the fitted scaler.
    4. Instantiate and train :class:`model.FaultDetector`.
    5. Evaluate on the test split and log results.
    6. Save training-curve plots.

    Parameters
    ----------
    experiment  : dict – one entry from ``config.TRAINING_RUNS``.
    features_df : pd.DataFrame – output of :func:`data_processing.engineer_features`.
    labels      : np.ndarray   – binary target array (float32).
    device      : torch.device – CPU or CUDA.
    """
    name = experiment["name"]
    selected_features = experiment["features"]

    # ── 0. Seed every RNG that affects this run ───────────────────────────────
    # Without this, weight initialisation and batch shuffling differ silently
    # between runs, so repeated runs of the same configuration are not
    # comparable and their spread cannot be attributed.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── 1. Create run directory ───────────────────────────────────────────────
    codename = f"{utils.generate_run_codename(name)}_s{seed}"
    run_dir = os.path.join(config.RUNS_DIR, codename)
    os.makedirs(run_dir, exist_ok=True)

    # ── 2. Set up logging ─────────────────────────────────────────────────────
    logger = utils.setup_logging(run_dir, codename)
    logger.info("=" * 60)
    logger.info("Experiment : %s", name)
    logger.info("Run codename: %s", codename)
    logger.info("Seed: %d", seed)
    logger.info("Features (%d): %s", len(selected_features), selected_features)
    logger.info("Log-transform: %s | dropout: %.2f | monitor: %s | best-ckpt: %s",
                config.LOG_TRANSFORM, config.DROPOUT,
                config.LR_MONITOR, config.USE_BEST_CHECKPOINT)
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    run_start = time()

    # ── 3. Data preparation ───────────────────────────────────────────────────
    logger.info("Preparing data …")
    train_loader, val_loader, X_test, y_test, _ = dp.prepare_dataloaders(
        features_df=features_df,
        labels=labels,
        selected_features=selected_features,
        run_dir=run_dir,
    )
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(X_test)
    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        n_train, n_val, n_test,
    )
    logger.info("Class balance — faulted: %.2f%%", float(y_test.mean()) * 100)

    # ── 4. Model instantiation & training ─────────────────────────────────────
    logger.info("Initialising FaultDetector (input_dim=%d) …",
                len(selected_features))
    model = FaultDetector(input_dim=len(selected_features)).to(device)

    logger.info("Training for up to %d epochs …", config.EPOCHS)
    history = tr.train_model(
        model, train_loader, val_loader, device, run_dir, logger)

    # ── 5. Test-set evaluation ────────────────────────────────────────────────
    tr.evaluate_on_test(model, X_test, y_test, device, logger)

    # ── 6. Plots & timing ────────────────────────────────────────────────────
    utils.plot_training_curves(history, run_dir)
    logger.info("Training curves saved → %s",
                os.path.join(run_dir, "training_curves.png"))

    elapsed = timedelta(seconds=time() - run_start)
    logger.info("Total training time: %s", elapsed)
    logger.info("Artefacts saved in: %s", run_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Optional: export a stable hold-out test CSV ───────────────────────────
    # Uncomment the line below to generate test_data_export.csv.
    # export_test_data()

    # ── Shared setup ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("Using device: %s", device)

    os.makedirs(config.RUNS_DIR, exist_ok=True)

    # ── Load & engineer features once (shared across all runs) ───────────────
    _log.info("Loading data from '%s' …", config.INPUT_FILE)
    raw_df = dp.load_raw_data(config.INPUT_FILE)

    _log.info("Engineering features …")
    features_df = dp.engineer_features(raw_df)
    labels = raw_df["faulted"].values.astype("float32")

    _log.info(
        "Dataset ready — %d samples | class balance (faulted): %.2f%%",
        len(labels), float(labels.mean()) * 100,
    )

    # ── Run all experiments, once per seed ────────────────────────────────────
    n_exp = len(config.TRAINING_RUNS)
    n_seeds = len(config.SEEDS)
    _log.info("Found %d experiment(s) x %d seed(s) = %d run(s).",
              n_exp, n_seeds, n_exp * n_seeds)

    for i, experiment in enumerate(config.TRAINING_RUNS, start=1):
        if experiment.get("skip", False):
            _log.info("Skipping experiment: %s", experiment["name"])
            continue
        for seed in config.SEEDS:
            _log.info("\n[%d/%d] Starting experiment: %s (seed %d)",
                      i, n_exp, experiment["name"], seed)
            run_experiment(experiment, features_df, labels, device, seed=seed)

    _log.info("\nAll experiments complete.")
