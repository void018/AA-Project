# utils.py
"""
Shared helper utilities:
  - generate_run_codename : timestamped, slug-based run identifier.
  - setup_logging         : dual-sink logger (console + run_log.txt).
  - plot_training_curves  : saves training_curves.png for a run.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Run identification
# ─────────────────────────────────────────────────────────────────────────────

def generate_run_codename(experiment_name: str) -> str:
    """Return a unique, filesystem-safe identifier for a training run.

    The codename has the form ``YYYYMMDD_HHMM_<experiment_name>``, e.g.
    ``20231027_1130_currents_only``.

    Parameters
    ----------
    experiment_name : str
        The ``"name"`` field from the experiment dict in ``config.py``.

    Returns
    -------
    str
        Timestamp-prefixed run codename.
    """
    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    return f"{timestamp}_{experiment_name}"


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(run_dir: str, run_codename: str) -> logging.Logger:
    """Configure a named logger that writes to both the console and a log file.

    Creates ``<run_dir>/run_log.txt``.  The logger name matches
    *run_codename* so each experiment keeps an isolated logger.

    Parameters
    ----------
    run_dir      : str – directory where ``run_log.txt`` is written.
    run_codename : str – unique run identifier used as the logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_path = os.path.join(run_dir, "run_log.txt")

    logger = logging.getLogger(run_codename)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called more than once
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: Dict[str, list], run_dir: str) -> None:
    """Save a two-panel training-curves figure to *run_dir*.

    The left panel shows training vs. validation **loss**;
    the right panel shows training vs. validation **accuracy**.

    Parameters
    ----------
    history : dict
        Keys ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``;
        values are lists of per-epoch scalars (as returned by
        :func:`training.train_model`).
    run_dir : str
        Destination directory; the file is saved as ``training_curves.png``.
    """
    # Import here to keep the module import-time cost low and avoid
    # matplotlib initialisation when plotting is not needed.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
