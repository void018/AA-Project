# training.py
"""
Implements the per-epoch training / validation loop and the final
test-set evaluation with classification report and confusion matrix.

All logging is done through a caller-supplied ``logging.Logger`` so that
every line is guaranteed to reach both the console and the run's log file.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import config


# ─────────────────────────────────────────────────────────────────────────────
# Single-epoch helper
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool = True,
) -> Tuple[float, float]:
    """Run one full pass over *loader* in either train or eval mode.

    Parameters
    ----------
    model     : The FaultDetector instance.
    loader    : DataLoader for the current split.
    criterion : Loss function (BCEWithLogitsLoss).
    optimizer : Adam optimiser — only used when ``training=True``.
    device    : CPU or CUDA device.
    training  : If ``True``, perform backpropagation and parameter updates.

    Returns
    -------
    avg_loss : float – mean BCE loss per sample.
    accuracy : float – fraction of correctly classified samples.
    """
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Full training run
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: str,
    logger: logging.Logger,
) -> Dict[str, list]:
    """Train *model* for the number of epochs defined in config.

    Saves ``model_last.pt`` (final-epoch weights, not necessarily best) to
    *run_dir*.  Early stopping triggers when validation accuracy stagnates for
    ``config.EARLY_STOP_PATIENCE`` consecutive epochs.

    Parameters
    ----------
    model        : Freshly initialised FaultDetector.
    train_loader : DataLoader for the training split.
    val_loader   : DataLoader for the validation split.
    device       : CPU or CUDA device.
    run_dir      : Experiment output directory for artefact persistence.
    logger       : Run-specific logger (writes to console AND run_log.txt).

    Returns
    -------
    history : dict with keys ``train_loss``, ``val_loss``,
              ``train_acc``, ``val_acc`` — one float per epoch.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.LR_PATIENCE,
        factor=config.LR_FACTOR,
    )

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 20 == 0:
            logger.info(
                "Epoch %3d | Train acc %.4f  loss %.4f | Val acc %.4f  loss %.4f",
                epoch, tr_acc, tr_loss, vl_acc, vl_loss,
            )

        if patience_count >= config.EARLY_STOP_PATIENCE:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    # ── Save final-epoch weights ──────────────────────────────────────────────
    model_path = os.path.join(run_dir, "model_last.pt")
    torch.save(model.state_dict(), model_path)
    logger.info("Saved final model weights → %s", model_path)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Test-set evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Evaluate *model* on the held-out test set and log every result line.

    Logs a full sklearn classification report (precision, recall, F1, support)
    and the confusion matrix.  Both are written to the console **and** to the
    run's ``run_log.txt`` via the supplied logger.

    Parameters
    ----------
    model  : Trained FaultDetector (set to eval mode internally).
    X_test : Scaled test-feature tensor, shape ``(n, input_dim)``.
    y_test : Test-label tensor, shape ``(n,)``.
    device : CPU or CUDA device.
    logger : Run-specific logger (writes to console AND run_log.txt).
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

    y_true = y_test.numpy()

    report = classification_report(
        y_true, preds, target_names=["Healthy", "Faulted"]
    )
    cm = confusion_matrix(y_true, preds)

    # Format confusion matrix as a labelled table for readability in the log
    cm_lines = [
        "                 Pred Healthy  Pred Faulted",
        f"  True Healthy       {cm[0, 0]:<12d}{cm[0, 1]}",
        f"  True Faulted       {cm[1, 0]:<12d}{cm[1, 1]}",
    ]
    cm_str = "\n".join(cm_lines)

    logger.info("")
    logger.info("─" * 55)
    logger.info("TEST SET RESULTS")
    logger.info("─" * 55)
    logger.info("\n%s", report)
    logger.info("Confusion Matrix:\n%s", cm_str)
    logger.info("─" * 55)
