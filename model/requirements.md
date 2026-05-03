# Project Requirements: Modular ML Experimentation Pipeline

## 1. Overview

A monolithic fault-detection training script has been refactored into a
configuration-driven, modular pipeline.  The goal is to efficiently run,
track, and compare multiple experiments with different feature sets and
hyperparameters **without touching any source file other than `config.py`**.

---

## 2. File Structure

```
project_folder/
├── runs/                   # Auto-created. One subdirectory per experiment run.
├── main_data.csv           # Input data source (see Section 5 for column spec).
├── main.py                 # Orchestrator — entry point, calls all modules.
├── config.py               # Control panel — hyperparameters & experiment list.
├── data_processing.py      # Data loading, rounding, feature engineering, splitting.
├── model.py                # FaultDetector nn.Module definition.
├── training.py             # Training loop and test-set evaluation.
└── utils.py                # Shared helpers: codename, logging, plotting.
```

---

## 3. Module Responsibilities

### `config.py`
- Non-executable configuration file; no logic.
- Defines global constants: `INPUT_FILE`, `RUNS_DIR`, `RANDOM_STATE`,
  `TEST_SIZE`, `VAL_FRACTION`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`,
  `WEIGHT_DECAY`, `LR_PATIENCE`, `LR_FACTOR`, `EARLY_STOP_PATIENCE`.
- Defines `TRAINING_RUNS`: a list of experiment dicts, each with:
  - `"name"` (str) — unique slug for directory naming and logging.
  - `"features"` (list[str]) — subset of `ALL_ENGINEERED_FEATURES` to use.

### `data_processing.py`
- `load_raw_data(filepath)` — loads CSV and applies rounding (see Section 6).
- `engineer_features(df)` — computes **all** possible features; returns a
  DataFrame whose columns are exactly `ALL_ENGINEERED_FEATURES`.
- `prepare_dataloaders(features_df, labels, selected_features, run_dir)` —
  selects the feature subset, performs the stratified split, fits a
  `StandardScaler` on the training split only, saves `scaler.pkl`, converts
  everything to tensors, and returns `(train_loader, val_loader, X_test,
  y_test, scaler)`.

### `model.py`
- Contains only the `FaultDetector(nn.Module)` class.
- Architecture: `Linear(input_dim→64) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64→1)` — outputs a raw logit; sigmoid is applied externally.
- `input_dim` is set at runtime from the length of the experiment's feature list.

### `training.py`
- `run_epoch(model, loader, criterion, optimizer, device, training)` — one
  pass over a DataLoader; returns `(avg_loss, accuracy)`.
- `train_model(model, train_loader, val_loader, device, run_dir, logger)` —
  full training loop with early stopping; saves `model_last.pt`; returns
  `history` dict.
- `evaluate_on_test(model, X_test, y_test, device, logger)` — runs inference
  on the test split and logs the classification report + labelled confusion
  matrix through the supplied logger (guaranteeing file output).
- **All logging uses the caller-supplied `logger`** — never a module-level
  logger — so every line reaches both the console and `run_log.txt`.

### `utils.py`
- `generate_run_codename(experiment_name)` → `"YYYYMMDD_HHMM_<name>"`.
- `setup_logging(run_dir, run_codename)` — creates a named logger with a
  `StreamHandler` (console) and a `FileHandler` (`run_log.txt`); returns it.
- `plot_training_curves(history, run_dir)` — saves a two-panel PNG
  (loss + accuracy, train vs val) as `training_curves.png`.

### `main.py`
- Loads data and engineers features **once** before the experiment loop
  (shared across all runs for efficiency).
- Iterates `config.TRAINING_RUNS`; for each experiment calls `run_experiment()`.
- `run_experiment()` wires everything together: creates the run directory,
  sets up logging, calls `prepare_dataloaders`, instantiates `FaultDetector`,
  calls `train_model` and `evaluate_on_test` (passing the run logger to both),
  saves plots, and logs elapsed time.
- Contains `export_test_data()` — a **standalone utility** (not part of the
  training loop) that exports the stratified test split (all original columns)
  to a CSV for final holdout validation. It is **commented out by default**
  in the `__main__` block.

---

## 4. Experiment Configuration (how to add a new run)

Open `config.py` and append a dict to `TRAINING_RUNS`:

```python
{
    "name": "my_new_experiment",          # must be unique
    "features": ["Ia", "Ib", "Ic", "vuf"],  # any subset of ALL_ENGINEERED_FEATURES
},
```

No other file needs to be changed.

---

## 5. Dataset Column Reference (`main_data.csv`)

```
No, vuf, V_A, V_B, V_C, I_A, I_B, I_C,
V_a, V_b, V_c, I_a, I_b, I_c,
phase_B, phase_C, phase_a, phase_b, phase_c,
faulted
```

| Column group | Columns | Side |
|---|---|---|
| Voltages | `V_A`, `V_B`, `V_C` | Pre-transformer |
| Currents | `I_A`, `I_B`, `I_C` | Pre-transformer |
| Phase angles | `phase_B`, `phase_C` | Pre-transformer (`phase_A` absent) |
| Voltages | `V_a`, `V_b`, `V_c` | Post-transformer |
| Currents | `I_a`, `I_b`, `I_c` | Post-transformer |
| Phase angles | `phase_a`, `phase_b`, `phase_c` | Post-transformer |
| Raw VUF | `vuf` | Measured (not derived) |
| Target | `faulted` | 0 = Healthy, 1 = Faulted |
| Index | `No` | Row number (not used as feature) |

> **Convention**: CAPITAL letters (e.g. `V_A`, `I_B`) = measurements **before**
> the transformer.  Small letters (e.g. `V_a`, `I_b`) = measurements **after**
> the transformer.

---

## 6. Mandatory Rounding (applied in `load_raw_data`)

| Columns | Rule |
|---|---|
| `V_a`, `V_b`, `V_c` | Round to **1 decimal place** |
| `V_A`, `V_B`, `V_C` | Round to **1 decimal place** |
| `I_a`, `I_b`, `I_c` | Round to **nearest integer** |
| `I_A`, `I_B`, `I_C` | Round to **nearest integer** |

---

## 7. Full Engineered Feature Catalogue (`ALL_ENGINEERED_FEATURES`)

All features below are produced by `engineer_features()` and available for
selection in `config.TRAINING_RUNS[*]["features"]`.

### Post-transformer raw
| Feature | Source column(s) | Description |
|---|---|---|
| `Va`, `Vb`, `Vc` | `V_a`, `V_b`, `V_c` | Raw post-transformer voltages |
| `Ia`, `Ib`, `Ic` | `I_a`, `I_b`, `I_c` | Raw post-transformer currents |
| `phase_a`, `phase_b`, `phase_c` | same | Post-transformer phase angles |

### Post-transformer derived
| Feature | Description |
|---|---|
| `diff_ab`, `diff_bc`, `diff_ca` | Pairwise absolute voltage differences |
| `dev_a`, `dev_b`, `dev_c` | Normalised deviation from mean voltage: `(Va − V̄) / V̄` |
| `min_max_ratio` | `min(Va,Vb,Vc) / max(Va,Vb,Vc)` — ≈1 when balanced |
| `vuf` | NEMA voltage unbalance factor (derived): `max_deviation / V̄` |
| `neg_seq_ratio` | `|V₂| / |V₁|` via symmetrical components — ≈0 healthy |

### Pre-transformer raw
| Feature | Source column(s) | Description |
|---|---|---|
| `VA`, `VB`, `VC` | `V_A`, `V_B`, `V_C` | Raw pre-transformer voltages |
| `IA`, `IB`, `IC` | `I_A`, `I_B`, `I_C` | Raw pre-transformer currents |
| `phase_B`, `phase_C` | same | Pre-transformer phase angles (`phase_A` not in data) |

### Pre-transformer derived
| Feature | Description |
|---|---|
| `diff_AB`, `diff_BC`, `diff_CA` | Pairwise absolute voltage differences |
| `dev_A`, `dev_B`, `dev_C` | Normalised deviation from mean voltage |
| `min_max_ratio_pre` | `min(VA,VB,VC) / max(VA,VB,VC)` |
| `vuf_pre` | NEMA VUF computed from pre-transformer voltages |
| `neg_seq_ratio_pre` | Negative-sequence ratio from pre-transformer voltages |

### Other
| Feature | Source | Description |
|---|---|---|
| `vuf_raw` | `vuf` column | VUF as recorded in the CSV (not re-derived) |

---

## 8. Per-Run Output Artefacts

Each experiment run creates a directory: `runs/<YYYYMMDD_HHMM_name>/`

| File | Contents |
|---|---|
| `model_last.pt` | PyTorch `state_dict` at the **final epoch** (not best epoch) |
| `scaler.pkl` | Fitted `StandardScaler` — required for inference on new data |
| `training_curves.png` | Two-panel plot: train/val loss (left) + train/val accuracy (right) |
| `run_log.txt` | Full run log: features used, per-epoch metrics, classification report, labelled confusion matrix, total training time |

---

## 9. Data Split

| Split | Fraction | Role |
|---|---|---|
| Train | 70 % | Model training |
| Validation | 15 % | LR scheduling, early stopping |
| Test | 15 % | Final evaluation (never seen during training) |

Split is **stratified** on the `faulted` label.  `RANDOM_STATE = 42` for
reproducibility.

---

## 10. Model Architecture

```
Input (input_dim)
  → Linear(input_dim, 64)
  → BatchNorm1d(64)
  → ReLU
  → Dropout(0.3)
  → Linear(64, 1)        ← raw logit
```

Loss: `BCEWithLogitsLoss` (sigmoid applied internally).  
Optimiser: `Adam(lr=1e-3, weight_decay=1e-4)`.  
LR schedule: `ReduceLROnPlateau(patience=10, factor=0.5)`.

---

## 11. Best Practices Enforced

- **Type hints** on all function signatures.
- **Docstrings** on all functions and classes.
- **Paths via `os.path.join`** — no string concatenation.
- **`runs/` auto-created** with `os.makedirs(exist_ok=True)`.
- **Scaler saved** alongside model weights for reproducible inference.
- **Logger passed as a parameter** to `train_model` and `evaluate_on_test`
  so every log line reaches both console and `run_log.txt`.
- **Feature validation**: `prepare_dataloaders` raises `ValueError` for
  unknown feature names before any training begins.
