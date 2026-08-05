# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration file. Edit this file to change hyperparameters or
# define new training experiments. No other source file needs to be modified.
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FILE: str = "../main_data.csv"
RUNS_DIR: str = "runs_round2"

# ── Data split ────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.30     # fraction held out as test + val combined
VAL_FRACTION: float = 0.50  # fraction of TEST_SIZE that becomes validation
DATA_COUNT: int = 0

# ── Raw-value rounding ────────────────────────────────────────────────────────
# These MUST mirror the rounding already applied in main.m when main_data.csv
# was exported, otherwise the Python pipeline silently re-quantises the data:
#   main.m does  round(V)      -> voltages to the nearest integer
#                round(I, 1)   -> currents to one decimal place
ROUND_VOLTAGE_DECIMALS: int = 0
ROUND_CURRENT_DECIMALS: int = 1

# ── Feature conditioning ──────────────────────────────────────────────────────
# The load sweep is logarithmic over a 1000x range (40 W … 40 kW), so the raw
# current and voltage magnitudes are heavy-tailed. Applying StandardScaler to
# them directly compresses the healthy/faulted decision boundary to ~0.014 std
# units on a feature spanning ~9 std units, which makes training unstable: the
# boundary is far narrower than the weight jitter produced by Adam + dropout,
# so validation accuracy oscillates between ~0.75 and 1.00 instead of settling.
#
# log1p() is applied to the non-negative magnitude features before scaling,
# which widens that boundary by roughly 20x and removes the oscillation.
# It is deliberately NOT applied to:
#   - phase angles (phase_*)          -> can be negative
#   - normalised deviations (dev_*)   -> can be negative
#   - bounded ratios (vuf, min_max_ratio, neg_seq_ratio, ...) -> already O(1)
LOG_TRANSFORM: bool = True
LOG_TRANSFORM_FEATURES: tuple[str, ...] = (
    "IA", "IB", "IC", "Ia", "Ib", "Ic",
    "VA", "VB", "VC", "Va", "Vb", "Vc",
)

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int = 64
EPOCHS: int = 400
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
DROPOUT: float = 0.1        # was 0.3; at 3-6 input features that dropped a
# whole feature in ~30% of forward passes
LR_PATIENCE: int = 10       # epochs before ReduceLROnPlateau halves the LR
LR_FACTOR: float = 0.5

# Metric the LR scheduler and early stopping both watch. Validation LOSS is
# smooth and converges early even while validation ACCURACY is still thrashing,
# so scheduling on loss meant the LR was never reduced when it mattered.
# "val_acc" (maximise) or "val_loss" (minimise)
LR_MONITOR: str = "val_acc"
EARLY_STOP_PATIENCE: int = 40        # real value; was EPOCHS, i.e. disabled

# Evaluate the best-validation checkpoint rather than whatever the final epoch
# happened to land on. With an oscillating curve the final-epoch score is close
# to a coin flip, which is what produced +-0.26 swings between identical reruns.
USE_BEST_CHECKPOINT: bool = True

# ── Repeats ───────────────────────────────────────────────────────────────────
# Every experiment is run once per seed so that run-to-run variance can be
# reported as mean +- std instead of being mistaken for a real effect.
SEEDS: tuple[int, ...] = (0, 1, 2)


# ── Experiment definitions ────────────────────────────────────────────────────
# Each entry requires:
#   "name"     – unique slug used in directory names and log output.
#   "features" – subset of columns produced by engineer_features().
#                See ALL_ENGINEERED_FEATURES in data_processing.py for valid names.

TRAINING_RUNS: list[dict] = [
    # ── Post-transformer only ─────────────────────────────────────────────────
    {
        "name": "pre_currents_only",
        "features": ["IA", "IB", "IC"],
    },
    {
        "name": "post_currents_only",
        "features": ["Ia", "Ib", "Ic"],
    },
    {
        "name": "pre_voltages_only",
        "features": ["VA", "VB", "VC"],
        "skip": False,
    },
    {
        "name": "post_voltages_only",
        "features": ["Va", "Vb", "Vc"],
    },
    {
        "name": "all_currents_pre_voltages",
        "features": ["IA", "IB", "IC",
                     "Ia", "Ib", "Ic",
                     "VA", "VB", "VC"],
    },
    {
        "name": "all_currents_post_voltages",
        "features": ["IA", "IB", "IC",
                     "Ia", "Ib", "Ic",
                     "Va", "Vb", "Vc"],
    },

    {
        "name": "pre_currents_voltages",
        "features": ["IA", "IB", "IC",
                     "VA", "VB", "VC"],
    },
    {
        "name": "post_currents_voltages",
        "features": ["Ia", "Ib", "Ic",
                     "Va", "Vb", "Vc"],
    },

    {
        "name": "pre_currents_all_voltages",
        "features": ["IA", "IB", "IC",
                     "VA", "VB", "VC",
                     "Va", "Vb", "Vc"],
    },
    {
        "name": "post_currents_all_voltages",
        "features": ["Ia", "Ib", "Ic",
                     "VA", "VB", "VC",
                     "Va", "Vb", "Vc"],
    },
    {
        "name": "all_currents_all_voltages",
        "skip": False,
        "features": ["IA", "IB", "IC",
                     "Ia", "Ib", "Ic",
                     "VA", "VB", "VC",
                     "Va", "Vb", "Vc"],
    },

]

TRAINING_RUNS_TMP: list[dict] = [
    {
        "name": "post_volt_deviations_and_vuf",
        "features": ["dev_a", "dev_b", "dev_c", "vuf"],
    },
    {
        "name": "post_full",
        "features": [
            "dev_a", "dev_b", "dev_c",
            "min_max_ratio", "vuf", "neg_seq_ratio",
        ],
    },
    # ── Pre-transformer only ──────────────────────────────────────────────────
    # ── Combined pre + post ───────────────────────────────────────────────────
    {
        "name": "combined_full",
        "features": [
            "Ia", "Ib", "Ic",
            "dev_a", "dev_b", "dev_c",
            "min_max_ratio", "vuf", "neg_seq_ratio",
            "IA", "IB", "IC",
            "dev_A", "dev_B", "dev_C",
            "min_max_ratio_pre", "vuf_pre", "neg_seq_ratio_pre",
        ],
    },
    {
        "name": "combined_with_phases",
        "features": [
            "Ia", "Ib", "Ic",
            "phase_a", "phase_b", "phase_c",
            "dev_a", "dev_b", "dev_c",
            "vuf", "neg_seq_ratio",
            "IA", "IB", "IC",
            "phase_B", "phase_C",
            "dev_A", "dev_B", "dev_C",
            "vuf_pre", "neg_seq_ratio_pre",
        ],
    },
]

# TRAINING_RUNS_TMP: list[dict] = [
TRAINING_RUNS += [
    {
        "name": "derivations_post",
        "features": [
            "dev_a", "dev_b", "dev_c",
            "phase_a", "phase_b", "phase_c",
            "min_max_ratio", "vuf", "neg_seq_ratio",
        ],
    },
    {
        "name": "derivations_pre",
        "features": [
            "dev_A", "dev_B", "dev_C",
            "phase_B", "phase_C",
            "min_max_ratio_pre", "vuf_pre", "neg_seq_ratio_pre",
        ],
    },
    {
        "name": "derivations_full",
        "features": [
            "dev_a", "dev_b", "dev_c",
            "dev_A", "dev_B", "dev_C",

            "phase_a", "phase_b", "phase_c",
            "phase_B", "phase_C",

            "min_max_ratio", "vuf", "neg_seq_ratio",
            "min_max_ratio_pre", "vuf_pre", "neg_seq_ratio_pre",
        ],
    },
]

# Skip IA column
for r in TRAINING_RUNS:
    if "IA" in r["features"]:
        r["features"].remove("IA")
