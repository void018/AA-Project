# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration file. Edit this file to change hyperparameters or
# define new training experiments. No other source file needs to be modified.
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FILE: str = "../main_data.csv"
RUNS_DIR: str = "runs"

# ── Feature conditioning ──────────────────────────────────────────────────────
# Currents are heavy-tailed (log-spaced load sweep over 1000x), so log1p is the
# right compression for them. Voltages are NOT heavy-tailed — they are a narrow
# band sitting on a huge offset (~133 kV), where log1p does almost nothing and
# StandardScaler leaves the discriminative signal at ~0.01 std units. Expressing
# them as per-unit deviation from nominal puts that signal at O(1) instead.
LOG_TRANSFORM: bool = True
LOG_TRANSFORM_FEATURES: tuple[str, ...] = (
    "IA", "IB", "IC", "Ia", "Ib", "Ic",
)

# "per_unit_deviation" | "log1p" | "none"
VOLTAGE_CONDITIONING: str = "per_unit_deviation"
VOLTAGE_FEATURES: tuple[str, ...] = ("VA", "VB", "VC", "Va", "Vb", "Vc")

# Yg primary -> phase voltage = 230 kV / sqrt(3); Delta secondary -> 13.8 kV
V_NOMINAL_PRE: float = 230e3 / (3 ** 0.5)
V_NOMINAL_POST: float = 13.8e3

# ── Instrument model (IEC 61869-2 CT / 61869-3 VT) ────────────────────────────
# The exported dataset records ideal measurements. A real CT's error is referred
# to its RATED current, not to the reading, so a healthy pre-side current of
# 0.7 A on a 251 A CT is far below what any relay could resolve — which is
# precisely why open-phase conditions go undetected in service (Section 2.5.4).
# Without this, an open phase is trivially detectable by the rule "I_A < 0.4 A"
# and the whole feature-set comparison collapses to a measurement of optimiser
# quality rather than of information content.
INSTRUMENT_MODEL: bool = True
CT_ACCURACY_CLASS: float = 0.005      # 0.5% of rated current (class 0.5)
VT_ACCURACY_CLASS: float = 0.005      # 0.5% of reading      (class 0.5)
# proportional (gain) error on the reading
CT_RATIO_ERROR: float = 0.002
INSTRUMENT_RNG_SEED: int = 12345      # fixed so the noise draw is reproducible

S_RATED_VA: float = 100e6             # transformer nominal power (Table 3.1)
V_RATED_PRE: float = 230e3
V_RATED_POST: float = 13.8e3

# Accuracy classes swept as an experimental variable. 0.0 reproduces the
# original ideal-measurement dataset for reference.
CT_CLASS_SWEEP: tuple[float, ...] = (0.0, 0.001, 0.002, 0.005, 0.01)

# ── Degenerate features ───────────────────────────────────────────────────────
# phase_a is exactly -0.01 for all 10,000 faulted rows: the phase of a signal
# whose amplitude is zero is undefined, and the solver emits a constant. On its
# own it separates the classes with 99.2% accuracy, which is an artifact of the
# simulation rather than a physical signature.
EXCLUDE_DEGENERATE_FEATURES: bool = True
DEGENERATE_FEATURES: tuple[str, ...] = ("phase_a",)

# ── Data split ────────────────────────────────────────────────────────────────
# Each simulation run emits a healthy row and a faulted row sharing identical
# load conditions. Grouping on the run keeps that pair on the same side of the
# split. Measured leakage is small, but grouping is correct and costs nothing.
GROUPED_SPLIT: bool = True

# ── Load-stratified reporting ─────────────────────────────────────────────────
# Detection difficulty depends strongly on loading; averaging over a 1000x sweep
# hides exactly the light-load regime the thesis is about.
LOAD_BINS: int = 5

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
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


# ── Experiment definitions ────────────────────────────────────────────────────
# Each entry requires:
#   "name"     – unique slug used in directory names and log output.
#   "features" – subset of columns produced by engineer_features().
#                See ALL_ENGINEERED_FEATURES in data_processing.py for valid names.

TRAINING_RUNS_TMP: list[dict] = [
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

iTRAINING_RUNS: list[dict] = [
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

iTRAINING_RUNS: list[dict] = [
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
