# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration file. Edit this file to change hyperparameters or
# define new training experiments. No other source file needs to be modified.
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_FILE: str = "../main_data.csv"
RUNS_DIR: str = "runs"

# ── Data split ────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.30     # fraction held out as test + val combined
VAL_FRACTION: float = 0.50  # fraction of TEST_SIZE that becomes validation
DATA_COUNT: int = 0

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int = 64
EPOCHS: int = 400
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
LR_PATIENCE: int = 10       # epochs before ReduceLROnPlateau halves the LR
LR_FACTOR: float = 0.5
EARLY_STOP_PATIENCE: int = EPOCHS  # set higher than EPOCHS to effectively disable

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

TRAINING_RUNS: list[dict] = [
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

TRAINING_RUNS: list[dict] = [
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