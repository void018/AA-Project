import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

# Configure basic logging for the parser itself
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- REGEX PATTERNS ---
# We compile them here for efficiency, especially if parsing many files.

# Matches: Run codename: 18-04_1452_post_currents_voltages
RE_CODENAME = re.compile(r"Run codename: (.+)")

# Matches: Features (6): ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
RE_FEATURES = re.compile(r"Features \(\d+\): (.+)")

# Matches: Epoch 400 | Train acc 0.7955  loss 0.3672 | Val acc 0.7597  loss 0.3457
RE_LAST_EPOCH = re.compile(
    r"Epoch\s+\d+\s+\|\s+Train acc\s+([0-9.]+)\s+loss\s+([0-9.]+)\s+\|\s+Val acc\s+([0-9.]+)\s+loss\s+([0-9.]+)"
)

# Matches the lines of the classification report, e.g., "Healthy       0.66      1.00      0.79      1500"
RE_CLASS_REPORT_LINE = re.compile(
    r"^\s*(\w+|\w+\s\w+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)")

# Matches the overall accuracy line, e.g., "    accuracy                           0.74      3000"
RE_TEST_ACCURACY = re.compile(r"^\s*accuracy\s+([0-9.]+)")

# Matches the confusion matrix lines, e.g., "  True Healthy       1500        0"
RE_CM_LINE = re.compile(r"^\s+True\s(Healthy|Faulted)\s+([0-9]+)\s+([0-9]+)")


def parse_classification_report(report_str: str) -> Dict[str, Any]:
    """
    Parses a multi-line string from sklearn's classification_report into a structured dictionary.
    """
    report_data = {}
    lines = report_str.strip().split('\n')
    # --- CHANGE 1: Modified the loop to read all summary lines ---
    # The original `lines[2:-3]` was cutting off the summary. `lines[2:]` includes everything after the header.
    for line in lines[2:]:
        # Skip blank lines that might be between sections
        if not line.strip():
            continue

        # --- CHANGE 2: Added a check for the special 'accuracy' line ---
        accuracy_match = RE_TEST_ACCURACY.search(line)
        if accuracy_match:
            report_data['accuracy'] = float(accuracy_match.group(1))
            # This line is processed, so we can skip to the next iteration
            continue

        # Use the existing regex for all other lines (Healthy, Faulted, weighted avg, etc.)
        match = RE_CLASS_REPORT_LINE.search(line)
        if match:
            # The key can be "Healthy", "Faulted", or "weighted avg".
            # Replace space with underscore for a cleaner JSON key (e.g., "weighted_avg").
            key_name, precision, recall, f1_score, support = match.groups()
            clean_key = key_name.strip().replace(" ", "_")
            report_data[clean_key] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1_score),
                "support": int(support),
            }
    return report_data


def parse_log_file(log_path: str) -> Optional[Dict[str, Any]]:
    """
    Parses a single run_log.txt file to extract key experiment data.
    """
    if not os.path.exists(log_path):
        logging.warning(f"Log file not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    run_data: Dict[str, Any] = {
        "run_codename": None,
        "features_used": [],
        "last_train_acc": None,
        "last_train_loss": None,
        "last_val_acc": None,
        "last_val_loss": None,
        "test_results": {},
        "confusion_matrix": None,
        "training_curves_path": None,
    }

    # --- METADATA EXTRACTION ---
    codename_match = RE_CODENAME.search(content)
    if codename_match:
        run_data["run_codename"] = codename_match.group(1).strip()

    features_match = RE_FEATURES.search(content)
    if features_match:
        # Use eval to safely parse the string representation of a list
        run_data["features_used"] = eval(features_match.group(1))

    # --- LAST EPOCH METRICS ---
    # Find all epoch lines and take the last one
    epoch_matches = RE_LAST_EPOCH.findall(content)
    if epoch_matches:
        last_epoch = epoch_matches[-1]
        run_data["last_train_acc"] = float(last_epoch[0])
        run_data["last_train_loss"] = float(last_epoch[1])
        run_data["last_val_acc"] = float(last_epoch[2])
        run_data["last_val_loss"] = float(last_epoch[3])

    # --- TEST RESULTS EXTRACTION ---
    # Find the block of text for the classification report and confusion matrix
    try:
        test_results_start = content.index("TEST SET RESULTS")
        test_results_end = content.index(
            "Confusion Matrix:", test_results_start)
        report_text = content[test_results_start:test_results_end]
        run_data["test_results"] = parse_classification_report(report_text)

        # Parse confusion matrix
        cm_text_start = test_results_end
        cm_lines = content[cm_text_start:].split('\n')

        cm_data = {"Healthy": {}, "Faulted": {}}
        for line in cm_lines:
            cm_match = RE_CM_LINE.search(line)
            if cm_match:
                true_label, pred_healthy, pred_faulted = cm_match.groups()
                cm_data[true_label]["pred_healthy"] = int(pred_healthy)
                cm_data[true_label]["pred_faulted"] = int(pred_faulted)

        # Convert to a simple nested list for easy rendering [ [TN, FP], [FN, TP] ]
        tn = cm_data.get("Healthy", {}).get("pred_healthy", 0)
        fp = cm_data.get("Healthy", {}).get("pred_faulted", 0)
        fn = cm_data.get("Faulted", {}).get("pred_healthy", 0)
        tp = cm_data.get("Faulted", {}).get("pred_faulted", 0)
        run_data["confusion_matrix"] = [[tn, fp], [fn, tp]]

    except ValueError:
        logging.warning(f"Could not find test results in {
                        log_path}. Run may be incomplete.")

    # Store path to the image
    run_dir = os.path.dirname(log_path)
    run_data["training_curves_path"] = os.path.join(
        run_dir, "training_curves.png").replace("\\", "/")

    return run_data


def main(runs_directory: str = "./runs_round2/", output_file: str = "runs_summary_round2.json"):
    """
    Main function to parse all subdirectories in the runs directory
    and save the aggregated results to a JSON file.
    """
    if not os.path.isdir(runs_directory):
        logging.error(f"Directory not found: '{
                      runs_directory}'. Please run your experiments first.")
        return

    all_runs_data: List[Dict[str, Any]] = []

    # Get a sorted list of run directories, newest first
    run_dirs = sorted(
        [d for d in os.listdir(runs_directory) if os.path.isdir(
            os.path.join(runs_directory, d))],
        reverse=True
    )

    logging.info(f"Found {len(run_dirs)} experiment directories in '{
                 runs_directory}'. Parsing...")

    for run_name in run_dirs:
        log_file_path = os.path.join(runs_directory, run_name, "run_log.txt")
        parsed_data = parse_log_file(log_file_path)
        # Ensure the run was parsed successfully
        if parsed_data and parsed_data["run_codename"]:
            all_runs_data.append(parsed_data)
        else:
            logging.warning(f"Skipping directory '{
                            run_name}' due to parsing errors or missing codename.")

    # --- SAVE RESULTS ---
    try:
        with open(output_file, 'w') as f:
            json.dump(all_runs_data, f, indent=2)
        logging.info(f"Successfully parsed {
                     len(all_runs_data)} runs. Summary saved to '{output_file}'.")
    except Exception as e:
        logging.error(f"Failed to write summary to '{output_file}': {e}")


if __name__ == "__main__":
    main()
