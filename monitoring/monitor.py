"""
monitoring/monitor.py
================================================================
WHAT THIS FILE DOES:
This is the "Monitoring" stage of the MLOps lifecycle. Once a
model is deployed, its job isn't done - the data it sees in
production can change over time (this is called "data drift"),
which can silently hurt model accuracy. This script uses
Evidently AI to compare:

    - reference_data.csv : data the model was TRAINED on
    - current_data.csv   : newer / incoming data (in this demo,
                            the held-out test split acts as a
                            simple stand-in for "new" data)

...and generates an HTML report showing whether the two datasets
still look statistically similar, or whether drift has occurred.

WHY THIS MATTERS IN MLOPS:
A model's accuracy on Day 1 doesn't guarantee accuracy on Day
100. If the distribution of incoming data changes (e.g. sensors
recalibrated, new flower varieties measured, a bug upstream),
the model can start making bad predictions without any code
"error" ever being thrown. Monitoring tools like Evidently give
early warning so a team knows when to investigate or retrain.
================================================================
"""

import os

import pandas as pd
import yaml
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[CONFIG ERROR] Could not find '{config_path}'. Run this "
            "script from the project root, e.g.:\n"
            "    python monitoring/monitor.py"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_datasets(cfg: dict):
    """
    Loads the reference and current datasets that were exported by
    src/train.py. Gives a clear, actionable error if training
    hasn't been run yet.
    """
    reference_path = cfg["paths"]["reference_data"]
    current_path = cfg["paths"]["current_data"]

    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        raise FileNotFoundError(
            "[DATA ERROR] Reference/current monitoring datasets not "
            "found. These are generated automatically when you train "
            "the model. Please run:\n    python src/train.py\n"
            "before running the monitoring script."
        )

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    return reference_df, current_df


def main():
    print("=" * 60)
    print("MLOps Iris Pipeline - Evidently AI Monitoring")
    print("=" * 60)

    cfg = load_config("config.yaml")

    print("\n[1/3] Loading reference and current datasets...")
    reference_df, current_df = load_datasets(cfg)
    print(f"      Reference dataset: {reference_df.shape[0]} rows")
    print(f"      Current dataset:   {current_df.shape[0]} rows")

    # ------------------------------------------------------------
    # Build an Evidently Report combining two useful presets:
    #
    #   - DataDriftPreset: statistically compares the distribution
    #     of each feature between reference and current data
    #     (e.g. using tests like Kolmogorov-Smirnov) and flags
    #     features that have "drifted".
    #
    #   - DataQualityPreset: checks for basic data quality issues
    #     such as missing values, duplicate rows, and unexpected
    #     value ranges.
    # ------------------------------------------------------------
    print("\n[2/3] Running data drift and data quality analysis...")
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    report.run(reference_data=reference_df, current_data=current_df)

    # ------------------------------------------------------------
    # Save the report as an interactive HTML file that can be
    # opened in any browser - this is the artifact used as
    # evidence/screenshots for the CCA report.
    # ------------------------------------------------------------
    print("\n[3/3] Saving monitoring report...")
    output_dir = "monitoring"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "monitoring_report.html")
    report.save_html(output_path)

    print(f"\n      Report saved to: {output_path}")
    print("      Open this file in a web browser to view the results.")

    # Also print a short, human-readable summary to the console so
    # results can be discussed/explained quickly in a viva without
    # needing to open the HTML file.
    result = report.as_dict()
    try:
        drift_metric = result["metrics"][0]["result"]
        dataset_drift = drift_metric.get("dataset_drift")
        n_drifted = drift_metric.get("number_of_drifted_columns")
        n_columns = drift_metric.get("number_of_columns")
        print("\n      --- Quick Summary ---")
        print(f"      Dataset-level drift detected: {dataset_drift}")
        print(f"      Drifted columns: {n_drifted} / {n_columns}")
    except (KeyError, IndexError, TypeError):
        print(
            "      (Could not parse a quick summary - open the HTML "
            "report for full details.)"
        )

    print("\nMonitoring run complete.")


if __name__ == "__main__":
    main()
