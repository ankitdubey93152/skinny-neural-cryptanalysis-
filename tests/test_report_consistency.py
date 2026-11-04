"""Test suite for verifying that report.md numbers match their source CSV tables.

Rules:
1. Every numerical value in report.md must be sourced directly from saved metric files.
2. No hard-coded metrics or fabricated statistics.
3. Comparative wording must align with statistical significance (p < 0.05 or non-overlapping CIs).
4. Zero references to SM4 in the generated report.
"""

import os
import re
import pytest
import pandas as pd


def test_no_sm4_in_report():
    """Verify that report.md contains zero references to SM4."""
    report_path = os.path.join(os.path.dirname(__file__), "..", "results_skinny", "report.md")
    if not os.path.exists(report_path):
        pytest.skip("report.md does not exist yet.")

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "SM4" not in content, "Found legacy SM4 reference in report.md!"
    assert "sm4" not in content, "Found legacy sm4 reference in report.md!"


def test_report_numbers_match_difference_propagation_csv():
    """Verify that diffusion and avalanche metrics in report match difference_propagation_avalanche.csv."""
    diff_csv = os.path.join(os.path.dirname(__file__), "..", "results_skinny", "difference_propagation_avalanche.csv")
    report_path = os.path.join(os.path.dirname(__file__), "..", "results_skinny", "report.md")

    if not os.path.exists(diff_csv) or not os.path.exists(report_path):
        pytest.skip("diff_csv or report.md does not exist.")

    df_diff = pd.read_csv(diff_csv)
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    for _, row in df_diff.iterrows():
        r = int(row.get("Round", row.get("Rounds", 0)))
        hw_val = float(row.get("Mean HW (/128)", row.get("Mean_HW", 0.0)))
        mean_hw = f"{hw_val:.2f}"

        # Verify active cells or mean HW appear in the report
        assert mean_hw in report_text or f"{hw_val:.1f}" in report_text or f"{round(hw_val, 1)}" in report_text, (
            f"Mean HW {mean_hw} for round {r} not found in report.md"
        )


def test_report_numbers_match_experiment_a_sweep_csv():
    """Verify that Experiment A metrics in report match table_experiment_a_sweep.csv."""
    sweep_csv = os.path.join(os.path.dirname(__file__), "..", "results_skinny", "table_experiment_a_sweep.csv")
    report_path = os.path.join(os.path.dirname(__file__), "..", "results_skinny", "report.md")

    if not os.path.exists(sweep_csv) or not os.path.exists(report_path):
        pytest.skip("sweep_csv or report.md does not exist.")

    df_sweep = pd.read_csv(sweep_csv)
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    for _, row in df_sweep.iterrows():
        acc = float(row["Accuracy"])
        # Format as percentage or decimal
        acc_pct = f"{acc * 100:.2f}"
        acc_pct_alt = f"{acc * 100:.1f}"

        # Either percentage string or raw decimal should be in the report table
        assert (acc_pct in report_text or acc_pct_alt in report_text or f"{acc:.4f}" in report_text), (
            f"Accuracy {acc} for {row.get('Model')} at round {row.get('Rounds')} not found in report.md"
        )


def test_report_has_required_sections_and_metadata():
    """Verify that report.md contains all required scientific sections and run metadata."""
    report_path = os.path.join(os.path.dirname(__file__), "..", "results_skinny", "report.md")
    if not os.path.exists(report_path):
        pytest.skip("report.md does not exist.")

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    required_headers = [
        "Executive Summary",
        "Empirical Difference Propagation & Avalanche Analysis",
        "Experiment A: Single-Pair Differential Distinguisher",
        "Non-Neural Baseline Distinguisher Comparisons",
        "Scientific Observations & Literature Framing",
        "Execution Environment & Package Versions",
    ]

    for header in required_headers:
        assert header in content, f"Missing required section header: '{header}'"

