"""Unit and integration tests for generate_report.py."""

import os
import sys
import tempfile
import pytest
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import generate_report
from generate_report import (
    df_to_markdown_table,
    load_and_normalize_table,
    analyze_experiment_data,
    generate_markdown_report,
)


class TestReportGenerator:
    """Test suite for Markdown report generation and statistical calculations."""

    def test_sys_path_points_to_project_root(self):
        """Verify that generate_report sets sys.path to the project root, not its parent."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        assert project_root in sys.path

    def test_df_to_markdown_table_formatting(self):
        """Test markdown table conversion and cell formatting."""
        df = pd.DataFrame([
            {"Rounds": 1, "Model": "1D CNN", "Accuracy": 0.9995, "CI": "[0.99, 1.0]", "p-value": 0.0, "AUC": 1.0},
            {"Rounds": 4, "Model": "MLP", "Accuracy": 0.5020, "CI": "[0.49, 0.51]", "p-value": 0.4567, "AUC": 0.5012},
        ])
        md_table = df_to_markdown_table(df)
        assert "| Rounds | Model | Accuracy | CI | p-value | AUC |" in md_table
        assert "| --- | --- | --- | --- | --- | --- |" in md_table
        assert "0.9995" in md_table
        assert "0.0" in md_table
        assert "0.4567" in md_table

    def test_analyze_experiment_data_identifies_highest_significant_round(self):
        """Verify Bonferroni correction and identification of highest significant round."""
        df = pd.DataFrame([
            {"Rounds": 1, "Model": "MLP (Dense=2)", "Accuracy": 0.9995, "p-value": 1e-20, "AUC": 1.0},
            {"Rounds": 2, "Model": "MLP (Dense=2)", "Accuracy": 0.9200, "p-value": 1e-10, "AUC": 0.98},
            {"Rounds": 3, "Model": "1D CNN", "Accuracy": 0.6500, "p-value": 1e-05, "AUC": 0.70},
            {"Rounds": 4, "Model": "MLP (Dense=2)", "Accuracy": 0.5010, "p-value": 0.4000, "AUC": 0.501},
            {"Rounds": 5, "Model": "MLP (Dense=2)", "Accuracy": 0.4990, "p-value": 0.5000, "AUC": 0.499},
        ])

        stats = analyze_experiment_data(df, alpha=0.05)
        # 5 tests -> bonferroni alpha = 0.05 / 5 = 0.01
        assert stats["bonferroni_alpha"] == 0.01
        assert stats["has_significance"] is True
        # Highest significant round is Round 3 (p-value 1e-5 < 0.01 and Accuracy 0.65 > 0.5)
        assert stats["highest_significant_round"] == 3
        assert stats["significant_models_at_max_round"] == ["1D CNN"]
        assert stats["max_round_acc"] == 0.6500
        assert stats["min_acc"] == 0.4990
        assert stats["max_acc"] == 0.9995
        assert stats["min_auc"] == 0.4990
        assert stats["max_auc"] == 1.0

    def test_analyze_experiment_data_no_significant_round(self):
        """Verify correct report when no model is statistically significant."""
        df = pd.DataFrame([
            {"Rounds": 1, "Model": "MLP", "Accuracy": 0.5020, "p-value": 0.35, "AUC": 0.501},
            {"Rounds": 2, "Model": "MLP", "Accuracy": 0.4980, "p-value": 0.40, "AUC": 0.498},
            {"Rounds": 4, "Model": "MLP", "Accuracy": 0.5010, "p-value": 0.60, "AUC": 0.502},
        ])

        stats = analyze_experiment_data(df, alpha=0.05)
        assert stats["has_significance"] is False
        assert stats["highest_significant_round"] is None
        assert stats["significant_models_at_max_round"] == []
        assert stats["max_round_acc"] is None

    def test_generate_markdown_report_end_to_end(self):
        """Test full end-to-end report generation with mock sweep tables."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create synthetic sweep CSVs
            df_a = pd.DataFrame([
                {"Rounds": 1, "Model": "MLP (Dense=2)", "Accuracy": 0.9995, "CI": "[0.998, 1.0]", "p-value": 1e-25, "AUC": 1.0},
                {"Rounds": 2, "Model": "MLP (Dense=2)", "Accuracy": 0.9100, "CI": "[0.890, 0.930]", "p-value": 1e-12, "AUC": 0.96},
                {"Rounds": 4, "Model": "MLP (Dense=2)", "Accuracy": 0.5010, "CI": "[0.490, 0.512]", "p-value": 0.45, "AUC": 0.502},
            ])
            df_b = pd.DataFrame([
                {"Rounds": 1, "Model": "1D CNN", "Accuracy": 0.9850, "CI": "[0.975, 0.995]", "p-value": 1e-18, "AUC": 0.99},
                {"Rounds": 4, "Model": "1D CNN", "Accuracy": 0.5030, "CI": "[0.492, 0.514]", "p-value": 0.30, "AUC": 0.504},
            ])
            df_c = pd.DataFrame([
                {"Rounds": 1, "Model": "2D CNN (Dense=2)", "Accuracy": 0.9700, "CI": "[0.955, 0.985]", "p-value": 1e-15, "AUC": 0.98},
                {"Rounds": 4, "Model": "2D CNN (Dense=2)", "Accuracy": 0.4990, "CI": "[0.488, 0.510]", "p-value": 0.70, "AUC": 0.499},
            ])

            df_a.to_csv(os.path.join(tmp_dir, "table_experiment_a_sweep.csv"), index=False)
            df_b.to_csv(os.path.join(tmp_dir, "table_experiment_b_sweep.csv"), index=False)
            df_c.to_csv(os.path.join(tmp_dir, "table_experiment_c_sweep.csv"), index=False)

            report_path = os.path.join(tmp_dir, "report.md")
            report_text = generate_markdown_report(results_dir=tmp_dir, output_path=report_path)

            assert os.path.exists(report_path)

            # 1. Setup Section Checks
            assert "## Experimental Setup" in report_text
            assert "SKINNY-128-128 tweakable block cipher" in report_text
            assert "128" in report_text
            assert "Delta" in report_text
            assert "100,000 samples" in report_text
            assert "1D CNN" in report_text
            assert "2D CNN" in report_text
            assert "MLP" in report_text
            assert "Evaluation Protocol" in report_text

            # 2. Per-Experiment Tables & Figures
            assert "## Experiment A: Single-Pair Differential Distinguisher" in report_text
            assert "### Table 1: Experiment A Sweep Results" in report_text
            assert "![Experiment A: Accuracy vs Rounds](exp_a_accuracy_vs_rounds.png)" in report_text

            assert "## Experiment B: Multi-Pair Bundled Distinguisher" in report_text
            assert "### Table 2: Experiment B Sweep Results" in report_text
            assert "![Experiment B: Accuracy vs Rounds](exp_b_accuracy_vs_rounds.png)" in report_text

            assert "## Experiment C: Cipher Identification" in report_text
            assert "### Table 3: Experiment C Sweep Results" in report_text
            assert "![Experiment C: Accuracy vs Rounds](exp_c_accuracy_vs_rounds.png)" in report_text

            # 3. Data-driven Conclusion Checks
            assert "## Conclusion & Statistical Assessment" in report_text
            assert "Round 2" in report_text  # Exp A highest significant round is 2
            assert "Round 1" in report_text  # Exp B/C highest significant round is 1
            assert "0.9995" in report_text   # Max accuracy quoted
            assert "0.4990" in report_text   # Min accuracy quoted
            assert "Bonferroni" in report_text

            # 4. Strict absence of forbidden hardcoded phrases
            assert "matches published research findings" not in report_text
            assert "48.5% - 50.8%" not in report_text
            assert "~48.5%" not in report_text
            assert "HIGHT" not in report_text

            # 5. Limitations Section Checks
            assert "## Limitations" in report_text
            assert "Limited Training Budget" in report_text
            assert "Single Fixed Input Difference" in report_text
            assert "Three Simple Model Architectures" in report_text
            assert "Results Do Not Prove Security" in report_text

