"""Unit and integration tests for experiments pipeline, CLI parsing, and positive control gate."""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEFAULT_ROUND_SWEEP, DEFAULT_DELTA
from experiments.exp_utils import (
    parse_rounds_arg,
    parse_delta_arg,
    filter_model_configs,
    plot_accuracy_vs_rounds,
    check_positive_control_gate,
    get_or_generate_dataset,
)
from experiments.run_experiment_a import run_experiment_a
from experiments.run_experiment_b import run_experiment_b
from experiments.run_experiment_c import run_experiment_c


class TestCLIAndExpUtils:
    """Test CLI parsing helpers and utility functions."""

    def test_parse_rounds_arg_defaults_and_formats(self):
        # Default when None
        assert parse_rounds_arg(None) == list(DEFAULT_ROUND_SWEEP)

        # Single integer
        assert parse_rounds_arg(4) == [4]

        # List of integers
        assert parse_rounds_arg([3, 1, 2]) == [1, 2, 3]

        # List of strings
        assert parse_rounds_arg(["1", "2", "4"]) == [1, 2, 4]

        # Comma-separated string
        assert parse_rounds_arg("1, 2, 4") == [1, 2, 4]
        assert parse_rounds_arg("1 2 4") == [1, 2, 4]

        # Empty / invalid
        with pytest.raises(ValueError):
            parse_rounds_arg([])
        with pytest.raises(TypeError):
            parse_rounds_arg(object())

    def test_parse_delta_arg(self):
        # Default
        assert parse_delta_arg(None) == DEFAULT_DELTA

        # 32 hex characters
        hex_str = "00000000000000000000000000000001"
        res = parse_delta_arg(hex_str)
        assert isinstance(res, bytes)
        assert len(res) == 16
        assert res == DEFAULT_DELTA

        # 16 raw bytes
        assert parse_delta_arg(DEFAULT_DELTA) == DEFAULT_DELTA

        # Invalid length
        with pytest.raises(ValueError, match="32 hex"):
            parse_delta_arg("0001")

        # Invalid hex
        with pytest.raises(ValueError, match="Invalid hex"):
            parse_delta_arg("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")

        # Invalid type
        with pytest.raises(TypeError):
            parse_delta_arg(12345)

    def test_filter_model_configs(self):
        configs = [
            {"name": "1D CNN", "model_type": "cnn_1d"},
            {"name": "2D CNN (Dense=2)", "model_type": "cnn_2d"},
            {"name": "2D CNN (Dense=3)", "model_type": "cnn_2d"},
            {"name": "MLP (Dense=2)", "model_type": "mlp"},
            {"name": "MLP (Dense=3)", "model_type": "mlp"},
        ]

        # None or empty returns all
        assert len(filter_model_configs(configs, None)) == 5
        assert len(filter_model_configs(configs, [])) == 5

        # Filter by name
        res = filter_model_configs(configs, ["1D CNN"])
        assert len(res) == 1
        assert res[0]["name"] == "1D CNN"

        # Filter by family
        res_mlp = filter_model_configs(configs, ["mlp"])
        assert len(res_mlp) == 2
        assert {c["name"] for c in res_mlp} == {"MLP (Dense=2)", "MLP (Dense=3)"}

        # Filter by specific dense model
        res_d2 = filter_model_configs(configs, ["MLP (Dense=2)"])
        assert len(res_d2) == 1
        assert res_d2[0]["name"] == "MLP (Dense=2)"

        # Invalid name raises ValueError
        with pytest.raises(ValueError, match="No model configs matched"):
            filter_model_configs(configs, ["transformer"])


class TestPositiveControlGate:
    """Test the positive-control gate validation rule."""

    def test_positive_control_passes_above_90_percent(self):
        sweep_data = pd.DataFrame([
            {"Rounds": 1, "Model": "MLP (Dense=2)", "Accuracy": 0.9995},
            {"Rounds": 2, "Model": "MLP (Dense=2)", "Accuracy": 0.9500},
            {"Rounds": 4, "Model": "MLP (Dense=2)", "Accuracy": 0.5010},
        ])
        # Should execute cleanly without exception
        check_positive_control_gate(sweep_data)

    def test_positive_control_fails_below_90_percent(self):
        sweep_data = pd.DataFrame([
            {"Rounds": 1, "Model": "MLP (Dense=2)", "Accuracy": 0.5200},
            {"Rounds": 2, "Model": "MLP (Dense=2)", "Accuracy": 0.5100},
            {"Rounds": 3, "Model": "MLP (Dense=2)", "Accuracy": 0.5050},
        ])
        with pytest.raises(RuntimeError, match="POSITIVE-CONTROL GATE FAILED"):
            check_positive_control_gate(sweep_data)

    def test_positive_control_skips_when_rounds_1_3_not_tested(self):
        sweep_data = pd.DataFrame([
            {"Rounds": 8, "Model": "MLP (Dense=2)", "Accuracy": 0.5000},
            {"Rounds": 10, "Model": "MLP (Dense=2)", "Accuracy": 0.5000},
        ])
        # Should not raise even if accuracy is low because rounds 1-3 were not in sweep
        check_positive_control_gate(sweep_data)


class TestPlottingAndCaching:
    """Test accuracy vs rounds plotting and dataset caching."""

    def test_plot_accuracy_vs_rounds_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sweep_data = pd.DataFrame([
                {"Rounds": 1, "Model": "1D CNN", "Accuracy": 0.99, "CI": "[0.98, 1.0]", "p-value": 0.0, "AUC": 1.0},
                {"Rounds": 2, "Model": "1D CNN", "Accuracy": 0.85, "CI": "[0.82, 0.88]", "p-value": 1e-10, "AUC": 0.92},
                {"Rounds": 4, "Model": "1D CNN", "Accuracy": 0.50, "CI": "[0.48, 0.52]", "p-value": 0.8, "AUC": 0.51},
                {"Rounds": 1, "Model": "MLP", "Accuracy": 1.00, "CI": "[0.99, 1.0]", "p-value": 0.0, "AUC": 1.0},
                {"Rounds": 2, "Model": "MLP", "Accuracy": 0.92, "CI": "[0.89, 0.94]", "p-value": 1e-15, "AUC": 0.97},
                {"Rounds": 4, "Model": "MLP", "Accuracy": 0.51, "CI": "[0.49, 0.53]", "p-value": 0.5, "AUC": 0.52},
            ])
            plot_path = os.path.join(tmp_dir, "exp_a_accuracy_vs_rounds.png")
            plot_accuracy_vs_rounds(sweep_data, save_path=plot_path, exp_name="A")
            assert os.path.exists(plot_path)
            assert os.path.getsize(plot_path) > 1000

    def test_get_or_generate_dataset_caches_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from data.generators import generate_experiment_a
            X1, y1 = get_or_generate_dataset(
                exp_id="a",
                rounds=1,
                n_samples=50,
                delta_bytes=DEFAULT_DELTA,
                data_dir=tmp_dir,
                generator_fn=generate_experiment_a,
            )
            assert len(y1) == 50
            cache_file = os.path.join(tmp_dir, "exp_a_r1_n50.npz")
            assert os.path.exists(cache_file)

            # Second call should load from cache without re-running generator
            X2, y2 = get_or_generate_dataset(
                exp_id="a",
                rounds=1,
                n_samples=50,
                delta_bytes=DEFAULT_DELTA,
                data_dir=tmp_dir,
                generator_fn=None,  # generator_fn is None, proves it loaded from cache!
            )
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)


class TestExperimentsIntegration:
    """Fast integration test of experiment scripts using small synthetic/quick configurations."""

    def test_run_experiment_a_round_sweep(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = os.path.join(tmp_dir, "results")
            data_dir = os.path.join(tmp_dir, "data")

            # Run 50 samples, round 4, 1 epoch, single fast model
            sweep_df = run_experiment_a(
                rounds=[4],
                n_samples=50,
                eval_mode="holdout",
                epochs=1,
                models=["MLP (Dense=2)"],
                results_dir=results_dir,
                data_dir=data_dir,
            )

            # Check return DataFrame
            assert isinstance(sweep_df, pd.DataFrame)
            expected_cols = ["Rounds", "Model", "Accuracy", "CI", "p-value", "AUC"]
            assert list(sweep_df.columns) == expected_cols
            assert len(sweep_df) == 1
            assert sweep_df.iloc[0]["Rounds"] == 4
            assert sweep_df.iloc[0]["Model"] == "MLP (Dense=2)"

            # Check generated files
            assert os.path.exists(os.path.join(results_dir, "table_experiment_a_r4.csv"))
            assert os.path.exists(os.path.join(results_dir, "table_experiment_a_sweep.csv"))
            assert os.path.exists(os.path.join(results_dir, "table_experiment_a.csv"))
            assert os.path.exists(os.path.join(results_dir, "exp_a_accuracy_vs_rounds.png"))

    def test_run_experiment_b_round_sweep(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = os.path.join(tmp_dir, "results")
            data_dir = os.path.join(tmp_dir, "data")

            # Run 50 samples, round 1, 1 epoch, single fast model
            sweep_df = run_experiment_b(
                rounds=[1],
                n_samples=50,
                pairs_per_sample=4,
                eval_mode="holdout",
                epochs=1,
                models=["MLP (Dense=2)"],
                results_dir=results_dir,
                data_dir=data_dir,
            )

            expected_cols = ["Rounds", "Model", "Accuracy", "CI", "p-value", "AUC"]
            assert list(sweep_df.columns) == expected_cols
            assert len(sweep_df) == 1
            assert os.path.exists(os.path.join(results_dir, "table_experiment_b_r1.csv"))
            assert os.path.exists(os.path.join(results_dir, "table_experiment_b_sweep.csv"))
            assert os.path.exists(os.path.join(results_dir, "exp_b_accuracy_vs_rounds.png"))

    def test_run_experiment_c_round_sweep(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = os.path.join(tmp_dir, "results")
            data_dir = os.path.join(tmp_dir, "data")

            # Run 50 samples, round 1, 1 epoch, single fast model
            sweep_df = run_experiment_c(
                rounds=[1],
                n_samples=50,
                eval_mode="holdout",
                epochs=1,
                models=["MLP (Dense=2)"],
                results_dir=results_dir,
                data_dir=data_dir,
            )

            expected_cols = ["Rounds", "Model", "Accuracy", "CI", "p-value", "AUC"]
            assert list(sweep_df.columns) == expected_cols
            assert len(sweep_df) == 1
            assert os.path.exists(os.path.join(results_dir, "table_experiment_c_r1.csv"))
            assert os.path.exists(os.path.join(results_dir, "table_experiment_c_sweep.csv"))
            assert os.path.exists(os.path.join(results_dir, "exp_c_accuracy_vs_rounds.png"))
