"""Unit tests for training and evaluation pipeline, statistical metrics, checkpointing, and sanity controls."""

import os
import json
import tempfile
import numpy as np
import pytest
from scipy.stats import binomtest
from training.train_eval import (
    accuracy_significance,
    compute_full_metrics,
    get_model_configs,
    run_kfold,
    run_holdout,
    run_multiseed_evaluation,
    run_sanity_controls,
    summarize_run,
    plot_training_curves,
)
from models.architectures import build_mlp, build_cnn_1d, build_cnn_2d


def test_accuracy_significance():
    """Verify accuracy_significance calculations, Wilson CI bounds, and p-values."""
    # 1. Pure chance: 50 / 100
    acc, ci, pval = accuracy_significance(50, 100)
    assert acc == 0.5
    assert pval == 1.0
    assert 0.40 <= ci.low <= 0.41
    assert 0.59 <= ci.high <= 0.60
    assert ci[0] == ci.low and ci[1] == ci.high

    # 2. Strong signal: 950 / 1000
    acc_high, ci_high, pval_high = accuracy_significance(950, 1000)
    assert acc_high == 0.95
    assert pval_high < 1e-15
    assert ci_high.low > 0.93
    assert ci_high.high < 0.97

    # 3. Edge case: 0 samples
    acc_zero, ci_zero, pval_zero = accuracy_significance(0, 0)
    assert acc_zero == 0.0
    assert pval_zero == 1.0


def test_compute_full_metrics():
    """Verify all metrics: accuracy, precision, recall, f1, ROC-AUC, PR-AUC, confusion matrix, CI, p-value."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.8, 0.9, 0.95], dtype=float)

    metrics = compute_full_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["n_samples"] == 8
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "ci_low" in metrics and "ci_high" in metrics
    assert "p_value" in metrics
    assert "confusion_matrix" in metrics

    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 3  # 0.1, 0.2, 0.3 <= 0.5 (y=0)
    assert cm["fp"] == 1  # 0.7 > 0.5 (y=0)
    assert cm["fn"] == 1  # 0.4 <= 0.5 (y=1)
    assert cm["tp"] == 3  # 0.8, 0.9, 0.95 > 0.5 (y=1)
    assert metrics["n_correct"] == 6
    assert metrics["accuracy"] == 0.75


def test_holdout_checkpoints_and_history_saving():
    """Verify that run_holdout saves best-weights checkpoints and history JSON/CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        hist_path = os.path.join(tmpdir, "histories", "history.json")

        n_samples = 100
        n_features = 256
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=(n_samples, n_features)).astype(np.float32)
        y = rng.integers(0, 2, size=n_samples).astype(np.float32)

        res = run_holdout(
            model_builder=lambda: build_mlp(input_dim=n_features, n_dense_layers=2),
            X=X,
            y=y,
            val_split=0.2,
            test_split=0.2,
            epochs=2,
            batch_size=32,
            checkpoint_dir=ckpt_dir,
            history_path=hist_path,
            model_name="TestModel",
        )

        assert res["checkpoint_path"] is not None
        assert os.path.exists(res["checkpoint_path"])
        assert os.path.exists(hist_path)
        csv_path = os.path.splitext(hist_path)[0] + ".csv"
        assert os.path.exists(csv_path)

        with open(hist_path, "r") as f:
            hist_data = json.load(f)
        assert "loss" in hist_data and "val_loss" in hist_data


def test_all_five_architectures_train_on_proper_shapes():
    """Verify that all five canonical architectures train on their exact input shapes."""
    configs = get_model_configs(experiment="A")
    assert len(configs) == 5

    N = 64
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=N).astype(np.float32)

    # 1. 1D CNN: (N, 128, 2)
    X_1d = rng.integers(0, 2, size=(N, 128, 2)).astype(np.float32)
    res_1d = run_holdout(
        model_builder=configs["1D CNN"]["builder"],
        X=X_1d, y=y, epochs=1, batch_size=32, model_name="1D_CNN"
    )
    assert 0.0 <= res_1d["test_accuracy"] <= 1.0

    # 2. 2D CNN (dense 2): (N, 2, 128, 1)
    X_2d = rng.integers(0, 2, size=(N, 2, 128, 1)).astype(np.float32)
    res_2d2 = run_holdout(
        model_builder=configs["2D CNN (dense 2)"]["builder"],
        X=X_2d, y=y, epochs=1, batch_size=32, model_name="2D_CNN_d2"
    )
    assert 0.0 <= res_2d2["test_accuracy"] <= 1.0

    # 3. 2D CNN (dense 3): (N, 2, 128, 1)
    res_2d3 = run_holdout(
        model_builder=configs["2D CNN (dense 3)"]["builder"],
        X=X_2d, y=y, epochs=1, batch_size=32, model_name="2D_CNN_d3"
    )
    assert 0.0 <= res_2d3["test_accuracy"] <= 1.0

    # 4. MLP (dense 2): (N, 256)
    X_mlp = rng.integers(0, 2, size=(N, 256)).astype(np.float32)
    res_mlp2 = run_holdout(
        model_builder=configs["MLP (dense 2)"]["builder"],
        X=X_mlp, y=y, epochs=1, batch_size=32, model_name="MLP_d2"
    )
    assert 0.0 <= res_mlp2["test_accuracy"] <= 1.0

    # 5. MLP (dense 3): (N, 256)
    res_mlp3 = run_holdout(
        model_builder=configs["MLP (dense 3)"]["builder"],
        X=X_mlp, y=y, epochs=1, batch_size=32, model_name="MLP_d3"
    )
    assert 0.0 <= res_mlp3["test_accuracy"] <= 1.0


def test_run_multiseed_evaluation():
    """Verify multi-seed runner executes across 5 seeds and computes mean, std, and 95% CI."""
    N = 100
    rng = np.random.default_rng(42)
    X = rng.integers(0, 2, size=(N, 256)).astype(np.float32)
    y = rng.integers(0, 2, size=N).astype(np.float32)

    splits = {
        "train": (X[:60], y[:60]),
        "val": (X[60:80], y[60:80]),
        "test": (X[80:], y[80:]),
    }

    seeds = [101, 102, 103, 104, 105]
    results = run_multiseed_evaluation(
        model_builder=lambda: build_mlp(input_dim=256, n_dense_layers=2),
        splits=splits,
        model_seeds=seeds,
        epochs=1,
        batch_size=32,
        model_name="MultiSeedMLP",
    )

    assert results["n_seeds"] == 5
    assert len(results["individual_runs"]) == 5
    assert "mean_accuracy" in results
    assert "std_accuracy" in results
    assert "ci_95_mean_accuracy" in results
    ci = results["ci_95_mean_accuracy"]
    assert ci[0] <= results["mean_accuracy"] <= ci[1]


def test_sanity_controls_toy_run():
    """Verify that all three cryptographic sanity controls execute and produce valid reports."""
    report = run_sanity_controls(
        experiment="A",
        sample_size=2000,
        epochs=5,
        batch_size=64,
        seed=42,
        verbose=0,
    )

    assert "positive_control" in report
    assert "label_shuffled_control" in report
    assert "random_ciphertext_control" in report
    # Positive control on round 1 must detect differential bias
    assert report["positive_control"]["test_accuracy"] >= 0.80
    assert report["positive_control"]["passed"]
    # Shuffled labels must be consistent with chance
    assert report["label_shuffled_control"]["passed"]
    # Random bytes must be consistent with chance
    assert report["random_ciphertext_control"]["passed"]
    assert report["all_controls_passed"]
