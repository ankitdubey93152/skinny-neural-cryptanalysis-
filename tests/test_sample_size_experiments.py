"""Tests for small-sample experiments D, E, F: split balance, leakage safety, and power analysis."""
import numpy as np
import pytest
from data.generators import generate_split_dataset
from experiments.run_experiment_sample_size import compute_power_analysis, holm_bonferroni


def test_sample_size_splits_and_leakage():
    """Verify stratified balance and zero leakage across splits at N=50, 500, 5000."""
    for n_tot, (n_tr, n_va, n_te) in [(50, (30, 10, 10)), (500, (300, 100, 100)), (5000, (3000, 1000, 1000))]:
        splits, meta, leakage = generate_split_dataset(
            "a", rounds=1, n_train=n_tr, n_val=n_va, n_test=n_te,
            seed=42, run_leakage_check=True, return_meta=True
        )
        assert len(splits["train"][1]) == n_tr
        assert len(splits["val"][1]) == n_va
        assert len(splits["test"][1]) == n_te

        # Check label balance (50% ones, 50% zeros in each split)
        assert np.sum(splits["train"][1] == 1) == n_tr // 2
        assert np.sum(splits["val"][1] == 1) == n_va // 2
        assert np.sum(splits["test"][1] == 1) == n_te // 2

        # Check zero leakage
        assert not leakage["tweakey_leakage"]
        assert not leakage["plaintext_leakage"]
        assert not leakage["ciphertext_leakage"]


def test_power_analysis_calculation():
    """Verify analytical power analysis formulas against theoretical limits."""
    # At N=50, n_test=10:
    pa = compute_power_analysis(50, 10)
    assert pa["total_pairs_N"] == 50
    assert pa["n_test_single"] == 10
    assert pa["min_acc_sig_p05"] > 0.80  # requires >80% accuracy for p<0.05 on 10 samples
    assert pa["min_detectable_acc_80power"] > pa["min_acc_sig_p05"]

    # At pooled N=300 (30 replicates of 10):
    pa_pooled = compute_power_analysis(50, 300)
    assert pa_pooled["min_acc_sig_p05"] < 0.60  # pooled requires much lower threshold ~56%


def test_holm_bonferroni_properties():
    """Verify Holm-Bonferroni correction maintains monotonicity and bounds."""
    raw = [0.001, 0.01, 0.03, 0.04, 0.5]
    adj = holm_bonferroni(raw)
    assert len(adj) == len(raw)
    assert np.all(adj >= raw)
    assert np.all(adj <= 1.0)
    # Order preservation
    assert adj[0] <= adj[1] <= adj[2] <= adj[3] <= adj[4]
