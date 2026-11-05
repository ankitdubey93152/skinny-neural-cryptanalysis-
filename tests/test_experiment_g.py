"""Tests for Experiment G: Multi-cipher identification, block-size parity, and artifact audit."""
import numpy as np
import pytest
from data.generators import generate_experiment_g, run_artifact_audit, bits
from models.architectures import build_mlp, build_cnn_1d, build_cnn_2d
from config import CIPHERS_NATIVE_128, CIPHERS_COMMON_64


def test_experiment_g_native_128_shapes_and_balance():
    """Verify scope native-128 generates 32-byte pairs with balanced class distribution."""
    n_samples = 300
    X, y = generate_experiment_g(n_samples=n_samples, ciphers=CIPHERS_NATIVE_128, rounds=2, scope="native-128")
    assert X.shape == (300, 32)
    assert y.shape == (300,)
    for c in range(3):
        assert np.sum(y == c) == 100


def test_experiment_g_common_64_no_zero_padding():
    """Verify scope common-64 generates 16-byte pairs with no zero-padding across all 5 ciphers."""
    n_samples = 500
    for slice_half in ["low", "high"]:
        # At round 4, difference has propagated across both halves for all 5 ciphers
        X, y = generate_experiment_g(n_samples=n_samples, ciphers=CIPHERS_COMMON_64, rounds=4, scope="common-64", slice_half=slice_half)
        assert X.shape == (500, 16)
        assert y.shape == (500,)
        for c in range(5):
            assert np.sum(y == c) == 100

        # Audit: verify no bit position is constant zero or constant one
        audit = run_artifact_audit(X, y, scope="common-64")
        assert audit["passed"] is True
        assert audit["constant_bits"] == 0
        assert audit["n_bits"] == 128


def test_artifact_audit_catches_corruptions():
    """Verify artifact audit fails loudly when constant bits or identical halves are injected."""
    # 1. Injected constant bit
    clean_X = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
    clean_y = np.random.randint(0, 2, 100, dtype=np.uint8)
    corrupt_const = clean_X.copy()
    corrupt_const[:, 0] = 0x00
    with pytest.raises(ValueError, match="constant bits found"):
        run_artifact_audit(corrupt_const, clean_y, scope="native-128")

    # 2. Injected identical ciphertext halves C1 == C2
    corrupt_halves = clean_X.copy()
    corrupt_halves[:, 16:] = corrupt_halves[:, :16]
    with pytest.raises(ValueError, match="identical ciphertext halves"):
        run_artifact_audit(corrupt_halves, clean_y, scope="native-128")


def test_multiclass_model_forward_passes_and_compilation():
    """Verify multiclass model forward pass and output shapes for K=3 and K=5 classes."""
    # Multiclass 3-class model
    m_3class = build_cnn_1d(input_shape=(128, 2), num_classes=3)
    dummy_x = np.zeros((4, 128, 2), dtype=np.float32)
    pred_3 = m_3class(dummy_x)
    assert pred_3.shape == (4, 3)

    # Multiclass 5-class model
    m_5class = build_cnn_1d(input_shape=(64, 2), num_classes=5)
    dummy_x64 = np.zeros((4, 64, 2), dtype=np.float32)
    pred_5 = m_5class(dummy_x64)
    assert pred_5.shape == (4, 5)

    # Verify standard binary model compilation (num_classes=1) produces single sigmoid output
    mlp_bin = build_mlp(input_dim=256, n_dense_layers=3)
    assert mlp_bin.output_shape == (None, 1)

    cnn1d_bin = build_cnn_1d(input_shape=(128, 2), n_dense_layers=2)
    assert cnn1d_bin.output_shape == (None, 1)

    cnn2d_bin = build_cnn_2d(input_shape=(2, 128, 1), n_dense_layers=2)
    assert cnn2d_bin.output_shape == (None, 1)
