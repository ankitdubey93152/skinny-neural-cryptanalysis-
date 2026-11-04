"""Unit tests for neural distinguisher model architectures and input shapers for 128-bit blocks."""

import pytest
import numpy as np
from models.architectures import (
    build_mlp,
    build_cnn_1d,
    build_cnn_2d,
    prepare_input
)


def test_build_mlp_compilation_and_forward_pass():
    """Verify MLP model compiles and outputs valid sigmoid probabilities for 256-bit and 1024-bit inputs."""
    # Exp A/C (256-bit input)
    model_ac = build_mlp(input_dim=256, n_dense_layers=3, units=640)
    assert model_ac.output_shape == (None, 1)

    dummy_input_ac = np.random.binomial(1, 0.5, size=(8, 256)).astype(np.float32)
    preds_ac = model_ac.predict(dummy_input_ac, verbose=0)
    assert preds_ac.shape == (8, 1)
    assert np.all((preds_ac >= 0.0) & (preds_ac <= 1.0))

    # Exp B (1024-bit input)
    model_b = build_mlp(input_dim=1024, n_dense_layers=3, units=640)
    assert model_b.output_shape == (None, 1)

    dummy_input_b = np.random.binomial(1, 0.5, size=(8, 1024)).astype(np.float32)
    preds_b = model_b.predict(dummy_input_b, verbose=0)
    assert preds_b.shape == (8, 1)
    assert np.all((preds_b >= 0.0) & (preds_b <= 1.0))


def test_build_mlp_layer_sweep():
    """Verify MLP can be constructed with varying dense layer counts (2 and 3)."""
    m2 = build_mlp(input_dim=256, n_dense_layers=2, units=640)
    m3 = build_mlp(input_dim=256, n_dense_layers=3, units=640)
    assert len(m2.layers) < len(m3.layers)


def test_build_cnn_1d_compilation_and_forward_pass():
    """Verify 1D CNN model compiles and performs forward pass on Exp A/C and Exp B shapes."""
    # Exp A/C: (128, 2)
    model_ac = build_cnn_1d(input_shape=(128, 2), n_dense_layers=2)
    assert model_ac.output_shape == (None, 1)

    dummy_ac = np.random.binomial(1, 0.5, size=(8, 128, 2)).astype(np.float32)
    preds_ac = model_ac.predict(dummy_ac, verbose=0)
    assert preds_ac.shape == (8, 1)
    assert np.all((preds_ac >= 0.0) & (preds_ac <= 1.0))

    # Exp B: (128, 8)
    model_b = build_cnn_1d(input_shape=(128, 8), n_dense_layers=2)
    assert model_b.output_shape == (None, 1)

    dummy_b = np.random.binomial(1, 0.5, size=(8, 128, 8)).astype(np.float32)
    preds_b = model_b.predict(dummy_b, verbose=0)
    assert preds_b.shape == (8, 1)
    assert np.all((preds_b >= 0.0) & (preds_b <= 1.0))


def test_build_cnn_2d_compilation_and_forward_pass():
    """Verify 2D CNN model compiles with 3x3 filter size on Exp A/C (2, 128, 1) and Exp B (4, 256, 1)."""
    # Exp A/C shape: (2, 128, 1)
    model_ac = build_cnn_2d(input_shape=(2, 128, 1), n_dense_layers=2, kernel_size=3)
    assert model_ac.output_shape == (None, 1)

    dummy_ac = np.random.binomial(1, 0.5, size=(8, 2, 128, 1)).astype(np.float32)
    preds_ac = model_ac.predict(dummy_ac, verbose=0)
    assert preds_ac.shape == (8, 1)
    assert np.all((preds_ac >= 0.0) & (preds_ac <= 1.0))

    # Exp B shape: (4, 256, 1)
    model_b = build_cnn_2d(input_shape=(4, 256, 1), n_dense_layers=2, kernel_size=3)
    assert model_b.output_shape == (None, 1)

    dummy_b = np.random.binomial(1, 0.5, size=(8, 4, 256, 1)).astype(np.float32)
    preds_b = model_b.predict(dummy_b, verbose=0)
    assert preds_b.shape == (8, 1)
    assert np.all((preds_b >= 0.0) & (preds_b <= 1.0))


def test_prepare_input_shapes_and_semantics():
    """Verify prepare_input produces exact expected shapes and channel/grid semantics."""
    N = 10

    # 1. Experiments A and C: input shape (N, 256)
    X_bits_ac = np.random.randint(0, 2, size=(N, 256), dtype=np.uint8)

    # MLP: (N, 256)
    out_mlp_a = prepare_input(X_bits_ac, "mlp", "a")
    assert out_mlp_a.shape == (N, 256)
    assert out_mlp_a.dtype == np.float32

    # CNN 1D: (N, 128, 2) with C1 as channel 0 and C2 as channel 1
    out_cnn1d_a = prepare_input(X_bits_ac, "cnn_1d", "a")
    assert out_cnn1d_a.shape == (N, 128, 2)
    assert np.array_equal(out_cnn1d_a[:, :, 0], X_bits_ac[:, :128].astype(np.float32))
    assert np.array_equal(out_cnn1d_a[:, :, 1], X_bits_ac[:, 128:].astype(np.float32))

    # CNN 2D: (N, 2, 128, 1)
    out_cnn2d_a = prepare_input(X_bits_ac, "cnn_2d", "a")
    assert out_cnn2d_a.shape == (N, 2, 128, 1)
    assert np.array_equal(out_cnn2d_a[:, 0, :, 0], X_bits_ac[:, :128].astype(np.float32))
    assert np.array_equal(out_cnn2d_a[:, 1, :, 0], X_bits_ac[:, 128:].astype(np.float32))

    # Experiment C produces identical shapes
    assert prepare_input(X_bits_ac, "mlp", "c").shape == (N, 256)
    assert prepare_input(X_bits_ac, "cnn_1d", "c").shape == (N, 128, 2)
    assert prepare_input(X_bits_ac, "cnn_2d", "c").shape == (N, 2, 128, 1)

    # 2. Experiment B: input shape (N, 4, 256)
    X_bits_b = np.random.randint(0, 2, size=(N, 4, 256), dtype=np.uint8)

    # MLP: (N, 1024)
    out_mlp_b = prepare_input(X_bits_b, "mlp", "b")
    assert out_mlp_b.shape == (N, 1024)

    # CNN 1D: (N, 128, 8)
    out_cnn1d_b = prepare_input(X_bits_b, "cnn_1d", "b")
    assert out_cnn1d_b.shape == (N, 128, 8)

    # CNN 2D: (N, 4, 256, 1)
    out_cnn2d_b = prepare_input(X_bits_b, "cnn_2d", "b")
    assert out_cnn2d_b.shape == (N, 4, 256, 1)


def test_prepare_input_block_bits_parameter():
    """Verify prepare_input properly honors block_bits without hardcoded lengths."""
    N = 6
    # 64-bit block cipher input (pair = 128 bits)
    X_bits_64 = np.random.randint(0, 2, size=(N, 128), dtype=np.uint8)
    assert prepare_input(X_bits_64, "mlp", "a", block_bits=64).shape == (N, 128)
    assert prepare_input(X_bits_64, "cnn_1d", "a", block_bits=64).shape == (N, 64, 2)
    assert prepare_input(X_bits_64, "cnn_2d", "a", block_bits=64).shape == (N, 2, 64, 1)

    # 64-bit multi-pair (4 pairs x 128 bits)
    X_bits_b_64 = np.random.randint(0, 2, size=(N, 4, 128), dtype=np.uint8)
    assert prepare_input(X_bits_b_64, "mlp", "b", block_bits=64).shape == (N, 512)
    assert prepare_input(X_bits_b_64, "cnn_1d", "b", block_bits=64).shape == (N, 64, 8)
    assert prepare_input(X_bits_b_64, "cnn_2d", "b", block_bits=64).shape == (N, 4, 128, 1)


def test_prepare_input_invalid_args():
    """Verify ValueError is raised on unknown model types or experiment codes."""
    X = np.zeros((4, 256), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown combination"):
        prepare_input(X, "transformer", "a")
    with pytest.raises(ValueError, match="Unknown combination"):
        prepare_input(X, "mlp", "z")
