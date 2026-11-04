"""Unit tests for SKINNY-128-128 dataset generators, leakage safety, bit expansion, and metadata."""

import os
import tempfile
import numpy as np
import pytest

from config import DEFAULT_DELTA, DEFAULT_ROUNDS
from ciphers.skinny128 import skinny128_encrypt_batch
from ciphers.aes_ref import aes128_encrypt_batch
from data.generators import (
    DELTA,
    bits,
    reshape_bits_for_model,
    get_dataset_filename,
    generate_experiment_a,
    generate_experiment_b,
    generate_experiment_c,
    generate_split_dataset,
    check_dataset_leakage,
    save_dataset,
    load_dataset,
    save_dataset_bundle,
    load_dataset_bundle,
)


def test_delta_constant():
    """Verify DEFAULT_DELTA is exactly 16 bytes with LSB set."""
    assert len(DEFAULT_DELTA) == 16
    assert DEFAULT_DELTA == bytes.fromhex("00000000000000000000000000000001")
    assert DELTA == DEFAULT_DELTA


def test_bits_expansion():
    """Verify byte array expansion to 0/1 bits."""
    # 0x01 = 00000001, 0x80 = 10000000
    arr = np.array([[0x01, 0x80]], dtype=np.uint8)
    b = bits(arr)
    assert b.shape == (1, 16)
    expected = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    assert np.array_equal(b, expected)


def test_reshape_bits_for_models():
    """Verify bit tensor shapes for all 5 model configurations across Experiments A/C and B."""
    N = 10

    # Exp A and C: 32 bytes -> 256 bits
    raw_a = np.zeros((N, 32), dtype=np.uint8)
    bits_a = bits(raw_a)  # (N, 256)
    assert bits_a.shape == (N, 256)

    mlp_a = reshape_bits_for_model(bits_a, "mlp", experiment="A")
    assert mlp_a.shape == (N, 256)

    cnn1d_a = reshape_bits_for_model(bits_a, "1d_cnn", experiment="A")
    assert cnn1d_a.shape == (N, 128, 2)

    cnn2d_a = reshape_bits_for_model(bits_a, "2d_cnn", experiment="A")
    assert cnn2d_a.shape == (N, 2, 128, 1)

    # Exp B: 4 pairs * 32 bytes -> (N, 4, 256) bits
    raw_b = np.zeros((N, 4, 32), dtype=np.uint8)
    bits_b = bits(raw_b)  # (N, 4, 256)
    assert bits_b.shape == (N, 4, 256)

    mlp_b = reshape_bits_for_model(bits_b, "mlp", experiment="B")
    assert mlp_b.shape == (N, 1024)

    cnn1d_b = reshape_bits_for_model(bits_b, "1d_cnn", experiment="B")
    assert cnn1d_b.shape == (N, 128, 8)

    cnn2d_b = reshape_bits_for_model(bits_b, "2d_cnn", experiment="B")
    assert cnn2d_b.shape == (N, 4, 256, 1)


def test_get_dataset_filename():
    """Verify standardized dataset filename formatting."""
    fn = get_dataset_filename("a", rounds=5, n_samples=100000, data_dir="data")
    assert os.path.normpath(fn) == os.path.normpath("data/exp_a_r5_n100000.npz")

    fn_exp = get_dataset_filename("exp_b", rounds=40, n_samples=20000, data_dir="custom_dir")
    assert os.path.normpath(fn_exp) == os.path.normpath("custom_dir/exp_b_r40_n20000.npz")


def test_rounds_validation():
    """Verify ValueError is raised when rounds is outside [1, 40]."""
    for bad_round in [0, 41, -1, 100]:
        with pytest.raises(ValueError, match="rounds must be in 1..40"):
            generate_experiment_a(n_samples=10, rounds=bad_round)
        with pytest.raises(ValueError, match="rounds must be in 1..40"):
            generate_experiment_b(n_samples=10, rounds=bad_round)
        with pytest.raises(ValueError, match="rounds must be in 1..40"):
            generate_experiment_c(n_samples=10, rounds=bad_round)


def test_generate_experiment_a_shape_dtype_balance_determinism():
    """Verify Experiment A generator shape, dtype, 50/50 balance, and seed determinism."""
    n_samples = 100
    rounds = 16
    seed = 42

    X1, y1 = generate_experiment_a(n_samples=n_samples, rounds=rounds, seed=seed)
    assert X1.shape == (n_samples, 32)
    assert X1.dtype == np.uint8
    assert y1.shape == (n_samples,)
    assert y1.dtype == np.uint8
    assert np.sum(y1 == 1) == 50
    assert np.sum(y1 == 0) == 50

    # Determinism check with same seed
    X2, y2 = generate_experiment_a(n_samples=n_samples, rounds=rounds, seed=seed)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)

    # Different seed should yield different data
    X3, y3 = generate_experiment_a(n_samples=n_samples, rounds=rounds, seed=999)
    assert not np.array_equal(X1, X3)


def test_generate_experiment_a_label_correctness_and_negative_modes():
    """Verify Experiment A label correctness for both 'delta_prime' and 'random_p2' modes."""
    n_samples = 100
    rounds = 4
    seed = 123
    delta = DEFAULT_DELTA
    delta_arr = np.frombuffer(delta, dtype=np.uint8)

    # 1. Mode: delta_prime
    X_dp, y_dp, meta_dp = generate_experiment_a(
        n_samples=n_samples, rounds=rounds, delta=delta, seed=seed, negative_mode="delta_prime", return_meta=True
    )
    p1 = meta_dp.p1
    p2 = meta_dp.p2
    diffs = p1 ^ p2

    y1_mask = (y_dp == 1)
    y0_mask = (y_dp == 0)

    # Label 1: exactly equals delta
    assert np.all(diffs[y1_mask] == delta_arr)
    # Label 0: non-zero and not equal to delta
    assert not np.any(np.all(diffs[y0_mask] == delta_arr, axis=-1))
    assert not np.any(np.all(diffs[y0_mask] == 0, axis=-1))

    # Ciphertexts match cipher encryption
    c1_exp = skinny128_encrypt_batch(p1, meta_dp.keys, rounds=rounds)
    c2_exp = skinny128_encrypt_batch(p2, meta_dp.keys, rounds=rounds)
    assert np.array_equal(X_dp[:, :16], c1_exp)
    assert np.array_equal(X_dp[:, 16:], c2_exp)

    # 2. Mode: random_p2
    X_rp, y_rp, meta_rp = generate_experiment_a(
        n_samples=n_samples, rounds=rounds, delta=delta, seed=seed, negative_mode="random_p2", return_meta=True
    )
    p1_rp = meta_rp.p1
    p2_rp = meta_rp.p2
    diffs_rp = p1_rp ^ p2_rp
    y1_m = (y_rp == 1)
    y0_m = (y_rp == 0)

    assert np.all(diffs_rp[y1_m] == delta_arr)
    # With random p2, probability of hitting delta is 2^-128
    assert not np.any(np.all(diffs_rp[y0_m] == delta_arr, axis=-1))


def test_generate_experiment_b_tweakey_strategies_and_labels():
    """Verify Experiment B shapes, shared vs independent tweakey strategies, and label correctness."""
    n_samples = 60
    pairs_per_sample = 4
    rounds = 4
    seed = 88
    delta = DEFAULT_DELTA
    delta_arr = np.frombuffer(delta, dtype=np.uint8)

    # Strategy 1: shared tweakey (default)
    X_sh, y_sh, meta_sh = generate_experiment_b(
        n_samples=n_samples, pairs_per_sample=pairs_per_sample, rounds=rounds,
        delta=delta, seed=seed, tweakey_strategy="shared", return_meta=True
    )
    assert X_sh.shape == (n_samples, pairs_per_sample, 32)
    assert meta_sh.keys.shape == (n_samples, 16)
    assert meta_sh.keys_expanded.shape == (n_samples, pairs_per_sample, 16)
    # In shared strategy, all 4 pairs in sample have the same key
    for p in range(pairs_per_sample):
        assert np.array_equal(meta_sh.keys_expanded[:, p, :], meta_sh.keys)

    # Check label 1 diffs
    y1_mask = (y_sh == 1)
    diffs_y1 = meta_sh.p1[y1_mask] ^ meta_sh.p2[y1_mask]
    assert np.all(diffs_y1 == delta_arr)

    # Check label 0 diffs
    y0_mask = (y_sh == 0)
    diffs_y0 = meta_sh.p1[y0_mask] ^ meta_sh.p2[y0_mask]
    assert not np.any(np.all(diffs_y0 == delta_arr, axis=-1))
    assert not np.any(np.all(diffs_y0 == 0, axis=-1))

    # Strategy 2: independent tweakeys per pair
    X_ind, y_ind, meta_ind = generate_experiment_b(
        n_samples=n_samples, pairs_per_sample=pairs_per_sample, rounds=rounds,
        delta=delta, seed=seed, tweakey_strategy="independent", return_meta=True
    )
    assert meta_ind.keys.shape == (n_samples, pairs_per_sample, 16)
    # Verify pairs in a sample have distinct keys
    assert not np.array_equal(meta_ind.keys[:, 0, :], meta_ind.keys[:, 1, :])


def test_generate_experiment_c_cipher_identification():
    """Verify Experiment C: same delta for both classes, SKINNY-128-128 vs AES-128 ciphertexts."""
    n_samples = 60
    rounds = 10
    seed = 66
    delta = DEFAULT_DELTA
    delta_arr = np.frombuffer(delta, dtype=np.uint8)

    X, y, meta = generate_experiment_c(
        n_samples=n_samples, rounds=rounds, delta=delta, reference="aes128", seed=seed, return_meta=True
    )
    assert X.shape == (n_samples, 32)
    assert y.shape == (n_samples,)
    assert np.sum(y == 1) == 30
    assert np.sum(y == 0) == 30

    p1 = meta.p1
    p2 = meta.p2
    keys = meta.keys

    # Both classes must satisfy P1 ^ P2 == delta
    assert np.all((p1 ^ p2) == delta_arr)

    # Label 1: SKINNY-128-128
    skinny_idx = np.where(y == 1)[0]
    c1_skinny = skinny128_encrypt_batch(p1[skinny_idx], keys[skinny_idx], rounds=rounds)
    c2_skinny = skinny128_encrypt_batch(p2[skinny_idx], keys[skinny_idx], rounds=rounds)
    assert np.array_equal(X[skinny_idx, :16], c1_skinny)
    assert np.array_equal(X[skinny_idx, 16:], c2_skinny)

    # Label 0: AES-128
    ref_idx = np.where(y == 0)[0]
    c1_ref = aes128_encrypt_batch(p1[ref_idx], keys[ref_idx])
    c2_ref = aes128_encrypt_batch(p2[ref_idx], keys[ref_idx])
    assert np.array_equal(X[ref_idx, :16], c1_ref)
    assert np.array_equal(X[ref_idx, 16:], c2_ref)


def test_chunked_generation_consistency():
    """Verify that chunked generation is deterministic and preserves exact 50/50 balance and shapes."""
    n_samples = 200
    chunk_size = 50
    rounds = 4
    seed = 31415

    X1, y1 = generate_experiment_a(
        n_samples=n_samples, rounds=rounds, seed=seed, chunk_size=chunk_size
    )
    X2, y2 = generate_experiment_a(
        n_samples=n_samples, rounds=rounds, seed=seed, chunk_size=chunk_size
    )

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert X1.shape == (n_samples, 32)
    assert np.sum(y1 == 1) == n_samples // 2
    assert np.sum(y1 == 0) == n_samples // 2


def test_generate_split_dataset_and_leakage_safety():
    """Verify split generation using SeedSequence.spawn and cross-split leakage detection."""
    splits, meta, report = generate_split_dataset(
        experiment="a",
        rounds=4,
        n_train=200,
        n_val=50,
        n_test=50,
        seed=12345,
        run_leakage_check=True,
        return_meta=True,
    )

    assert "train" in splits and "val" in splits and "test" in splits
    X_tr, y_tr, meta_tr = splits["train"]
    X_va, y_va, meta_va = splits["val"]
    X_te, y_te, meta_te = splits["test"]

    assert len(y_tr) == 200
    assert len(y_va) == 50
    assert len(y_te) == 50

    # Leakage report must be completely clean
    assert not report["tweakey_leakage"]
    assert not report["plaintext_leakage"]
    assert not report["ciphertext_leakage"]
    assert report["shared_tweakeys"]["train_val"] == 0
    assert report["shared_tweakeys"]["train_test"] == 0
    assert report["shared_tweakeys"]["val_test"] == 0

    # Metadata must have all required fields
    assert meta["cipher"] == "SKINNY-128-128"
    assert meta["variant"] == "128/128-TK1"
    assert meta["rounds"] == 4
    assert meta["split_method"] == "independent_rng_streams_seedsequence_spawn"
    assert "seeds" in meta
    assert meta["sizes"]["train"] == 200


def test_leakage_check_raises_on_leak():
    """Verify that check_dataset_leakage catches and raises ValueError on synthetic leakage."""
    keys = np.random.randint(0, 256, size=(10, 16), dtype=np.uint8)
    p1 = np.random.randint(0, 256, size=(10, 16), dtype=np.uint8)
    p2 = np.random.randint(0, 256, size=(10, 16), dtype=np.uint8)

    # Make train and val share a key
    train_meta = {"keys": keys.copy(), "p1": p1.copy(), "p2": p2.copy()}
    val_meta = {"keys": keys.copy(), "p1": p1.copy(), "p2": p2.copy()}
    test_meta = {
        "keys": np.random.randint(0, 256, size=(10, 16), dtype=np.uint8),
        "p1": np.random.randint(0, 256, size=(10, 16), dtype=np.uint8),
        "p2": np.random.randint(0, 256, size=(10, 16), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match="Cross-split tweakey leakage detected"):
        check_dataset_leakage(train_meta, val_meta, test_meta, raise_on_leakage=True)


def test_dataset_bundle_save_load():
    """Verify dataset bundle save/load roundtrip with strict cipher validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = os.path.join(tmpdir, "bundle.npz")

        splits, meta, _ = generate_split_dataset(
            experiment="a", rounds=5, n_train=40, n_val=20, n_test=20, seed=42, return_meta=False
        )
        save_dataset_bundle(bundle_path, splits, meta)
        assert os.path.exists(bundle_path)

        loaded_splits, loaded_meta = load_dataset_bundle(bundle_path, expected_cipher="SKINNY-128-128")
        assert np.array_equal(loaded_splits["train"][0], splits["train"][0])
        assert np.array_equal(loaded_splits["train"][1], splits["train"][1])
        assert loaded_meta["cipher"] == "SKINNY-128-128"
        assert loaded_meta["rounds"] == 5

        # Must reject incorrect cipher expectation
        with pytest.raises(ValueError, match="Dataset cipher mismatch"):
            load_dataset_bundle(bundle_path, expected_cipher="AES-128")
