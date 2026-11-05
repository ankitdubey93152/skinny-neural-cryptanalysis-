"""Unit tests, official Known-Answer Test, and 1000-sample scalar cross-check for SM4."""
import numpy as np
import pytest
from ciphers.sm4 import sm4_encrypt, sm4_encrypt_batch, FULL_ROUNDS
from tests.sm4_scalar_ref import sm4_encrypt_scalar


def test_sm4_official_kat():
    """Verify SM4 against the official standard test vector in GB/T 32907-2016.

    Standard: GB/T 32907-2016, Appendix A.1
    Key:       01 23 45 67 89 ab cd ef fe dc ba 98 76 54 32 10
    Plaintext: 01 23 45 67 89 ab cd ef fe dc ba 98 76 54 32 10
    Expected:  68 1e df 34 d2 06 96 5e 86 b3 e9 4f 53 6e 42 46
    """
    key = bytes.fromhex("0123456789abcdeffedcba9876543210")
    pt = bytes.fromhex("0123456789abcdeffedcba9876543210")
    expected_ct = bytes.fromhex("681edf34d206965e86b3e94f536e4246")

    # Scalar check
    ct_scalar = sm4_encrypt_scalar(pt, key, rounds=32)
    assert ct_scalar == expected_ct

    # Single-block API
    ct_single = sm4_encrypt(pt, key, rounds=32)
    assert ct_single == expected_ct

    # Vectorized batch API
    ct_batch = sm4_encrypt_batch(np.frombuffer(pt, dtype=np.uint8).reshape(1, 16),
                                 np.frombuffer(key, dtype=np.uint8).reshape(1, 16),
                                 rounds=32)
    assert bytes(ct_batch[0]) == expected_ct


def test_sm4_scalar_cross_check_1000_triples():
    """Independent cross-check comparing vectorized SM4 with pure scalar implementation on 1000 triples."""
    rng = np.random.default_rng(20261105)
    n_triples = 1000

    pts = rng.integers(0, 256, (n_triples, 16), dtype=np.uint8)
    keys = rng.integers(0, 256, (n_triples, 16), dtype=np.uint8)
    rounds_arr = rng.integers(0, 33, size=n_triples)

    # Batch test per round group for vectorization efficiency
    for r in range(33):
        mask = (rounds_arr == r)
        if not np.any(mask):
            continue
        sub_pts = pts[mask]
        sub_keys = keys[mask]
        cts_vec = sm4_encrypt_batch(sub_pts, sub_keys, rounds=r)

        for i in range(len(sub_pts)):
            pt_b = bytes(sub_pts[i])
            k_b = bytes(sub_keys[i])
            expected = sm4_encrypt_scalar(pt_b, k_b, rounds=r)
            assert bytes(cts_vec[i]) == expected, f"SM4 mismatch at round {r}"


def test_sm4_broadcasting_and_shapes():
    """Verify key broadcasting and shape checks for SM4."""
    key = bytes.fromhex("0123456789abcdeffedcba9876543210")
    pt = bytes.fromhex("0123456789abcdeffedcba9876543210")
    expected_ct = bytes.fromhex("681edf34d206965e86b3e94f536e4246")

    pts = np.tile(np.frombuffer(pt, dtype=np.uint8), (5, 1))
    cts = sm4_encrypt_batch(pts, key, rounds=32)
    assert cts.shape == (5, 16)
    for row in cts:
        assert bytes(row) == expected_ct

    with pytest.raises(ValueError):
        sm4_encrypt_batch(pts, key, rounds=33)
