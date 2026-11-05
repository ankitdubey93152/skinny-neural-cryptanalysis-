"""Unit tests, official FIPS 46-3 Known-Answer Test, and 1000-sample scalar cross-check for DES."""
import numpy as np
import pytest
from Crypto.Cipher import DES as CryptoDES
from ciphers.des import des_encrypt, des_encrypt_batch, FULL_ROUNDS
from tests.des_scalar_ref import des_encrypt_scalar


def test_des_official_fips46_3_kat():
    """Verify DES against official FIPS 46-3 Known-Answer Test vector.

    Key:       13 34 57 79 9B BC DF F1
    Plaintext: 01 23 45 67 89 AB CD EF
    Expected:  85 E8 13 54 0F 0A B4 05
    """
    key = bytes.fromhex("133457799BBCDFF1")
    pt = bytes.fromhex("0123456789ABCDEF")
    expected_ct = bytes.fromhex("85E813540F0AB405")

    ct_scalar = des_encrypt_scalar(pt, key, rounds=16)
    assert ct_scalar == expected_ct

    ct_single = des_encrypt(pt, key, rounds=16)
    assert ct_single == expected_ct

    ct_batch = des_encrypt_batch(
        np.frombuffer(pt, dtype=np.uint8).reshape(1, 8),
        np.frombuffer(key, dtype=np.uint8).reshape(1, 8),
        rounds=16,
    )
    assert bytes(ct_batch[0]) == expected_ct


def test_des_pycryptodome_cross_check_100_blocks():
    """Cross-check scalar and vectorized DES against PyCryptodome DES across 100 random blocks."""
    rng = np.random.default_rng(20261105)
    for _ in range(100):
        k = rng.bytes(8)
        p = rng.bytes(8)
        expected = CryptoDES.new(k, CryptoDES.MODE_ECB).encrypt(p)
        assert des_encrypt_scalar(p, k, rounds=16) == expected
        assert des_encrypt(p, k, rounds=16) == expected


def test_des_scalar_cross_check_1000_triples():
    """Independent cross-check comparing vectorized DES with pure scalar implementation on 1000 triples."""
    rng = np.random.default_rng(20261105)
    n_triples = 1000

    pts = rng.integers(0, 256, (n_triples, 8), dtype=np.uint8)
    keys = rng.integers(0, 256, (n_triples, 8), dtype=np.uint8)
    rounds_arr = rng.integers(0, 17, size=n_triples)

    for r in range(17):
        mask = (rounds_arr == r)
        if not np.any(mask):
            continue
        sub_pts = pts[mask]
        sub_keys = keys[mask]
        cts_vec = des_encrypt_batch(sub_pts, sub_keys, rounds=r)

        for i in range(len(sub_pts)):
            pt_b = bytes(sub_pts[i])
            k_b = bytes(sub_keys[i])
            expected = des_encrypt_scalar(pt_b, k_b, rounds=r)
            assert bytes(cts_vec[i]) == expected, f"DES mismatch at round {r}"


def test_des_broadcasting_and_shapes():
    """Verify key broadcasting and shape checks for DES."""
    key = bytes.fromhex("133457799BBCDFF1")
    pt = bytes.fromhex("0123456789ABCDEF")
    expected_ct = bytes.fromhex("85E813540F0AB405")

    pts = np.tile(np.frombuffer(pt, dtype=np.uint8), (5, 1))
    cts = des_encrypt_batch(pts, key, rounds=16)
    assert cts.shape == (5, 8)
    for row in cts:
        assert bytes(row) == expected_ct

    with pytest.raises(ValueError):
        des_encrypt_batch(pts, key, rounds=17)
