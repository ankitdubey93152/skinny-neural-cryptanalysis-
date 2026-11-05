"""Unit tests, official Known-Answer Tests, and 1000-sample scalar cross-check for HIGHT."""
import numpy as np
import pytest
from ciphers.hight import hight_encrypt, hight_encrypt_batch, FULL_ROUNDS
from tests.hight_scalar_ref import hight_encrypt_scalar


def test_hight_official_kat1():
    """Verify HIGHT against official test vector 1 (draft-kisa-hight-00 / ISO/IEC 18033-3).

    Key:       00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
    Plaintext: 00 00 00 00 00 00 00 00
    Expected:  00 f4 18 ae d9 4f 03 f2
    """
    key = bytes.fromhex("00112233445566778899aabbccddeeff")
    pt = bytes.fromhex("0000000000000000")
    expected_ct = bytes.fromhex("00f418aed94f03f2")

    ct_scalar = hight_encrypt_scalar(pt, key, rounds=32)
    assert ct_scalar == expected_ct

    ct_single = hight_encrypt(pt, key, rounds=32)
    assert ct_single == expected_ct

    ct_batch = hight_encrypt_batch(
        np.frombuffer(pt, dtype=np.uint8).reshape(1, 8),
        np.frombuffer(key, dtype=np.uint8).reshape(1, 16),
        rounds=32,
    )
    assert bytes(ct_batch[0]) == expected_ct


def test_hight_official_kat2():
    """Verify HIGHT against official test vector 2 (draft-kisa-hight-00 / ISO/IEC 18033-3).

    Key:       ff ee dd cc bb aa 99 88 77 66 55 44 33 22 11 00
    Plaintext: 00 11 22 33 44 55 66 77
    Expected:  23 ce 9f 72 e5 43 e6 d8
    """
    key = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    pt = bytes.fromhex("0011223344556677")
    expected_ct = bytes.fromhex("23ce9f72e543e6d8")

    ct_scalar = hight_encrypt_scalar(pt, key, rounds=32)
    assert ct_scalar == expected_ct

    ct_single = hight_encrypt(pt, key, rounds=32)
    assert ct_single == expected_ct

    ct_batch = hight_encrypt_batch(
        np.frombuffer(pt, dtype=np.uint8).reshape(1, 8),
        np.frombuffer(key, dtype=np.uint8).reshape(1, 16),
        rounds=32,
    )
    assert bytes(ct_batch[0]) == expected_ct


def test_hight_scalar_cross_check_1000_triples():
    """Independent cross-check comparing vectorized HIGHT with pure scalar implementation on 1000 triples."""
    rng = np.random.default_rng(20261105)
    n_triples = 1000

    pts = rng.integers(0, 256, (n_triples, 8), dtype=np.uint8)
    keys = rng.integers(0, 256, (n_triples, 16), dtype=np.uint8)
    rounds_arr = rng.integers(0, 33, size=n_triples)

    for r in range(33):
        mask = (rounds_arr == r)
        if not np.any(mask):
            continue
        sub_pts = pts[mask]
        sub_keys = keys[mask]
        cts_vec = hight_encrypt_batch(sub_pts, sub_keys, rounds=r)

        for i in range(len(sub_pts)):
            pt_b = bytes(sub_pts[i])
            k_b = bytes(sub_keys[i])
            expected = hight_encrypt_scalar(pt_b, k_b, rounds=r)
            assert bytes(cts_vec[i]) == expected, f"HIGHT mismatch at round {r}"


def test_hight_broadcasting_and_shapes():
    """Verify key broadcasting and shape checks for HIGHT."""
    key = bytes.fromhex("00112233445566778899aabbccddeeff")
    pt = bytes.fromhex("0000000000000000")
    expected_ct = bytes.fromhex("00f418aed94f03f2")

    pts = np.tile(np.frombuffer(pt, dtype=np.uint8), (5, 1))
    cts = hight_encrypt_batch(pts, key, rounds=32)
    assert cts.shape == (5, 8)
    for row in cts:
        assert bytes(row) == expected_ct

    with pytest.raises(ValueError):
        hight_encrypt_batch(pts, key, rounds=33)
