"""Unit tests and FIPS-197 Known-Answer Test for AES-128 baseline reference."""

import numpy as np
import pytest
from ciphers.aes_ref import aes128_encrypt_batch


def test_aes128_fips197_test_vector():
    """Verify AES-128 implementation against the official NIST FIPS-197 standard test vector.

    Standard: NIST FIPS-197, Appendix C.1 (AES-128 Example)
    Key:       2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c
    Plaintext: 32 43 f6 a8 88 5a 30 8d 31 31 98 a2 e0 37 07 34
    Expected:  39 25 84 1d 02 dc 09 fb dc 11 85 97 19 6a 0b 32
    """
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    pt = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    expected_ct = bytes.fromhex("3925841d02dc09fbdc118597196a0b32")

    pt_arr = np.frombuffer(pt, dtype=np.uint8).reshape((1, 16))
    key_arr = np.frombuffer(key, dtype=np.uint8).reshape((1, 16))

    ct = aes128_encrypt_batch(pt_arr, key_arr)[0]
    assert bytes(ct) == expected_ct, f"FIPS-197 vector failed: got {bytes(ct).hex()}, expected {expected_ct.hex()}"


def test_aes128_batch_and_broadcasting():
    """Verify AES-128 batch encryption and key broadcasting across multiple blocks."""
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    pt = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    expected_ct = bytes.fromhex("3925841d02dc09fbdc118597196a0b32")

    n_samples = 8
    pts = np.tile(np.frombuffer(pt, dtype=np.uint8), (n_samples, 1))
    key_single = np.frombuffer(key, dtype=np.uint8)

    # 1. 1D key broadcasting
    cts_1d = aes128_encrypt_batch(pts, key_single)
    assert cts_1d.shape == (n_samples, 16)
    for i in range(n_samples):
        assert bytes(cts_1d[i]) == expected_ct

    # 2. 2D single key broadcasting
    cts_2d = aes128_encrypt_batch(pts, key_single.reshape(1, 16))
    assert np.array_equal(cts_1d, cts_2d)


def test_aes128_dimension_validation():
    """Verify ValueError is raised on invalid array dimensions."""
    pts_invalid = np.zeros((4, 15), dtype=np.uint8)
    key_valid = np.zeros(16, dtype=np.uint8)
    with pytest.raises(ValueError):
        aes128_encrypt_batch(pts_invalid, key_valid)

    pts_valid = np.zeros((4, 16), dtype=np.uint8)
    key_invalid = np.zeros(15, dtype=np.uint8)
    with pytest.raises(ValueError):
        aes128_encrypt_batch(pts_valid, key_invalid)

