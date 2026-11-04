"""Comprehensive test suite for SKINNY-128-128 cryptographic implementation.

Verifies:
1. Official Known-Answer Tests (KAT) from primary sources (NIST LWC / CRYPTO 2016).
2. Independent cross-check against an isolated pure-Python scalar reference on 1000 random triples.
3. Unit tests for each individual round transformation and its inverse.
4. Hand-derived step-by-step intermediate round values.
5. Flexible input interfaces (bytes, bytearray, lists, ndarrays) and non-mutation guarantees.
6. Boundary conditions, extreme inputs, tweakey broadcasting, and strict validation.
"""

import os
import json
import random
import numpy as np
import pytest

from ciphers.skinny128 import (
    skinny128_encrypt,
    skinny128_encrypt_batch,
    sub_cells,
    inv_sub_cells,
    add_constants,
    add_round_tweakey,
    shift_rows,
    inv_shift_rows,
    mix_columns,
    inv_mix_columns,
    tweakey_permutation,
    inv_tweakey_permutation,
    SBOX,
    INV_SBOX,
    ROUND_CONSTANTS,
    TK_PERM,
    INV_TK_PERM,
    FULL_ROUNDS,
    BLOCK_BYTES,
    TWEAKEY_BYTES,
)
from tests.skinny128_scalar_ref import skinny128_128_scalar_encrypt


# ==============================================================================
# 1. Known-Answer Tests (Primary Source Validation)
# ==============================================================================

def test_kat_from_primary_source_nist_lwc():
    """Verify full 40-round SKINNY-128-128 against official NIST LWC Remus specification test vector.

    Primary Source:
        Document: NIST Lightweight Cryptography (LWC) Candidate Submission: Remus Specification
        Authors:  Tetsu Iwata, Mustafa Khairallah, Kazuhiko Minematsu, Thomas Peyrin
        Section:  Appendix A: Test Vectors for Primitive SKINNY-128-128
        URL:      https://csrc.nist.gov/CSRC/media/Projects/lightweight-cryptography/documents/round-1/remus-spec.pdf
    """
    key = bytes.fromhex("4f55cfb0520cac52fd92c15f37073e93")
    pt = bytes.fromhex("f20adb0eb08b648a3b2eeed1f0adda14")
    expected_ct = bytes.fromhex("22ff30d498ea62d7e45b476e33675b74")

    # 1. Test single-block interface
    ct_single = skinny128_encrypt(pt, key, rounds=40)
    assert bytes(ct_single) == expected_ct, (
        f"NIST LWC KAT failed on single-block: got {bytes(ct_single).hex()}, expected {expected_ct.hex()}"
    )

    # 2. Test batch interface
    ct_batch = skinny128_encrypt_batch(pt, key, rounds=40)
    assert bytes(ct_batch[0]) == expected_ct, (
        f"NIST LWC KAT failed on batch: got {bytes(ct_batch[0]).hex()}, expected {expected_ct.hex()}"
    )


def test_kat_vectors_json_file():
    """Verify all test vectors stored in tests/vectors/skinny128_128_vectors.json."""
    vector_path = os.path.join(os.path.dirname(__file__), "vectors", "skinny128_128_vectors.json")
    assert os.path.exists(vector_path), f"Vectors file missing: {vector_path}"

    with open(vector_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["cipher"] == "SKINNY-128-128"
    assert data["full_rounds"] == 40

    for vec in data["vectors"]:
        r = vec["rounds"]
        tk = bytes.fromhex(vec["tweakey_hex"])
        pt = bytes.fromhex(vec["plaintext_hex"])
        expected = bytes.fromhex(vec["ciphertext_hex"])

        result = skinny128_encrypt(pt, tk, rounds=r)
        assert bytes(result) == expected, (
            f"Vector '{vec['description']}' failed at round {r}: "
            f"got {bytes(result).hex()}, expected {expected.hex()}"
        )


def test_hand_derived_first_round():
    """Verify round 1 output on all-zero plaintext and tweakey against step-by-step derivation.

    Derivation from Beierle et al., CRYPTO 2016, Section 2:
      State initialized to 0.
      1. SubCells: SBOX[0] = 0x65 for all 16 cells.
      2. AddConstants: RC0 = 0x01.
         IS[0,0] ^= 0x01 -> 0x64
         IS[1,0] ^= 0x00 -> 0x65
         IS[2,0] ^= 0x02 -> 0x67
         IS[3,0] ^= 0x00 -> 0x65
      3. AddRoundTweakey: TK=0, unchanged.
      4. ShiftRows:
         Row 0: [0x64, 0x65, 0x65, 0x65]
         Row 1: [0x65, 0x65, 0x65, 0x65]
         Row 2: [0x65, 0x65, 0x67, 0x65]
         Row 3: [0x65, 0x65, 0x65, 0x65]
      5. MixColumns:
         Col 0: a'=0x64, b'=0x64, c'=0x00, d'=0x01
         Col 1: a'=0x65, b'=0x65, c'=0x00, d'=0x00
         Col 2: a'=0x67, b'=0x65, c'=0x02, d'=0x02
         Col 3: a'=0x65, b'=0x65, c'=0x00, d'=0x00
      Result bytes: 64 65 67 65 64 65 65 65 00 00 02 00 01 00 02 00
    """
    zero_block = bytes(16)
    expected_r1 = bytes.fromhex("64656765646565650000020001000200")
    actual_r1 = skinny128_encrypt(zero_block, zero_block, rounds=1)
    assert bytes(actual_r1) == expected_r1


# ==============================================================================
# 2. Independent Cross-Check (1000 Random Triples against Scalar Reference)
# ==============================================================================

def test_independent_cross_check_1000_triples():
    """Cross-check NumPy vectorized implementation against independent scalar pure-Python reference.

    Evaluates 1000 uniformly random (plaintext, tweakey, rounds) triples covering
    every round from 1 to 40.
    """
    rng = random.Random(42)
    mismatches = 0

    for i in range(1000):
        # Round-robin through all 40 round values to ensure complete round coverage
        r = (i % 40) + 1
        pt = bytes(rng.getrandbits(8) for _ in range(16))
        tk = bytes(rng.getrandbits(8) for _ in range(16))

        # 1. Independent scalar reference output
        expected = skinny128_128_scalar_encrypt(pt, tk, rounds=r)

        # 2. Vectorized NumPy output
        actual = bytes(skinny128_encrypt(pt, tk, rounds=r))

        if actual != expected:
            mismatches += 1
            break

    assert mismatches == 0, f"Independent cross-check encountered mismatch at test {i}, round {r}!"


# ==============================================================================
# 3. Unit Tests per Transformation Step
# ==============================================================================

def test_sub_cells_and_inverse():
    """Verify SubCells bijective property, table lookups, and inverse round-trip."""
    # S-box length and uniqueness
    assert len(SBOX) == 256
    assert len(np.unique(SBOX)) == 256
    assert SBOX[0] == 0x65
    assert SBOX[255] == 0xFF

    # Inverse table correctness
    assert len(INV_SBOX) == 256
    assert np.array_equal(INV_SBOX[SBOX], np.arange(256))

    # Test round-trip on random state
    rng = np.random.default_rng(123)
    state = rng.integers(0, 256, size=(8, 4, 4), dtype=np.uint8)
    forward = sub_cells(state)
    recovered = inv_sub_cells(forward)
    assert np.array_equal(recovered, state)


def test_add_constants_lfsr_and_column_isolation():
    """Verify AddConstants 6-bit LFSR derivation and column-0 exclusivity."""
    # Check all 40 constants against mathematical LFSR
    rc = 0
    derived = []
    for _ in range(40):
        rc5 = (rc >> 5) & 1
        rc4 = (rc >> 4) & 1
        rc = ((rc << 1) & 0x3F) | (rc4 ^ rc5 ^ 1)
        derived.append(rc)
    assert list(ROUND_CONSTANTS) == derived

    # Verify that ONLY column 0 is modified by add_constants
    state = np.zeros((1, 4, 4), dtype=np.uint8)
    for r_idx in range(40):
        rc_val = int(ROUND_CONSTANTS[r_idx])
        out = add_constants(state, rc_val)
        # Column 0 checks
        assert out[0, 0, 0] == (rc_val & 0x0F)
        assert out[0, 1, 0] == ((rc_val >> 4) & 0x03)
        assert out[0, 2, 0] == 0x02
        assert out[0, 3, 0] == 0x00
        # Columns 1, 2, 3 must be untouched (all zeros)
        assert np.all(out[0, :, 1:] == 0)


def test_add_round_tweakey_row_isolation():
    """Verify AddRoundTweakey only affects rows 0 and 1; rows 2 and 3 remain untouched."""
    state = np.full((1, 4, 4), 0xAA, dtype=np.uint8)
    tk = np.full((1, 4, 4), 0x55, dtype=np.uint8)

    out = add_round_tweakey(state, tk)
    # Rows 0 and 1: 0xAA ^ 0x55 = 0xFF
    assert np.all(out[:, :2, :] == 0xFF)
    # Rows 2 and 3: unchanged (0xAA)
    assert np.all(out[:, 2:, :] == 0xAA)


def test_shift_rows_and_inverse():
    """Verify ShiftRows rotation amounts and inverse round-trip."""
    # Create deterministic row patterns: row 0=0..3, row 1=4..7, row 2=8..11, row 3=12..15
    state = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)
    shifted = shift_rows(state)

    # Row 0: no shift [0, 1, 2, 3]
    assert np.array_equal(shifted[0, 0, :], [0, 1, 2, 3])
    # Row 1: right shift by 1 [7, 4, 5, 6]
    assert np.array_equal(shifted[0, 1, :], [7, 4, 5, 6])
    # Row 2: right shift by 2 [10, 11, 8, 9]
    assert np.array_equal(shifted[0, 2, :], [10, 11, 8, 9])
    # Row 3: right shift by 3 [13, 14, 15, 12]
    assert np.array_equal(shifted[0, 3, :], [13, 14, 15, 12])

    # Inverse ShiftRows round-trip
    recovered = inv_shift_rows(shifted)
    assert np.array_equal(recovered, state)


def test_mix_columns_and_inverse():
    """Verify MixColumns binary matrix multiplication and inverse round-trip."""
    rng = np.random.default_rng(456)
    state = rng.integers(0, 256, size=(10, 4, 4), dtype=np.uint8)

    mc_state = mix_columns(state)
    recovered = inv_mix_columns(mc_state)
    assert np.array_equal(recovered, state)

    # Hand-checked column: col = [1, 2, 4, 8]
    # a=1, b=2, c=4, d=8
    # a' = 1 ^ 4 ^ 8 = 13 (0x0D)
    # b' = 1
    # c' = 2 ^ 4 = 6
    # d' = 1 ^ 4 = 5
    col_test = np.array([[[1], [2], [4], [8]]], dtype=np.uint8)  # shape (1, 4, 1)
    # tile across 4 cols
    state_col = np.tile(col_test, (1, 1, 4))
    res = mix_columns(state_col)
    for c in range(4):
        assert res[0, 0, c] == 13
        assert res[0, 1, c] == 1
        assert res[0, 2, c] == 6
        assert res[0, 3, c] == 5


def test_tweakey_permutation_and_inverse():
    """Verify P_T cell permutation and inverse round-trip."""
    assert list(TK_PERM) == [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    assert len(set(TK_PERM)) == 16

    tk = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)
    permuted = tweakey_permutation(tk)
    recovered = inv_tweakey_permutation(permuted)
    assert np.array_equal(recovered, tk)


# ==============================================================================
# 4. Interface and Input Flexibility Tests
# ==============================================================================

def test_interface_types_bytes_bytearray_list_ndarray():
    """Verify that bytes, bytearray, list, and ndarray inputs produce identical ciphertexts."""
    pt_bytes = bytes(range(16))
    tk_bytes = bytes(range(16, 32))

    ct_bytes = skinny128_encrypt(pt_bytes, tk_bytes, rounds=10)
    ct_bytearray = skinny128_encrypt(bytearray(pt_bytes), bytearray(tk_bytes), rounds=10)
    ct_list = skinny128_encrypt(list(pt_bytes), list(tk_bytes), rounds=10)
    ct_ndarray = skinny128_encrypt(np.array(list(pt_bytes), dtype=np.uint8),
                                  np.array(list(tk_bytes), dtype=np.uint8), rounds=10)

    assert np.array_equal(ct_bytes, ct_bytearray)
    assert np.array_equal(ct_bytes, ct_list)
    assert np.array_equal(ct_bytes, ct_ndarray)


def test_single_and_batch_interfaces_match():
    """Verify single-block and batch interfaces produce identical results across rounds."""
    p = bytes(range(16))
    tk = bytes(range(16, 32))
    for r in [1, 2, 7, 20, 40]:
        single_res = skinny128_encrypt(p, tk, rounds=r)
        batch_res = skinny128_encrypt_batch(p, tk, rounds=r)[0]
        assert np.array_equal(single_res, batch_res)


def test_tweakey_broadcasting():
    """Verify a single 16-byte tweakey broadcasts across a batch of N plaintexts."""
    n_samples = 5
    pts = (np.arange(n_samples * 16, dtype=int) % 256).astype(np.uint8).reshape(n_samples, 16)
    single_tk = bytes(range(16))

    # Batch encrypt with 1 tweakey
    cts = skinny128_encrypt_batch(pts, single_tk, rounds=5)
    assert cts.shape == (n_samples, 16)

    # Check each individual block
    for i in range(n_samples):
        expected_ct = skinny128_encrypt(pts[i], single_tk, rounds=5)
        assert np.array_equal(cts[i], expected_ct)


def test_input_non_mutation():
    """Verify encryption functions never mutate caller input arrays or buffers."""
    pt_orig = np.arange(16, dtype=np.uint8).reshape(1, 16)
    tk_orig = np.full((1, 16), 0x42, dtype=np.uint8)

    pt_copy = pt_orig.copy()
    tk_copy = tk_orig.copy()

    _ = skinny128_encrypt_batch(pt_orig, tk_orig, rounds=40)

    assert np.array_equal(pt_orig, pt_copy), "Plaintext input was mutated!"
    assert np.array_equal(tk_orig, tk_copy), "Tweakey input was mutated!"


# ==============================================================================
# 5. Boundary Conditions, Extreme Inputs, and Error Handling
# ==============================================================================

@pytest.mark.parametrize("r", [1, 2, 39, 40])
def test_valid_round_boundaries(r):
    """Verify valid boundary rounds (1, 2, 39, 40) execute without error."""
    pt = bytes(16)
    tk = bytes(16)
    res = skinny128_encrypt(pt, tk, rounds=r)
    assert res.shape == (16,)
    assert res.dtype == np.uint8


@pytest.mark.parametrize("bad_r", [0, 41, -1, 100, 3.5, "40", True, False])
def test_invalid_rounds_fail_loudly(bad_r):
    """Verify invalid round counts raise ValueError or TypeError."""
    pt = bytes(16)
    tk = bytes(16)
    with pytest.raises((ValueError, TypeError)):
        skinny128_encrypt(pt, tk, rounds=bad_r)


def test_invalid_tweakey_sizes_fail_loudly():
    """Verify tweakeys of sizes other than 16 bytes (128 bits) raise ValueError."""
    pt = bytes(16)
    for bad_size in [8, 12, 15, 17, 24, 32]:
        with pytest.raises(ValueError, match="16"):
            skinny128_encrypt(pt, bytes(bad_size), rounds=40)


def test_invalid_plaintext_sizes_fail_loudly():
    """Verify plaintexts of sizes other than 16 bytes raise ValueError."""
    tk = bytes(16)
    for bad_size in [8, 15, 17, 32]:
        with pytest.raises(ValueError, match="16"):
            skinny128_encrypt(bytes(bad_size), tk, rounds=40)


def test_extreme_inputs_all_zeros_and_all_ones():
    """Verify encryption of all-zero and all-one blocks completes deterministically."""
    zeros = bytes(16)
    ones = bytes([0xFF] * 16)

    c_00 = skinny128_encrypt(zeros, zeros, rounds=40)
    c_11 = skinny128_encrypt(ones, ones, rounds=40)
    c_01 = skinny128_encrypt(zeros, ones, rounds=40)
    c_10 = skinny128_encrypt(ones, zeros, rounds=40)

    # Distinct outputs
    assert not np.array_equal(c_00, c_11)
    assert not np.array_equal(c_00, c_01)
    assert not np.array_equal(c_00, c_10)


def test_batch_sizes_1_and_large():
    """Verify execution on batch size 1 and batch size 256."""
    rng = np.random.default_rng(789)
    # Batch size 1
    p1 = rng.integers(0, 256, size=(1, 16), dtype=np.uint8)
    k1 = rng.integers(0, 256, size=(1, 16), dtype=np.uint8)
    c1 = skinny128_encrypt_batch(p1, k1, rounds=40)
    assert c1.shape == (1, 16)

    # Batch size 256
    p256 = rng.integers(0, 256, size=(256, 16), dtype=np.uint8)
    k256 = rng.integers(0, 256, size=(256, 16), dtype=np.uint8)
    c256 = skinny128_encrypt_batch(p256, k256, rounds=40)
    assert c256.shape == (256, 16)
