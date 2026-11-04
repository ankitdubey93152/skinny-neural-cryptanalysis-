"""Vectorized NumPy implementation of SKINNY-128-128 (40 rounds, TK1 only).
Reference: Beierle et al., CRYPTO 2016 / IACR ePrint 2016/660.
"""
from __future__ import annotations
from typing import Union, Sequence, Optional
import numpy as np

BLOCK_BYTES = BLOCK_BITS = KEY_BYTES = TWEAKEY_BYTES = 16
FULL_ROUNDS = 40

SBOX = np.array([
    0x65, 0x4c, 0x6a, 0x42, 0x4b, 0x63, 0x43, 0x6b, 0x55, 0x75, 0x5a, 0x7a, 0x53, 0x73, 0x5b, 0x7b,
    0x35, 0x8c, 0x3a, 0x81, 0x89, 0x33, 0x80, 0x3b, 0x95, 0x25, 0x98, 0x2a, 0x90, 0x23, 0x99, 0x2b,
    0xe5, 0xcc, 0xe8, 0xc1, 0xc9, 0xe0, 0xc0, 0xe9, 0xd5, 0xf5, 0xd8, 0xf8, 0xd0, 0xf0, 0xd9, 0xf9,
    0xa5, 0x1c, 0xa8, 0x12, 0x1b, 0xa0, 0x13, 0xa9, 0x05, 0xb5, 0x0a, 0xb8, 0x03, 0xb0, 0x0b, 0xb9,
    0x32, 0x88, 0x3c, 0x85, 0x8d, 0x34, 0x84, 0x3d, 0x91, 0x22, 0x9c, 0x2c, 0x94, 0x24, 0x9d, 0x2d,
    0x62, 0x4a, 0x6c, 0x45, 0x4d, 0x64, 0x44, 0x6d, 0x52, 0x72, 0x5c, 0x7c, 0x54, 0x74, 0x5d, 0x7d,
    0xa1, 0x1a, 0xac, 0x15, 0x1d, 0xa4, 0x14, 0xad, 0x02, 0xb1, 0x0c, 0xbc, 0x04, 0xb4, 0x0d, 0xbd,
    0xe1, 0xc8, 0xec, 0xc5, 0xcd, 0xe4, 0xc4, 0xed, 0xd1, 0xf1, 0xdc, 0xfc, 0xd4, 0xf4, 0xdd, 0xfd,
    0x36, 0x8e, 0x38, 0x82, 0x8b, 0x30, 0x83, 0x39, 0x96, 0x26, 0x9a, 0x28, 0x93, 0x20, 0x9b, 0x29,
    0x66, 0x4e, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69, 0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79,
    0xa6, 0x1e, 0xaa, 0x11, 0x19, 0xa3, 0x10, 0xab, 0x06, 0xb6, 0x08, 0xba, 0x00, 0xb3, 0x09, 0xbb,
    0xe6, 0xce, 0xea, 0xc2, 0xcb, 0xe3, 0xc3, 0xeb, 0xd6, 0xf6, 0xda, 0xfa, 0xd3, 0xf3, 0xdb, 0xfb,
    0x31, 0x8a, 0x3e, 0x86, 0x8f, 0x37, 0x87, 0x3f, 0x92, 0x21, 0x9e, 0x2e, 0x97, 0x27, 0x9f, 0x2f,
    0x61, 0x48, 0x6e, 0x46, 0x4f, 0x67, 0x47, 0x6f, 0x51, 0x71, 0x5e, 0x7e, 0x57, 0x77, 0x5f, 0x7f,
    0xa2, 0x18, 0xae, 0x16, 0x1f, 0xa7, 0x17, 0xaf, 0x01, 0xb2, 0x0e, 0xbe, 0x07, 0xb7, 0x0f, 0xbf,
    0xe2, 0xca, 0xee, 0xc6, 0xcf, 0xe7, 0xc7, 0xef, 0xd2, 0xf2, 0xde, 0xfe, 0xd7, 0xf7, 0xdf, 0xff,
], dtype=np.uint8)

INV_SBOX = np.zeros(256, dtype=np.uint8)
INV_SBOX[SBOX] = np.arange(256, dtype=np.uint8)

ROUND_CONSTANTS = np.array([
    0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3e, 0x3d, 0x3b, 0x37, 0x2f, 0x1e, 0x3c, 0x39, 0x33, 0x27, 0x0e,
    0x1d, 0x3a, 0x35, 0x2b, 0x16, 0x2c, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0b, 0x17, 0x2e, 0x1c, 0x38,
    0x31, 0x23, 0x06, 0x0d, 0x1b, 0x36, 0x2d, 0x1a,
], dtype=np.uint8)

TK_PERM = np.array([9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.intp)
INV_TK_PERM = np.array([8, 9, 10, 11, 12, 13, 14, 15, 2, 0, 4, 7, 6, 3, 5, 1], dtype=np.intp)


def _as_batch(x: Union[bytes, bytearray, Sequence, np.ndarray], name: str) -> np.ndarray:
    if isinstance(x, (bytes, bytearray, memoryview)):
        raw = bytes(x)
        if len(raw) % 16 != 0 or len(raw) == 0:
            raise ValueError(f"{name} byte string length must be a non-zero multiple of 16, got {len(raw)}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 16).copy()
    if isinstance(x, list):
        if len(x) == 0:
            raise ValueError(f"{name} list cannot be empty")
        if isinstance(x[0], (int, np.integer)):
            if len(x) != 16:
                raise ValueError(f"{name} 1D list must have 16 elements, got {len(x)}")
            return np.array([x], dtype=np.uint8)
        arr = np.asarray(x, dtype=np.uint8)
        if arr.ndim != 2 or arr.shape[1] != 16:
            raise ValueError(f"{name} list must have shape (N, 16) or (16,), got {arr.shape}")
        return arr.copy()
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must contain integers, got dtype {arr.dtype}")
    if arr.ndim == 1:
        if arr.shape[0] != 16:
            raise ValueError(f"{name} 1D array must have length 16, got {arr.shape[0]}")
        arr = arr.reshape(1, 16)
    elif arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"{name} array must have shape (N, 16) or (16,), got {arr.shape}")
    return arr.astype(np.uint8, copy=True)


def sub_cells(state: np.ndarray) -> np.ndarray:
    return SBOX[state]


def inv_sub_cells(state: np.ndarray) -> np.ndarray:
    return INV_SBOX[state]


def add_constants(state: np.ndarray, rc: int) -> np.ndarray:
    st = state.copy()
    st[:, 0, 0] ^= rc & 0x0F
    st[:, 1, 0] ^= (rc >> 4) & 0x03
    st[:, 2, 0] ^= 0x02
    return st


def add_round_tweakey(state: np.ndarray, tweakey: np.ndarray) -> np.ndarray:
    st = state.copy()
    st[:, :2, :] ^= tweakey[:, :2, :]
    return st


def shift_rows(state: np.ndarray) -> np.ndarray:
    st = state.copy()
    for r in range(1, 4):
        st[:, r, :] = np.roll(st[:, r, :], r, axis=1)
    return st


def inv_shift_rows(state: np.ndarray) -> np.ndarray:
    st = state.copy()
    for r in range(1, 4):
        st[:, r, :] = np.roll(st[:, r, :], -r, axis=1)
    return st


def mix_columns(state: np.ndarray) -> np.ndarray:
    st = state.copy()
    a, b, c, d = st[:, 0, :].copy(), st[:, 1, :].copy(), st[:, 2, :].copy(), st[:, 3, :].copy()
    st[:, 0, :] = a ^ c ^ d
    st[:, 1, :] = a
    st[:, 2, :] = b ^ c
    st[:, 3, :] = a ^ c
    return st


def inv_mix_columns(state: np.ndarray) -> np.ndarray:
    st = state.copy()
    a_p, b_p, c_p, d_p = st[:, 0, :].copy(), st[:, 1, :].copy(), st[:, 2, :].copy(), st[:, 3, :].copy()
    st[:, 0, :] = b_p
    st[:, 1, :] = b_p ^ c_p ^ d_p
    st[:, 2, :] = b_p ^ d_p
    st[:, 3, :] = a_p ^ d_p
    return st


def tweakey_permutation(tweakey: np.ndarray) -> np.ndarray:
    return tweakey.reshape(len(tweakey), 16)[:, TK_PERM].reshape(len(tweakey), 4, 4)


def inv_tweakey_permutation(tweakey: np.ndarray) -> np.ndarray:
    return tweakey.reshape(len(tweakey), 16)[:, INV_TK_PERM].reshape(len(tweakey), 4, 4)


def skinny128_encrypt_batch(
    plaintexts: Union[bytes, bytearray, Sequence, np.ndarray],
    tweakeys: Optional[Union[bytes, bytearray, Sequence, np.ndarray]] = None,
    rounds: int = FULL_ROUNDS,
    *,
    tweakey: Optional[Union[bytes, bytearray, Sequence, np.ndarray]] = None,
    keys: Optional[Union[bytes, bytearray, Sequence, np.ndarray]] = None,
    key: Optional[Union[bytes, bytearray, Sequence, np.ndarray]] = None,
) -> np.ndarray:
    """Encrypt a batch of blocks under SKINNY-128-128."""
    tk_arg = tweakeys if tweakeys is not None else (tweakey if tweakey is not None else (keys if keys is not None else key))
    if tk_arg is None:
        raise TypeError("skinny128_encrypt_batch missing required argument: 'tweakeys' (or 'tweakey')")

    p = _as_batch(plaintexts, "plaintexts")
    tk = _as_batch(tk_arg, "tweakeys")
    if len(tk) == 1 and len(p) > 1:
        tk = np.repeat(tk, len(p), axis=0)
    elif len(tk) != len(p):
        raise ValueError(f"Batch sizes must match: plaintexts {len(p)} vs tweakeys {len(tk)}")

    if not isinstance(rounds, (int, np.integer)) or isinstance(rounds, bool) or not (1 <= int(rounds) <= FULL_ROUNDS):
        raise ValueError(f"rounds must be an integer in 1..{FULL_ROUNDS}, got {rounds}")

    state, tk_state = p.reshape(-1, 4, 4).copy(), tk.reshape(-1, 4, 4).copy()
    for r_idx in range(int(rounds)):
        state = mix_columns(shift_rows(add_round_tweakey(add_constants(sub_cells(state), int(ROUND_CONSTANTS[r_idx])), tk_state)))
        tk_state = tweakey_permutation(tk_state)
    return state.reshape(-1, 16)


def skinny128_encrypt(
    plaintext: Union[bytes, bytearray, Sequence, np.ndarray],
    tweakey: Union[bytes, bytearray, Sequence, np.ndarray],
    rounds: int = FULL_ROUNDS,
    *,
    key: Optional[Union[bytes, bytearray, Sequence, np.ndarray]] = None,
) -> np.ndarray:
    """Encrypt a single 16-byte block with a 16-byte tweakey."""
    tk_arg = tweakey if tweakey is not None else key
    if tk_arg is None:
        raise TypeError("skinny128_encrypt missing required argument: 'tweakey'")
    p, tk = _as_batch(plaintext, "plaintext"), _as_batch(tk_arg, "tweakey")
    if p.shape[0] != 1 or tk.shape[0] != 1:
        raise ValueError("skinny128_encrypt expects a single 16-byte plaintext and tweakey")
    return skinny128_encrypt_batch(p, tk, rounds=rounds)[0]


encrypt = skinny128_encrypt
encrypt_batch = skinny128_encrypt_batch
