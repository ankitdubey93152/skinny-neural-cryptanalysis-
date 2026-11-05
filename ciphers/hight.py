"""Vectorized NumPy implementation of the HIGHT block cipher (ISO/IEC 18033-3).

64-bit block (8 bytes), 128-bit key (16 bytes), 32 rounds.
ARX structure with modular addition mod 256, bitwise XOR, and 8-bit circular rotations.
"""
from __future__ import annotations
from typing import Union, Sequence, Optional
import numpy as np

BLOCK_BYTES = 8
KEY_BYTES = 16
FULL_ROUNDS = 32

# Precompute 128 7-bit constants delta_i from 7-bit LFSR
# s0=0, s1=1, s2=0, s3=1, s4=1, s5=0, s6=1; s_{i+6} = s_{i+2} ^ s_{i-1} for i=1..127
_s = [0, 1, 0, 1, 1, 0, 1]
for _i in range(1, 128):
    _s.append(_s[_i + 2] ^ _s[_i - 1])

DELTA = np.array([
    (_s[i + 6] << 6)
    | (_s[i + 5] << 5)
    | (_s[i + 4] << 4)
    | (_s[i + 3] << 3)
    | (_s[i + 2] << 2)
    | (_s[i + 1] << 1)
    | _s[i]
    for i in range(128)
], dtype=np.uint8)


def _as_batch_8(x: Union[bytes, bytearray, Sequence, np.ndarray], name: str) -> np.ndarray:
    if isinstance(x, (bytes, bytearray, memoryview)):
        raw = bytes(x)
        if len(raw) % 8 != 0 or len(raw) == 0:
            raise ValueError(f"{name} byte string length must be a non-zero multiple of 8, got {len(raw)}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8).copy()
    if isinstance(x, list):
        if len(x) == 0:
            raise ValueError(f"{name} list cannot be empty")
        if isinstance(x[0], (int, np.integer)):
            if len(x) != 8:
                raise ValueError(f"{name} 1D list must have 8 elements, got {len(x)}")
            return np.array([x], dtype=np.uint8)
        arr = np.asarray(x, dtype=np.uint8)
        if arr.ndim != 2 or arr.shape[1] != 8:
            raise ValueError(f"{name} list must have shape (N, 8) or (8,), got {arr.shape}")
        return arr.copy()
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must contain integers, got dtype {arr.dtype}")
    if arr.ndim == 1:
        if arr.shape[0] != 8:
            raise ValueError(f"{name} 1D array must have length 8, got {arr.shape[0]}")
        arr = arr.reshape(1, 8)
    elif arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"{name} array must have shape (N, 8) or (8,), got {arr.shape}")
    return arr.astype(np.uint8, copy=True)


def _as_batch_16(x: Union[bytes, bytearray, Sequence, np.ndarray], name: str) -> np.ndarray:
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


def _rol8(x: np.ndarray, n: int) -> np.ndarray:
    return ((x << n) | (x >> (8 - n))).astype(np.uint8)


def _F0(x: np.ndarray) -> np.ndarray:
    return _rol8(x, 1) ^ _rol8(x, 2) ^ _rol8(x, 7)


def _F1(x: np.ndarray) -> np.ndarray:
    return _rol8(x, 3) ^ _rol8(x, 4) ^ _rol8(x, 6)


def hight_key_schedule(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive whitening keys WK of shape (8, N) and subkeys SK of shape (128, N) from keys (N, 16)."""
    n = keys.shape[0]
    # K15 is byte 0, K0 is byte 15
    k = [keys[:, 15 - i].copy() for i in range(16)]

    wk = np.empty((8, n), dtype=np.uint8)
    for i in range(4):
        wk[i] = k[i + 12]
    for i in range(4, 8):
        wk[i] = k[i - 4]

    sk = np.empty((128, n), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            sk[16 * i + j] = k[(j - i) % 8] + DELTA[16 * i + j]
            sk[16 * i + j + 8] = k[((j - i) % 8) + 8] + DELTA[16 * i + j + 8]
    return wk, sk


def hight_encrypt_batch(
    plaintexts: Union[bytes, bytearray, Sequence, np.ndarray],
    keys: Union[bytes, bytearray, Sequence, np.ndarray],
    rounds: int = FULL_ROUNDS,
    *,
    key: Optional[Union[bytes, bytearray, Sequence, np.ndarray]] = None,
) -> np.ndarray:
    """Encrypt a batch of blocks under HIGHT with arbitrary rounds (0 to 32)."""
    if key is not None and keys is None:
        keys = key
    if not (0 <= rounds <= 32):
        raise ValueError(f"Rounds must be in range [0, 32], got {rounds}")

    pts = _as_batch_8(plaintexts, "plaintexts")
    n = pts.shape[0]
    ks = _as_batch_16(keys, "keys")
    if ks.shape[0] == 1 and n > 1:
        ks = np.repeat(ks, n, axis=0)
    elif ks.shape[0] != n:
        raise ValueError(f"Key count ({ks.shape[0]}) does not match plaintext count ({n})")

    if rounds == 0:
        return pts.copy()

    wk, sk = hight_key_schedule(ks)
    # P7 is byte 0, P0 is byte 7
    p = [pts[:, 7 - i].copy() for i in range(8)]

    # Initial whitening
    x = [
        p[0] + wk[0],
        p[1].copy(),
        p[2] ^ wk[1],
        p[3].copy(),
        p[4] + wk[2],
        p[5].copy(),
        p[6] ^ wk[3],
        p[7].copy(),
    ]

    for i in range(min(rounds, 31)):
        x_next = [
            x[7] ^ (_F0(x[6]) + sk[4 * i + 3]),
            x[0].copy(),
            x[1] + (_F1(x[0]) ^ sk[4 * i + 0]),
            x[2].copy(),
            x[3] ^ (_F0(x[2]) + sk[4 * i + 1]),
            x[4].copy(),
            x[5] + (_F1(x[4]) ^ sk[4 * i + 2]),
            x[6].copy(),
        ]
        x = x_next

    if rounds == 32:
        # Final round 31 (no rotation):
        x_final = [
            x[0].copy(),
            x[1] + (_F1(x[0]) ^ sk[124]),
            x[2].copy(),
            x[3] ^ (_F0(x[2]) + sk[125]),
            x[4].copy(),
            x[5] + (_F1(x[4]) ^ sk[126]),
            x[6].copy(),
            x[7] ^ (_F0(x[6]) + sk[127]),
        ]
        # Final transformation with WK4..WK7:
        c = [
            x_final[0] + wk[4],
            x_final[1].copy(),
            x_final[2] ^ wk[5],
            x_final[3].copy(),
            x_final[4] + wk[6],
            x_final[5].copy(),
            x_final[6] ^ wk[7],
            x_final[7].copy(),
        ]
        # C = C7 || C6 || ... || C0
        out_cols = [c[7 - i] for i in range(8)]
        return np.column_stack(out_cols).copy()

    # For reduced rounds < 32:
    out_cols = [x[7 - i] for i in range(8)]
    return np.column_stack(out_cols).copy()


def hight_encrypt(
    plaintext: Union[bytes, bytearray, Sequence, np.ndarray],
    key: Union[bytes, bytearray, Sequence, np.ndarray],
    rounds: int = FULL_ROUNDS,
) -> bytes:
    """Encrypt a single 8-byte block under HIGHT."""
    pts = _as_batch_8(plaintext, "plaintext")
    if pts.shape[0] != 1:
        raise ValueError(f"Expected single block, got {pts.shape[0]}")
    ks = _as_batch_16(key, "key")
    if ks.shape[0] != 1:
        raise ValueError(f"Expected single key, got {ks.shape[0]}")
    ct = hight_encrypt_batch(pts, ks, rounds=rounds)
    return bytes(ct[0])
