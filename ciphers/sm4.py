"""Vectorized NumPy implementation of the SM4 block cipher (GB/T 32907-2016).

128-bit block, 128-bit key, 32 rounds, generalized Feistel structure with
non-linear substitution tau (8-bit S-box) and linear diffusion L / L'.
"""
from __future__ import annotations
from typing import Union, Sequence, Optional
import numpy as np

BLOCK_BYTES = BLOCK_BITS = KEY_BYTES = 16
FULL_ROUNDS = 32

SBOX = np.array([
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48,
], dtype=np.uint8)

FK = np.array([0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc], dtype=np.uint32)

CK = np.array([
    (((4 * i + 0) * 7 % 256) << 24)
    | (((4 * i + 1) * 7 % 256) << 16)
    | (((4 * i + 2) * 7 % 256) << 8)
    | (((4 * i + 3) * 7 % 256))
    for i in range(32)
], dtype=np.uint32)


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


def _rol(w: np.ndarray, n: int) -> np.ndarray:
    return ((w << n) | (w >> (32 - n))).astype(np.uint32)


def _tau(w: np.ndarray) -> np.ndarray:
    b0 = SBOX[(w >> 24) & 0xFF].astype(np.uint32)
    b1 = SBOX[(w >> 16) & 0xFF].astype(np.uint32)
    b2 = SBOX[(w >> 8) & 0xFF].astype(np.uint32)
    b3 = SBOX[w & 0xFF].astype(np.uint32)
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3


def _L(b: np.ndarray) -> np.ndarray:
    return b ^ _rol(b, 2) ^ _rol(b, 10) ^ _rol(b, 18) ^ _rol(b, 24)


def _L_prime(b: np.ndarray) -> np.ndarray:
    return b ^ _rol(b, 13) ^ _rol(b, 23)


def sm4_key_schedule(keys: np.ndarray) -> np.ndarray:
    """Expand keys of shape (N, 16) uint8 into round keys of shape (32, N) uint32."""
    n = keys.shape[0]
    mk = np.ascontiguousarray(keys).view(">u4").reshape(n, 4)
    k = [mk[:, i] ^ FK[i] for i in range(4)]
    rk = []
    for i in range(32):
        tmp = k[i + 1] ^ k[i + 2] ^ k[i + 3] ^ CK[i]
        k_next = k[i] ^ _L_prime(_tau(tmp))
        k.append(k_next)
        rk.append(k_next)
    return np.array(rk, dtype=np.uint32)  # shape (32, N)


def sm4_encrypt_batch(
    plaintexts: Union[bytes, bytearray, Sequence, np.ndarray],
    keys: Union[bytes, bytearray, Sequence, np.ndarray],
    rounds: int = FULL_ROUNDS,
    *,
    key: Optional[Union[bytes, bytearray, Sequence, np.ndarray]] = None,
) -> np.ndarray:
    """Encrypt a batch of blocks under SM4 with arbitrary rounds (0 to 32)."""
    if key is not None and keys is None:
        keys = key
    if not (0 <= rounds <= 32):
        raise ValueError(f"Rounds must be in range [0, 32], got {rounds}")

    pts = _as_batch(plaintexts, "plaintexts")
    n = pts.shape[0]
    ks = _as_batch(keys, "keys")
    if ks.shape[0] == 1 and n > 1:
        ks = np.repeat(ks, n, axis=0)
    elif ks.shape[0] != n:
        raise ValueError(f"Key count ({ks.shape[0]}) does not match plaintext count ({n})")

    if rounds == 0:
        return pts.copy()

    rk = sm4_key_schedule(ks)  # shape (32, N)
    x = [col.copy() for col in np.ascontiguousarray(pts).view(">u4").reshape(n, 4).T]

    for i in range(rounds):
        tmp = x[i + 1] ^ x[i + 2] ^ x[i + 3] ^ rk[i]
        x_next = x[i] ^ _L(_tau(tmp))
        x.append(x_next)

    # Reverse transformation R: (X_{r+3}, X_{r+2}, X_{r+1}, X_r)
    y = np.stack([x[rounds + 3], x[rounds + 2], x[rounds + 1], x[rounds]], axis=1)
    return np.ascontiguousarray(y).astype(">u4").view(np.uint8).reshape(n, 16).copy()


def sm4_encrypt(
    plaintext: Union[bytes, bytearray, Sequence, np.ndarray],
    key: Union[bytes, bytearray, Sequence, np.ndarray],
    rounds: int = FULL_ROUNDS,
) -> bytes:
    """Encrypt a single 16-byte block under SM4."""
    pts = _as_batch(plaintext, "plaintext")
    if pts.shape[0] != 1:
        raise ValueError(f"Expected single block, got {pts.shape[0]}")
    ks = _as_batch(key, "key")
    if ks.shape[0] != 1:
        raise ValueError(f"Expected single key, got {ks.shape[0]}")
    ct = sm4_encrypt_batch(pts, ks, rounds=rounds)
    return bytes(ct[0])
