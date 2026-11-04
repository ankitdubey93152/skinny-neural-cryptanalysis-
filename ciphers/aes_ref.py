"""AES-128 Reference Implementation using PyCryptodome AES-ECB.

Used as a standard baseline and for cipher identification experiments.
"""

from typing import Union
import numpy as np
from Crypto.Cipher import AES


def aes128_encrypt_batch(
    plaintexts: np.ndarray,
    keys: np.ndarray
) -> np.ndarray:
    """Encrypt N 128-bit blocks using AES-128 in ECB mode.

    Args:
        plaintexts (np.ndarray): Plaintext blocks of shape (N, 16) with dtype uint8.
        keys (np.ndarray): Master keys of shape (N, 16) or (16,) with dtype uint8.

    Returns:
        np.ndarray: Ciphertext blocks of shape (N, 16) with dtype uint8.
    """
    pts = np.ascontiguousarray(plaintexts)
    if pts.ndim != 2 or pts.shape[1] != 16:
        raise ValueError(f"Expected plaintexts of shape (N, 16), got {pts.shape}")

    N = pts.shape[0]
    keys_arr = np.ascontiguousarray(keys)

    if keys_arr.ndim == 1:
        if keys_arr.shape[0] != 16:
            raise ValueError(f"Expected key of length 16, got {keys_arr.shape[0]}")
        cipher = AES.new(bytes(keys_arr), AES.MODE_ECB)
        ct_bytes = cipher.encrypt(pts.tobytes())
        return np.frombuffer(ct_bytes, dtype=np.uint8).reshape((N, 16))

    if keys_arr.ndim == 2:
        if keys_arr.shape[1] != 16:
            raise ValueError(f"Expected keys of second dimension 16, got {keys_arr.shape}")
        if keys_arr.shape[0] == 1 and N > 1:
            cipher = AES.new(bytes(keys_arr[0]), AES.MODE_ECB)
            ct_bytes = cipher.encrypt(pts.tobytes())
            return np.frombuffer(ct_bytes, dtype=np.uint8).reshape((N, 16))
        if keys_arr.shape[0] != N:
            raise ValueError(f"Expected keys length {N}, got {keys_arr.shape[0]}")

        out = np.empty((N, 16), dtype=np.uint8)
        for i in range(N):
            cipher = AES.new(bytes(keys_arr[i]), AES.MODE_ECB)
            out[i] = np.frombuffer(cipher.encrypt(bytes(pts[i])), dtype=np.uint8)
        return out

    raise ValueError(f"Invalid keys dimension: {keys_arr.ndim}")
