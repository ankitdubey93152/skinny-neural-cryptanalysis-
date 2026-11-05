"""Scalar reference implementation of the DES block cipher for testing cross-checks."""
from __future__ import annotations
from ciphers.des import (
    IP, FP, PC1, PC2, SHIFTS, E, P, S_BOXES,
    _bytes_to_bits, _bits_to_bytes, _permute
)


def des_key_schedule_scalar(key_bytes: bytes) -> list[list[int]]:
    key_bits = _bytes_to_bits(key_bytes)
    permuted = _permute(key_bits, PC1)
    c, d = permuted[:28], permuted[28:]
    subkeys = []
    for shift in SHIFTS:
        c = c[shift:] + c[:shift]
        d = d[shift:] + d[:shift]
        subkeys.append(_permute(c + d, PC2))
    return subkeys


def _feistel_f_scalar(r: list[int], subkey: list[int]) -> list[int]:
    expanded = _permute(r, E)
    xored = [a ^ b for a, b in zip(expanded, subkey)]
    s_output = []
    for i in range(8):
        block = xored[i * 6 : (i + 1) * 6]
        row = (block[0] << 1) | block[5]
        col = (block[1] << 3) | (block[2] << 2) | (block[3] << 1) | block[4]
        val = S_BOXES[i][row * 16 + col]
        for shift in (3, 2, 1, 0):
            s_output.append((val >> shift) & 1)
    return _permute(s_output, P)


def des_encrypt_scalar(pt_bytes: bytes, key_bytes: bytes, rounds: int = 16) -> bytes:
    if rounds == 0:
        return pt_bytes
    subkeys = des_key_schedule_scalar(key_bytes)
    pt_bits = _bytes_to_bits(pt_bytes)
    perm = _permute(pt_bits, IP)
    left = perm[:32]
    right = perm[32:]
    for r in range(rounds):
        f = _feistel_f_scalar(right, subkeys[r])
        new_right = [l ^ f_bit for l, f_bit in zip(left, f)]
        left, right = right, new_right
    combined = right + left
    out_bits = _permute(combined, FP)
    return _bits_to_bytes(out_bits)
