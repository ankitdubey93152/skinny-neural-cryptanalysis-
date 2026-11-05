"""Scalar reference implementation of the HIGHT block cipher for testing cross-checks."""
from __future__ import annotations

_s = [0, 1, 0, 1, 1, 0, 1]
for _i in range(1, 128):
    _s.append(_s[_i + 2] ^ _s[_i - 1])

DELTA = [
    (_s[i + 6] << 6)
    | (_s[i + 5] << 5)
    | (_s[i + 4] << 4)
    | (_s[i + 3] << 3)
    | (_s[i + 2] << 2)
    | (_s[i + 1] << 1)
    | _s[i]
    for i in range(128)
]


def _rol8(x: int, n: int) -> int:
    return ((x << n) | (x >> (8 - n))) & 0xFF


def _F0(x: int) -> int:
    return _rol8(x, 1) ^ _rol8(x, 2) ^ _rol8(x, 7)


def _F1(x: int) -> int:
    return _rol8(x, 3) ^ _rol8(x, 4) ^ _rol8(x, 6)


def hight_key_schedule_scalar(key_bytes: bytes) -> tuple[list[int], list[int]]:
    K = [key_bytes[15 - i] for i in range(16)]
    WK = [0] * 8
    for i in range(4):
        WK[i] = K[i + 12]
    for i in range(4, 8):
        WK[i] = K[i - 4]
    SK = [0] * 128
    for i in range(8):
        for j in range(8):
            SK[16 * i + j] = (K[(j - i) % 8] + DELTA[16 * i + j]) & 0xFF
            SK[16 * i + j + 8] = (K[((j - i) % 8) + 8] + DELTA[16 * i + j + 8]) & 0xFF
    return WK, SK


def hight_encrypt_scalar(pt_bytes: bytes, key_bytes: bytes, rounds: int = 32) -> bytes:
    if rounds == 0:
        return pt_bytes
    WK, SK = hight_key_schedule_scalar(key_bytes)
    P = [pt_bytes[7 - i] for i in range(8)]
    X = [0] * 8
    X[0] = (P[0] + WK[0]) & 0xFF
    X[1] = P[1]
    X[2] = P[2] ^ WK[1]
    X[3] = P[3]
    X[4] = (P[4] + WK[2]) & 0xFF
    X[5] = P[5]
    X[6] = P[6] ^ WK[3]
    X[7] = P[7]

    for i in range(min(rounds, 31)):
        X_next = [0] * 8
        X_next[0] = X[7] ^ ((_F0(X[6]) + SK[4 * i + 3]) & 0xFF)
        X_next[1] = X[0]
        X_next[2] = (X[1] + (_F1(X[0]) ^ SK[4 * i])) & 0xFF
        X_next[3] = X[2]
        X_next[4] = X[3] ^ ((_F0(X[2]) + SK[4 * i + 1]) & 0xFF)
        X_next[5] = X[4]
        X_next[6] = (X[5] + (_F1(X[4]) ^ SK[4 * i + 2])) & 0xFF
        X_next[7] = X[6]
        X = X_next

    if rounds == 32:
        X_final = [0] * 8
        X_final[0] = X[0]
        X_final[1] = (X[1] + (_F1(X[0]) ^ SK[124])) & 0xFF
        X_final[2] = X[2]
        X_final[3] = X[3] ^ ((_F0(X[2]) + SK[125]) & 0xFF)
        X_final[4] = X[4]
        X_final[5] = (X[5] + (_F1(X[4]) ^ SK[126])) & 0xFF
        X_final[6] = X[6]
        X_final[7] = X[7] ^ ((_F0(X[6]) + SK[127]) & 0xFF)

        C = [0] * 8
        C[0] = (X_final[0] + WK[4]) & 0xFF
        C[1] = X_final[1]
        C[2] = X_final[2] ^ WK[5]
        C[3] = X_final[3]
        C[4] = (X_final[4] + WK[6]) & 0xFF
        C[5] = X_final[5]
        C[6] = X_final[6] ^ WK[7]
        C[7] = X_final[7]
        return bytes([C[7 - i] for i in range(8)])

    return bytes([X[7 - i] for i in range(8)])
