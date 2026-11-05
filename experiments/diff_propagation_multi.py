"""Empirical Difference Propagation and Avalanche Profiler across Multiple Ciphers.

Compares diffusion speeds of SKINNY-128-128, AES-128, SM4, HIGHT, and DES under
standard input differences to explain neural distinguisher round horizons.
"""
import os, sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    DEFAULT_DELTA, DEFAULT_DELTA_64, CIPHER_METADATA, CIPHERS_COMMON_64
)
from ciphers.skinny128 import skinny128_encrypt_batch
from ciphers.aes_ref import aes128_encrypt_batch
from ciphers.sm4 import sm4_encrypt_batch
from ciphers.hight import hight_encrypt_batch
from ciphers.des import des_encrypt_batch
from data.generators import bits


def measure_multi_cipher_diffusion(
    ciphers: Optional[List[str]] = None,
    max_rounds: int = 8,
    n_samples: int = 5000,
    seed: int = 42,
    results_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Measure empirical bit diffusion and avalanche curves across ciphers."""
    c_list = ciphers or CIPHERS_COMMON_64
    rng = np.random.default_rng(seed)
    rows = []

    for c_name in c_list:
        clean = c_name.lower().replace("-", "").strip()
        meta = CIPHER_METADATA[clean]
        b_bytes = meta["block_bytes"]
        b_bits = b_bytes * 8
        k_bytes = meta["key_bytes"]
        full_r = meta["full_rounds"]

        d_bytes = np.frombuffer(
            DEFAULT_DELTA if b_bytes == 16 else DEFAULT_DELTA_64, dtype=np.uint8
        )
        keys = rng.integers(0, 256, size=(n_samples, k_bytes), dtype=np.uint8)
        p1 = rng.integers(0, 256, size=(n_samples, b_bytes), dtype=np.uint8)
        p2 = p1 ^ d_bytes[None, :]

        rounds_to_test = range(1, min(max_rounds, full_r) + 1)
        # AES-128 reference has fixed 10 rounds; test full round as well
        if clean == "aes128":
            rounds_to_test = [10]

        for r in rounds_to_test:
            if clean == "skinny128":
                c1 = skinny128_encrypt_batch(p1, keys, rounds=r)
                c2 = skinny128_encrypt_batch(p2, keys, rounds=r)
            elif clean == "aes128":
                c1 = aes128_encrypt_batch(p1, keys)
                c2 = aes128_encrypt_batch(p2, keys)
            elif clean == "sm4":
                c1 = sm4_encrypt_batch(p1, keys, rounds=r)
                c2 = sm4_encrypt_batch(p2, keys, rounds=r)
            elif clean == "hight":
                c1 = hight_encrypt_batch(p1, keys, rounds=r)
                c2 = hight_encrypt_batch(p2, keys, rounds=r)
            elif clean == "des":
                c1 = des_encrypt_batch(p1, keys, rounds=r)
                c2 = des_encrypt_batch(p2, keys, rounds=r)
            else:
                continue

            diff = c1 ^ c2
            diff_b = bits(diff)
            hw = np.sum(diff_b, axis=1)
            b_probs = np.mean(diff_b, axis=0)
            active_bytes = float(np.mean(np.sum(diff != 0, axis=1)))
            mean_hw = float(np.mean(hw))
            norm_avalanche = float(mean_hw / b_bits)
            max_bias = float(np.max(np.abs(b_probs - 0.5)))
            full_diff = bool(active_bytes >= (b_bytes - 0.05) and 0.45 <= norm_avalanche <= 0.55 and max_bias < 0.05)

            rows.append({
                "Cipher": meta["name"],
                "Cipher_Key": clean,
                "Block_Bits": b_bits,
                "Round": r,
                "Full_Rounds": full_r,
                "Active_Bytes": round(active_bytes, 2),
                "Total_Bytes": b_bytes,
                "Mean_HW": round(mean_hw, 2),
                "Normalized_Avalanche": round(norm_avalanche, 4),
                "Max_Bit_Bias": round(max_bias, 4),
                "Full_Diffusion_Achieved": full_diff,
            })

    df = pd.DataFrame(rows)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "difference_propagation_multi.csv")
        df.to_csv(csv_path, index=False)
        print(f"[+] Saved multi-cipher diffusion profile to: {csv_path}")
    return df


if __name__ == "__main__":
    measure_multi_cipher_diffusion(results_dir="results_skinny")
