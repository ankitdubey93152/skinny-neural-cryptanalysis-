"""Difference Propagation and Strict Avalanche Criterion (SAC) Profiler for SKINNY-128-128."""
import os, sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEFAULT_DELTA, BLOCK_BYTES
from ciphers.skinny128 import skinny128_encrypt_batch
from data.generators import bits

CANDIDATE_DELTAS: Dict[str, bytes] = {
    "delta_lsb": DEFAULT_DELTA,
    "delta_msb": bytes.fromhex("80000000000000000000000000000000"),
    "delta_cell0_byte": bytes.fromhex("01000000000000000000000000000000"),
    "delta_cell8_tweakey_row": bytes.fromhex("00000000000000000100000000000000"),
    "delta_high_weight": bytes.fromhex("ffffffffffffffffffffffffffffffff"),
}

def measure_difference_propagation(candidate_deltas: Optional[Dict[str, bytes]] = None,
                                   max_rounds: int = 8, n_samples: int = 5000,
                                   seed: int = 42, results_dir: Optional[str] = None) -> pd.DataFrame:
    """Measure empirical difference propagation statistics across rounds for candidate deltas."""
    deltas = candidate_deltas or CANDIDATE_DELTAS
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 256, size=(n_samples, BLOCK_BYTES), dtype=np.uint8)
    p1 = rng.integers(0, 256, size=(n_samples, BLOCK_BYTES), dtype=np.uint8)
    rows = []

    for name, d_bytes in deltas.items():
        p2 = p1 ^ np.frombuffer(d_bytes, dtype=np.uint8)[None, :]
        for r in range(1, max_rounds + 1):
            c_diff = skinny128_encrypt_batch(p1, keys, rounds=r) ^ skinny128_encrypt_batch(p2, keys, rounds=r)
            diff_bits = bits(c_diff)
            hw = np.sum(diff_bits, axis=1)
            b_probs = np.mean(diff_bits, axis=0)
            rows.append({
                "Delta Name": name, "Delta Hex": d_bytes.hex(), "Round": r,
                "Active Bytes (/16)": round(float(np.mean(np.sum(c_diff != 0, axis=1))), 2),
                "Inactive Bits (/128)": int(np.sum(np.all(diff_bits == 0, axis=0))),
                "Mean HW (/128)": round(float(np.mean(hw)), 2), "Std HW": round(float(np.std(hw)), 2),
                "Max Bit Bias": round(float(np.max(np.abs(b_probs - 0.5))), 4),
                "Mean SAC Error": round(float(np.mean(np.abs(b_probs - 0.5))), 4),
            })

    df = pd.DataFrame(rows)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "difference_propagation_avalanche.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved difference propagation analysis to: {csv_path}")
    return df

if __name__ == "__main__":
    df_res = measure_difference_propagation(max_rounds=8, n_samples=5000, results_dir="results_skinny")
    print("\nDIFFERENCE PROPAGATION & AVALANCHE REPORT (SKINNY-128-128):\n" + "=" * 80)
    print(df_res[df_res["Delta Name"] == "delta_lsb"].to_string(index=False))
