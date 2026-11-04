"""Benchmark the vectorized NumPy implementation of SKINNY-128-128."""
import time
from typing import Dict, Any, List
import numpy as np
from ciphers.skinny128 import skinny128_encrypt_batch, FULL_ROUNDS, BLOCK_BYTES

def benchmark_configuration(n_blocks: int = 50000, rounds: int = FULL_ROUNDS, warmup: int = 1, trials: int = 3) -> Dict[str, Any]:
    """Benchmark SKINNY-128-128 encryption for a specific batch size and round depth."""
    rng = np.random.default_rng(42)
    pt = rng.integers(0, 256, size=(n_blocks, BLOCK_BYTES), dtype=np.uint8)
    tk = rng.integers(0, 256, size=(n_blocks, BLOCK_BYTES), dtype=np.uint8)
    for _ in range(warmup): skinny128_encrypt_batch(pt, tk, rounds=rounds)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        skinny128_encrypt_batch(pt, tk, rounds=rounds)
        times.append(time.perf_counter() - t0)
    avg_t = float(np.mean(times))
    return {
        "n_blocks": n_blocks, "rounds": rounds, "avg_time_sec": avg_t, "std_time_sec": float(np.std(times)),
        "blocks_per_sec": n_blocks / avg_t, "mb_per_sec": (n_blocks * BLOCK_BYTES) / (avg_t * 1024 * 1024),
        "time_100k_pairs_sec": (200000 / n_blocks) * avg_t,
    }

def run_full_benchmark():
    """Execute complete benchmark suite across round sweep and batch sizes."""
    print("=" * 85 + "\nSKINNY-128-128 ENCRYPTION THROUGHPUT BENCHMARK (Pure Vectorized NumPy)\n" + "=" * 85)
    print(f"\n[1] Round Scaling Benchmark (Batch Size: 50,000 blocks, 3 trials):\n{'Rounds':<8} | {'Avg Time (s)':<14} | {'Throughput (blocks/s)':<22} | {'Speed (MB/s)':<14} | {'100k Pairs Est.':<16}\n" + "-" * 85)
    for r in [1, 2, 4, 8, 16, 20, 32, 40]:
        res = benchmark_configuration(n_blocks=50000, rounds=r)
        print(f"{r:<8} | {res['avg_time_sec']:<14.4f} | {res['blocks_per_sec']:<22.1f} | {res['mb_per_sec']:<14.2f} | {res['time_100k_pairs_sec']:<16.2f}s")
    print("\n[2] Batch Size Scaling Benchmark (Full 40 Rounds):\n{'Batch Size':<12} | {'Avg Time (s)':<14} | {'Throughput (blocks/s)':<22} | {'Speed (MB/s)':<14}\n" + "-" * 70)
    for b in [1000, 10000, 50000, 100000]:
        res = benchmark_configuration(n_blocks=b, rounds=FULL_ROUNDS)
        print(f"{b:<12,d} | {res['avg_time_sec']:<14.4f} | {res['blocks_per_sec']:<22.1f} | {res['mb_per_sec']:<14.2f}")

if __name__ == "__main__":
    run_full_benchmark()
