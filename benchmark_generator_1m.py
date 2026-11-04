"""Benchmark dataset generation throughput and memory footprint for 1,000,000 pairs."""

import time
import tracemalloc
import numpy as np

from config import DEFAULT_DELTA
from data.generators import generate_experiment_a


def run_benchmark(n_samples: int = 1_000_000, rounds: int = 4, chunk_size: int = 50_000):
    print(f"Benchmarking generation of {n_samples:,} pairs (rounds={rounds}, chunk_size={chunk_size:,})...")

    tracemalloc.start()
    t0 = time.perf_counter()

    X, y = generate_experiment_a(
        n_samples=n_samples,
        rounds=rounds,
        delta=DEFAULT_DELTA,
        seed=42,
        negative_mode="delta_prime",
        chunk_size=chunk_size,
        return_meta=False,
    )

    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    throughput = n_samples / elapsed
    peak_mb = peak_mem / (1024 * 1024)
    x_mb = X.nbytes / (1024 * 1024)
    y_mb = y.nbytes / (1024 * 1024)

    print("=" * 60)
    print("BENCHMARK REPORT: 1,000,000 PAIRS GENERATION")
    print("=" * 60)
    print(f"Samples generated : {len(y):,}")
    print(f"X shape & dtype   : {X.shape}, {X.dtype} ({x_mb:.2f} MB)")
    print(f"y shape & dtype   : {y.shape}, {y.dtype} ({y_mb:.2f} MB)")
    print(f"Label balance     : Y=1: {np.sum(y == 1):,} | Y=0: {np.sum(y == 0):,}")
    print(f"Elapsed time      : {elapsed:.3f} seconds")
    print(f"Throughput        : {throughput:,.1f} pairs/second")
    print(f"Peak memory traced: {peak_mb:.2f} MB")
    print("=" * 60)

    return elapsed, peak_mb, throughput


if __name__ == "__main__":
    run_benchmark()

