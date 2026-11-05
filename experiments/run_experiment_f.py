"""Experiment F: Small-Sample Single-Pair Differential Distinguisher (N=5000 pairs)."""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEFAULT_DELTA, SAMPLE_SIZE_F
from experiments.run_experiment_sample_size import run_sample_size_experiment, update_sample_size_scaling_summary


def run_experiment_f(
    rounds=None, replicates=None, delta=DEFAULT_DELTA, epochs=10,
    models=None, results_dir="results_skinny", quick_mode=False
):
    df = run_sample_size_experiment(
        total_pairs=SAMPLE_SIZE_F, rounds=rounds, replicates=replicates,
        delta=delta, epochs=epochs, models=models, results_dir=results_dir,
        quick_mode=quick_mode
    )
    update_sample_size_scaling_summary(results_dir=results_dir)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Experiment F (N=5000 pairs)")
    p.add_argument("--rounds", nargs="+", default=None)
    p.add_argument("--replicates", type=int, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--delta", type=str, default=DEFAULT_DELTA.hex())
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--results-dir", type=str, default="results_skinny")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run_experiment_f(
        rounds=a.rounds, replicates=a.replicates, delta=a.delta,
        epochs=a.epochs, models=a.models, results_dir=a.results_dir, quick_mode=a.quick
    )
