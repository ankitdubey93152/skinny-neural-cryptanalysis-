"""Top-level master script to run Experiments A, B, and C and compile the final report."""
import os, sys, argparse, time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from experiments.run_experiment_a import run_experiment_a
from experiments.run_experiment_b import run_experiment_b
from experiments.run_experiment_c import run_experiment_c
from generate_report import generate_markdown_report
from config import DEFAULT_DELTA

def main():
    p = argparse.ArgumentParser(description="Master Execution Pipeline for SKINNY-128-128 Neural Cryptanalysis")
    p.add_argument("--rounds", nargs="+", default=None, help="Cipher rounds to evaluate")
    p.add_argument("--samples", type=int, default=100000, help="Number of dataset samples per experiment")
    p.add_argument("--delta", type=str, default=None, help="32 hex characters input difference Delta")
    p.add_argument("--eval", dest="eval_mode", choices=["holdout", "kfold"], default="holdout", help="Evaluation method")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--models", nargs="+", default=None, help="Subset of model configs to evaluate")
    p.add_argument("--results-dir", type=str, default="results_skinny", help="Target results directory")
    p.add_argument("--data-dir", type=str, default="data_skinny", help="Directory to cache datasets")
    p.add_argument("--quick", action="store_true", help="Quick execution mode")
    args = p.parse_args()

    delta = args.delta if args.delta is not None else DEFAULT_DELTA
    t0 = time.time()
    print("=" * 80 + f"\nSTARTING FULL REPRODUCIBILITY PIPELINE: SKINNY-128-128 NEURAL CRYPTANALYSIS\n"
          f"Configurations: samples={args.samples}, results_dir='{args.results_dir}', quick_mode={args.quick}\n" + "=" * 80)

    # 1. Run Experiment A (Single-pair differential distinguisher)
    print("\n>>> [1/3] EXECUTING EXPERIMENT A...")
    run_experiment_a(rounds=args.rounds, n_samples=args.samples, delta=delta, eval_mode=args.eval_mode,
                     epochs=args.epochs, models=args.models, results_dir=args.results_dir, data_dir=args.data_dir, quick_mode=args.quick)

    # 2. Run Experiment B (Multi-pair bundled distinguisher)
    print("\n>>> [2/3] EXECUTING EXPERIMENT B...")
    run_experiment_b(rounds=args.rounds, n_samples=args.samples, pairs_per_sample=4, delta=delta, eval_mode=args.eval_mode,
                     epochs=args.epochs, models=args.models, results_dir=args.results_dir, data_dir=args.data_dir, quick_mode=args.quick)

    # 3. Run Experiment C (Cipher identification distinguisher)
    print("\n>>> [3/3] EXECUTING EXPERIMENT C...")
    run_experiment_c(rounds=args.rounds, n_samples=args.samples, delta=delta, eval_mode=args.eval_mode,
                     epochs=args.epochs, models=args.models, results_dir=args.results_dir, data_dir=args.data_dir, quick_mode=args.quick)

    # 4. Generate Synthesis Report
    print("\n>>> [4/4] GENERATING SYNTHESIS REPORT...")
    generate_markdown_report(results_dir=args.results_dir, output_path=os.path.join(args.results_dir, "report.md"))
    print("\n" + "=" * 80 + f"\nFULL PIPELINE EXECUTION COMPLETED IN {time.time() - t0:.1f}s!\n"
          f"CSVs, plots, and markdown report are located in: {args.results_dir}/\n" + "=" * 80)

if __name__ == "__main__":
    main()
