"""Experiment A: Single-Pair Differential Distinguisher on SKINNY-128-128."""
import os, sys, argparse
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEFAULT_ROUND_SWEEP, DEFAULT_DELTA, DEFAULT_SEED
from utils.seeding import set_global_seed
from data.generators import generate_split_dataset, bits
from models.architectures import build_mlp, build_cnn_1d, build_cnn_2d, prepare_input
from training.train_eval import run_holdout, summarize_run, plot_training_curves
from experiments.exp_utils import (
    parse_rounds_arg, parse_delta_arg, filter_model_configs,
    plot_accuracy_vs_rounds, check_positive_control_gate, run_non_neural_baselines,
)
from experiments.diff_propagation import measure_difference_propagation


def run_experiment_a(rounds: Optional[Union[List[int], str, int]] = None, n_samples: int = 100000, delta: Union[str, bytes] = DEFAULT_DELTA, eval_mode: str = "holdout", epochs: int = 10, models: Optional[List[str]] = None, results_dir: str = "results_skinny", data_dir: str = "data_skinny", quick_mode: bool = False, **kwargs: Any) -> pd.DataFrame:
    set_global_seed(DEFAULT_SEED)
    os.makedirs(results_dir, exist_ok=True); os.makedirs(data_dir, exist_ok=True)
    d_bytes = parse_delta_arg(delta)
    r_list = ([1, 2, 4] if rounds is None else parse_rounds_arg(rounds)) if quick_mode else parse_rounds_arg(rounds)
    if quick_mode:
        if n_samples == 100000: n_samples = 20000
        if epochs == 10: epochs = 3

    print(f"EXPERIMENT A: SKINNY-128-128 | Rounds: {r_list} | N: {n_samples:,} | Epochs: {epochs} | Delta: {d_bytes.hex()}")

    av_csv = os.path.join(results_dir, "difference_propagation_avalanche.csv")
    if not os.path.exists(av_csv):
        measure_difference_propagation(max_rounds=8, n_samples=5000, results_dir=results_dir)

    all_cfgs = [
        {"name": "1D CNN", "model_type": "cnn_1d", "builder": build_cnn_1d, "kwargs": {"input_shape": (128, 2), "n_dense_layers": 2}, "epochs": epochs, "batch_size": 256},
        {"name": "2D CNN (Dense=2)", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": (2, 128, 1), "n_dense_layers": 2, "kernel_size": 3}, "epochs": epochs, "batch_size": 256},
        {"name": "2D CNN (Dense=3)", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": (2, 128, 1), "n_dense_layers": 3, "kernel_size": 3}, "epochs": epochs, "batch_size": 256},
        {"name": "MLP (Dense=2)", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": 256, "n_dense_layers": 2, "units": 640}, "epochs": epochs, "batch_size": 256},
        {"name": "MLP (Dense=3)", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": 256, "n_dense_layers": 3, "units": 640}, "epochs": epochs, "batch_size": 256},
    ]
    cfgs = filter_model_configs(all_cfgs, models)
    sweep_rows = []

    for r in r_list:
        n_val = n_test = int(n_samples * 0.15)
        splits, _, _ = generate_split_dataset("a", rounds=r, n_train=n_samples - 2*n_val, n_val=n_val, n_test=n_test, delta=d_bytes, seed=DEFAULT_SEED, return_meta=False, run_leakage_check=True)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits["train"], splits["val"], splits["test"]

        if models is None or "baseline" in str(models).lower():
            for b_name, bm in run_non_neural_baselines(X_tr, y_tr, X_te, y_te).items():
                sweep_rows.append({"Rounds": r, "Model": b_name, "Accuracy": round(float(bm["accuracy"]), 4), "CI": f"[{bm['ci_low']:.4f}, {bm['ci_high']:.4f}]", "p-value": float(bm["p_value"]), "AUC": round(float(bm["roc_auc"]), 4)})

        xt_b, xv_b, xte_b = bits(X_tr), bits(X_va), bits(X_te)
        r_sums = []
        for c in cfgs:
            x_tr_in, x_va_in, x_te_in = prepare_input(xt_b, c["model_type"], "a"), prepare_input(xv_b, c["model_type"], "a"), prepare_input(xte_b, c["model_type"], "a")
            safe = c["name"].lower().replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
            res = run_holdout(c["builder"], splits={"train": (x_tr_in, y_tr), "val": (x_va_in, y_va), "test": (x_te_in, y_te)}, epochs=c["epochs"], batch_size=c["batch_size"], patience=3, model_name=c["name"], random_state=DEFAULT_SEED, checkpoint_dir=os.path.join(results_dir, f"checkpoints_exp_a_r{r}"), history_path=os.path.join(results_dir, f"history_exp_a_r{r}_{safe}.json"), **c["kwargs"])
            plot_training_curves(res, save_path=os.path.join(results_dir, f"exp_a_r{r}_{safe}_curves.png"), title_prefix=f"[Exp A r={r}]")
            r_sums.append(summarize_run(res))
            sweep_rows.append({"Rounds": r, "Model": c["name"], "Accuracy": round(float(res["test_accuracy"]), 4), "CI": f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]", "p-value": float(res["p_value"]), "AUC": round(float(res["test_auc"]), 4)})

        pd.DataFrame(r_sums).to_csv(os.path.join(results_dir, f"table_experiment_a_r{r}.csv"), index=False)

    sweep_df = pd.DataFrame(sweep_rows)[["Rounds", "Model", "Accuracy", "CI", "p-value", "AUC"]]
    for f in ("table_experiment_a_sweep.csv", "table_experiment_a.csv"): sweep_df.to_csv(os.path.join(results_dir, f), index=False)
    neural_df = sweep_df[~sweep_df["Model"].str.contains("Logistic|Bias", case=False, na=False)]
    plot_accuracy_vs_rounds(neural_df if not neural_df.empty else sweep_df, save_path=os.path.join(results_dir, "exp_a_accuracy_vs_rounds.png"), exp_name="A")
    check_positive_control_gate(neural_df if not neural_df.empty else sweep_df)
    return sweep_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", nargs="+", default=None); p.add_argument("--samples", type=int, default=100000); p.add_argument("--delta", type=str, default=DEFAULT_DELTA.hex())
    p.add_argument("--eval", dest="eval_mode", choices=["holdout", "kfold"], default="holdout"); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--models", nargs="+", default=None); p.add_argument("--results-dir", type=str, default="results_skinny"); p.add_argument("--data-dir", type=str, default="data_skinny"); p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run_experiment_a(rounds=a.rounds, n_samples=a.samples, delta=a.delta, eval_mode=a.eval_mode, epochs=a.epochs, models=a.models, results_dir=a.results_dir, data_dir=a.data_dir, quick_mode=a.quick)
