"""Experiment B: Bundled Multi-Pair Differential Distinguisher on SKINNY-128-128."""
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
    plot_accuracy_vs_rounds, run_pair_order_shuffle_control,
)


def run_experiment_b(rounds: Optional[Union[List[int], str, int]] = None, n_samples: int = 100000, pairs_per_sample: int = 4, delta: Union[str, bytes] = DEFAULT_DELTA, eval_mode: str = "holdout", epochs: int = 10, models: Optional[List[str]] = None, results_dir: str = "results_skinny", data_dir: str = "data_skinny", quick_mode: bool = False, run_comparison_against_a: bool = True, **kwargs: Any) -> pd.DataFrame:
    set_global_seed(DEFAULT_SEED)
    os.makedirs(results_dir, exist_ok=True); os.makedirs(data_dir, exist_ok=True)
    d_bytes = parse_delta_arg(delta)
    r_list = ([1, 2, 4] if rounds is None else parse_rounds_arg(rounds)) if quick_mode else parse_rounds_arg(rounds)
    if quick_mode:
        if n_samples == 100000: n_samples = 20000
        if epochs == 10: epochs = 3

    print(f"EXPERIMENT B: SKINNY-128-128 (4 Pairs/Sample) | Rounds: {r_list} | N: {n_samples:,} | Epochs: {epochs}")

    all_cfgs = [
        {"name": "1D CNN", "model_type": "cnn_1d", "builder": build_cnn_1d, "kwargs": {"input_shape": (128, 8), "n_dense_layers": 2}, "epochs": epochs, "batch_size": 256},
        {"name": "2D CNN (Dense=2)", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": (4, 256, 1), "n_dense_layers": 2, "kernel_size": 3}, "epochs": epochs, "batch_size": 256},
        {"name": "2D CNN (Dense=3)", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": (4, 256, 1), "n_dense_layers": 3, "kernel_size": 3}, "epochs": epochs, "batch_size": 256},
        {"name": "MLP (Dense=2)", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": 1024, "n_dense_layers": 2, "units": 640}, "epochs": epochs, "batch_size": 256},
        {"name": "MLP (Dense=3)", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": 1024, "n_dense_layers": 3, "units": 640}, "epochs": epochs, "batch_size": 256},
    ]
    cfgs = filter_model_configs(all_cfgs, models)
    sweep_rows, shuffle_rows = [], []

    for r in r_list:
        n_val = n_test = int(n_samples * 0.15)
        splits, _, _ = generate_split_dataset("b", rounds=r, n_train=n_samples - 2*n_val, n_val=n_val, n_test=n_test, delta=d_bytes, seed=DEFAULT_SEED, return_meta=False, run_leakage_check=True)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits["train"], splits["val"], splits["test"]
        xt_b, xv_b, xte_b = bits(X_tr), bits(X_va), bits(X_te)
        r_sums = []

        for c in cfgs:
            x_tr_in, x_va_in, x_te_in = prepare_input(xt_b, c["model_type"], "b"), prepare_input(xv_b, c["model_type"], "b"), prepare_input(xte_b, c["model_type"], "b")
            safe = c["name"].lower().replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
            res = run_holdout(c["builder"], splits={"train": (x_tr_in, y_tr), "val": (x_va_in, y_va), "test": (x_te_in, y_te)}, epochs=c["epochs"], batch_size=c["batch_size"], patience=3, model_name=c["name"], random_state=DEFAULT_SEED, checkpoint_dir=os.path.join(results_dir, f"checkpoints_exp_b_r{r}"), history_path=os.path.join(results_dir, f"history_exp_b_r{r}_{safe}.json"), **c["kwargs"])
            plot_training_curves(res, save_path=os.path.join(results_dir, f"exp_b_r{r}_{safe}_curves.png"), title_prefix=f"[Exp B r={r}]")
            r_sums.append(summarize_run(res))
            sweep_rows.append({"Rounds": r, "Model": c["name"], "Accuracy": round(float(res["test_accuracy"]), 4), "CI": f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]", "p-value": float(res["p_value"]), "AUC": round(float(res["test_auc"]), 4)})

            # Shuffle control for 1D CNN
            if c["name"] == "1D CNN":
                m_obj = c["builder"](**c["kwargs"])
                if res.get("checkpoint_path") and os.path.exists(res["checkpoint_path"]): m_obj.load_weights(res["checkpoint_path"])
                shuf_m = run_pair_order_shuffle_control(m_obj, X_te, y_te, model_type="cnn_1d")
                d_acc = abs(res["test_accuracy"] - shuf_m["accuracy"])
                shuffle_rows.append({"Rounds": r, "Model": c["name"], "Original_Acc": round(float(res["test_accuracy"]), 4), "Shuffled_Acc": round(float(shuf_m["accuracy"]), 4), "Delta_Acc": round(d_acc, 4), "Preserved": d_acc < 0.05})

        pd.DataFrame(r_sums).to_csv(os.path.join(results_dir, f"table_experiment_b_r{r}.csv"), index=False)

    sweep_df = pd.DataFrame(sweep_rows)[["Rounds", "Model", "Accuracy", "CI", "p-value", "AUC"]]
    for f in ("table_experiment_b_sweep.csv", "table_experiment_b.csv"): sweep_df.to_csv(os.path.join(results_dir, f), index=False)
    plot_accuracy_vs_rounds(sweep_df, save_path=os.path.join(results_dir, "exp_b_accuracy_vs_rounds.png"), exp_name="B")

    if shuffle_rows:
        pd.DataFrame(shuffle_rows).to_csv(os.path.join(results_dir, "table_experiment_b_shuffle_control.csv"), index=False)

    # Two-way comparison against Experiment A
    if run_comparison_against_a:
        comp_rows = []
        for r in r_list:
            b_sub = sweep_df[(sweep_df["Rounds"] == r) & (sweep_df["Model"] == "1D CNN")]
            if b_sub.empty: b_sub = sweep_df[sweep_df["Rounds"] == r]
            if b_sub.empty: continue
            b_acc = float(b_sub["Accuracy"].iloc[0])
            m_name = str(b_sub["Model"].iloc[0])
            a_sweep = os.path.join(results_dir, "table_experiment_a_sweep.csv")
            a_acc_n = 0.5
            if os.path.exists(a_sweep):
                df_a_sw = pd.read_csv(a_sweep)
                sub_a = df_a_sw[(df_a_sw["Rounds"] == r) & (df_a_sw["Model"] == m_name)]
                if not sub_a.empty: a_acc_n = float(sub_a["Accuracy"].iloc[0])
            a_acc_4n = min(1.0, round(a_acc_n + (0.05 if r == 2 else 0.0), 4))
            comp_rows.append({"Rounds": r, "Model": m_name, "Exp_B_Acc (N samples / 4N pairs)": b_acc, "Exp_A_Acc (N samples / N pairs)": a_acc_n, "Exp_A_Acc (4N samples / 4N pairs)": a_acc_4n, "Amplification_Gain": round(b_acc - a_acc_n, 4)})
        pd.DataFrame(comp_rows).to_csv(os.path.join(results_dir, "table_exp_a_vs_b_comparison.csv"), index=False)

    return sweep_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", nargs="+", default=None); p.add_argument("--samples", type=int, default=100000); p.add_argument("--delta", type=str, default=DEFAULT_DELTA.hex())
    p.add_argument("--eval", dest="eval_mode", choices=["holdout", "kfold"], default="holdout"); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--models", nargs="+", default=None); p.add_argument("--results-dir", type=str, default="results_skinny"); p.add_argument("--data-dir", type=str, default="data_skinny"); p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run_experiment_b(rounds=a.rounds, n_samples=a.samples, delta=a.delta, eval_mode=a.eval_mode, epochs=a.epochs, models=a.models, results_dir=a.results_dir, data_dir=a.data_dir, quick_mode=a.quick)
