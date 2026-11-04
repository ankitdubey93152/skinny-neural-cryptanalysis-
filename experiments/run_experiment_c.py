"""Experiment C: Cipher Identification (SKINNY-128-128 vs AES-128)."""
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
from experiments.exp_utils import parse_rounds_arg, parse_delta_arg, filter_model_configs, plot_accuracy_vs_rounds


def run_experiment_c(rounds: Optional[Union[List[int], str, int]] = None, n_samples: int = 100000, delta: Union[str, bytes] = DEFAULT_DELTA, eval_mode: str = "holdout", epochs: int = 10, models: Optional[List[str]] = None, results_dir: str = "results_skinny", data_dir: str = "data_skinny", quick_mode: bool = False, run_random_control: bool = True, **kwargs: Any) -> pd.DataFrame:
    set_global_seed(DEFAULT_SEED)
    os.makedirs(results_dir, exist_ok=True); os.makedirs(data_dir, exist_ok=True)
    d_bytes = parse_delta_arg(delta)
    r_list = ([1, 2, 4] if rounds is None else parse_rounds_arg(rounds)) if quick_mode else parse_rounds_arg(rounds)
    if quick_mode:
        if n_samples == 100000: n_samples = 20000
        if epochs == 10: epochs = 3

    print(f"EXPERIMENT C: CIPHER IDENTIFICATION (SKINNY-128-128 vs AES-128) | Rounds: {r_list} | N: {n_samples:,} | Epochs: {epochs}")

    all_cfgs = [
        {"name": "1D CNN", "model_type": "cnn_1d", "builder": build_cnn_1d, "kwargs": {"input_shape": (128, 2), "n_dense_layers": 2}, "epochs": epochs, "batch_size": 256},
        {"name": "2D CNN (Dense=2)", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": (2, 128, 1), "n_dense_layers": 2, "kernel_size": 3}, "epochs": epochs, "batch_size": 256},
        {"name": "2D CNN (Dense=3)", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": (2, 128, 1), "n_dense_layers": 3, "kernel_size": 3}, "epochs": epochs, "batch_size": 256},
        {"name": "MLP (Dense=2)", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": 256, "n_dense_layers": 2, "units": 640}, "epochs": epochs, "batch_size": 256},
        {"name": "MLP (Dense=3)", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": 256, "n_dense_layers": 3, "units": 640}, "epochs": epochs, "batch_size": 256},
    ]
    cfgs = filter_model_configs(all_cfgs, models)
    sweep_rows, rnd_ctrl_rows = [], []

    for r in r_list:
        n_val = n_test = int(n_samples * 0.15)
        splits, _, _ = generate_split_dataset("c", rounds=r, n_train=n_samples - 2*n_val, n_val=n_val, n_test=n_test, delta=d_bytes, seed=DEFAULT_SEED, reference="aes128", return_meta=False, run_leakage_check=True)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits["train"], splits["val"], splits["test"]
        xt_b, xv_b, xte_b = bits(X_tr), bits(X_va), bits(X_te)
        r_sums = []

        for c in cfgs:
            x_tr_in, x_va_in, x_te_in = prepare_input(xt_b, c["model_type"], "c"), prepare_input(xv_b, c["model_type"], "c"), prepare_input(xte_b, c["model_type"], "c")
            safe = c["name"].lower().replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
            res = run_holdout(c["builder"], splits={"train": (x_tr_in, y_tr), "val": (x_va_in, y_va), "test": (x_te_in, y_te)}, epochs=c["epochs"], batch_size=c["batch_size"], patience=3, model_name=c["name"], random_state=DEFAULT_SEED, checkpoint_dir=os.path.join(results_dir, f"checkpoints_exp_c_r{r}"), history_path=os.path.join(results_dir, f"history_exp_c_r{r}_{safe}.json"), **c["kwargs"])
            plot_training_curves(res, save_path=os.path.join(results_dir, f"exp_c_r{r}_{safe}_curves.png"), title_prefix=f"[Exp C r={r}]")
            r_sums.append(summarize_run(res))
            sweep_rows.append({"Rounds": r, "Model": c["name"], "Accuracy": round(float(res["test_accuracy"]), 4), "CI": f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]", "p-value": float(res["p_value"]), "AUC": round(float(res["test_auc"]), 4)})

        if run_random_control:
            # Control: SKINNY vs Random Bytes (replace AES-128 samples with random bytes)
            rng = np.random.default_rng(DEFAULT_SEED + r)
            x_te_rnd = x_te_in.copy()
            ref_mask = (y_te == 0)
            x_te_rnd[ref_mask] = rng.integers(0, 2, size=x_te_rnd[ref_mask].shape, dtype=np.uint8).astype(np.float32)
            c1d = cfgs[0]
            m1d = c1d["builder"](**c1d["kwargs"])
            ckpt = os.path.join(results_dir, f"checkpoints_exp_c_r{r}", f"{c1d['name']}_best.weights.h5")
            if os.path.exists(ckpt): m1d.load_weights(ckpt)
            p_rnd = m1d.predict(x_te_rnd, batch_size=256, verbose=0).flatten()
            acc_rnd = float(np.mean((p_rnd >= 0.5).astype(int) == y_te))
            rnd_ctrl_rows.append({"Rounds": r, "Model": "1D CNN", "Acc_vs_AES": sweep_rows[-len(cfgs)]["Accuracy"], "Acc_vs_Random_Bytes": round(acc_rnd, 4)})

        pd.DataFrame(r_sums).to_csv(os.path.join(results_dir, f"table_experiment_c_r{r}.csv"), index=False)

    sweep_df = pd.DataFrame(sweep_rows)[["Rounds", "Model", "Accuracy", "CI", "p-value", "AUC"]]
    for f in ("table_experiment_c_sweep.csv", "table_experiment_c.csv"): sweep_df.to_csv(os.path.join(results_dir, f), index=False)
    plot_accuracy_vs_rounds(sweep_df, save_path=os.path.join(results_dir, "exp_c_accuracy_vs_rounds.png"), exp_name="C")

    if rnd_ctrl_rows:
        pd.DataFrame(rnd_ctrl_rows).to_csv(os.path.join(results_dir, "table_experiment_c_random_control.csv"), index=False)

    return sweep_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", nargs="+", default=None); p.add_argument("--samples", type=int, default=100000); p.add_argument("--delta", type=str, default=DEFAULT_DELTA.hex())
    p.add_argument("--eval", dest="eval_mode", choices=["holdout", "kfold"], default="holdout"); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--models", nargs="+", default=None); p.add_argument("--results-dir", type=str, default="results_skinny"); p.add_argument("--data-dir", type=str, default="data_skinny"); p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run_experiment_c(rounds=a.rounds, n_samples=a.samples, delta=a.delta, eval_mode=a.eval_mode, epochs=a.epochs, models=a.models, results_dir=a.results_dir, data_dir=a.data_dir, quick_mode=a.quick)
