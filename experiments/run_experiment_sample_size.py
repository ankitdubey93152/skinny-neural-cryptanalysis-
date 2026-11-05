"""Core engine for small-sample neural distinguisher experiments (D: N=50, E: N=500, F: N=5000)."""
import os, sys, argparse, time
from typing import List, Dict, Any, Optional, Union, Tuple, Sequence
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    DEFAULT_DELTA, DEFAULT_SEED, SAMPLE_SIZE_D, SAMPLE_SIZE_E, SAMPLE_SIZE_F,
    DEFAULT_REPLICATES
)
from utils.seeding import set_global_seed
from data.generators import generate_split_dataset, bits
from models.architectures import build_mlp, build_cnn_1d, build_cnn_2d, prepare_input
from training.train_eval import accuracy_significance, compute_full_metrics, run_holdout
from experiments.exp_utils import (
    parse_rounds_arg, parse_delta_arg, filter_model_configs,
    run_non_neural_baselines
)


def holm_bonferroni(p_vals: Sequence[float]) -> np.ndarray:
    """Apply Holm-Bonferroni family-wise error rate correction."""
    p = np.asarray(p_vals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([])
    order = np.argsort(p)
    sorted_p = p[order]
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        val = min(1.0, float((m - i) * sorted_p[i]))
        running_max = max(running_max, val)
        adj[i] = running_max
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def compute_power_analysis(n_total: int, n_test: int, alpha: float = 0.05, power_target: float = 0.80) -> Dict[str, Any]:
    """Analytical power calculation for one-sample binomial test vs p0=0.5."""
    z_alpha = 1.95996  # two-sided 0.05
    z_beta = 0.84162   # 80% power
    sigma = 0.5 / np.sqrt(n_test)
    crit_acc = 0.5 + z_alpha * sigma
    det_acc = 0.5 + (z_alpha + z_beta) * sigma
    return {
        "total_pairs_N": n_total,
        "n_test_single": n_test,
        "alpha": alpha,
        "target_power": power_target,
        "min_acc_sig_p05": round(float(crit_acc), 4),
        "min_detectable_acc_80power": round(float(det_acc), 4),
    }


def run_sample_size_experiment(
    total_pairs: int,
    rounds: Optional[Union[List[int], str, int]] = None,
    replicates: Optional[int] = None,
    split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    delta: Union[str, bytes] = DEFAULT_DELTA,
    epochs: int = 10,
    models: Optional[List[str]] = None,
    results_dir: str = "results_skinny",
    quick_mode: bool = False,
    run_null_control: bool = True,
) -> pd.DataFrame:
    """Execute small-sample differential distinguisher experiment across multiple replicates."""
    os.makedirs(results_dir, exist_ok=True)
    d_bytes = parse_delta_arg(delta)
    r_list = ([1, 2, 4] if rounds is None else parse_rounds_arg(rounds)) if quick_mode else ([1, 2, 3, 4, 5] if rounds is None else parse_rounds_arg(rounds))

    if replicates is None:
        if quick_mode:
            replicates = 5 if total_pairs <= 50 else (3 if total_pairs <= 500 else 2)
        else:
            replicates = DEFAULT_REPLICATES.get(total_pairs, 10)

    if quick_mode and epochs > 3:
        epochs = 3

    # Stratified split sizes
    tr_r, va_r, te_r = split_ratio
    n_tr = int(round(total_pairs * tr_r))
    n_va = int(round(total_pairs * va_r))
    n_te = total_pairs - n_tr - n_va
    # Ensure even numbers for balance
    n_tr = (n_tr // 2) * 2
    n_va = (n_va // 2) * 2
    n_te = total_pairs - n_tr - n_va
    if n_te % 2 != 0:
        n_te += 1
        n_tr -= 1

    adaptive_batch = min(256, max(8, n_tr // 4))

    exp_letter = "d" if total_pairs <= 50 else ("e" if total_pairs <= 500 else "f")
    print(f"\n=======================================================")
    print(f"EXPERIMENT {exp_letter.upper()}: N={total_pairs} pairs | Split: {n_tr}/{n_va}/{n_te} | Replicates: {replicates} | Batch: {adaptive_batch}")
    print(f"Rounds: {r_list} | Epochs: {epochs} | Quick: {quick_mode}")
    print(f"=======================================================")

    all_cfgs = [
        {"name": "1D CNN", "model_type": "cnn_1d", "builder": build_cnn_1d, "kwargs": {"input_shape": (128, 2), "n_dense_layers": 2}},
        {"name": "2D CNN (Dense=2)", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": (2, 128, 1), "n_dense_layers": 2, "kernel_size": 3}},
        {"name": "2D CNN (Dense=3)", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": (2, 128, 1), "n_dense_layers": 3, "kernel_size": 3}},
        {"name": "MLP (Dense=2)", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": 256, "n_dense_layers": 2, "units": 640}},
        {"name": "MLP (Dense=3)", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": 256, "n_dense_layers": 3, "units": 640}},
    ]
    cfgs = filter_model_configs(all_cfgs, models)

    rows = []
    p_values_to_correct = []
    indices_for_correction = []

    for r in r_list:
        print(f"\n--- Round {r} ---")
        for c in cfgs:
            rep_test_accs = []
            rep_train_accs = []
            pooled_correct = 0
            pooled_total = 0

            for rep in range(replicates):
                rep_seed = DEFAULT_SEED + rep * 1000 + r * 17
                splits, _, _ = generate_split_dataset(
                    "a", rounds=r, n_train=n_tr, n_val=n_va, n_test=n_te,
                    delta=d_bytes, seed=rep_seed, return_meta=False, run_leakage_check=True
                )
                (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits["train"], splits["val"], splits["test"]

                xt_b, xv_b, xte_b = bits(X_tr), bits(X_va), bits(X_te)
                x_tr_in = prepare_input(xt_b, c["model_type"], "a")
                x_va_in = prepare_input(xv_b, c["model_type"], "a")
                x_te_in = prepare_input(xte_b, c["model_type"], "a")

                tf_seed = rep_seed + 1
                res = run_holdout(
                    c["builder"],
                    splits={"train": (x_tr_in, y_tr), "val": (x_va_in, y_va), "test": (x_te_in, y_te)},
                    epochs=epochs,
                    batch_size=adaptive_batch,
                    patience=2,
                    model_name=c["name"],
                    random_state=tf_seed,
                    verbose=0,
                    **c["kwargs"]
                )

                rep_test_accs.append(res["test_accuracy"])
                h = res.get("history", {})
                tr_acc = h.get("accuracy", [0.5])[-1] if "accuracy" in h else 0.5
                rep_train_accs.append(tr_acc)
                pooled_correct += res["n_correct"]
                pooled_total += res["n_test"]

            # Compute pooled statistics
            pooled_acc, pooled_ci, pooled_pval = accuracy_significance(pooled_correct, pooled_total)
            m_test = float(np.mean(rep_test_accs))
            s_test = float(np.std(rep_test_accs, ddof=1)) if replicates > 1 else 0.0
            med_test = float(np.median(rep_test_accs))
            min_test = float(np.min(rep_test_accs))
            max_test = float(np.max(rep_test_accs))
            m_train = float(np.mean(rep_train_accs))
            overfit_gap = float(m_train - m_test)

            indices_for_correction.append(len(rows))
            p_values_to_correct.append(pooled_pval)

            row = {
                "Total_N": total_pairs,
                "Rounds": r,
                "Model": c["name"],
                "Replicates": replicates,
                "Batch_Size": adaptive_batch,
                "Mean_Acc": round(m_test, 4),
                "Std_Acc": round(s_test, 4),
                "Median_Acc": round(med_test, 4),
                "Min_Acc": round(min_test, 4),
                "Max_Acc": round(max_test, 4),
                "Train_Acc": round(m_train, 4),
                "Overfit_Gap": round(overfit_gap, 4),
                "Pooled_Correct": pooled_correct,
                "Pooled_Total": pooled_total,
                "Pooled_Acc": round(pooled_acc, 4),
                "CI_Low": round(float(pooled_ci.low), 4),
                "CI_High": round(float(pooled_ci.high), 4),
                "p-value_raw": float(pooled_pval),
                "p-value_holm": float(pooled_pval),  # updated below
            }
            rows.append(row)

        # Baseline evaluation at this N and round
        splits_b, _, _ = generate_split_dataset(
            "a", rounds=r, n_train=n_tr, n_val=n_va, n_test=n_te,
            delta=d_bytes, seed=DEFAULT_SEED, return_meta=False, run_leakage_check=True
        )
        (X_tr_b, y_tr_b), _, (X_te_b, y_te_b) = splits_b["train"], splits_b["val"], splits_b["test"]
        baselines = run_non_neural_baselines(X_tr_b, y_tr_b, X_te_b, y_te_b)
        for b_name, bm in baselines.items():
            rows.append({
                "Total_N": total_pairs,
                "Rounds": r,
                "Model": b_name,
                "Replicates": 1,
                "Batch_Size": adaptive_batch,
                "Mean_Acc": round(float(bm["accuracy"]), 4),
                "Std_Acc": 0.0,
                "Median_Acc": round(float(bm["accuracy"]), 4),
                "Min_Acc": round(float(bm["accuracy"]), 4),
                "Max_Acc": round(float(bm["accuracy"]), 4),
                "Train_Acc": round(float(bm["accuracy"]), 4),
                "Overfit_Gap": 0.0,
                "Pooled_Correct": int(bm["n_correct"]),
                "Pooled_Total": int(bm["n_samples"]),
                "Pooled_Acc": round(float(bm["accuracy"]), 4),
                "CI_Low": round(float(bm["ci_low"]), 4),
                "CI_High": round(float(bm["ci_high"]), 4),
                "p-value_raw": float(bm["p_value"]),
                "p-value_holm": float(bm["p_value"]),
            })

    # Apply Holm-Bonferroni across neural models in this experiment
    if p_values_to_correct:
        adj_p = holm_bonferroni(p_values_to_correct)
        for idx, p_adj in zip(indices_for_correction, adj_p):
            rows[idx]["p-value_holm"] = round(float(p_adj), 6)

    res_df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, f"table_experiment_{exp_letter}_n{total_pairs}_sweep.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved Experiment {exp_letter.upper()} results to {out_csv}")

    # Power analysis CSV
    pa = compute_power_analysis(total_pairs, n_te)
    pa_pooled = compute_power_analysis(total_pairs, n_te * replicates)
    pa_rows = [
        {"Scope": "Single Replicate Test Set", **pa},
        {"Scope": f"Pooled Test Set ({replicates} Reps)", **pa_pooled},
    ]
    pa_df = pd.DataFrame(pa_rows)
    pa_csv = os.path.join(results_dir, f"table_sample_size_power_n{total_pairs}.csv")
    pa_df.to_csv(pa_csv, index=False)

    # Label-shuffled empirical null control
    if run_null_control:
        null_rows = []
        c = cfgs[0]  # use 1D CNN
        for r in r_list[:2]:
            null_accs = []
            for rep in range(min(5, replicates)):
                s, _, _ = generate_split_dataset("a", rounds=r, n_train=n_tr, n_val=n_va, n_test=n_te, delta=d_bytes, seed=DEFAULT_SEED + rep, return_meta=False)
                (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = s["train"], s["val"], s["test"]
                y_tr_shuf = np.random.default_rng(DEFAULT_SEED + rep).permutation(y_tr)
                x_tr_in = prepare_input(bits(X_tr), c["model_type"], "a")
                x_va_in = prepare_input(bits(X_va), c["model_type"], "a")
                x_te_in = prepare_input(bits(X_te), c["model_type"], "a")
                res_null = run_holdout(c["builder"], splits={"train": (x_tr_in, y_tr_shuf), "val": (x_va_in, y_va), "test": (x_te_in, y_te)}, epochs=epochs, batch_size=adaptive_batch, patience=1, random_state=DEFAULT_SEED + rep, verbose=0, **c["kwargs"])
                null_accs.append(res_null["test_accuracy"])
            null_rows.append({
                "Total_N": total_pairs, "Rounds": r, "Model": c["name"],
                "Control": "Label-Shuffled Empirical Null",
                "Mean_Acc": round(float(np.mean(null_accs)), 4),
                "Std_Acc": round(float(np.std(null_accs, ddof=1)) if len(null_accs) > 1 else 0.0, 4),
            })
        null_df = pd.DataFrame(null_rows)
        null_csv = os.path.join(results_dir, f"table_sample_size_null_n{total_pairs}.csv")
        null_df.to_csv(null_csv, index=False)

    return res_df


def update_sample_size_scaling_summary(results_dir: str = "results_skinny") -> Optional[pd.DataFrame]:
    """Generate scaling summary table across N=50, 500, 5000 and Exp A full-size if present."""
    records = []
    sizes = [("D", 50), ("E", 500), ("F", 5000)]
    for let, sz in sizes:
        f = os.path.join(results_dir, f"table_experiment_{let.lower()}_n{sz}_sweep.csv")
        if os.path.exists(f):
            df = pd.read_csv(f)
            neural = df[~df["Model"].str.contains("Logistic|Bias", case=False, na=False)]
            for r in neural["Rounds"].unique():
                sub = neural[neural["Rounds"] == r]
                best = sub.sort_values("Pooled_Acc", ascending=False).iloc[0]
                records.append({
                    "Experiment": f"Exp_{let}",
                    "Total_N": sz,
                    "Rounds": r,
                    "Best_Model": best["Model"],
                    "Pooled_Acc": best["Pooled_Acc"],
                    "CI_Low": best["CI_Low"],
                    "CI_High": best["CI_High"],
                    "p-value_holm": best.get("p-value_holm", best.get("p-value_raw", 1.0)),
                    "Overfit_Gap": best["Overfit_Gap"],
                })

    # Add Exp A if exists
    exp_a_csv = os.path.join(results_dir, "table_experiment_a_sweep.csv")
    if os.path.exists(exp_a_csv):
        df_a = pd.read_csv(exp_a_csv)
        neural_a = df_a[~df_a["Model"].str.contains("Logistic|Bias", case=False, na=False)]
        for r in neural_a["Rounds"].unique():
            sub = neural_a[neural_a["Rounds"] == r]
            best = sub.sort_values("Accuracy", ascending=False).iloc[0]
            records.append({
                "Experiment": "Exp_A",
                "Total_N": 100000,
                "Rounds": r,
                "Best_Model": best["Model"],
                "Pooled_Acc": best["Accuracy"],
                "CI_Low": float(best["CI"].split(",")[0].replace("[", "").strip()),
                "CI_High": float(best["CI"].split(",")[1].replace("]", "").strip()),
                "p-value_holm": best["p-value"],
                "Overfit_Gap": 0.0,
            })

    if records:
        sum_df = pd.DataFrame(records)
        sum_csv = os.path.join(results_dir, "table_sample_size_scaling_summary.csv")
        sum_df.to_csv(sum_csv, index=False)
        print(f"[+] Updated scaling summary table at {sum_csv}")
        return sum_df
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total-pairs", type=int, default=50)
    p.add_argument("--rounds", nargs="+", default=None)
    p.add_argument("--replicates", type=int, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--delta", type=str, default=DEFAULT_DELTA.hex())
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--results-dir", type=str, default="results_skinny")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run_sample_size_experiment(
        total_pairs=a.total_pairs, rounds=a.rounds, replicates=a.replicates,
        delta=a.delta, epochs=a.epochs, models=a.models, results_dir=a.results_dir,
        quick_mode=a.quick
    )
    update_sample_size_scaling_summary(results_dir=a.results_dir)
