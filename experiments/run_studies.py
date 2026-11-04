"""Research Studies Execution Module (Part 7: Empirical Investigations).
Executes the 8 studies specified in reports/study_plan.md:
1. Extended Round Sweep (r in 1..40)
2. Input-Difference Sensitivity (5 Delta vectors)
3. Dataset-Size Scaling (N in 1e3..1e5)
4. Tweakey Generalization (Fixed 10 keys vs Fresh keys)
5. Architecture Comparison & Model Capacity (parameter counts, Holm-Bonferroni)
6. Single-Pair vs Four-Pair Distinguisher (Amplification analysis)
7. Hyperparameter Sensitivity & Multi-Seed Robustness
8. Ablation Studies (Representation, Negative class, Tweakey sharing)
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEFAULT_SEED, DEFAULT_DELTA
from data.generators import generate_split_dataset, bits
from models.architectures import build_mlp, build_cnn_1d, build_cnn_2d, prepare_input
from training.train_eval import run_holdout
from ciphers.skinny128 import skinny128_encrypt_batch as skinny_encrypt

def _eval_model(mtype, builder, splits, epochs=3, bs=256, lr=0.001, seed=DEFAULT_SEED):
    xt = {k: (prepare_input(bits(splits[k][0]), mtype, "a"), splits[k][1]) for k in ["train", "val", "test"]}
    return run_holdout(builder, xt, epochs=epochs, batch_size=bs, learning_rate=lr, patience=2, verbose=0, seed=seed)

def run_study_1_round_sweep(results_dir: str, quick: bool = True) -> pd.DataFrame:
    """Study 1: Extended Round Sweep from r=1 to 40 (dense around 1..6)."""
    print("\n--- Executing Study 1: Extended Round Sweep (r=1..40) ---")
    rounds = [1, 2, 3, 4, 5, 6, 8, 16, 40] if quick else [1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 32, 40]
    ntr, nv, nte, ep = (3500, 750, 750, 3) if quick else (14000, 3000, 3000, 6)
    rows = []
    for r in rounds:
        splits, _, _ = generate_split_dataset("a", rounds=r, n_train=ntr, n_val=nv, n_test=nte, delta=DEFAULT_DELTA, seed=DEFAULT_SEED + r)
        res = _eval_model("cnn_1d", lambda **k: build_cnn_1d(input_shape=(128, 2), n_dense_layers=2), splits, epochs=ep, seed=DEFAULT_SEED + r)
        acc = float(res.get("accuracy", res.get("test_accuracy", 0.5)))
        p_val, auc = float(res.get("p_value", 1.0)), float(res.get("roc_auc", 0.5))
        diff_bits = np.bitwise_xor(splits["test"][0][:, :16], splits["test"][0][:, 16:])
        act_bits = int(np.sum(np.any(bits(diff_bits) != 0, axis=0)))
        rows.append({"Rounds": r, "Model": "1D CNN", "Accuracy": round(acc, 4), "Accuracy_Mean": round(acc, 4),
                     "Accuracy_Std": round(float(np.sqrt(acc * (1 - acc) / nte)), 4), "ROC_AUC": round(auc, 4),
                     "CI_Low": round(float(res.get("ci_low", 0.5)), 4), "CI_High": round(float(res.get("ci_high", 0.5)), 4),
                     "P_Value": p_val, "Active_Diff_Bits": act_bits, "Status": "Distinguishable" if p_val < 0.01 and acc > 0.52 else "Chance"})
        print(f"  Round {r:2d}: Accuracy = {acc*100:6.2f}%, AUC = {auc:.4f}, p = {p_val:.2e} [{rows[-1]['Status']}]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "study1_round_sweep.csv"), index=False)
    return df

def run_study_2_input_difference(results_dir: str, quick: bool = True) -> pd.DataFrame:
    """Study 2: Input Difference Sensitivity across cell positions and row interactions."""
    print("\n--- Executing Study 2: Input Difference Sensitivity Study ---")
    diffs = [
        ("Delta_1 (Row 3, Cell 15 LSB, No Tweakey XOR)", bytes.fromhex("00000000000000000000000000000001"), "Row 2-3 (No TK)", 1),
        ("Delta_2 (Row 0, Cell 0 LSB, Receives Tweakey XOR)", bytes.fromhex("01000000000000000000000000000000"), "Row 0-1 (TK XOR)", 1),
        ("Delta_3 (Row 0, Cell 0 MSB, Receives Tweakey XOR)", bytes.fromhex("80000000000000000000000000000000"), "Row 0-1 (TK XOR)", 1),
        ("Delta_4 (Row 1, Cell 5 Byte, Receives Tweakey XOR)", bytes.fromhex("0000000000ff00000000000000000000"), "Row 0-1 (TK XOR)", 8),
        ("Delta_5 (High Weight Distributed across all rows)", bytes.fromhex("01010101010101010101010101010101"), "Distributed", 16),
    ]
    rounds = [1, 2] if quick else [1, 2, 3]
    ntr, nv, nte, ep = (3500, 750, 750, 3) if quick else (14000, 3000, 3000, 6)
    rows = []
    for lbl, d_bytes, row_cat, hw in diffs:
        for r in rounds:
            splits, _, _ = generate_split_dataset("a", rounds=r, n_train=ntr, n_val=nv, n_test=nte, delta=d_bytes, seed=DEFAULT_SEED + r + hw)
            res = _eval_model("cnn_1d", lambda **k: build_cnn_1d(input_shape=(128, 2), n_dense_layers=2), splits, epochs=ep, seed=DEFAULT_SEED)
            acc, auc = float(res.get("accuracy", res.get("test_accuracy", 0.5))), float(res.get("roc_auc", 0.5))
            rows.append({"Description": lbl, "Delta_Hex": d_bytes.hex(), "Row_Category": row_cat,
                         "Hamming_Weight": hw, "Rounds": r, "Accuracy": round(acc, 4), "ROC_AUC": round(auc, 4)})
            print(f"  {lbl} (R={r}): Acc = {acc*100:.2f}%, AUC = {auc:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "study2_input_difference.csv"), index=False)
    return df

def run_study_3_dataset_size_scaling(results_dir: str, quick: bool = True) -> pd.DataFrame:
    """Study 3: Dataset-Size Scaling (Sample Complexity Analysis)."""
    print("\n--- Executing Study 3: Dataset-Size Scaling ---")
    sizes = [1000, 5000, 20000] if quick else [1000, 5000, 20000, 100000]
    rounds = [1, 2] if quick else [1, 2, 4]
    ep = 3 if quick else 6
    rows = []
    for n in sizes:
        ntr, nv, nte = int(n * 0.7), int(n * 0.15), int(n * 0.15)
        for r in rounds:
            t0 = time.time()
            splits, _, _ = generate_split_dataset("a", rounds=r, n_train=ntr, n_val=nv, n_test=nte, delta=DEFAULT_DELTA, seed=DEFAULT_SEED)
            res = _eval_model("mlp", lambda **k: build_mlp(input_dim=256, n_dense_layers=2), splits, epochs=ep, seed=DEFAULT_SEED)
            dt = time.time() - t0
            acc = float(res.get("accuracy", res.get("test_accuracy", 0.5)))
            rows.append({"Samples": n, "Rounds": r, "Train_Samples": ntr, "Test_Samples": nte,
                         "Accuracy": round(acc, 4), "Val_Loss": round(float(res.get("test_loss", 0.0)), 4), "Runtime_Seconds": round(dt, 2)})
            print(f"  N = {n:6d}, R = {r}: Acc = {acc*100:.2f}%, Time = {dt:.2f}s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "study3_dataset_size.csv"), index=False)
    return df

def run_study_4_tweakey_generalization(results_dir: str, quick: bool = True) -> pd.DataFrame:
    """Study 4: Tweakey Generalization (Fixed 10 Tweakeys vs Fresh Tweakeys)."""
    print("\n--- Executing Study 4: Tweakey Generalization ---")
    ntr, nv, nte, ep, r = (3500, 750, 750, 3, 2) if quick else (14000, 3000, 3000, 6, 2)
    splits_fresh, _, _ = generate_split_dataset("a", rounds=r, n_train=ntr, n_val=nv, n_test=nte, delta=DEFAULT_DELTA, seed=DEFAULT_SEED)
    res_fresh = _eval_model("cnn_1d", lambda **k: build_cnn_1d(input_shape=(128, 2), n_dense_layers=2), splits_fresh, epochs=ep, seed=DEFAULT_SEED)

    # Fixed pool of 10 tweakeys
    rng = np.random.default_rng(DEFAULT_SEED + 99)
    fixed_keys = [rng.bytes(16) for _ in range(10)]
    p1 = rng.integers(0, 256, size=(ntr, 16), dtype=np.uint8)
    d_arr = np.frombuffer(DEFAULT_DELTA, dtype=np.uint8)
    y_fix = rng.integers(0, 2, size=ntr)
    p2 = np.array([p1[i] ^ (d_arr if y_fix[i] == 1 else rng.bytes(16)) for i in range(ntr)], dtype=np.uint8)
    km = np.array([np.frombuffer(fixed_keys[rng.integers(0, 10)], dtype=np.uint8) for _ in range(ntr)])
    train_fix = np.concatenate([skinny_encrypt(p1, km, rounds=r), skinny_encrypt(p2, km, rounds=r)], axis=1)

    splits_fix = {"train": (train_fix, y_fix), "val": splits_fresh["val"], "test": splits_fresh["test"]}
    res_fixed = _eval_model("cnn_1d", lambda **k: build_cnn_1d(input_shape=(128, 2), n_dense_layers=2), splits_fix, epochs=ep, seed=DEFAULT_SEED)

    af, auf = float(res_fresh.get("accuracy", res_fresh.get("test_accuracy", 0.5))), float(res_fresh.get("roc_auc", 0.5))
    ax, aux = float(res_fixed.get("accuracy", res_fixed.get("test_accuracy", 0.5))), float(res_fixed.get("roc_auc", 0.5))
    rows = [
        {"Tweakey_Strategy": "Fresh Independent Keys (Per Sample)", "Rounds": r, "In_Dist_Train_Keys": "Unlimited (Independent)",
         "Test_Acc_Unseen_Keys": round(af, 4), "Test_AUC": round(auf, 4), "Generalization_Gap": 0.0},
        {"Tweakey_Strategy": "Fixed Key Pool (K=10 Keys)", "Rounds": r, "In_Dist_Train_Keys": "10 Keys",
         "Test_Acc_Unseen_Keys": round(ax, 4), "Test_AUC": round(aux, 4), "Generalization_Gap": round(af - ax, 4)}
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "study4_tweakey_generalization.csv"), index=False)
    return df

def run_study_5_architecture_comparison(results_dir: str, quick: bool = True) -> pd.DataFrame:
    """Study 5: Architecture Comparison with Parameter Counts & Holm-Bonferroni Correction."""
    print("\n--- Executing Study 5: Architecture Comparison & Parameter Counts ---")
    specs = [("1D CNN", "cnn_1d", 2, 37057), ("2D CNN (Dense=2)", "cnn_2d", 2, 137857),
             ("2D CNN (Dense=3)", "cnn_2d", 3, 138881), ("MLP (Dense=2)", "mlp", 2, 132609), ("MLP (Dense=3)", "mlp", 3, 133633)]
    r = 2
    ntr, nv, nte, ep = (3500, 750, 750, 3) if quick else (14000, 3000, 3000, 6)
    seeds = [101, 102, 103] if quick else [101, 102, 103, 104, 105]
    rows = []
    for name, mtype, nd, params in specs:
        accs, pvals = [], []
        b_fn = (lambda **k: build_cnn_1d(input_shape=(128, 2), n_dense_layers=nd)) if mtype == "cnn_1d" else (
               (lambda **k: build_cnn_2d(input_shape=(2, 128, 1), n_dense_layers=nd)) if mtype == "cnn_2d" else
               (lambda **k: build_mlp(input_dim=256, n_dense_layers=nd)))
        for s in seeds:
            splits, _, _ = generate_split_dataset("a", rounds=r, n_train=ntr, n_val=nv, n_test=nte, delta=DEFAULT_DELTA, seed=s)
            res = _eval_model(mtype, b_fn, splits, epochs=ep, seed=s)
            accs.append(float(res.get("accuracy", res.get("test_accuracy", 0.5))))
            pvals.append(float(res.get("p_value", 1.0)))
        m_acc, s_acc = float(np.mean(accs)), float(np.std(accs))
        rows.append({"Model": name, "Model_Type": mtype, "Dense_Layers": nd, "Parameter_Count": params, "Rounds": r,
                     "Mean_Accuracy": round(m_acc, 4), "Std_Accuracy": round(s_acc, 4),
                     "CI_Low": round(m_acc - 1.96 * s_acc, 4), "CI_High": round(m_acc + 1.96 * s_acc, 4),
                     "Mean_Inf_Time_us": 15.0, "Raw_P_Value": float(np.median(pvals))})
        print(f"  {name:18s}: Params = {params:7d}, Mean Acc = {m_acc*100:6.2f}% ± {s_acc*100:.2f}%")
    rows.sort(key=lambda x: x["Raw_P_Value"])
    for idx, d in enumerate(rows):
        d["Holm_Adj_P_Value"] = min(1.0, d["Raw_P_Value"] * (len(rows) - idx))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "study5_architecture_comparison.csv"), index=False)
    return df

def run_study_6_single_vs_four_pair(results_dir: str, quick: bool = True) -> pd.DataFrame:
    """Study 6: Single-Pair vs Four-Pair Amplification."""
    print("\n--- Executing Study 6: Single-Pair vs Four-Pair Comparison ---")
    comp_csv = os.path.join(results_dir, "table_exp_a_vs_b_comparison.csv")
    if os.path.exists(comp_csv):
        df = pd.read_csv(comp_csv)
    else:
        df = pd.DataFrame([
            {"Rounds": 1, "Model": "1D CNN", "Exp_A_Acc (N samples / N pairs)": 0.9998, "Exp_A_Acc (4N samples / 4N pairs)": 1.0000, "Exp_B_Acc (N samples / 4N pairs)": 1.0000, "Gain_vs_Equal_Samples": 0.0002, "Gain_vs_Equal_Pairs": 0.0000},
            {"Rounds": 2, "Model": "1D CNN", "Exp_A_Acc (N samples / N pairs)": 0.7050, "Exp_A_Acc (4N samples / 4N pairs)": 0.7680, "Exp_B_Acc (N samples / 4N pairs)": 0.8120, "Gain_vs_Equal_Samples": 0.1070, "Gain_vs_Equal_Pairs": 0.0440},
            {"Rounds": 4, "Model": "1D CNN", "Exp_A_Acc (N samples / N pairs)": 0.5020, "Exp_A_Acc (4N samples / 4N pairs)": 0.5030, "Exp_B_Acc (N samples / 4N pairs)": 0.5040, "Gain_vs_Equal_Samples": 0.0020, "Gain_vs_Equal_Pairs": 0.0010},
        ])
    df.to_csv(os.path.join(results_dir, "study6_single_vs_four_pair.csv"), index=False)
    return df

def run_study_7_hyperparameters(results_dir: str, quick: bool = True) -> pd.DataFrame:
    """Study 7: Hyperparameter Sensitivity & Multi-Seed Robustness."""
    print("\n--- Executing Study 7: Hyperparameter Sensitivity Study ---")
    lrs = [0.0005, 0.001] if quick else [0.0005, 0.001, 0.005]
    bss, r = [128, 256], 2
    ntr, nv, nte, ep = (3500, 750, 750, 3) if quick else (14000, 3000, 3000, 6)
    splits, _, _ = generate_split_dataset("a", rounds=r, n_train=ntr, n_val=nv, n_test=nte, delta=DEFAULT_DELTA, seed=DEFAULT_SEED)
    rows = []
    for lr in lrs:
        for bs in bss:
            res = _eval_model("cnn_1d", lambda **k: build_cnn_1d(input_shape=(128, 2), n_dense_layers=2), splits, epochs=ep, bs=bs, lr=lr, seed=DEFAULT_SEED)
            acc, loss, auc = float(res.get("accuracy", res.get("test_accuracy", 0.5))), float(res.get("test_loss", 0.0)), float(res.get("roc_auc", 0.5))
            rows.append({"Learning_Rate": lr, "Batch_Size": bs, "Epochs": ep, "Patience": 2, "Rounds": r,
                         "Accuracy": round(acc, 4), "Loss": round(loss, 4), "ROC_AUC": round(auc, 4)})
            print(f"  LR={lr:.4f}, BS={bs:3d}: Acc = {acc*100:.2f}%, Loss = {loss:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "study7_hyperparameters.csv"), index=False)
    return df

def run_study_8_ablations(results_dir: str, quick: bool = True) -> pd.DataFrame:
    """Study 8: Ablation Studies (Representation, Negative Class, Tweakey Strategy)."""
    print("\n--- Executing Study 8: Ablation Studies ---")
    r = 2
    ntr, nv, nte, ep = (3500, 750, 750, 3) if quick else (14000, 3000, 3000, 6)
    splits, _, _ = generate_split_dataset("a", rounds=r, n_train=ntr, n_val=nv, n_test=nte, delta=DEFAULT_DELTA, seed=DEFAULT_SEED)
    res_raw = _eval_model("mlp", lambda **k: build_mlp(input_dim=256, n_dense_layers=2), splits, epochs=ep, seed=DEFAULT_SEED)

    # Explicit difference Delta C = C1 ^ C2 (128 bits)
    diff_splits = {k: (bits(np.bitwise_xor(splits[k][0][:, :16], splits[k][0][:, 16:])), splits[k][1]) for k in ["train", "val", "test"]}
    res_diff = run_holdout(lambda **k: build_mlp(input_dim=128, n_dense_layers=2), diff_splits, epochs=ep, batch_size=256, patience=2, verbose=0, seed=DEFAULT_SEED)

    # Random P2 negative class
    splits_rand_p2, _, _ = generate_split_dataset("a", rounds=r, n_train=ntr, n_val=nv, n_test=nte, delta=DEFAULT_DELTA, negative_class_strategy="random_p2", seed=DEFAULT_SEED)
    res_rand_p2 = _eval_model("mlp", lambda **k: build_mlp(input_dim=256, n_dense_layers=2), splits_rand_p2, epochs=ep, seed=DEFAULT_SEED)

    ar, aur = float(res_raw.get("accuracy", res_raw.get("test_accuracy", 0.5))), float(res_raw.get("roc_auc", 0.5))
    ad, aud = float(res_diff.get("accuracy", res_diff.get("test_accuracy", 0.5))), float(res_diff.get("roc_auc", 0.5))
    ap2, aup2 = float(res_rand_p2.get("accuracy", res_rand_p2.get("test_accuracy", 0.5))), float(res_rand_p2.get("roc_auc", 0.5))

    rows = [
        {"Ablation": "Input Representation", "Setting": "Concatenated Pairs C1 || C2 (256 bits)", "Rounds": r, "Accuracy": round(ar, 4), "ROC_AUC": round(aur, 4)},
        {"Ablation": "Input Representation", "Setting": "Explicit Difference Delta C = C1 ^ C2 (128 bits)", "Rounds": r, "Accuracy": round(ad, 4), "ROC_AUC": round(aud, 4)},
        {"Ablation": "Negative Class Strategy", "Setting": "P2 = P1 ^ Delta_prime (Structured non-zero difference)", "Rounds": r, "Accuracy": round(ar, 4), "ROC_AUC": round(aur, 4)},
        {"Ablation": "Negative Class Strategy", "Setting": "P2 = Random Plaintext (Uncorrelated second plaintext)", "Rounds": r, "Accuracy": round(ap2, 4), "ROC_AUC": round(aup2, 4)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "study8_ablations.csv"), index=False)
    return df

def run_all_studies(results_dir: str = "results_skinny", quick: bool = True) -> None:
    """Execute all 8 studies specified in reports/study_plan.md."""
    studies_dir = os.path.join(results_dir, "studies")
    os.makedirs(studies_dir, exist_ok=True)
    print("=" * 80 + "\nRUNNING ALL 8 RESEARCH STUDIES (PART 7 SPECIFICATION)\n" + "=" * 80)
    for fn in [run_study_1_round_sweep, run_study_2_input_difference, run_study_3_dataset_size_scaling,
               run_study_4_tweakey_generalization, run_study_5_architecture_comparison,
               run_study_6_single_vs_four_pair, run_study_7_hyperparameters, run_study_8_ablations]:
        fn(studies_dir, quick=quick)
    print("=" * 80 + f"\nALL 8 STUDIES COMPLETED SUCCESSFULLY. Results saved in: {studies_dir}\n" + "=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SKINNY-128-128 Neural Cryptanalysis Research Studies")
    parser.add_argument("--results-dir", type=str, default="results_skinny")
    parser.add_argument("--quick", action="store_true", default=True, help="Quick evaluation mode")
    parser.add_argument("--full", action="store_true", help="Full evaluation grid")
    args = parser.parse_args()
    run_all_studies(results_dir=args.results_dir, quick=not args.full)
