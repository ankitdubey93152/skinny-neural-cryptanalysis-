"""Experiment G: Multi-Cipher Identification (SKINNY-128-128, AES-128, SM4, HIGHT, DES)."""
import os, sys, argparse, itertools
from typing import List, Dict, Any, Optional, Union, Tuple, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    DEFAULT_SEED, CIPHER_METADATA, CIPHERS_NATIVE_128, CIPHERS_COMMON_64
)
from utils.seeding import set_global_seed
from data.generators import (
    generate_experiment_g, generate_random_control, run_artifact_audit,
    bits, reshape_bits_for_model
)
from models.architectures import build_mlp, build_cnn_1d, build_cnn_2d, prepare_input
from training.train_eval import accuracy_significance, compute_full_metrics, run_holdout
from experiments.exp_utils import parse_rounds_arg, filter_model_configs
from experiments.run_experiment_sample_size import holm_bonferroni
from experiments.diff_propagation_multi import measure_multi_cipher_diffusion


def run_experiment_g(
    scope: str = "native-128",
    slice_half: str = "low",
    rounds_mode: str = "matched",
    rounds: Optional[Union[List[int], str, int]] = None,
    cipher_rounds: Optional[Dict[str, int]] = None,
    n_samples: int = 20000,
    epochs: int = 10,
    models: Optional[List[str]] = None,
    results_dir: str = "results_skinny",
    quick_mode: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Execute Multi-Cipher Identification Experiment G."""
    os.makedirs(results_dir, exist_ok=True)
    scope = scope.lower().strip()
    if scope not in ("native-128", "common-64"):
        raise ValueError(f"Unknown scope: '{scope}'. Must be 'native-128' or 'common-64'")

    scope_ciphers = CIPHERS_NATIVE_128 if scope == "native-128" else CIPHERS_COMMON_64
    block_bits = 128 if scope == "native-128" else 64
    scope_suffix = "native128" if scope == "native-128" else "common64"

    if quick_mode:
        if n_samples > 3000:
            n_samples = 3000
        if epochs > 3:
            epochs = 3
        r_list = [1, 2, 4] if rounds is None else parse_rounds_arg(rounds)
    else:
        r_list = [1, 2, 3, 4] if rounds is None else parse_rounds_arg(rounds)

    print(f"\n=======================================================")
    print(f"EXPERIMENT G: Multi-Cipher Identification | Scope: {scope} ({len(scope_ciphers)} ciphers)")
    print(f"Rounds Mode: {rounds_mode} | Rounds: {r_list} | N: {n_samples:,} | Epochs: {epochs}")
    print(f"Ciphers: {scope_ciphers}")
    print(f"=======================================================")

    # 1. Structural Artifact Audit
    print("\n[1/4] Running Structural Artifact Audit...")
    audit_data_X, audit_data_y = generate_experiment_g(
        n_samples=min(5000, n_samples),
        ciphers=scope_ciphers,
        rounds=4,
        scope=scope,
        slice_half=slice_half,
        seed=DEFAULT_SEED,
    )
    audit_result = run_artifact_audit(audit_data_X, audit_data_y, scope=scope)
    audit_rows = [{
        "Scope": scope,
        "Slice_Half": slice_half,
        "N_Samples_Audited": audit_result["n_samples"],
        "N_Bits_Audited": audit_result["n_bits"],
        "Constant_Bits_Found": audit_result["constant_bits"],
        "Identical_Halves_Found": audit_result["identical_pairs"],
        "Overall_Bit_Mean": round(audit_result["bit_mean_overall"], 4),
        "Min_Bit_Std": round(audit_result["bit_std_min"], 4),
        "Audit_Passed": audit_result["passed"],
    }]
    audit_df = pd.DataFrame(audit_rows)
    audit_csv = os.path.join(results_dir, f"table_experiment_g_artifact_audit_{scope_suffix}.csv")
    audit_df.to_csv(audit_csv, index=False)
    print(f"[+] Artifact audit passed! Saved to {audit_csv}")

    # 2. Diffusion profile
    diff_csv = os.path.join(results_dir, "difference_propagation_multi.csv")
    if not os.path.exists(diff_csv):
        print("\n[2/4] Generating Multi-Cipher Diffusion Profile...")
        measure_multi_cipher_diffusion(results_dir=results_dir)

    # Setup model architectures for this block size
    s_1d = (block_bits, 2)
    s_2d = (2, block_bits, 1)
    dim_mlp = 2 * block_bits

    all_binary_cfgs = [
        {"name": "1D CNN", "model_type": "cnn_1d", "builder": build_cnn_1d, "kwargs": {"input_shape": s_1d, "n_dense_layers": 2}},
        {"name": "2D CNN", "model_type": "cnn_2d", "builder": build_cnn_2d, "kwargs": {"input_shape": s_2d, "n_dense_layers": 2, "kernel_size": 3}},
        {"name": "MLP", "model_type": "mlp", "builder": build_mlp, "kwargs": {"input_dim": dim_mlp, "n_dense_layers": 2, "units": 640}},
    ]
    binary_cfgs = filter_model_configs(all_binary_cfgs, models)

    # 3. Pairwise Binary Classification for ALL pairs C(K, 2)
    print(f"\n[3/4] Running Pairwise Binary Classification (C({len(scope_ciphers)}, 2) pairs)...")
    cipher_pairs = list(itertools.combinations(scope_ciphers, 2))
    pairwise_rows = []
    p_vals_pairwise = []

    for c1_name, c2_name in cipher_pairs:
        pair_label = f"{c1_name}_vs_{c2_name}"
        for r in r_list:
            if rounds_mode == "full":
                r1 = CIPHER_METADATA[c1_name]["full_rounds"]
                r2 = CIPHER_METADATA[c2_name]["full_rounds"]
                r_dict = {c1_name: r1, c2_name: r2}
                r_display = f"full({r1},{r2})"
            elif rounds_mode == "custom" and cipher_rounds:
                r_dict = cipher_rounds
                r_display = str(cipher_rounds)
            else:
                r_dict = r
                r_display = str(r)

            # Generate pair dataset (class 0: c1, class 1: c2)
            X_all, y_all = generate_experiment_g(
                n_samples=n_samples,
                ciphers=[c1_name, c2_name],
                rounds=r_dict,
                scope=scope,
                slice_half=slice_half,
                seed=DEFAULT_SEED + r * 10,
            )

            # 70/15/15 split
            n_tot = len(y_all)
            n_tr = int(n_tot * 0.70)
            n_va = int(n_tot * 0.15)
            n_te = n_tot - n_tr - n_va

            X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]
            X_va, y_va = X_all[n_tr:n_tr + n_va], y_all[n_tr:n_tr + n_va]
            X_te, y_te = X_all[n_tr + n_va:], y_all[n_tr + n_va:]

            X_tr_b, X_va_b, X_te_b = bits(X_tr), bits(X_va), bits(X_te)

            for cfg in binary_cfgs:
                x_tr_in = reshape_bits_for_model(X_tr_b, cfg["model_type"], "G", block_bits=block_bits)
                x_va_in = reshape_bits_for_model(X_va_b, cfg["model_type"], "G", block_bits=block_bits)
                x_te_in = reshape_bits_for_model(X_te_b, cfg["model_type"], "G", block_bits=block_bits)

                res = run_holdout(
                    cfg["builder"],
                    splits={"train": (x_tr_in, y_tr), "val": (x_va_in, y_va), "test": (x_te_in, y_te)},
                    epochs=epochs,
                    batch_size=256 if n_tr >= 1000 else 64,
                    patience=2,
                    model_name=cfg["name"],
                    random_state=DEFAULT_SEED,
                    verbose=0,
                    **cfg["kwargs"]
                )

                p_vals_pairwise.append(res["p_value"])
                pairwise_rows.append({
                    "Scope": scope,
                    "Slice_Half": slice_half,
                    "Cipher_1": c1_name,
                    "Cipher_2": c2_name,
                    "Pair": pair_label,
                    "Round_Mode": rounds_mode,
                    "Round": r_display,
                    "Model": cfg["name"],
                    "N_Test": res["n_samples"],
                    "Accuracy": round(float(res["test_accuracy"]), 4),
                    "CI_Low": round(float(res["ci_low"]), 4),
                    "CI_High": round(float(res["ci_high"]), 4),
                    "ROC_AUC": round(float(res["test_auc"]), 4),
                    "p-value_raw": float(res["p_value"]),
                    "p-value_holm": float(res["p_value"]),
                })

    if p_vals_pairwise:
        adj_p = holm_bonferroni(p_vals_pairwise)
        for i, p_adj in enumerate(adj_p):
            pairwise_rows[i]["p-value_holm"] = round(float(p_adj), 6)

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_csv = os.path.join(results_dir, f"table_experiment_g_pairwise_{scope_suffix}.csv")
    pairwise_df.to_csv(pairwise_csv, index=False)
    print(f"[+] Saved pairwise classification results to {pairwise_csv}")

    # 4. K-way Multiclass Identification
    print(f"\n[4/4] Running K-way Multiclass Identification (K={len(scope_ciphers)})...")
    k_classes = len(scope_ciphers)
    multi_rows = []

    for r in r_list:
        X_all, y_all = generate_experiment_g(
            n_samples=n_samples,
            ciphers=scope_ciphers,
            rounds=r if rounds_mode == "matched" else 4,
            scope=scope,
            slice_half=slice_half,
            seed=DEFAULT_SEED + r * 100,
        )

        n_tot = len(y_all)
        n_tr = int(n_tot * 0.70)
        n_va = int(n_tot * 0.15)
        n_te = n_tot - n_tr - n_va

        X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]
        X_va, y_va = X_all[n_tr:n_tr + n_va], y_all[n_tr:n_tr + n_va]
        X_te, y_te = X_all[n_tr + n_va:], y_all[n_tr + n_va:]

        y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes=k_classes)
        y_va_cat = tf.keras.utils.to_categorical(y_va, num_classes=k_classes)

        X_tr_b, X_va_b, X_te_b = bits(X_tr), bits(X_va), bits(X_te)
        x_tr_in = reshape_bits_for_model(X_tr_b, "1d_cnn", "G", block_bits=block_bits)
        x_va_in = reshape_bits_for_model(X_va_b, "1d_cnn", "G", block_bits=block_bits)
        x_te_in = reshape_bits_for_model(X_te_b, "1d_cnn", "G", block_bits=block_bits)

        # Build multiclass 1D CNN
        tf.keras.backend.clear_session()
        set_global_seed(DEFAULT_SEED)
        model = build_cnn_1d(input_shape=s_1d, n_dense_layers=2, num_classes=k_classes)
        model.fit(
            x_tr_in, y_tr_cat,
            validation_data=(x_va_in, y_va_cat),
            epochs=epochs,
            batch_size=256 if n_tr >= 1000 else 64,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        )

        y_prob = model.predict(x_te_in, batch_size=256, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = float(np.mean(y_pred == y_te))
        macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_te, y_pred, labels=list(range(k_classes))).tolist()

        multi_rows.append({
            "Scope": scope,
            "K_Classes": k_classes,
            "Ciphers": "-".join(scope_ciphers),
            "Round": r,
            "Model": "1D CNN",
            "Chance_Level": round(1.0 / k_classes, 4),
            "Test_Accuracy": round(acc, 4),
            "Macro_F1": round(macro_f1),
            "Confusion_Matrix": str(cm),
        })

    multi_df = pd.DataFrame(multi_rows)
    multi_csv = os.path.join(results_dir, f"table_experiment_g_multiclass_{scope_suffix}.csv")
    multi_df.to_csv(multi_csv, index=False)
    print(f"[+] Saved multiclass identification results to {multi_csv}")

    # 5. Random Baseline Control
    print("\nRunning Random-Baseline Controls...")
    ctrl_rows = []
    for c_name in scope_ciphers:
        for r in r_list[:2]:
            X_ctrl, y_ctrl = generate_random_control(
                cipher_name=c_name,
                n_samples=min(4000, n_samples),
                rounds=r,
                scope=scope,
                slice_half=slice_half,
                seed=DEFAULT_SEED,
            )
            n_tot = len(y_ctrl)
            n_tr = int(n_tot * 0.70)
            n_va = int(n_tot * 0.15)
            X_tr, y_tr = X_ctrl[:n_tr], y_ctrl[:n_tr]
            X_va, y_va = X_ctrl[n_tr:n_tr + n_va], y_ctrl[n_tr:n_tr + n_va]
            X_te, y_te = X_ctrl[n_tr + n_va:], y_ctrl[n_tr + n_va:]

            x_tr_in = reshape_bits_for_model(bits(X_tr), "1d_cnn", "G", block_bits=block_bits)
            x_va_in = reshape_bits_for_model(bits(X_va), "1d_cnn", "G", block_bits=block_bits)
            x_te_in = reshape_bits_for_model(bits(X_te), "1d_cnn", "G", block_bits=block_bits)

            cfg = binary_cfgs[0]
            res = run_holdout(
                cfg["builder"],
                splits={"train": (x_tr_in, y_tr), "val": (x_va_in, y_va), "test": (x_te_in, y_te)},
                epochs=epochs,
                batch_size=256 if n_tr >= 1000 else 64,
                patience=2,
                model_name=cfg["name"],
                random_state=DEFAULT_SEED,
                verbose=0,
                **cfg["kwargs"]
            )
            ctrl_rows.append({
                "Scope": scope,
                "Cipher": c_name,
                "Control": "Cipher_vs_Uniform_Random",
                "Round": r,
                "Model": cfg["name"],
                "Accuracy": round(float(res["test_accuracy"]), 4),
                "CI_Low": round(float(res["ci_low"]), 4),
                "CI_High": round(float(res["ci_high"]), 4),
                "p-value": float(res["p_value"]),
            })

    ctrl_df = pd.DataFrame(ctrl_rows)
    ctrl_csv = os.path.join(results_dir, "table_experiment_g_random_control.csv")
    ctrl_df.to_csv(ctrl_csv, index=False)
    print(f"[+] Saved random controls to {ctrl_csv}")

    return {
        "pairwise": pairwise_df,
        "multiclass": multi_df,
        "random_control": ctrl_df,
        "artifact_audit": audit_df,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Experiment G (Multi-Cipher Identification)")
    p.add_argument("--scope", type=str, default="native-128", choices=["native-128", "common-64"])
    p.add_argument("--slice-half", type=str, default="low", choices=["low", "high"])
    p.add_argument("--rounds-mode", type=str, default="matched", choices=["matched", "full", "custom"])
    p.add_argument("--rounds", nargs="+", default=None)
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--results-dir", type=str, default="results_skinny")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run_experiment_g(
        scope=a.scope,
        slice_half=a.slice_half,
        rounds_mode=a.rounds_mode,
        rounds=a.rounds,
        n_samples=a.samples,
        epochs=a.epochs,
        models=a.models,
        results_dir=a.results_dir,
        quick_mode=a.quick,
    )
