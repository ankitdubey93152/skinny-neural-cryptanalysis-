"""Training, validation, and evaluation pipeline for SKINNY-128-128 neural cryptanalysis."""
import os, sys, json, time
from typing import Callable, Dict, Any, Optional, Tuple, Union, List, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, t as student_t
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix,
)
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.seeding import set_global_seed, detect_hardware
from models.architectures import build_mlp, build_cnn_1d, build_cnn_2d, prepare_input


def accuracy_significance(n_correct: int, n: int) -> Tuple[float, Any, float]:
    """Calculate accuracy, 95% Wilson confidence interval, and two-sided binomial p-value vs chance (0.5)."""
    if n == 0: return 0.0, (0.0, 0.0), 1.0
    acc = float(n_correct) / float(n)
    res = binomtest(k=int(n_correct), n=int(n), p=0.5, alternative="two-sided")
    ci = res.proportion_ci(confidence_level=0.95, method="wilson")
    return acc, ci, float(res.pvalue)


def compute_full_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate evaluation metrics: acc, prec, rec, f1, ROC-AUC, PR-AUC, CM, Wilson CI, p-value."""
    yt = np.asarray(y_true, dtype=int).flatten()
    yp_prob = np.asarray(y_pred_prob, dtype=float).flatten()
    yp = (yp_prob >= threshold).astype(int)
    n_tot = len(yt)
    n_cor = int(np.sum(yp == yt))
    acc, ci, pval = accuracy_significance(n_cor, n_tot)

    try: roc_auc = float(roc_auc_score(yt, yp_prob))
    except ValueError: roc_auc = 0.5
    try: pr_auc = float(average_precision_score(yt, yp_prob))
    except ValueError: pr_auc = 0.5

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return {
        "n_samples": n_tot, "n_correct": n_cor, "accuracy": acc,
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "ci_low": float(ci.low), "ci_high": float(ci.high), "ci_center": acc,
        "p_value": float(pval),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }


def get_model_configs(experiment: str = "A") -> Dict[str, Dict[str, Any]]:
    """Return factory definitions for the 5 canonical model architectures."""
    exp = experiment.upper().replace("EXP_", "")
    dim_mlp = 1024 if exp == "B" else 256
    s_1d = (128, 8) if exp == "B" else (128, 2)
    s_2d = (4, 256, 1) if exp == "B" else (2, 128, 1)

    return {
        "1D CNN": {"builder": lambda **kw: build_cnn_1d(input_shape=s_1d, n_dense_layers=2, **kw), "model_type": "cnn_1d", "n_dense_layers": 2},
        "2D CNN (dense 2)": {"builder": lambda **kw: build_cnn_2d(input_shape=s_2d, n_dense_layers=2, kernel_size=3, **kw), "model_type": "cnn_2d", "n_dense_layers": 2},
        "2D CNN (dense 3)": {"builder": lambda **kw: build_cnn_2d(input_shape=s_2d, n_dense_layers=3, kernel_size=3, **kw), "model_type": "cnn_2d", "n_dense_layers": 3},
        "MLP (dense 2)": {"builder": lambda **kw: build_mlp(input_dim=dim_mlp, n_dense_layers=2, **kw), "model_type": "mlp", "n_dense_layers": 2},
        "MLP (dense 3)": {"builder": lambda **kw: build_mlp(input_dim=dim_mlp, n_dense_layers=3, **kw), "model_type": "mlp", "n_dense_layers": 3},
    }


def train_and_evaluate(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 20, batch_size: int = 256, patience: int = 3, checkpoint_dir: Optional[str] = None, history_path: Optional[str] = None, verbose: int = 0) -> Dict[str, Any]:
    callbacks: List[tf.keras.callbacks.Callback] = []
    ckpt_file = None
    if patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=verbose))
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_file = os.path.join(checkpoint_dir, f"{getattr(model, 'name', 'model')}_best.weights.h5")
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_file, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=verbose))

    t0 = time.perf_counter()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    train_time = time.perf_counter() - t0

    test_loss, _ = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    y_prob = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()

    metrics = compute_full_metrics(y_test, y_prob)
    metrics.update({"test_loss": float(test_loss), "train_time_sec": float(train_time), "epochs_trained": len(history.history.get("loss", [])), "history": history.history, "checkpoint_path": ckpt_file})

    if history_path:
        os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
        with open(history_path, "w") as f: json.dump(history.history, f, indent=2)
        pd.DataFrame(history.history).to_csv(os.path.splitext(history_path)[0] + ".csv", index=False)

    return metrics


def run_holdout(model_builder: Callable[..., tf.keras.Model], X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, splits: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None, val_split: float = 0.15, test_split: float = 0.15, epochs: int = 20, batch_size: int = 256, patience: int = 3, random_state: int = 42, checkpoint_dir: Optional[str] = None, history_path: Optional[str] = None, verbose: int = 0, model_name: Optional[str] = None, **model_kwargs: Any) -> Dict[str, Any]:
    if splits is not None:
        X_tr, y_tr = splits["train"][:2]
        X_va, y_va = splits["val"][:2]
        X_te, y_te = splits["test"][:2]
    else:
        if X is None or y is None: raise ValueError("Must provide either 'splits' or both 'X' and 'y'")
        X_tr_va, X_te, y_tr_va, y_te = train_test_split(X, y, test_size=test_split, random_state=random_state, stratify=y)
        rel_val = val_split / (1.0 - test_split)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tr_va, y_tr_va, test_size=rel_val, random_state=random_state, stratify=y_tr_va)

    tf.keras.backend.clear_session()
    seed_info = set_global_seed(random_state)
    model = model_builder(**model_kwargs)
    disp_name = model_name or getattr(model, "name", "Model")

    res = train_and_evaluate(model, X_tr, y_tr, X_va, y_va, X_te, y_te, epochs=epochs, batch_size=batch_size, patience=patience, checkpoint_dir=checkpoint_dir, history_path=history_path, verbose=verbose)
    dense_count = model_kwargs.get("n_dense_layers", 3 if "mlp" in disp_name.lower() else 2)

    return {
        "model_name": disp_name, "n_dense_layers": dense_count, "epochs": res["epochs_trained"], "max_epochs": epochs, "batch_size": batch_size,
        "accuracy": res["accuracy"], "test_accuracy": res["accuracy"],
        "precision": res["precision"], "test_precision": res["precision"],
        "recall": res["recall"], "test_recall": res["recall"],
        "f1": res["f1"], "test_f1": res["f1"],
        "roc_auc": res["roc_auc"], "test_auc": res["roc_auc"],
        "pr_auc": res["pr_auc"], "test_pr_auc": res["pr_auc"],
        "test_loss": res["test_loss"], "confusion_matrix": res["confusion_matrix"],
        "ci_low": res["ci_low"], "ci_high": res["ci_high"], "ci_center": res["ci_center"],
        "p_value": res["p_value"], "n_test": res["n_samples"], "n_correct": res["n_correct"],
        "history": res["history"], "histories": [res["history"]], "checkpoint_path": res["checkpoint_path"], "hardware": seed_info.get("hardware", {}),
        "mean_accuracy": res["accuracy"], "std_accuracy": 0.0, "mean_auc": res["roc_auc"], "std_auc": 0.0, "mean_loss": res["test_loss"], "std_loss": 0.0,
    }


def run_kfold(model_builder: Callable[..., tf.keras.Model], X: np.ndarray, y: np.ndarray, n_folds: int = 5, epochs: int = 5, batch_size: int = 256, random_state: int = 42, verbose: int = 0, model_name: Optional[str] = None, **model_kwargs: Any) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    accs, losses, aucs, histories = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        tf.keras.backend.clear_session()
        set_global_seed(random_state + fold)
        model = model_builder(**model_kwargs)
        disp = model_name or getattr(model, "name", "Model")
        h = model.fit(X[tr_idx], y[tr_idx], validation_data=(X[va_idx], y[va_idx]), epochs=epochs, batch_size=batch_size, verbose=verbose)
        histories.append(h.history)
        vl, va = model.evaluate(X[va_idx], y[va_idx], batch_size=batch_size, verbose=0)
        yp = model.predict(X[va_idx], batch_size=batch_size, verbose=0).flatten()
        try: v_auc = float(roc_auc_score(y[va_idx], yp))
        except ValueError: v_auc = 0.5
        accs.append(float(va)); losses.append(float(vl)); aucs.append(v_auc)

    disp = model_name or "Model"
    dense_count = model_kwargs.get("n_dense_layers", 3 if "mlp" in disp.lower() else 2)
    return {
        "model_name": disp, "n_dense_layers": dense_count, "n_folds": n_folds, "epochs": epochs, "batch_size": batch_size,
        "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "mean_loss": float(np.mean(losses)), "std_loss": float(np.std(losses)),
        "fold_accuracies": accs, "fold_losses": losses, "fold_aucs": aucs, "histories": histories,
    }


def run_multiseed_evaluation(model_builder: Callable[..., tf.keras.Model], splits: Dict[str, Tuple[np.ndarray, np.ndarray]], model_seeds: Sequence[int] = (42, 43, 44, 45, 46), epochs: int = 15, batch_size: int = 256, patience: int = 3, model_name: Optional[str] = None, results_dir: Optional[str] = None, verbose: int = 0, **model_kwargs: Any) -> Dict[str, Any]:
    seed_results, accs, aucs = [], [], []
    for s in model_seeds:
        ckpt = os.path.join(results_dir, f"checkpoints_seed_{s}") if results_dir else None
        hf = os.path.join(results_dir, f"history_{model_name}_seed_{s}.json") if results_dir else None
        res = run_holdout(model_builder=model_builder, splits=splits, epochs=epochs, batch_size=batch_size, patience=patience, random_state=s, checkpoint_dir=ckpt, history_path=hf, verbose=verbose, model_name=model_name, **model_kwargs)
        seed_results.append(res); accs.append(res["test_accuracy"]); aucs.append(res["test_auc"])

    accs, aucs, n_s = np.array(accs), np.array(aucs), len(model_seeds)
    m_acc, s_acc = float(np.mean(accs)), (float(np.std(accs, ddof=1)) if n_s > 1 else 0.0)
    if n_s > 1 and s_acc > 0:
        se = s_acc / np.sqrt(n_s)
        t_crit = float(student_t.ppf(0.975, df=n_s - 1))
        ci_mean = (m_acc - t_crit * se, m_acc + t_crit * se)
    else: ci_mean = (m_acc, m_acc)

    return {
        "model_name": model_name or "Model", "n_seeds": n_s, "seeds": list(model_seeds),
        "mean_accuracy": m_acc, "std_accuracy": s_acc, "ci_95_mean_accuracy": ci_mean,
        "mean_auc": float(np.mean(aucs)), "std_auc": (float(np.std(aucs, ddof=1)) if n_s > 1 else 0.0),
        "individual_runs": seed_results,
    }


def run_sanity_controls(experiment: str = "A", sample_size: int = 4000, epochs: int = 4, batch_size: int = 128, seed: int = 42, verbose: int = 0) -> Dict[str, Any]:
    from data.generators import generate_split_dataset, bits, reshape_bits_for_model
    sp_r1, _, _ = generate_split_dataset(experiment=experiment, rounds=1, n_train=int(sample_size * 0.7), n_val=int(sample_size * 0.15), n_test=int(sample_size * 0.15), seed=seed, return_meta=False)
    x_tr = reshape_bits_for_model(bits(sp_r1["train"][0]), "mlp", experiment)
    x_va = reshape_bits_for_model(bits(sp_r1["val"][0]), "mlp", experiment)
    x_te = reshape_bits_for_model(bits(sp_r1["test"][0]), "mlp", experiment)
    s_fmt = {"train": (x_tr, sp_r1["train"][1]), "val": (x_va, sp_r1["val"][1]), "test": (x_te, sp_r1["test"][1])}

    pos = run_holdout(lambda: build_mlp(input_dim=256, n_dense_layers=2), splits=s_fmt, epochs=epochs, batch_size=batch_size, patience=2, random_state=seed, verbose=verbose, model_name="PositiveControl_R1")
    pos_ok = pos["test_accuracy"] >= 0.80

    rng = np.random.default_rng(seed)
    s_shuf = {"train": (x_tr, rng.permutation(sp_r1["train"][1])), "val": (x_va, sp_r1["val"][1]), "test": (x_te, sp_r1["test"][1])}
    shuf = run_holdout(lambda: build_mlp(input_dim=256, n_dense_layers=2), splits=s_shuf, epochs=epochs, batch_size=batch_size, patience=2, random_state=seed, verbose=verbose, model_name="LabelShuffledControl")
    shuf_ok = shuf["p_value"] > 0.01

    x_tr_rnd = rng.integers(0, 2, size=x_tr.shape, dtype=np.uint8).astype(np.float32)
    x_va_rnd = rng.integers(0, 2, size=x_va.shape, dtype=np.uint8).astype(np.float32)
    x_te_rnd = rng.integers(0, 2, size=x_te.shape, dtype=np.uint8).astype(np.float32)
    s_rnd = {"train": (x_tr_rnd, sp_r1["train"][1]), "val": (x_va_rnd, sp_r1["val"][1]), "test": (x_te_rnd, sp_r1["test"][1])}
    rnd = run_holdout(lambda: build_mlp(input_dim=256, n_dense_layers=2), splits=s_rnd, epochs=epochs, batch_size=batch_size, patience=2, random_state=seed, verbose=verbose, model_name="RandomCiphertextControl")
    rnd_ok = rnd["p_value"] > 0.01

    return {
        "positive_control": {"passed": bool(pos_ok), "test_accuracy": pos["test_accuracy"], "ci_low": pos["ci_low"], "ci_high": pos["ci_high"], "expected": ">= 0.80"},
        "label_shuffled_control": {"passed": bool(shuf_ok), "test_accuracy": shuf["test_accuracy"], "ci_low": shuf["ci_low"], "ci_high": shuf["ci_high"], "p_value": shuf["p_value"], "expected": "p > 0.01"},
        "random_ciphertext_control": {"passed": bool(rnd_ok), "test_accuracy": rnd["test_accuracy"], "ci_low": rnd["ci_low"], "ci_high": rnd["ci_high"], "p_value": rnd["p_value"], "expected": "p > 0.01"},
        "all_controls_passed": bool(pos_ok and shuf_ok and rnd_ok),
    }


def plot_training_curves(result_dict: Dict[str, Any], save_path: Optional[str] = None, title_prefix: str = "") -> None:
    histories = result_dict.get("histories", []) or ([result_dict["history"]] if "history" in result_dict else [])
    if not histories: return
    epochs = range(1, len(histories[0].get("accuracy", [])) + 1)
    is_kf, name = len(histories) > 1, result_dict.get("model_name", "Model")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    for i, h in enumerate(histories):
        sfx = f"Fold {i+1} " if is_kf else ""
        ax0.plot(epochs, h.get("accuracy", []), "--", alpha=0.5, label=f"{sfx}Train")
        ax0.plot(epochs, h.get("val_accuracy", []), "-", alpha=0.8, label=f"{sfx}Val")
        ax1.plot(epochs, h.get("loss", []), "--", alpha=0.5, label=f"{sfx}Train")
        ax1.plot(epochs, h.get("val_loss", []), "-", alpha=0.8, label=f"{sfx}Val")
    ax0.axhline(0.5, color="red", linestyle=":", label="Chance (50%)")
    ax0.set_title(f"{title_prefix} {name} - Accuracy".strip()); ax0.set_xlabel("Epoch"); ax0.grid(True, linestyle="--", alpha=0.5); ax0.legend(loc="lower right", fontsize=8)
    ax1.set_title(f"{title_prefix} {name} - Loss".strip()); ax1.set_xlabel("Epoch"); ax1.grid(True, linestyle="--", alpha=0.5); ax1.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def summarize_run(result_dict: Dict[str, Any]) -> pd.Series:
    row = {"Model Name": result_dict.get("model_name", "Model"), "Dense Layers": result_dict.get("n_dense_layers", 2)}
    if "n_folds" in result_dict: row["Folds"] = result_dict["n_folds"]
    row["Epochs"] = result_dict.get("epochs", 5)
    acc = result_dict.get("test_accuracy", result_dict.get("mean_accuracy", 0.0))
    row["Test Acc (%)" if "test_accuracy" in result_dict else "Accuracy (%)"] = round(acc * 100, 2)
    if result_dict.get("std_accuracy", 0.0) > 0: row["Std Dev (%)"] = round(result_dict["std_accuracy"] * 100, 2)
    auc = result_dict.get("test_auc", result_dict.get("mean_auc", 0.5))
    row["ROC-AUC"] = round(auc, 4)
    if "test_pr_auc" in result_dict: row["PR-AUC"] = round(result_dict["test_pr_auc"], 4)
    if "ci_low" in result_dict and "ci_high" in result_dict:
        row["CI low (%)"], row["CI high (%)"] = round(result_dict["ci_low"] * 100, 2), round(result_dict["ci_high"] * 100, 2)
    if "p_value" in result_dict: row["p-value"] = result_dict["p_value"]
    return pd.Series(row)
