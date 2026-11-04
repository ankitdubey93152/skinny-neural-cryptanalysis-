"""Shared utility routines, baselines, and CLI parsing for neural cryptanalysis experiments."""
import os, sys, json
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DEFAULT_ROUND_SWEEP, DEFAULT_DELTA, DEFAULT_SEED
from data.generators import save_dataset, load_dataset, bits, reshape_bits_for_model
from training.train_eval import compute_full_metrics, accuracy_significance


def parse_rounds_arg(rounds_input: Any) -> List[int]:
    """Parse round arguments into a sorted list of unique integers."""
    if rounds_input is None: return list(DEFAULT_ROUND_SWEEP)
    if isinstance(rounds_input, int): return [rounds_input]
    res = []
    if isinstance(rounds_input, (list, tuple)):
        for item in rounds_input:
            if isinstance(item, int): res.append(item)
            elif isinstance(item, str): res.extend(int(t) for t in item.replace(",", " ").split())
    elif isinstance(rounds_input, str):
        res.extend(int(t) for t in rounds_input.replace(",", " ").split())
    else: raise TypeError(f"Unsupported type: {type(rounds_input)}")
    if not res: raise ValueError("Rounds list cannot be empty.")
    return sorted(list(set(res)))


def parse_delta_arg(delta_input: Optional[Union[str, bytes]]) -> bytes:
    """Parse and validate 16-byte (32 hex characters) input difference Delta."""
    if delta_input is None: return DEFAULT_DELTA
    if isinstance(delta_input, bytes):
        if len(delta_input) != 16: raise ValueError(f"Delta must be 16 bytes, got {len(delta_input)}")
        return delta_input
    if isinstance(delta_input, str):
        s = delta_input.strip()
        if len(s) != 32: raise ValueError(f"Delta must be 32 hex characters, got {len(s)}")
        try: return bytes.fromhex(s)
        except ValueError as e: raise ValueError(f"Invalid hex string for delta: {e}")
    raise TypeError(f"Delta must be str or bytes, got {type(delta_input)}")


def filter_model_configs(all_configs: List[Dict[str, Any]], requested_models: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter model configurations against a requested subset list."""
    if not requested_models: return all_configs
    def cln(s: str) -> str: return "".join(c for c in s.lower() if c.isalnum())
    reqs = [cln(m) for m in requested_models]
    selected = [c for c in all_configs if any(r == cln(c["name"]) or r == cln(c.get("model_type", "")) or r in cln(c["name"]) or r in cln(c.get("model_type", "")) for r in reqs)]
    if not selected: raise ValueError(f"No model configs matched '{requested_models}'. Valid: {[c['name'] for c in all_configs]}")
    return selected


def get_or_generate_dataset(exp_id: str, rounds: int, n_samples: int, delta_bytes: bytes, data_dir: str = "data_skinny", generator_fn: Optional[Callable] = None, **gen_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached dataset or generate a fresh one if missing."""
    os.makedirs(data_dir, exist_ok=True)
    cname = f"exp_{exp_id.lower()}_r{rounds}_n{n_samples}.npz" if delta_bytes == DEFAULT_DELTA else f"exp_{exp_id.lower()}_r{rounds}_n{n_samples}_{delta_bytes.hex()[:8]}.npz"
    cpath = os.path.join(data_dir, cname)
    if os.path.exists(cpath):
        X_raw, y = load_dataset(cpath)
        if len(y) >= n_samples: return X_raw[:n_samples], y[:n_samples]
    if generator_fn is None: raise ValueError("generator_fn required when dataset cache is absent.")
    X_raw, y = generator_fn(n_samples=n_samples, rounds=rounds, delta=delta_bytes, seed=DEFAULT_SEED, **gen_kwargs)
    save_dataset(cpath, X_raw, y, rounds=rounds, delta=delta_bytes, seed=DEFAULT_SEED)
    return X_raw, y


def run_non_neural_baselines(X_tr_bytes: np.ndarray, y_tr: np.ndarray, X_te_bytes: np.ndarray, y_te: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Compute non-neural statistical baselines for Experiment A."""
    tr_raw, te_raw = bits(X_tr_bytes).astype(np.float32), bits(X_te_bytes).astype(np.float32)
    tr_diff = bits(X_tr_bytes[:, :16] ^ X_tr_bytes[:, 16:]).astype(np.float32)
    te_diff = bits(X_te_bytes[:, :16] ^ X_te_bytes[:, 16:]).astype(np.float32)

    baselines: Dict[str, Dict[str, Any]] = {}
    clf_diff = LogisticRegression(max_iter=200, random_state=DEFAULT_SEED).fit(tr_diff, y_tr)
    baselines["Logistic Regression (Diff Bits)"] = compute_full_metrics(y_te, clf_diff.predict_proba(te_diff)[:, 1])

    clf_raw = LogisticRegression(max_iter=200, random_state=DEFAULT_SEED).fit(tr_raw, y_tr)
    baselines["Logistic Regression (Raw Bits)"] = compute_full_metrics(y_te, clf_raw.predict_proba(te_raw)[:, 1])

    p_pos, p_neg = np.mean(tr_diff[y_tr == 1], axis=0), np.mean(tr_diff[y_tr == 0], axis=0)
    best_bit = int(np.argmax(np.abs(p_pos - p_neg)))
    pred_bias = te_diff[:, best_bit] if p_pos[best_bit] >= p_neg[best_bit] else 1.0 - te_diff[:, best_bit]
    baselines["Per-Bit Bias Distinguisher"] = compute_full_metrics(y_te, pred_bias)
    return baselines


def run_pair_order_shuffle_control(model: Any, X_test_bundle: np.ndarray, y_test: np.ndarray, model_type: str = "mlp", seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Test model invariance to pair-order permutations in Experiment B."""
    rng, N = np.random.default_rng(seed), X_test_bundle.shape[0]
    X_shuffled = np.empty_like(X_test_bundle)
    for i in range(N): X_shuffled[i] = X_test_bundle[i, rng.permutation(4), :]
    X_in = reshape_bits_for_model(bits(X_shuffled), model_type, experiment="B")
    return compute_full_metrics(y_test, model.predict(X_in, batch_size=256, verbose=0).flatten())


def plot_accuracy_vs_rounds(sweep_df: pd.DataFrame, save_path: str, exp_name: str = "A") -> None:
    """Plot distinguisher accuracy vs cipher rounds for swept models."""
    if sweep_df.empty: return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    models = sweep_df["Model"].unique()
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    for i, m in enumerate(models):
        sub = sweep_df[sweep_df["Model"] == m].sort_values("Rounds")
        ax.plot(sub["Rounds"], sub["Accuracy"], marker=markers[i % len(markers)], linewidth=1.8, label=m)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1.5, label="Chance (0.50)")
    ax.set_ylim([0.45, 1.05]); ax.set_xlabel("Rounds"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Experiment {exp_name.upper()}: Distinguisher Accuracy vs Cipher Rounds")
    ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="best", fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close()


def check_positive_control_gate(sweep_df: pd.DataFrame) -> None:
    """Validate positive-control gate on rounds 1-3 for Experiment A."""
    r13 = sweep_df[sweep_df["Rounds"].isin([1, 2, 3])]
    if r13.empty: return
    best_acc = float(r13["Accuracy"].max())
    if best_acc < 0.90:
        raise RuntimeError(f"POSITIVE-CONTROL GATE FAILED: Best accuracy at rounds 1-3 is {best_acc:.2%} (< 90%). Pipeline is broken.")
    print(f"\n[+] POSITIVE-CONTROL GATE PASSED: Best model accuracy at rounds 1-3 is {best_acc:.2%} (>= 90%).")
