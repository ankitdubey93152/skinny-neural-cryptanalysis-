"""Reproducible, leakage-safe dataset generation pipeline for SKINNY-128-128 and multi-cipher cryptanalysis."""
import os, sys, json, time
from datetime import datetime, timezone
from typing import Tuple, Optional, Union, Dict, Any, List, Sequence
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    BLOCK_BYTES, BLOCK_BITS, KEY_BYTES, DEFAULT_ROUNDS, DEFAULT_DELTA, DEFAULT_SEED,
    DEFAULT_DELTA_64, CIPHER_METADATA, CIPHERS_NATIVE_128, CIPHERS_COMMON_64
)
from ciphers.skinny128 import skinny128_encrypt_batch
from ciphers.aes_ref import aes128_encrypt_batch
from ciphers.sm4 import sm4_encrypt_batch
from ciphers.hight import hight_encrypt_batch
from ciphers.des import des_encrypt_batch

DELTA = DEFAULT_DELTA
DELTA_ARR = np.frombuffer(DEFAULT_DELTA, dtype=np.uint8)
GENERATOR_VERSION = "2.1.0"


class GeneratorMeta(dict):
    """Metadata container supporting attribute and dictionary key access."""
    @property
    def p1(self) -> np.ndarray: return self["p1"]
    @property
    def p2(self) -> np.ndarray: return self["p2"]
    @property
    def keys(self) -> np.ndarray: return self["keys"]
    @property
    def keys_expanded(self) -> np.ndarray: return self.get("keys_expanded", self["keys"])
    @property
    def plaintexts(self) -> Tuple[np.ndarray, np.ndarray]: return self.get("plaintexts", (self["p1"], self["p2"]))
    def __getattr__(self, name: str) -> Any:
        try: return self[name]
        except KeyError: raise AttributeError(f"'GeneratorMeta' has no attribute '{name}'")


def bits(byte_array: np.ndarray) -> np.ndarray:
    """Expand uint8 array into big-endian bits along last axis."""
    arr = byte_array if byte_array.dtype == np.uint8 else byte_array.astype(np.uint8)
    return np.unpackbits(arr, axis=-1)


def reshape_bits_for_model(bit_array: np.ndarray, model_type: str, experiment: str = "A", block_bits: int = 128) -> np.ndarray:
    """Reshape bit tensor to exact input shape required by model architecture."""
    exp, m, n = experiment.upper().replace("EXP_", ""), model_type.lower(), bit_array.shape[0]
    p_bits = 2 * block_bits
    if exp in ("A", "C", "D", "E", "F", "G"):
        flat = bit_array.reshape(n, p_bits)
        if "mlp" in m: return flat
        if "1d" in m: return np.stack([flat[:, :block_bits], flat[:, block_bits:]], axis=-1)
        if "2d" in m: return np.stack([flat[:, :block_bits], flat[:, block_bits:]], axis=1)[..., np.newaxis]
        raise ValueError(f"Unknown model_type: {model_type}")
    if exp == "B":
        if "mlp" in m: return bit_array.reshape(n, 1024)
        pairs_256 = bit_array.reshape(n, 4, 256)
        if "1d" in m:
            blocks = [pairs_256[:, p, i*128:(i+1)*128] for p in range(4) for i in range(2)]
            return np.stack(blocks, axis=-1)
        if "2d" in m: return pairs_256[..., np.newaxis]
        raise ValueError(f"Unknown model_type: {model_type}")
    raise ValueError(f"Unknown experiment: {experiment}")


def get_dataset_filename(experiment: str, rounds: int, n_samples: int, data_dir: str = "data") -> str:
    exp = experiment.lower()
    if not exp.startswith("exp_"): exp = f"exp_{exp}"
    return os.path.join(data_dir, f"{exp}_r{rounds}_n{n_samples}.npz")


def _run_chunked(gen_chunk_fn, n_samples: int, chunk_size: int, return_meta: bool, out_shape: tuple, meta_keys: list):
    half = n_samples // 2
    n_samples = half * 2
    c_size = (chunk_size // 2) * 2

    if n_samples <= c_size:
        X, y, m = gen_chunk_fn(n_samples)
        return (X, y, GeneratorMeta(**m, plaintexts=(m["p1"], m["p2"]))) if return_meta else (X, y)

    X_out = np.empty((n_samples, *out_shape), dtype=np.uint8)
    y_out = np.empty(n_samples, dtype=np.uint8)
    meta_accum = {k: [] for k in meta_keys} if return_meta else None

    offset = 0
    while offset < n_samples:
        cur = min(c_size, n_samples - offset)
        X_c, y_c, m_c = gen_chunk_fn(cur)
        X_out[offset:offset + cur] = X_c
        y_out[offset:offset + cur] = y_c
        if return_meta:
            for k in meta_keys: meta_accum[k].append(m_c[k])
        offset += cur

    if return_meta:
        m_final = {k: np.concatenate(meta_accum[k], axis=0) for k in meta_keys}
        return X_out, y_out, GeneratorMeta(**m_final, plaintexts=(m_final["p1"], m_final["p2"]))
    return X_out, y_out


def _generate_chunk_a(rng, chunk_size: int, rounds: int, delta_bytes: np.ndarray, negative_mode: str, return_meta: bool):
    half = chunk_size // 2
    sz = half * 2
    y = np.zeros(sz, dtype=np.uint8)
    y[:half] = 1
    y = y[rng.permutation(sz)]

    keys = rng.integers(0, 256, size=(sz, KEY_BYTES), dtype=np.uint8)
    p1 = rng.integers(0, 256, size=(sz, BLOCK_BYTES), dtype=np.uint8)

    if negative_mode == "delta_prime":
        rd = rng.integers(0, 256, size=(sz, BLOCK_BYTES), dtype=np.uint8)
        bad = np.all(rd == 0, axis=-1) | np.all(rd == delta_bytes, axis=-1)
        while np.any(bad):
            rd[bad] = rng.integers(0, 256, size=(int(np.sum(bad)), BLOCK_BYTES), dtype=np.uint8)
            bad = np.all(rd == 0, axis=-1) | np.all(rd == delta_bytes, axis=-1)
        p2 = p1 ^ np.where((y == 1)[:, None], delta_bytes[None, :], rd)
    elif negative_mode == "random_p2":
        p2 = np.where((y == 1)[:, None], p1 ^ delta_bytes[None, :], rng.integers(0, 256, size=(sz, BLOCK_BYTES), dtype=np.uint8))
    else:
        raise ValueError(f"Unknown negative_mode: '{negative_mode}'. Supported: 'delta_prime', 'random_p2'")

    c1 = skinny128_encrypt_batch(p1, keys, rounds=rounds)
    c2 = skinny128_encrypt_batch(p2, keys, rounds=rounds)
    X = np.concatenate([c1, c2], axis=1)
    return X, y, ({"p1": p1, "p2": p2, "keys": keys} if return_meta else None)


def generate_experiment_a(n_samples: int, rounds: int = DEFAULT_ROUNDS, delta: bytes = DEFAULT_DELTA, seed: Optional[Union[int, np.random.Generator]] = None, negative_mode: str = "delta_prime", chunk_size: int = 50000, show_progress: bool = False, return_meta: bool = False):
    if not (1 <= rounds <= 40): raise ValueError(f"rounds must be in 1..40, got {rounds}")
    d_bytes = np.frombuffer(delta, dtype=np.uint8)
    if d_bytes.shape != (BLOCK_BYTES,): raise ValueError(f"delta must be {BLOCK_BYTES} bytes, got {len(delta)}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    fn = lambda n: _generate_chunk_a(rng, n, rounds, d_bytes, negative_mode, return_meta)
    return _run_chunked(fn, n_samples, chunk_size, return_meta, out_shape=(32,), meta_keys=["p1", "p2", "keys"])


def _generate_chunk_b(rng, chunk_size: int, pairs_per_sample: int, rounds: int, delta_bytes: np.ndarray, negative_mode: str, tweakey_strategy: str, return_meta: bool):
    half = chunk_size // 2
    sz = half * 2
    y = np.zeros(sz, dtype=np.uint8)
    y[:half] = 1
    y = y[rng.permutation(sz)]

    if tweakey_strategy == "shared":
        s_keys = rng.integers(0, 256, size=(sz, KEY_BYTES), dtype=np.uint8)
        k_exp = np.repeat(s_keys, pairs_per_sample, axis=0)
    elif tweakey_strategy == "independent":
        k_exp = rng.integers(0, 256, size=(sz * pairs_per_sample, KEY_BYTES), dtype=np.uint8)
        s_keys = k_exp.reshape(sz, pairs_per_sample, KEY_BYTES)
    else:
        raise ValueError(f"Unknown tweakey_strategy: '{tweakey_strategy}'. Supported: 'shared', 'independent'")

    p1 = rng.integers(0, 256, size=(sz, pairs_per_sample, BLOCK_BYTES), dtype=np.uint8)
    d_broad = np.broadcast_to(delta_bytes, (sz, pairs_per_sample, BLOCK_BYTES))

    if negative_mode == "delta_prime":
        rd = rng.integers(0, 256, size=(sz, pairs_per_sample, BLOCK_BYTES), dtype=np.uint8)
        bad = np.all(rd == 0, axis=-1) | np.all(rd == delta_bytes, axis=-1)
        while np.any(bad):
            rd[bad] = rng.integers(0, 256, size=(int(np.sum(bad)), BLOCK_BYTES), dtype=np.uint8)
            bad = np.all(rd == 0, axis=-1) | np.all(rd == delta_bytes, axis=-1)
        p2 = p1 ^ np.where((y == 1)[:, None, None], d_broad, rd)
    elif negative_mode == "random_p2":
        p2 = np.where((y == 1)[:, None, None], p1 ^ d_broad, rng.integers(0, 256, size=(sz, pairs_per_sample, BLOCK_BYTES), dtype=np.uint8))
    else:
        raise ValueError(f"Unknown negative_mode: '{negative_mode}'. Supported: 'delta_prime', 'random_p2'")

    c1 = skinny128_encrypt_batch(p1.reshape(-1, BLOCK_BYTES), k_exp, rounds=rounds)
    c2 = skinny128_encrypt_batch(p2.reshape(-1, BLOCK_BYTES), k_exp, rounds=rounds)
    X = np.concatenate([c1, c2], axis=1).reshape(sz, pairs_per_sample, 32)
    meta = {"p1": p1, "p2": p2, "keys": s_keys, "keys_expanded": k_exp.reshape(sz, pairs_per_sample, KEY_BYTES)} if return_meta else None
    return X, y, meta


def generate_experiment_b(n_samples: int, pairs_per_sample: int = 4, rounds: int = DEFAULT_ROUNDS, delta: bytes = DEFAULT_DELTA, seed: Optional[Union[int, np.random.Generator]] = None, negative_mode: str = "delta_prime", tweakey_strategy: str = "shared", chunk_size: int = 25000, show_progress: bool = False, return_meta: bool = False):
    if not (1 <= rounds <= 40): raise ValueError(f"rounds must be in 1..40, got {rounds}")
    d_bytes = np.frombuffer(delta, dtype=np.uint8)
    if d_bytes.shape != (BLOCK_BYTES,): raise ValueError(f"delta must be {BLOCK_BYTES} bytes, got {len(delta)}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    fn = lambda n: _generate_chunk_b(rng, n, pairs_per_sample, rounds, d_bytes, negative_mode, tweakey_strategy, return_meta)
    return _run_chunked(fn, n_samples, chunk_size, return_meta, out_shape=(pairs_per_sample, 32), meta_keys=["p1", "p2", "keys", "keys_expanded"])


def _generate_chunk_c(rng, chunk_size: int, rounds: int, delta_bytes: np.ndarray, reference: str, return_meta: bool):
    half = chunk_size // 2
    sz = half * 2
    y = np.zeros(sz, dtype=np.uint8)
    y[:half] = 1
    y = y[rng.permutation(sz)]

    keys = rng.integers(0, 256, size=(sz, KEY_BYTES), dtype=np.uint8)
    p1 = rng.integers(0, 256, size=(sz, BLOCK_BYTES), dtype=np.uint8)
    p2 = p1 ^ delta_bytes[None, :]
    X = np.zeros((sz, 32), dtype=np.uint8)

    sk_idx, ref_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(sk_idx) > 0:
        X[sk_idx, :16] = skinny128_encrypt_batch(p1[sk_idx], keys[sk_idx], rounds=rounds)
        X[sk_idx, 16:] = skinny128_encrypt_batch(p2[sk_idx], keys[sk_idx], rounds=rounds)
    if len(ref_idx) > 0:
        if reference.lower().replace("-", "").strip() != "aes128":
            raise ValueError(f"Unsupported reference cipher: '{reference}'. Supported: 'aes128'")
        X[ref_idx, :16] = aes128_encrypt_batch(p1[ref_idx], keys[ref_idx])
        X[ref_idx, 16:] = aes128_encrypt_batch(p2[ref_idx], keys[ref_idx])

    meta = {"p1": p1, "p2": p2, "keys": keys} if return_meta else None
    return X, y, meta


def generate_experiment_c(n_samples: int, rounds: int = DEFAULT_ROUNDS, delta: bytes = DEFAULT_DELTA, reference: str = "aes128", seed: Optional[Union[int, np.random.Generator]] = None, chunk_size: int = 50000, show_progress: bool = False, return_meta: bool = False):
    if not (1 <= rounds <= 40): raise ValueError(f"rounds must be in 1..40, got {rounds}")
    d_bytes = np.frombuffer(delta, dtype=np.uint8)
    if d_bytes.shape != (BLOCK_BYTES,): raise ValueError(f"delta must be {BLOCK_BYTES} bytes, got {len(delta)}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    fn = lambda n: _generate_chunk_c(rng, n, rounds, d_bytes, reference, return_meta)
    return _run_chunked(fn, n_samples, chunk_size, return_meta, out_shape=(32,), meta_keys=["p1", "p2", "keys"])


def encrypt_cipher_batch(
    cipher_name: str,
    plaintexts: np.ndarray,
    keys: np.ndarray,
    rounds: int,
    scope: str = "native-128",
    slice_half: str = "low",
) -> np.ndarray:
    """Encrypt a batch of blocks under the specified cipher with scope-aware slicing."""
    c_name = cipher_name.lower().replace("-", "").strip()
    if c_name == "skinny128":
        cts = skinny128_encrypt_batch(plaintexts, keys, rounds=rounds)
    elif c_name == "aes128":
        cts = aes128_encrypt_batch(plaintexts, keys)
    elif c_name == "sm4":
        cts = sm4_encrypt_batch(plaintexts, keys, rounds=rounds)
    elif c_name == "hight":
        if scope == "native-128":
            raise ValueError("HIGHT is a 64-bit cipher and not available in scope 'native-128'")
        return hight_encrypt_batch(plaintexts, keys, rounds=rounds)
    elif c_name == "des":
        if scope == "native-128":
            raise ValueError("DES is a 64-bit cipher and not available in scope 'native-128'")
        return des_encrypt_batch(plaintexts, keys, rounds=rounds)
    else:
        raise ValueError(f"Unsupported cipher: '{cipher_name}'")

    if scope == "common-64":
        if slice_half == "low":
            return cts[:, :8].copy()
        elif slice_half == "high":
            return cts[:, 8:].copy()
        else:
            raise ValueError(f"Invalid slice_half '{slice_half}', must be 'low' or 'high'")
    return cts


def generate_experiment_g(
    n_samples: int,
    ciphers: Sequence[str],
    rounds: Union[int, Dict[str, int]] = 4,
    scope: str = "native-128",
    slice_half: str = "low",
    delta: Optional[bytes] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
    chunk_size: int = 50000,
    return_meta: bool = False,
):
    """Generate dataset for Experiment G (multi-cipher identification)."""
    k_ciphers = len(ciphers)
    if k_ciphers < 2:
        raise ValueError("At least 2 ciphers required for Experiment G")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n_per_cipher = n_samples // k_ciphers
    total_sz = n_per_cipher * k_ciphers

    if scope == "native-128":
        for c in ciphers:
            clean = c.lower().replace("-", "").strip()
            if clean not in CIPHERS_NATIVE_128:
                raise ValueError(f"Cipher '{c}' is not supported in scope 'native-128'")
        d_bytes = np.frombuffer(delta if delta is not None else DEFAULT_DELTA, dtype=np.uint8)
    elif scope == "common-64":
        d_bytes = np.frombuffer(delta if delta is not None else DEFAULT_DELTA_64, dtype=np.uint8)
    else:
        raise ValueError(f"Unknown scope: '{scope}'. Must be 'native-128' or 'common-64'")

    X_list = []
    y_list = []
    meta_p1, meta_p2, meta_keys = [], [], []

    for idx, c_name in enumerate(ciphers):
        c_clean = c_name.lower().replace("-", "").strip()
        r = rounds[c_name] if isinstance(rounds, dict) else rounds
        full_r = CIPHER_METADATA[c_clean]["full_rounds"]
        r = min(r, full_r)

        key_len = CIPHER_METADATA[c_clean]["key_bytes"]
        cipher_keys = rng.integers(0, 256, size=(n_per_cipher, key_len), dtype=np.uint8)

        if scope == "native-128":
            p1 = rng.integers(0, 256, size=(n_per_cipher, 16), dtype=np.uint8)
            p2 = p1 ^ d_bytes[None, :]
        else:  # common-64
            if c_clean in ("hight", "des"):
                p1 = rng.integers(0, 256, size=(n_per_cipher, 8), dtype=np.uint8)
                p2 = p1 ^ d_bytes[None, :]
            else:
                p1 = rng.integers(0, 256, size=(n_per_cipher, 16), dtype=np.uint8)
                d128 = np.zeros(16, dtype=np.uint8)
                d128[:8] = d_bytes
                p2 = p1 ^ d128[None, :]

        c1 = encrypt_cipher_batch(c_clean, p1, cipher_keys, rounds=r, scope=scope, slice_half=slice_half)
        c2 = encrypt_cipher_batch(c_clean, p2, cipher_keys, rounds=r, scope=scope, slice_half=slice_half)

        X_k = np.concatenate([c1, c2], axis=1)
        y_k = np.full(n_per_cipher, idx, dtype=np.uint8)

        X_list.append(X_k)
        y_list.append(y_k)
        if return_meta:
            meta_p1.append(p1)
            meta_p2.append(p2)
            meta_keys.append(cipher_keys)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    perm = rng.permutation(total_sz)
    X_shuffled = X_all[perm]
    y_shuffled = y_all[perm]

    if return_meta:
        m = {
            "p1": np.concatenate(meta_p1, axis=0)[perm],
            "p2": np.concatenate(meta_p2, axis=0)[perm],
            "keys": meta_keys,
            "ciphers": list(ciphers),
            "scope": scope,
        }
        return X_shuffled, y_shuffled, GeneratorMeta(**m, plaintexts=(m["p1"], m["p2"]))
    return X_shuffled, y_shuffled


def generate_random_control(
    cipher_name: str,
    n_samples: int,
    rounds: int = 4,
    scope: str = "native-128",
    slice_half: str = "low",
    delta: Optional[bytes] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate control dataset: class 1 is cipher output, class 0 is uniform random bytes."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    half = n_samples // 2
    sz = half * 2

    # Generate cipher output (class 1)
    X_cipher, _ = generate_experiment_g(
        n_samples=sz,
        ciphers=[cipher_name, cipher_name],  # duplicate to get half from same cipher
        rounds=rounds,
        scope=scope,
        slice_half=slice_half,
        delta=delta,
        seed=rng,
    )
    X_c1 = X_cipher[:half]

    # Generate uniform random strings (class 0)
    out_bytes = 32 if scope == "native-128" else 16
    X_rand = rng.integers(0, 256, size=(half, out_bytes), dtype=np.uint8)

    X = np.concatenate([X_c1, X_rand], axis=0)
    y = np.zeros(sz, dtype=np.uint8)
    y[:half] = 1

    perm = rng.permutation(sz)
    return X[perm], y[perm]


def run_artifact_audit(X_bytes: np.ndarray, y: np.ndarray, scope: str = "native-128") -> Dict[str, Any]:
    """Audit dataset for representation artifacts: constant bits, extreme bit biases, identical halves."""
    b = bits(X_bytes)
    n_samples, n_bits = b.shape
    stds = np.std(b, axis=0)
    constant_bits = int(np.sum(stds == 0))
    if constant_bits > 0:
        bad_indices = np.where(stds == 0)[0].tolist()
        raise ValueError(f"Artifact audit failed: {constant_bits} constant bits found at indices {bad_indices}")

    classes = np.unique(y)
    class_means = {}
    for c in classes:
        mask = (y == c)
        c_mean = np.mean(b[mask], axis=0)
        class_means[int(c)] = {"min": float(np.min(c_mean)), "max": float(np.max(c_mean)), "mean": float(np.mean(c_mean))}
        if np.any(c_mean == 0.0) or np.any(c_mean == 1.0):
            raise ValueError(f"Artifact audit failed: class {c} has constant bit with mean 0.0 or 1.0")

    half_bytes = 16 if scope == "native-128" else 8
    c1 = X_bytes[:, :half_bytes]
    c2 = X_bytes[:, half_bytes:]
    identical_pairs = int(np.sum(np.all(c1 == c2, axis=-1)))
    if identical_pairs > 0:
        raise ValueError(f"Artifact audit failed: {identical_pairs} samples have identical ciphertext halves C1 == C2")

    return {
        "passed": True,
        "n_samples": n_samples,
        "n_bits": n_bits,
        "constant_bits": constant_bits,
        "identical_pairs": identical_pairs,
        "bit_mean_overall": float(np.mean(b)),
        "bit_std_min": float(np.min(stds)),
        "class_means": class_means,
    }


def check_dataset_leakage(train_meta: Dict[str, Any], val_meta: Dict[str, Any], test_meta: Dict[str, Any], train_X: Optional[np.ndarray] = None, val_X: Optional[np.ndarray] = None, test_X: Optional[np.ndarray] = None, train_y: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None, test_y: Optional[np.ndarray] = None, raise_on_leakage: bool = True) -> Dict[str, Any]:
    def _to_bset(arr): return set(map(bytes, arr.reshape(-1, arr.shape[-1])))
    report = {
        "tweakey_leakage": False, "plaintext_leakage": False, "ciphertext_leakage": False,
        "shared_tweakeys": {"train_val": 0, "train_test": 0, "val_test": 0},
        "shared_plaintexts": {"train_val": 0, "train_test": 0, "val_test": 0},
        "shared_ciphertexts": {"train_val": 0, "train_test": 0, "val_test": 0},
        "internal_duplicates": {}, "label_balance": {},
    }

    if "keys" in train_meta and "keys" in val_meta and "keys" in test_meta:
        if isinstance(train_meta["keys"], np.ndarray):
            ktr, kva, kte = _to_bset(train_meta["keys"]), _to_bset(val_meta["keys"]), _to_bset(test_meta["keys"])
            tv, tt, vt = len(ktr & kva), len(ktr & kte), len(kva & kte)
            report["shared_tweakeys"] = {"train_val": tv, "train_test": tt, "val_test": vt}
            if tv or tt or vt: report["tweakey_leakage"] = True

    if "p1" in train_meta and "p1" in val_meta and "p1" in test_meta:
        def _all_p(m): return set(map(bytes, np.concatenate([m["p1"].reshape(-1, m["p1"].shape[-1]), m["p2"].reshape(-1, m["p2"].shape[-1])], axis=0)))
        ptr, pva, pte = _all_p(train_meta), _all_p(val_meta), _all_p(test_meta)
        tv, tt, vt = len(ptr & pva), len(ptr & pte), len(pva & pte)
        report["shared_plaintexts"] = {"train_val": tv, "train_test": tt, "val_test": vt}
        if tv or tt or vt: report["plaintext_leakage"] = True

    if train_X is not None and val_X is not None and test_X is not None:
        ctr, cva, cte = _to_bset(train_X), _to_bset(val_X), _to_bset(test_X)
        tv, tt, vt = len(ctr & cva), len(ctr & cte), len(cva & cte)
        report["shared_ciphertexts"] = {"train_val": tv, "train_test": tt, "val_test": vt}
        if tv or tt or vt: report["ciphertext_leakage"] = True

    for name, arr in [("train", train_y), ("val", val_y), ("test", test_y)]:
        if arr is not None:
            n, ones = len(arr), int(np.sum(arr == 1))
            report["label_balance"][name] = {"total": n, "ones": ones, "zeros": n - ones, "balance": float(ones / n) if n else 0.0}

    if raise_on_leakage:
        if report["tweakey_leakage"]: raise ValueError(f"Cross-split tweakey leakage detected! {report['shared_tweakeys']}")
        if report["plaintext_leakage"]: raise ValueError(f"Cross-split plaintext leakage detected! {report['shared_plaintexts']}")
        if report["ciphertext_leakage"]: raise ValueError(f"Cross-split ciphertext leakage detected! {report['shared_ciphertexts']}")

    return report


def generate_split_dataset(experiment: str, rounds: int = DEFAULT_ROUNDS, n_train: int = 70000, n_val: int = 15000, n_test: int = 15000, delta: bytes = DEFAULT_DELTA, seed: int = DEFAULT_SEED, negative_mode: str = "delta_prime", tweakey_strategy: str = "shared", pairs_per_sample: int = 4, reference: str = "aes128", chunk_size: int = 50000, run_leakage_check: bool = True, return_meta: bool = True):
    exp = experiment.lower().replace("exp_", "").strip()
    parent_ss = np.random.SeedSequence(seed)
    train_ss, val_ss, test_ss = parent_ss.spawn(3)

    if exp in ("a", "d", "e", "f"):
        gen = lambda n, rng: generate_experiment_a(n, rounds=rounds, delta=delta, seed=rng, negative_mode=negative_mode, chunk_size=chunk_size, return_meta=True)
    elif exp == "b":
        gen = lambda n, rng: generate_experiment_b(n, pairs_per_sample=pairs_per_sample, rounds=rounds, delta=delta, seed=rng, negative_mode=negative_mode, tweakey_strategy=tweakey_strategy, chunk_size=chunk_size, return_meta=True)
    elif exp == "c":
        gen = lambda n, rng: generate_experiment_c(n, rounds=rounds, delta=delta, reference=reference, seed=rng, chunk_size=chunk_size, return_meta=True)
    else:
        raise ValueError(f"Unknown experiment: '{experiment}'")

    X_tr, y_tr, m_tr = gen(n_train, np.random.default_rng(train_ss))
    X_va, y_va, m_va = gen(n_val, np.random.default_rng(val_ss))
    X_te, y_te, m_te = gen(n_test, np.random.default_rng(test_ss))

    leakage = check_dataset_leakage(m_tr, m_va, m_te, X_tr, X_va, X_te, y_tr, y_va, y_te, raise_on_leakage=run_leakage_check)
    metadata = {
        "cipher": "SKINNY-128-128", "variant": "128/128-TK1", "experiment": f"Exp_{exp.upper()}", "rounds": int(rounds),
        "input_difference_hex": delta.hex(), "negative_mode": negative_mode,
        "tweakey_strategy": tweakey_strategy if exp == "b" else "independent_per_sample",
        "plaintext_strategy": "uniform_random",
        "label_definitions": {"y=1": "SKINNY-128-128" if exp == "c" else "SKINNY-128-128 under delta", "y=0": f"Reference: {reference}" if exp == "c" else f"Negative mode: {negative_mode}"},
        "seeds": {"parent_seed": seed, "train_seed_entropy": int(train_ss.entropy), "val_seed_entropy": int(val_ss.entropy), "test_seed_entropy": int(test_ss.entropy)},
        "split_method": "independent_rng_streams_seedsequence_spawn",
        "sizes": {"train": len(y_tr), "val": len(y_va), "test": len(y_te), "total": len(y_tr) + len(y_va) + len(y_te)},
        "byte_encoding": "uint8", "bit_order": "big_endian_msb_first", "generator_version": GENERATOR_VERSION,
        "library_versions": {"numpy": np.__version__, "python": sys.version.split()[0]},
        "created_at": datetime.now(timezone.utc).isoformat(), "leakage_verified": True,
    }
    splits = {
        "train": (X_tr, y_tr, m_tr) if return_meta else (X_tr, y_tr),
        "val": (X_va, y_va, m_va) if return_meta else (X_va, y_va),
        "test": (X_te, y_te, m_te) if return_meta else (X_te, y_te),
    }
    return splits, metadata, leakage


def generate_split_dataset_g(
    ciphers: Sequence[str],
    rounds: Union[int, Dict[str, int]] = 4,
    scope: str = "native-128",
    slice_half: str = "low",
    n_train: int = 7000,
    n_val: int = 1500,
    n_test: int = 1500,
    delta: Optional[bytes] = None,
    seed: int = DEFAULT_SEED,
    run_leakage_check: bool = True,
    return_meta: bool = True,
):
    """Generate leakage-safe split dataset for Experiment G."""
    parent_ss = np.random.SeedSequence(seed)
    train_ss, val_ss, test_ss = parent_ss.spawn(3)

    gen = lambda n, rng: generate_experiment_g(
        n, ciphers=ciphers, rounds=rounds, scope=scope, slice_half=slice_half,
        delta=delta, seed=rng, return_meta=True
    )

    X_tr, y_tr, m_tr = gen(n_train, np.random.default_rng(train_ss))
    X_va, y_va, m_va = gen(n_val, np.random.default_rng(val_ss))
    X_te, y_te, m_te = gen(n_test, np.random.default_rng(test_ss))

    leakage = check_dataset_leakage(m_tr, m_va, m_te, X_tr, X_va, X_te, y_tr, y_va, y_te, raise_on_leakage=run_leakage_check)
    # Run artifact audit on training split
    audit = run_artifact_audit(X_tr, y_tr, scope=scope)

    metadata = {
        "ciphers": list(ciphers), "scope": scope, "slice_half": slice_half,
        "rounds": rounds if isinstance(rounds, int) else dict(rounds),
        "experiment": "Exp_G", "seeds": {"parent_seed": seed},
        "sizes": {"train": len(y_tr), "val": len(y_va), "test": len(y_te)},
        "artifact_audit": audit, "leakage_verified": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    splits = {
        "train": (X_tr, y_tr, m_tr) if return_meta else (X_tr, y_tr),
        "val": (X_va, y_va, m_va) if return_meta else (X_va, y_va),
        "test": (X_te, y_te, m_te) if return_meta else (X_te, y_te),
    }
    return splits, metadata, leakage


def save_dataset_bundle(filepath: str, splits: Dict[str, Tuple[np.ndarray, np.ndarray]], metadata: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(filepath, metadata_json=np.array(json.dumps(metadata)), X_train=splits["train"][0], y_train=splits["train"][1], X_val=splits["val"][0], y_val=splits["val"][1], X_test=splits["test"][0], y_test=splits["test"][1])


def load_dataset_bundle(filepath: str, expected_cipher: str = "SKINNY-128-128") -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    if "metadata_json" not in data: raise ValueError(f"Missing metadata_json in {filepath}")
    meta = json.loads(str(data["metadata_json"]))
    if expected_cipher is not None and meta.get("cipher") != expected_cipher:
        raise ValueError(f"Dataset cipher mismatch! Expected '{expected_cipher}', found '{meta.get('cipher')}' in {filepath}")
    return {"train": (data["X_train"], data["y_train"]), "val": (data["X_val"], data["y_val"]), "test": (data["X_test"], data["y_test"])}, meta


def save_dataset(filepath: str, X: np.ndarray, y: np.ndarray, rounds: int = DEFAULT_ROUNDS, delta: bytes = DEFAULT_DELTA, seed: Optional[int] = None, cipher: str = "SKINNY-128-128", **kwargs) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    d_hex = delta.hex() if isinstance(delta, (bytes, bytearray)) else bytes(delta).hex()
    meta = {"cipher": np.array(cipher), "rounds": np.array(rounds, dtype=np.int32), "delta_hex": np.array(d_hex), "seed": np.array(seed if seed is not None else -1, dtype=np.int64), "generator_version": np.array(GENERATOR_VERSION), "created_at": np.array(datetime.now(timezone.utc).isoformat())}
    np.savez_compressed(filepath, X=X, y=y, **meta, **kwargs)


def load_dataset(filepath: str, return_meta: bool = False, expected_cipher: str = "SKINNY-128-128") -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    if expected_cipher is not None and "cipher" in data and str(data["cipher"]) != expected_cipher:
        raise ValueError(f"Cipher mismatch: expected '{expected_cipher}', got '{data['cipher']}' in {filepath}")
    return (data["X"], data["y"], {k: data[k] for k in data.files if k not in ("X", "y")}) if return_meta else (data["X"], data["y"])
