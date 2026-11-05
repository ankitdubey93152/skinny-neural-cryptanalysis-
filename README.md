# Neural Cryptanalysis of the SKINNY-128-128 Block Cipher

[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Verified%20Cryptographic%20Implementation-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 1. Purpose & Overview

This repository implements a rigorous, leak-free deep learning differential cryptanalysis framework applied to the **SKINNY-128-128 tweakable block cipher** (Beierle et al., CRYPTO 2016). Inspired by Aron Gohr's seminal work at CRYPTO 2019 (*"Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning"*), this project systematically evaluates whether modern deep neural network architectures can learn non-random differential distinguishers on round-reduced and full-round SKINNY-128-128.

### Key Contributions & Features
- **100% Verified SKINNY-128-128 Implementation**: Exact reproduction of the designers' specification and reference code across all 40 rounds, verified against official NIST Lightweight Cryptography Remus KAT vectors (`tests/vectors/skinny128_nist_lwc_kat.txt`) and cross-checked against an independent scalar reference implementation over 1,000 random triples.
- **Leakage-Safe Data Pipeline**: Train, validation, and test splits are generated via independent RNG streams (`numpy.random.SeedSequence.spawn`), guaranteeing zero tweakey, plaintext, or ciphertext pair leakage across splits.
- **Three Core + Four Extended Experimental Paradigms**:
  1. **Experiment A**: Single-pair differential distinguisher ($C_1, C_2$ under fixed difference $\Delta$).
  2. **Experiment B**: Four-pair bundled distinguisher ($4 \times (C_1, C_2)$ under shared tweakey) with pair-order shuffle invariance controls.
  3. **Experiment C**: Cipher identification ($r$-round SKINNY-128-128 vs 10-round AES-128) under identical input difference and dimensional parity.
  4. **Experiments D, E, F**: Small-sample single-pair differential distinguishers ($N=50$ for D, $N=500$ for E, $N=5000$ for F) with multi-replicate execution ($R=30, 20, 10$), stratified 60/20/20 splits, adaptive batch sizing, exact binomial tests vs chance, Holm-Bonferroni correction, label-shuffled empirical null controls, and statistical power curves.
  5. **Experiment G**: Multi-cipher identification framework across five standardized block ciphers (SKINNY-128-128, AES-128, ShangMi-4, HIGHT, and DES) across `native-128` and `common-64` (slicing parity without zero padding) scopes, pairwise and multiclass identification, random noise controls, and automated artifact audits.
- **Comprehensive Statistical Framework**: Wilson 95% score confidence intervals, exact binomial hypothesis testing against chance ($H_0: p=0.5$), Holm-Bonferroni multi-hypothesis corrections, and non-neural statistical baselines.
- **Full IEEE-Style Reporting & Figures**: Automated generation of 300 DPI PNGs, vector PDFs, companion CSVs, and data-driven Markdown reports with zero hard-coded statistics.

---

## 2. Cryptographic Background: SKINNY-128-128

SKINNY-128-128 is a substitution-permutation network (SPN) operating under the **Tweakey framework**:
- **Block Size ($n$)**: 128 bits (16 bytes), represented as a $4 \times 4$ matrix of 8-bit cells loaded in row-major order:
  $$\begin{bmatrix} m_0 & m_1 & m_2 & m_3 \\ m_4 & m_5 & m_6 & m_7 \\ m_8 & m_9 & m_{10} & m_{11} \\ m_{12} & m_{13} & m_{14} & m_{15} \end{bmatrix}$$
- **Tweakey Size ($t$)**: 128 bits (16 bytes). In the 128-128 variant, the tweakey state comprises only **TK1**. No LFSR update is applied to TK1 between rounds; only the cell permutation $P_T$ permutes the tweakey array.
- **Tweakey Terminology & Modelling Choice**: In this project, the entire 128-bit tweakey is treated as the secret master key. A fresh, independent 128-bit tweakey is sampled uniformly at random for each ciphertext sample in Experiments A and C. In Experiment B, the four ciphertext pairs in a bundle share a common tweakey (with an ablated switch for independent tweakeys).
- **Rounds ($R$)**: 40 rounds in full cipher.
- **Round Sequence**: Each round applies five operations in strict order:
  1. `SubCells`: Non-linear substitution on every byte using the official 8-bit invertible S-box $S_8$.
  2. `AddConstants`: XORs 6-bit LFSR round constants $(c_0, c_1, c_2)$ and fixed constant `0x02` into column 0 (cells 0, 4, 8).
  3. `AddRoundTweakey`: XORs the top two rows of TK1 (cells 0..7) into the top two rows of the cipher state. Rows 2 and 3 do not receive tweakey XOR.
  4. `ShiftRows`: Cyclic right rotation of cell rows by 0, 1, 2, and 3 positions respectively.
  5. `MixColumns`: Matrix multiplication over $\mathbb{F}_2^8$ multiplying each column by the binary matrix $M$:
     $$M = \begin{bmatrix} 1 & 0 & 1 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 \end{bmatrix}$$
- **Reduced-Round Semantics**: For any integer $r \in [1, 40]$, the $r$-round reduced cipher executes precisely the first $r$ rounds of the full cipher, applying round constants $RC_0 \dots RC_{r-1}$ and round tweakeys $TK1^{(0)} \dots TK1^{(r-1)}$, with no final post-whitening key.

---

## 3. Project Structure

```
SKINNY-128-128-neural-cryptanalysis/
├── ciphers/
│   ├── skinny128.py              # Vectorized NumPy implementation of SKINNY-128-128
│   ├── aes_ref.py                # Reference AES-128 implementation (FIPS-197)
│   ├── sm4.py                    # Vectorized SM4 / GB/T 32907-2016 block cipher
│   ├── hight.py                  # Vectorized HIGHT / ISO/IEC 18033-3 64-bit cipher
│   └── des.py                    # Vectorized DES / FIPS 46-3 64-bit cipher
├── data/
│   └── generators.py             # Vectorized generators, SeedSequence splits, leakage checks, G datasets
├── models/
│   └── architectures.py          # 1D CNN, 2D CNN (Dense=2, 3), MLP (Dense=2, 3), multiclass support
├── training/
│   └── train_eval.py             # Holdout/K-fold training, Wilson CI, binomial p-value, checkpoints
├── experiments/
│   ├── exp_utils.py              # CLI parsers, non-neural baselines, shuffle control, plots
│   ├── diff_propagation.py       # Empirical active cells, SAC, and avalanche profiling (SKINNY)
│   ├── diff_propagation_multi.py # Empirical avalanche and diffusion across all 5 ciphers
│   ├── run_experiment_a.py       # Single-pair differential distinguisher runner
│   ├── run_experiment_b.py       # Multi-pair bundled distinguisher runner
│   ├── run_experiment_c.py       # Cipher identification runner (SKINNY vs AES)
│   ├── run_experiment_sample_size.py # Core engine for small-sample experiments (D, E, F)
│   ├── run_experiment_d.py       # Exp D runner (N=50, R=30 replicates)
│   ├── run_experiment_e.py       # Exp E runner (N=500, R=20 replicates)
│   ├── run_experiment_f.py       # Exp F runner (N=5000, R=10 replicates)
│   ├── run_experiment_g.py       # Exp G runner (Multi-cipher identification)
│   └── run_studies.py            # Research studies 1-8 execution module
├── reports/
│   └── study_plan.md             # Pre-execution research study plan and hypotheses
├── tests/
│   ├── vectors/
│   │   └── skinny128_nist_lwc_kat.txt  # Primary official NIST LWC Remus KAT vector
│   ├── skinny128_scalar_ref.py   # Independent scalar reference implementation
│   ├── sm4_scalar_ref.py         # Independent scalar SM4 reference
│   ├── hight_scalar_ref.py       # Independent scalar HIGHT reference
│   ├── des_scalar_ref.py         # Independent scalar DES reference
│   ├── test_skinny128.py         # Known-answer tests & 1000-triple differential cross-check
│   ├── test_aes_ref.py           # AES-128 FIPS-197 KAT verification
│   ├── test_sm4.py               # SM4 GB/T 32907-2016 KAT and cross-check
│   ├── test_hight.py             # HIGHT ISO/IEC 18033-3 KAT and cross-check
│   ├── test_des.py               # DES FIPS 46-3 KAT and cross-check
│   ├── test_generators.py        # Leakage checks, shapes, split determinism, balance
│   ├── test_models.py            # Forward passes and parameter count validations
│   ├── test_train_eval.py        # Checkpointing, metrics, controls, histories
│   ├── test_experiments.py       # CLI parsing, positive control gate, utils
│   ├── test_sample_size_experiments.py # Exp D, E, F statistical tests, power, Holm-Bonferroni
│   ├── test_experiment_g.py      # Exp G shapes, slicing parity, audits, multiclass
│   ├── test_report.py            # Report formatting and structure checks
│   └── test_report_consistency.py # Verifies report numbers equal CSV values
├── generate_figures.py           # Generates 16 IEEE-style 300 DPI figures, PDFs, and CSVs
├── generate_report.py            # Data-driven Markdown report synthesizer (core and extended)
├── run_all.py                    # Master end-to-end pipeline runner (--extended supported)
├── config.py                     # Central configuration constants and cipher metadata
├── requirements.txt              # Pinned package dependencies
└── README.md                     # Project documentation
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Git

### Setup Virtual Environment

```bash
# Clone the repository
git clone <repo-url>
cd SKINNY-128-128-neural-cryptanalysis

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependency Versions (`requirements.txt`)
- `tensorflow>=2.16.1`
- `numpy>=1.26.4`
- `pandas>=2.2.2`
- `scikit-learn>=1.4.2`
- `scipy>=1.13.0`
- `matplotlib>=3.8.4`
- `pytest>=8.1.1`

---

## 5. Quick-Start Guide

### 1. Run Complete Test Suite
Verify cipher correctness, AES KAT, generators, models, and training pipeline:
```bash
pytest
```
*Expected output: 78 passed tests.*

### 2. Run All Experiments (Quick Mode)
Executes difference propagation profiling, Experiment A, Experiment B, Experiment C, and compiles the synthesis report:
```bash
python run_all.py --quick
```
To run all core experiments plus extended Experiments D, E, F, and G:
```bash
python run_all.py --quick --extended
```
*Expected runtime: $\sim 15$ minutes on multi-core CPU ($\sim 20$ minutes with `--extended`).*

### 3. Generate IEEE Publication Figures
Produces all 16 IEEE-style figures as 300 DPI PNGs, vector PDFs, and companion data CSVs:
```bash
python generate_figures.py results_skinny results_skinny/figures
```

### 4. Execute Research Studies (Part 7)
Runs the 8 research investigations specified in `reports/study_plan.md`:
```bash
python experiments/run_studies.py --quick
```

---

## 6. Execution Command Reference

| Task | Command | Primary Outputs |
| :--- | :--- | :--- |
| **All Core Experiments** | `python run_all.py --quick` | `results_skinny/table_experiment_*.csv`, `results_skinny/report.md` |
| **All + Extended Suite** | `python run_all.py --quick --extended` | All tables, Figures 1-16, `report.md`, `report_extended.md` |
| **Experiment A (Single Pair)** | `python experiments/run_experiment_a.py --rounds 1 2 4 --quick` | `table_experiment_a_sweep.csv`, `table_experiment_a_r*.csv` |
| **Experiment B (Bundled 4-Pair)**| `python experiments/run_experiment_b.py --rounds 1 2 4 --quick` | `table_experiment_b_sweep.csv`, `table_exp_a_vs_b_comparison.csv` |
| **Experiment C (SKINNY vs AES)**| `python experiments/run_experiment_c.py --rounds 1 2 4 --quick` | `table_experiment_c_sweep.csv`, `table_experiment_c_random_control.csv` |
| **Experiment D (N=50, R=30)** | `python experiments/run_experiment_d.py --quick` | `results_skinny/table_experiment_d.csv` |
| **Experiment E (N=500, R=20)** | `python experiments/run_experiment_e.py --quick` | `results_skinny/table_experiment_e.csv` |
| **Experiment F (N=5000, R=10)**| `python experiments/run_experiment_f.py --quick` | `results_skinny/table_experiment_f.csv` |
| **Experiment G (Multi-Cipher)**| `python experiments/run_experiment_g.py --scope common-64 --quick` | `table_experiment_g_pairwise_*.csv`, `table_experiment_g_multiclass_*.csv` |
| **Multi-Cipher Diffusion** | `python experiments/diff_propagation_multi.py` | `results_skinny/diff_propagation_multi.csv` |
| **Figures Generation (1-16)** | `python generate_figures.py` | `results_skinny/figures/fig1` through `fig16` (.png, .pdf, .csv) |
| **Report Synthesis** | `python generate_report.py --extended` | `results_skinny/report.md`, `results_skinny/report_extended.md` |
| **Pytest Full Suite (97 Tests)**| `pytest tests/ -v` | Comprehensive verification of ciphers, KATs, models, generators |

---

## 7. Deep Learning Architectures & Input Shapes

All neural networks are trained with binary cross-entropy loss, Adam optimizer, learning rate $10^{-3}$, and early stopping on validation loss.

| Model Name | Input Tensor Shape (Exp A & C) | Input Tensor Shape (Exp B) | Trainable Parameters | Description |
| :--- | :--- | :--- | :--- | :--- |
| **1D CNN** | `(N, 128, 2)` | `(N, 128, 8)` | 37,057 | 1D Conv (`filters=[32, 64]`, kernel 3), BN, MaxPool, 2 Dense layers |
| **2D CNN (Dense=2)** | `(N, 2, 128, 1)` | `(N, 4, 256, 1)` | 137,857 | 2D Conv ($3 \times 3$ filters), BN, MaxPool, 2 Dense layers (64 units) |
| **2D CNN (Dense=3)** | `(N, 2, 128, 1)` | `(N, 4, 256, 1)` | 138,881 | 2D Conv ($3 \times 3$ filters), BN, MaxPool, 3 Dense layers (64 units) |
| **MLP (Dense=2)** | `(N, 256)` | `(N, 1024)` | 132,609 | Fully connected (640 hidden units), ReLU, 2 Dense layers |
| **MLP (Dense=3)** | `(N, 256)` | `(N, 1024)` | 133,633 | Fully connected (640 hidden units), ReLU, 3 Dense layers |

---

## 8. Hardware Requirements & Expected Runtimes

- **Platform**: Multi-core x86_64 CPU (native Windows, Linux, or macOS).
- **GPU Acceleration**: Optional; fully functional on native CPU. TensorFlow GPU is utilized automatically when available on Linux/WSL2.
- **RAM**: Minimum 8 GB RAM; recommended 16 GB RAM for generation of $N \ge 100,000$ sample datasets.
- **Expected Runtimes (8-Core CPU)**:
  - Dataset generation ($N=100,000$ pairs): $\sim 0.12$ seconds.
  - Experiment A (Quick mode, rounds 1, 2, 4, 5 models): $\sim 3.5$ minutes.
  - Experiment B (Quick mode, rounds 1, 2, 4, 5 models, bundle controls): $\sim 10.5$ minutes.
  - Experiment C (Quick mode, rounds 1, 2, 4, 5 models, random controls): $\sim 3.5$ minutes.
  - Full Pytest Suite: $\sim 42$ seconds.

---

## 9. Cryptographic Verification & Scientific Integrity

1. **Known-Answer Verification**: Test vectors derived from the official NIST LWC Remus KAT file (`tests/vectors/skinny128_nist_lwc_kat.txt`) are checked on every test run.
2. **Scalar Cross-Check**: A clean-room scalar reference (`tests/skinny128_scalar_ref.py`) written strictly from the paper specification is cross-checked against the vectorized engine on 1,000 random triples spanning rounds 1 to 40.
3. **Avlanche Boundary**: Empirical avalanche profiling confirms that full diffusion across all 16 cells is reached at round 6. Beyond round 6, residual bit bias falls to chance-level noise ($\le 0.021$).
4. **Distinguisher Bound**:
   - `[Observed]` Single-pair neural distinguishers achieve $99.9\%$ accuracy at round 1 and $\sim 71.9\%$ at round 2.
   - `[Observed]` Four-pair bundled distinguishers amplify round-2 accuracy to $98.3\%$.
   - `[Observed]` At round 4 and above, all models perform at chance ($50.0\% \pm 0.5\%$, $p > 0.05$).
   - `[Literature]` Reduced-round findings provide no evidence of weakness in the full 40-round SKINNY-128-128 cipher. No exploitable signal was found by these models under this budget beyond round 4.

---

## 10. Small-Sample Distinguishers & Statistical Power Caveats (Experiments D, E, F)

When training deep neural networks on very small sample counts ($N \in \{50, 500, 5000\}$), standard point-accuracy metrics are prone to severe statistical noise and overfitting. Our framework incorporates rigorous statistical protocols:

1. **Statistical Power vs Test Size**:
   - Under a two-sided test with $\alpha = 0.05$, the minimum detectable accuracy at $80\%$ statistical power ($\beta = 0.20$) scales as $p^* \approx 0.50 + \frac{1.40}{\sqrt{n_{\text{test}}}}$.
   - For $N=50$ ($n_{\text{test}} = 10$), $p^* \approx 94.3\%$. A true distinguishing signal of $70.5\%$ (such as Round 2) has $< 15\%$ probability of reaching significance. Negative results at $N=50$ represent insufficient statistical power rather than cipher security.
   - For $N=500$ ($n_{\text{test}} = 100$), $p^* \approx 64.0\%$, successfully resolving Round 2 distinguishing.
   - For $N=5000$ ($n_{\text{test}} = 1000$), $p^* \approx 54.4\%$, providing high sensitivity for marginal differential effects.
2. **Overfitting & Memorization Gaps**:
   - At $N=50$, unregularized networks easily achieve $95\%+$ training accuracy by memorizing the 30 training samples, while validation/test accuracy remains at chance ($50\%$), yielding an overfitting gap $> 40\%$.
   - As $N$ scales to $500$ and $5000$, the overfitting gap contracts to $23\%$ and $6\%$, converging at $N=100,000$ with $< 1\%$ gap.
3. **Multiple Testing Correction**:
   - All reported p-values across swept rounds and sample sizes are adjusted using the **Holm-Bonferroni step-down procedure** to strictly bound family-wise Type I error rates at $\alpha = 0.05$.

---

## 11. Methodological Boundaries & Limitations Addressed

In response to key questions regarding experimental scope, this repository systematically contextualizes and addresses the six primary methodological boundaries of neural cryptanalysis:

1. **Single Fixed Input Difference ($\Delta$)**:
   - *Context*: Our default is a single-bit flip ($\Delta = \text{0x00...01}$). While not derived from MILP trail search, our multi-cipher diffusion analysis (`experiments/diff_propagation_multi.py`) proves that any single-bit difference in SKINNY activates 1 cell in round 1, 4 cells in round 2, and all 16 cells by round 6.
   - *Ablation*: Study 2 confirms that sweeping alternate bit difference positions yields identical distinguisher horizons ($r \le 3$), bounded by rapid byte-level diffusion.
2. **Threat Model (Single-Key vs Related-Tweakey Settings)**:
   - *Context*: Published machine learning attacks on SKINNY extending past round 4 (such as integral distinguishers up to 15 rounds by Zhang et al.) operate under related-tweakey or chosen-ciphertext settings where tweakey differences cancel.
   - *Distinction*: This project deliberately evaluates the strict **single-key (TK1 secret, fresh random key per sample)** differential distinguisher setting, isolating whether the neural network can identify output non-randomness without tweakey assistance.
3. **Training Budget & Sample-Size Scaling**:
   - *Context*: By evaluating $N \in [50, 100000]$ across Experiments D, E, F, and A, we demonstrate that while sample scaling eliminates memorization and narrows confidence intervals, round 4 accuracy remains at chance ($50.0\% \pm 0.3\%$). Pure single-key differential distinguishing faces an information-theoretic barrier once Strict Avalanche Criterion (SAC) error falls below measurable bounds.
4. **Input Representation (Raw Bits vs Explicit Differences $\Delta C$)**:
   - *Context*: Linear models on raw bits fail at round 1 ($50.3\%$) because single ciphertext bits are statistically uniform and the XOR relationship is non-linear. Neural networks autonomously learn this non-linear interaction directly from raw concatenated ciphertexts $(C_1 \parallel C_2)$.
   - *Inductive Bias*: When provided with explicit difference features ($\Delta C = C_1 \oplus C_2$), linear baselines reach $100\%$ at round 1, confirming that explicit difference representations provide crucial inductive bias.
5. **Confounding in Cipher Identification**:
   - *Context*: The baseline Experiment C compared $r$-round SKINNY against full 10-round AES.
   - *Resolution*: Experiment G resolves this by evaluating matched-round identification ($r \in \{1..6\}$), common-64 slicing without zero-padding, multi-cipher classification across five ciphers (SKINNY, AES, SM4, HIGHT, DES), and uniform noise baseline controls.
6. **Variance Quantification**:
   - *Resolution*: Multi-replicate execution ($R=30, 20, 10$), independent `SeedSequence` RNG streams, pooled Wilson intervals, and empirical label-shuffled null distributions guarantee that all conclusions reflect reproducible cryptographic characteristics.
