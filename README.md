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
- **Three Experimental Paradigms**:
  1. **Experiment A**: Single-pair differential distinguisher ($C_1, C_2$ under fixed difference $\Delta$).
  2. **Experiment B**: Four-pair bundled distinguisher ($4 \times (C_1, C_2)$ under shared tweakey) with pair-order shuffle invariance controls.
  3. **Experiment C**: Cipher identification ($r$-round SKINNY-128-128 vs 10-round AES-128) under identical input difference and dimensional parity (128-bit blocks, zero padding/truncation).
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
│   └── aes_ref.py                # Reference AES-128 implementation (FIPS-197)
├── data/
│   └── generators.py             # Vectorized generators, SeedSequence splits, leakage checks
├── models/
│   └── architectures.py          # 1D CNN, 2D CNN (Dense=2, 3), MLP (Dense=2, 3)
├── training/
│   └── train_eval.py             # Holdout/K-fold training, Wilson CI, binomial p-value, checkpoints
├── experiments/
│   ├── exp_utils.py              # CLI parsers, non-neural baselines, shuffle control, plots
│   ├── diff_propagation.py       # Empirical active cells, SAC, and avalanche profiling
│   ├── run_experiment_a.py       # Single-pair differential distinguisher runner
│   ├── run_experiment_b.py       # Multi-pair bundled distinguisher runner
│   ├── run_experiment_c.py       # Cipher identification runner (SKINNY vs AES)
│   └── run_studies.py            # Research studies 1-8 execution module
├── reports/
│   └── study_plan.md             # Pre-execution research study plan and hypotheses
├── tests/
│   ├── vectors/
│   │   └── skinny128_nist_lwc_kat.txt  # Primary official NIST LWC Remus KAT vector
│   ├── skinny128_scalar_ref.py   # Independent scalar reference implementation
│   ├── test_skinny128.py         # Known-answer tests & 1000-triple differential cross-check
│   ├── test_aes_ref.py           # AES-128 FIPS-197 KAT verification
│   ├── test_generators.py        # Leakage checks, shapes, split determinism, balance
│   ├── test_models.py            # Forward passes and parameter count validations
│   ├── test_train_eval.py        # Checkpointing, metrics, controls, histories
│   ├── test_experiments.py       # CLI parsing, positive control gate, utils
│   ├── test_report.py            # Report formatting and structure checks
│   └── test_report_consistency.py # Verifies report numbers equal CSV values
├── generate_figures.py           # Generates 9 IEEE-style 300 DPI figures, PDFs, and CSVs
├── generate_report.py            # Data-driven Markdown report synthesizer
├── run_all.py                    # Master end-to-end pipeline runner
├── config.py                     # Central configuration constants
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
*Expected runtime: $\sim 15$ minutes on multi-core CPU.*

### 3. Generate IEEE Publication Figures
Produces all 9 required figures as 300 DPI PNGs, vector PDFs, and companion data CSVs:
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
| **All Experiments (Quick)** | `python run_all.py --quick` | `results_skinny/table_experiment_*.csv`, `results_skinny/report.md` |
| **All Experiments (Full)** | `python run_all.py --samples 100000 --epochs 10` | Full sweep results across all rounds |
| **Difference Propagation** | `python experiments/diff_propagation.py` | `results_skinny/difference_propagation_avalanche.csv` |
| **Experiment A Only** | `python experiments/run_experiment_a.py --rounds 1 2 4 --quick` | `table_experiment_a_sweep.csv`, `table_experiment_a_r*.csv` |
| **Experiment B Only** | `python experiments/run_experiment_b.py --rounds 1 2 4 --quick` | `table_experiment_b_sweep.csv`, `table_exp_a_vs_b_comparison.csv` |
| **Experiment C Only** | `python experiments/run_experiment_c.py --rounds 1 2 4 --quick` | `table_experiment_c_sweep.csv`, `table_experiment_c_random_control.csv` |
| **Research Studies** | `python experiments/run_studies.py --quick` | `results_skinny/studies/study*.csv` |
| **Figures Generation** | `python generate_figures.py` | `results_skinny/figures/fig*.png`, `.pdf`, `.csv` |
| **Report Synthesis** | `python generate_report.py` | `results_skinny/report.md` |
| **Pytest Full Suite** | `pytest tests/ -v` | Unit & integration test verification |

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
