# SKINNY-128-128 Neural Cryptanalysis: Research Study Plan

**Document Version:** 1.0.0  
**Date:** November 2026  
**Status:** Pre-Execution Specification (Exit Gate: Written Before Study Execution)  
**Standard:** IEEE Transactions on Information Forensics and Security / IACR Transactions on Symmetric Cryptology Style  

---

## 1. Executive Summary & Hardware Budget

### 1.1 Scope and Objective
This document specifies the experimental design, hypotheses, evaluation metrics, parameter grids, seed allocations, and runtime budgets for the eight research studies investigating neural differential distinguishers on the lightweight block cipher SKINNY-128-128 (128-bit block, 128-bit TK1 tweakey, 40 rounds, SPN architecture).

### 1.2 Scientific Framing & Claim Discipline
- `[Literature]` Beierle et al. (CRYPTO 2016) established that SKINNY-128-128 achieves full diffusion across its 16 state cells within 6 rounds under the Tweakey framework, and that differential characteristics decay exponentially with round count.
- `[Observed]` In preliminary empirical avalanche profiling (`experiments/diff_propagation.py`), the number of active ciphertext difference cells reaches 16/16 at round 6, and maximum per-bit bias drops from 0.5000 at round 1 to 0.0576 at round 6 and $\le 0.021$ at rounds 7–8 (consistent with finite-sample binomial noise on $N=10^4$).
- `[Hypothesis]` Neural distinguishers without key guidance cannot distinguish SKINNY-128-128 ciphertexts from random pairs beyond round 6 under standard training budgets, because residual differential bias falls below the detectable threshold for finite training sets.
- `[Literature]` Reduced-round distinguisher results on rounds $r \le 6$ provide no evidence of weakness in the full 40-round SKINNY-128-128 cipher. If a model performs at chance on round 40 (or round 7+), the conclusion must be: *"No exploitable signal found by these neural architectures under this computational budget ($N$ samples, $E$ epochs)."*

### 1.3 Hardware Assessment & Compute Budget Estimation
- **Execution Platform:** Native Windows 11 x86_64 workstation, Intel/AMD multi-core CPU. TensorFlow 2.16+ runs in native CPU mode (TensorFlow GPU support is not available on native Windows for TF $\ge 2.11$).
- **Throughput Profiling:**
  - Data generation: $\sim 816,000$ pairs/second (chunked vectorized NumPy implementation).
  - Neural training throughput on CPU:
    - MLP (Dense=2, 3): $\sim 1.2$ seconds/epoch on $N=20,000$ samples.
    - 1D CNN: $\sim 2.5$ seconds/epoch on $N=20,000$ samples.
    - 2D CNN: $\sim 4.8$ seconds/epoch on $N=20,000$ samples.
- **Full Grid Runtime Reality Check:**
  - Evaluating 40 rounds $\times$ 5 architectures $\times$ 5 seeds $\times$ 10 epochs on $10^5$ samples would require approximately 180 CPU hours ($\sim 7.5$ days), which is unrealistic for synchronous iteration.
- **Principled Stratified Grid Proposal:**
  - Round sampling is stratified: dense evaluation over rounds $r \in [1, 2, 3, 4, 5, 6]$ where diffusion is active, plus anchor checkpoints at $r \in [8, 10, 16, 20, 32, 40]$ to verify the asymptotic chance baseline.
  - Core study datasets: $N = 20,000$ samples for rapid sensitivity/ablation sweeps, scaled to $N = 100,000$ for primary scaling anchors.
  - Multi-seed evaluations: 5 independent seeds for all core distinguishing tests ($r=1, 2, 4$).
  - Estimated Total Runtime for All 8 Studies: $\sim 45$–$60$ minutes on native CPU.

---

## 2. Study Specifications

### Study 1: Extended Round Sweep ($r \in [1..40]$)
- **Objective:** Measure the exact round boundary where neural distinguishing advantage vanishes against chance (0.50).
- **Hypothesis:** `[Hypothesis]` Distinguisher accuracy will sit near 1.00 at rounds 1–2, decay sharply over rounds 3–5, and match chance level (0.50) within Wilson 95% confidence intervals by round 6 and all subsequent rounds up to 40.
- **Metric:** Holdout Test Accuracy with 95% Wilson Score Interval, ROC-AUC, Exact Binomial $p$-value ($H_0: p=0.5$).
- **Parameter Grid:**
  - Rounds: $r \in [1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 32, 40]$
  - Models: 1D CNN, MLP (Dense=2)
  - Sample Size: $N = 20,000$ ($N_{train}=14,000, N_{val}=3,000, N_{test}=3,000$)
  - Seeds: 3 independent seeds per round
  - Delta: `00000000000000000000000000000001` (LSB of Cell 15)

### Study 2: Input Difference ($\Delta$) Sensitivity Study
- **Objective:** Investigate whether differential distinguisher performance depends on bit position, byte position, Hamming weight, and interaction with tweakey-addition rows.
- **Hypotheses:**
  - `[Hypothesis]` Differences placed in rows 0–1 (which undergo AddRoundTweakey XOR with TK1) will exhibit different propagation properties compared to differences placed in rows 2–3 (which do not receive round tweakey XOR).
  - `[Hypothesis]` High-Hamming-weight differences activate more S-box branches in round 1, accelerating diffusion and potentially decreasing round-2 distinguishability.
- **Metric:** Test Accuracy, ROC-AUC, Active Difference Bits at Round 1 & Round 2.
- **Parameter Grid:**
  - Differences tested ($\Delta$):
    1. $\Delta_1$ (Cell 15 LSB, Row 3 - No Tweakey XOR): `00000000000000000000000000000001` (HW=1)
    2. $\Delta_2$ (Cell 0 LSB, Row 0 - Receives Tweakey XOR): `01000000000000000000000000000000` (HW=1)
    3. $\Delta_3$ (Cell 0 MSB, Row 0 - Receives Tweakey XOR): `80000000000000000000000000000000` (HW=1)
    4. $\Delta_4$ (Cell 5 Byte, Row 1 - Receives Tweakey XOR): `0000000000ff00000000000000000000` (HW=8)
    5. $\Delta_5$ (High Weight Distributed, Rows 0–3): `01010101010101010101010101010101` (HW=16)
  - Rounds: $r \in [1, 2, 3]$
  - Model: 1D CNN, MLP (Dense=2)
  - Sample Size: $N = 20,000$, Seeds: 3

### Study 3: Dataset-Size Scaling ($N \in [10^3, 5\times 10^3, 2\times 10^4, 10^5]$)
- **Objective:** Quantify the sample complexity required for neural models to extract differential distinguishing features.
- **Hypothesis:** `[Hypothesis]` Round-1 patterns require minimal samples ($N \ge 1,000$ achieves $>90\%$ accuracy), whereas higher rounds require exponentially more samples to detect diminishing bias.
- **Metric:** Test Accuracy, Sample Efficiency Curve, Training Loss Convergence.
- **Parameter Grid:**
  - Dataset Sizes: $N \in [1000, 5000, 20000, 100000]$
  - Rounds: $r \in [1, 2, 4]$
  - Model: 1D CNN, MLP (Dense=2)
  - Epochs: 10 (early stopping patience 3)
  - Seeds: 3 seeds per size

### Study 4: Tweakey Generalization (Fixed Tweakeys vs Fresh Tweakeys)
- **Objective:** Determine whether neural distinguishers overfit to specific tweakeys or learn invariant differential propagation patterns.
- **Hypothesis:** `[Hypothesis]` Models trained on a small fixed tweakey pool ($K=10$) generalize to unseen tweakeys if differential propagation is tweakey-independent, but may exhibit slight test degradation if key-dependent S-box transitions are memorized.
- **Metric:** In-Distribution Test Accuracy (same 10 keys) vs Out-of-Distribution Test Accuracy (fresh independent keys).
- **Parameter Grid:**
  - Tweakey Strategies:
    - Fixed Key Pool: $K=10$ random tweakeys for training
    - Fresh Keys: Fresh independent tweakey per sample
  - Evaluation: Evaluated on test set with 100% disjoint, unseen tweakeys
  - Rounds: $r \in [1, 2]$
  - Model: MLP (Dense=2), 1D CNN
  - Sample Size: $N = 20,000$, Seeds: 3

### Study 5: Architecture Comparison & Model Capacity
- **Objective:** Evaluate inductive biases across 1D CNN, 2D CNN (Dense=2, 3), and MLP (Dense=2, 3), controlling for trainable parameter count.
- **Hypothesis:** `[Hypothesis]` 1D CNN achieves equal or superior distinguishing performance to MLP while requiring fewer parameters, due to local convolutional filters capturing bit correlations across ciphertext pairs.
- **Metric:** Test Accuracy, ROC-AUC, Trainable Parameter Count, Inference Time per Sample ($\mu$s).
- **Parameter Grid:**
  - Models:
    1. 1D CNN (2 dense layers)
    2. 2D CNN (Dense=2)
    3. 2D CNN (Dense=3)
    4. MLP (Dense=2)
    5. MLP (Dense=3)
  - Rounds: $r \in [1, 2, 4]$
  - Seeds: 5 independent seeds
  - Multiple Comparison Correction: Holm-Bonferroni adjusted $p$-values.

### Study 6: Single-Pair vs Four-Pair Bundled Distinguisher (Exp A vs Exp B)
- **Objective:** Measure the distinguishing amplification gained from bundling four ciphertext pairs under the same key versus a single pair.
- **Hypothesis:** `[Hypothesis]` Bundling 4 pairs improves distinguishing accuracy at round 2 over a single pair with equal sample count $N$, but the advantage diminishes when controlling for equal total ciphertext pairs ($4N$ single pairs in Exp A).
- **Metric:** Accuracy comparison: $\text{Acc}_B(N) - \text{Acc}_A(N)$ (equal samples) and $\text{Acc}_B(N) - \text{Acc}_A(4N)$ (equal pairs).
- **Parameter Grid:**
  - Rounds: $r \in [1, 2, 4]$
  - Comparison Modes: Equal Samples ($N=20,000$) and Equal Pairs ($4N=80,000$)
  - Shuffle Control: Verification that permuting pair order does not degrade performance.

### Study 7: Hyperparameter Sensitivity & Multi-Seed Robustness
- **Objective:** Assess stability across learning rates, batch sizes, optimizer patience, and random weight initializations.
- **Hypothesis:** `[Hypothesis]` The neural distinguisher is robust to learning rate variations within $[10^{-4}, 10^{-2}]$ and batch sizes within $[64, 512]$, with seed variance $\sigma < 0.02$.
- **Metric:** Mean Test Accuracy, Standard Deviation $\sigma$, 95% Confidence Interval across seeds.
- **Parameter Grid:**
  - Learning Rates: $\eta \in [0.0005, 0.001, 0.005]$
  - Batch Sizes: $B \in [128, 256, 512]$
  - Seeds: 5 independent seeds per cell
  - Round: $r = 2$ (transition round where sensitivity is highest)

### Study 8: Ablation Studies
- **Objective:** Isolate the necessity and impact of individual cryptographic and representation components.
- **Hypotheses:**
  - `[Hypothesis]` Inputting the explicit ciphertext difference $\Delta C = C_1 \oplus C_2$ accelerates convergence compared to concatenated raw pairs $C_1 \parallel C_2$, because the network does not need to learn the XOR operation.
  - `[Hypothesis]` Switching negative class from $P_2 = P_1 \oplus \Delta'$ to independent random $P_2$ does not degrade distinguishing accuracy at rounds 1–2.
- **Parameter Grid:**
  - Representation: $C_1 \parallel C_2$ (256 bits) vs $\Delta C$ (128 bits)
  - Negative Class: Structured $\Delta' \neq \Delta$ vs Uniformly Random $P_2$
  - Exp B Tweakey: Shared key per bundle vs Independent key per pair
  - Round: $r = 2$
  - Model: 1D CNN, MLP (Dense=2)

---

## 3. Results Output & Verification Protocol
- All numerical results are written directly to CSV and JSON files under `results_skinny/studies/`.
- Null results (accuracy within 95% Wilson interval of 0.50, binomial $p > 0.05$) are reported explicitly as null findings without embellishment.
- No statement regarding security may exceed the tested round bound.

