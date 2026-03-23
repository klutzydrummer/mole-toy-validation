# mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations

**arXiv:** 2601.05732
**Published:** January 9, 2026
**Authors:** Yongyi Yang, Jianyang Gao
**Code:** https://github.com/FFTYYY/mhc-lite

---

## Abstract (paraphrased — verbatim not rendered by arXiv HTML)

Identifies two limitations in DeepSeek's mHC (arXiv:2512.24880):

1. **Approximation gap**: Finite Sinkhorn-Knopp iterations (20) do not guarantee exact
   doubly stochastic matrices. Measured deviation: column sums deviate from 1.0 by up to
   100% per-layer; composite product across 24 layers deviates up to 220%.

2. **Engineering complexity**: Specialized CUDA kernels are required for efficient SK
   execution; not portable to standard PyTorch.

Proposes mHC-lite: replaces iterative SK with a direct construction of doubly stochastic
matrices as convex combinations of permutation matrices (Birkhoff-von Neumann theorem).
Guarantees exact doubly stochasticity using only standard softmax + matrix operations.

---

## Core Problem: SK Approximation Gap

**Eq. 3 — Relative range (convergence criterion):**

    ν := min_{i,j: x_{i,j}>0} x_{i,j} / max_{i,j} x_{i,j}

Theoretical convergence of Sinkhorn-Knopp to ε-accuracy requires
O(n² log(n/ν) / ε²) iterations. In practice:

> "~27.9% of SK inputs have log(1/ν) ≥ 30 (relative range 1/ν ≥ 10^13);
> 20 fixed iterations is provably insufficient for these inputs."

Evidence from Figure 3:
- mHC column-sum MAE per layer: up to 100%
- mHC product-of-matrices column-sum MAE across 24 layers: up to 220%

This means the "doubly stochastic" constraint in mHC is routinely violated in practice.

---

## Method: Birkhoff-von Neumann Construction

**Theorem 3.1 (Birkhoff-von Neumann):**
Any doubly stochastic matrix X ∈ B_n can be expressed as a convex combination of
permutation matrices:

    X = Σ_{k=1}^{n!} a_k P_k,   a_k ≥ 0, Σ a_k = 1

**Eq. 4 — mHC-lite weight vector:**

    a_l = softmax(α_l^{res} x̂_l' W_l^{res} + b_l^{res})

**Eq. 5 — Doubly stochastic construction:**

    H_l^{res} = Σ_{k=1}^{n!} a_{l,k} P_k

where {P_k}_{k=1}^{n!} enumerates all n×n permutation matrices. Since each P_k is
doubly stochastic, and the convex combination of doubly stochastic matrices is doubly
stochastic, H_l^{res} is exactly doubly stochastic by construction — no iterations needed.

**Initialization:**

    b_l^{res} initialized to -8 for all permutations except the identity permutation (set to 0)
    → at init, a_l ≈ one-hot on identity → H_l^{res} ≈ I (standard residual start)

**Practical constraint on n:**
For n=4: 4! = 24 permutation matrices — tractable.
For n=8: 8! = 40,320 — intractable. mHC-lite is only practical for n ≤ 4 or n ≤ 5.

**Special case n=2 (relevant to this project's n_streams=2):**
For n=2, there are exactly 2! = 2 permutation matrices: the identity I and the swap
matrix Swap. mHC-lite for n=2 reduces to:

    H_l^{res} = a · I + (1-a) · Swap,   a = softmax([logit_0, logit_1])[0]

This is a single scalar a ∈ (0, 1) — trivially exact and nearly free to compute.

---

## Empirical Results

**Models:** Small (~45M, 6 layers), Medium (~0.12B, 12 layers), Large (~0.36B, 24 layers),
n=4 streams, 8× A100, 10,000 steps (~1.3B tokens).

**FineWeb-Edu final validation loss (Large model):**

| Method | Val loss |
|--------|----------|
| Baseline | 3.240 |
| HC (2409.19606) | 3.248 |
| mHC (2512.24880) | 3.204 |
| **mHC-lite** | **3.185** |

mHC-lite is best.

**Throughput (Medium model, relative to baseline):**

| Method | Relative throughput |
|--------|-------------------|
| Baseline | 1.0× |
| HC | ~0.95× |
| mHC | ~0.90× |
| **mHC-lite** | **~1.02×** |

mHC-lite exceeds baseline throughput with no specialized kernels.

**Gradient norms:** mHC-lite shows lower mean and lower fluctuation than mHC.

**Doubly stochastic fidelity (column-sum MAE):**
- mHC: up to 100% per-layer deviation
- mHC-lite: 0% (exact by construction)

---

## Comparison to mHC

| Aspect | mHC (2512.24880) | mHC-lite (2601.05732) |
|--------|------------------|-----------------------|
| Doubly stochastic | Approximate (SK) | Exact (B-vN) |
| Column-sum deviation | Up to 100%/layer | 0% |
| Custom kernels | Yes | No (PyTorch native) |
| Throughput overhead | −6.7% | +2% |
| Gradient stability | Fluctuations | Lower, smoother |
| Max practical n | Any (SK always runs) | n ≤ 4–5 (n! grows) |
| Parameter complexity | O(n³C) | O(nC · n!) |

---

## Key Claims (Direct Quotes)

> "Finite SK iterations in mHC can leave a non-negligible approximation gap to the
> doubly stochastic constraint."

> "Stability issues persist in mHC despite the manifold constraint."

> "mHC can still exhibit instability in practice (though less severe than HC), whereas
> mHC-lite eliminates this issue entirely."

> "Perfect doubly stochasticity of H_l^{res} and its composition is guaranteed by
> construction via the Birkhoff-von Neumann theorem."
