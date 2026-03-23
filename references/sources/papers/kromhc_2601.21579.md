# KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices

**arXiv:** 2601.21579
**Published:** January 29, 2026
**Authors:** Wuyang Zhou, Yuxuan Gu, Giorgos Iacovides, Danilo Mandic
**Institution:** Imperial College London (inferred from Mandic)

---

## Abstract (verbatim)

> "The success of Hyper-Connections (HC) in neural networks (NN) has also highlighted
> issues related to its training instability and restricted scalability. The
> Manifold-Constrained Hyper-Connections (mHC) mitigate these challenges by projecting
> the residual connection space onto a Birkhoff polytope, however, it faces two issues:
> 1) its iterative Sinkhorn-Knopp (SK) algorithm does not always yield exact doubly
> stochastic residual matrices; 2) mHC incurs a prohibitive O(n³C) parameter complexity
> with n as the width of the residual stream and C as the feature dimension..."

(Remainder: proposes Kronecker-product factorization to achieve exact doubly stochasticity
at O(n²C) parameter complexity, beating both mHC and mHC-lite on downstream tasks with
fewer parameters.)

---

## Core Method: Kronecker-Product Factorization

**Theorem 4.2 (Kronecker closure — key theoretical result):**

Let B_n denote the set of n×n doubly stochastic matrices.
Let U_l^1 ∈ B_{i1} and U_l^2 ∈ B_{i2}.
Then their Kronecker product satisfies:

    U_l^1 ⊗ U_l^2 ∈ B_{i1·i2}

The Kronecker product of doubly stochastic matrices is doubly stochastic. This is the
key property enabling exact constraint enforcement.

**Eq. 14 — KromHC parametrization:**

    a_l^k = Softmax(α_l^{res} x_l' W_l^{res,k} + b_l^{res,k})     [learnable coefficients]
    U_l^k = Σ_{m=1}^{i_k!} a_l^k(m) P_m                            [factor matrix via B-vN]
    H_l^{res} = ⊗_{k} U_l^k                                         [Kronecker product]

Each factor U_l^k is constructed as a convex combination of permutation matrices
(like mHC-lite), but over a *smaller* dimension i_k, where n = ∏ i_k.

Example for n=4: factor into two n=2 matrices. Each n=2 factor has 2! = 2 permutations
(just identity and swap). The Kronecker product of two 2×2 doubly stochastic matrices is
a 4×4 doubly stochastic matrix. Exactly doubly stochastic; only 2×2 = 4 permutation
matrices needed per factor.

**Initialization:** Each factor U_l^k initialized near identity
(a_l^k[identity permutation] ≈ 1, others ≈ 0) → H_l^{res} ≈ I at start.

**Ablation finding:** Sharing a single global α_l^{res} across all factor matrices
(not separate α_l^{res,k} per factor) yields better performance.

---

## Parameter Complexity Comparison

| Method | Additional params | Formula |
|--------|------------------|---------|
| mHC | 1,844K | O(n³C) |
| mHC-lite | 2,433K | O(nC · n!) |
| **KromHC** | **959K** | **O(n²C)** |

KromHC uses 48% fewer parameters than mHC and 61% fewer than mHC-lite at n=4.

**Scaling to large n (Figure 4):**
- n=4: KromHC 959K, mHC-lite 2,433K, mHC 1,844K
- n=8: KromHC adds ~3.8M, mHC-lite would require n!-level blowup
- n=16: KromHC adds only ~11.37M — scales gracefully
- mHC-lite is intractable for n > 5 (n! = 120 at n=5, 720 at n=6)

**Practical constraint:** n must be composite (factorable) for Kronecker decomposition.
n=4 = 2×2 ✓, n=6 = 2×3 ✓, n=8 = 2×4 ✓, n=5 (prime) ✗ — workaround: use highly
composite n values.

---

## Empirical Results (D=12 blocks, n=4 streams)

**Commonsense reasoning average accuracy (Table 4):**

| Method | Accuracy |
|--------|----------|
| Baseline residual | 46.2% |
| mHC | 44.5% |
| mHC-lite | 44.4% |
| **KromHC** | **47.7%** |

**Language modeling (Table 5):**

| Method | Accuracy |
|--------|----------|
| Baseline residual | 23.7% |
| mHC | 22.9% |
| mHC-lite | 23.3% |
| **KromHC** | **24.0%** |

KromHC is best on all downstream tasks.

**Column-sum MAE across 24 layers (Figure 2):**

| Method | Column-sum MAE |
|--------|---------------|
| mHC | ~0.05 |
| mHC-lite | 0 |
| KromHC | 0 |

Both mHC-lite and KromHC are exactly doubly stochastic; mHC is not.

**Gradient norms:** KromHC consistently achieves the lowest gradient norm among all
HC variants, with smoother trajectories across 7,000 training steps.

---

## Comparison to mHC and mHC-lite

| Criterion | mHC | mHC-lite | KromHC |
|-----------|-----|----------|--------|
| Exact doubly stochastic | No | Yes | Yes |
| Parameter complexity | O(n³C) | O(nC · n!) | O(n²C) |
| PyTorch native | No | Yes | Yes |
| Downstream accuracy | Worst | Middle | Best |
| Gradient norm | Highest | Middle | Lowest |
| Scalable to large n | Moderately | No (n! blowup) | Yes |
| Requires composite n | No | No | Yes |

---

## Relation to Tucker Decomposition

The Kronecker factorization is mathematically related to Tucker decomposition and
tensor networks. The residual matrix H_l^{res} is treated as a tensor decomposed
into smaller factor tensors. This framing suggests future extensions: higher-order
Kronecker products, non-square factorizations, etc.
