# go-mHC: Direct Parameterization of Manifold-Constrained Hyper-Connections via Generalized Orthostochastic Matrices

**arXiv:** 2604.02309
**Published:** April 2, 2026
**Authors:** Torque Dandachi, Sophia Diggs-Galligan
**Categories:** cs.LG, cs.CL
**HTML:** https://arxiv.org/abs/2604.02309

---

## Abstract (summary)

Proposes a novel parameterization of doubly stochastic matrices (the Birkhoff polytope) for
learned stream mixing in multi-stream neural networks (mHC). Uses generalized orthostochastic
matrices via Cayley transform + block Frobenius projection. Achieves O(d³) computational
complexity vs. factorial O(d!) scaling of prior exact methods. A single hyperparameter s
interpolates between computational efficiency and full Birkhoff polytope coverage. Default s=2.

---

## Core method: three-step pipeline

### Step 1: Skew-symmetric matrix

Input: mean-pooled stream aggregate `x_agg ∈ ℝ^d` → project to skew params `z ∈ ℝ^{ns(ns-1)/2}`:
```
z = W_res(mean(streams))        # [B, L, ns*(ns-1)//2]
A = skew(z)                     # [B, L, ns, ns] — skew-symmetric
```
where `ns = n * s` (n = number of streams, s = expressivity parameter).

### Step 2: Cayley transform → orthogonal matrix

```
Q = (I − A)(I + A)⁻¹            # [B, L, ns, ns], Q ∈ SO(ns)
```

Equivalently (and used in our implementation for numerical convenience via `linalg.solve`):
```
Q = (I + A)⁻¹(I − A)
```
Both forms are identical for skew-symmetric A because (I−A) and (I+A) commute:
`(I−A)(I+A) = I − A² = (I+A)(I−A)`.

Non-singularity guaranteed: A skew-symmetric → eigenvalues purely imaginary → −1 ∉ spectrum
of A → (I+A) always invertible.

Init: W_res = 0, b_res = 0 → z = 0 → A = 0 → Q = I → H_res = I_n (identity at start).

### Step 3: Block Frobenius projection → doubly stochastic

Reshape Q as n×s blocks, take squared Frobenius norm per block:
```
Q_blocks = Q.reshape(n, s, n, s)                      # block structure
H_res[i, j] = (1/s) * ||Q[i*s:(i+1)*s, j*s:(j+1)*s]||²_F
```

**Exactly doubly stochastic by construction:**
- Non-negative: squared norms ≥ 0
- Row sums = 1: each row-block of Q is orthonormal → sum of squared norms = s → divided by s = 1
- Col sums = 1: same argument on columns (Q orthogonal)

---

## The s parameter

s controls the trade-off between expressivity and computational cost:
- **s = 1:** Orthostochastic matrices — Q is n×n, maps to Birkhoff boundary points
- **s = 2:** Default — better spectral coverage, O(n²·s²) block operations
- **s → ∞:** Approaches full Birkhoff polytope coverage

Paper recommendation: **s = 2** as the default, balancing geometry and expressivity.

go-mHC with s=2 covers substantially larger spectral regions (Karpelevič region) than
KromHC and Kronecker-based approaches. KromHC critically fails to represent cyclic
permutations beyond order 2 — go-mHC with s=2 does not have this limitation.

---

## Complexity

- Parameters: O(n²s²) free params in skew-symmetric matrix A
- FLOPs: O(n³s³) for the linear solve (Cayley transform) — "nontrivial overhead for large n"
- vs. KromHC: O(n²) params, O(n²) FLOPs — cheaper but expressivity-limited
- vs. mHC-lite (iterative Sinkhorn): factorial O(n!) in exact form, approximate in practice

At our scale (n=4, s=2): ns=8, skew params=28, linear solve is 8×8 — negligible.

---

## Experimental results

### Synthetic stream-mixing benchmarks

- Up to **10× faster convergence** vs prior exact parameterizations
- n=4 to n=32 streams tested
- go-mHC achieves theoretical minimum loss; KromHC plateaus above it
- Spectral coverage: go-mHC fills larger fraction of Karpelevič region than all competitors

### 30M parameter GPT language model

- Grammar and creativity metrics (Table 3): parity with baselines
- No specialized optimizer required — performance holds across SGD and Adam variants
- No separate parameter group needed (confirmed: standard optimizer config sufficient)

### Limitations

- Large-scale LLM experiments not reported — small-scale only
- O(d³) Cayley solve may add overhead at large n

---

## Relationship to our implementation

Our `GoMHCResidual` in `phase1/components/mhc.py` implements exactly the pipeline above:

```python
z   = self.W_res(x_agg)                          # Step 1
A   = self._skew(z)                              # Step 1
I   = self.I_ns.expand(B, L, ns, ns)
Q   = torch.linalg.solve(I + A, I - A)           # Step 2 (left form, equiv.)
Q_blocks = Q.reshape(B, L, n, s, n, s)
H_res = (Q_blocks ** 2).sum(dim=(3, 5)) / s      # Step 3
```

Uses s=2, W_res zero-init, registered I_ns buffer. Matches paper specification exactly.

---

## Key corrections vs. prior assumptions

- **Authors:** Dandachi & Diggs-Galligan — NOT DeepSeek researchers (mHC original was DeepSeek)
- **Cayley form:** Paper writes `(I−A)(I+A)⁻¹`; our code uses `(I+A)⁻¹(I−A)` — equivalent
  for skew-symmetric A (the two matrices commute)
