# Component: mHC

## Component

**mHC (Manifold-Constrained Hyper-Connections)** — multi-stream residual connections where
the stream-mixing matrix is constrained to the Birkhoff polytope (doubly stochastic), restoring
the identity-mapping property of standard residuals while allowing richer cross-stream information
flow.

---

## Sources

- **Paper:** `sources/papers/mhc_2512.24880.md` — "mHC: Manifold-Constrained Hyper-Connections",
  DeepSeek, arXiv:2512.24880, December 2025
- **Code:** `sources/code/mhc_hyper_connections.py` — author reference implementation from
  `tokenbender/mHC-manifold-constrained-hyper-connections` (GitHub)
- **Follow-up:** `sources/papers/mhc_lite_2601.05732.md` — "mHC-lite: You Don't Need 20
  Sinkhorn-Knopp Iterations", arXiv:2601.05732, January 2026. Demonstrates mHC's SK approximation
  gap and proposes Birkhoff-von Neumann exact construction. Key finding: mHC column sums deviate
  from 1.0 by up to 100% per-layer in 24-layer networks; mHC can still exhibit instability.
- **Follow-up:** `sources/papers/kromhc_2601.21579.md` — "KromHC: Manifold-Constrained
  Hyper-Connections with Kronecker-Product Residual Matrices", arXiv:2601.21579, January 2026.
  Achieves exact doubly stochasticity at O(n²C) parameter complexity via Kronecker factorization.
  Best downstream accuracy and lowest gradient norms among all HC variants.

---

## Authoritative equations

All equations below are from **arXiv:2512.24880** as transcribed in
`sources/papers/mhc_2512.24880.md`.

### Standard residual (Eq. 1)

    x_{l+1} = x_l + F(x_l, W_l)

### Core mHC layer update (Eq. 3)

    x_{l+1} = H^res_l · x_l  +  (H^post_l)^T · F(H^pre_l · x_l, W_l)

Where:
- `x_l ∈ ℝ^(n×C)` — n parallel residual streams, each of width C
- `H^res_l ∈ ℝ^(n×n)` — stream-mixing matrix (doubly stochastic via Sinkhorn)
- `H^pre_l ∈ ℝ^(1×n)` — combines n streams into one sublayer input
- `H^post_l ∈ ℝ^(1×n)` — distributes sublayer output back to n streams

### Multi-layer unrolled (Eq. 4)

    x_L = (∏_{i=1}^{L-l} H^res_{L-i}) · x_l
          + Σ_{i=l}^{L-1} (∏_{j=1}^{L-1-i} H^res_{L-j}) · (H^post_i)^T · F(H^pre_i · x_i, W_i)

### Manifold constraint — Birkhoff polytope (Eq. 6)

    P_M^res(H^res_l) := { H^res_l ∈ ℝ^(n×n) | H^res_l · 1_n = 1_n,
                                                1_n^T · H^res_l = 1_n^T,
                                                H^res_l ≥ 0 }

Row sums = 1, column sums = 1, all entries ≥ 0. Spectral norm ≤ 1. Closed under multiplication.

### Parameterization of H_pre and H_post (Eq. 8)

Paper formulation:

    H^pre_l  = σ(H̃^pre_l)
    H^post_l = 2σ(H̃^post_l)

where σ = sigmoid. **Note:** the reference code and our implementation use
`softmax(dim=-1)` instead of sigmoid for both H_pre and H_post (see Deviations below).

### Sinkhorn-Knopp projection (Eq. 9)

    M^(0) = exp(H̃^res_l)
    M^(t) = T_r(T_c(M^(t-1)))

where T_r and T_c are row and column normalization. Paper uses t_max = 20 iterations.

---

## Reference implementation

Source: `sources/code/mhc_hyper_connections.py` (tokenbender/mHC-manifold-constrained-hyper-connections).

### Log-space Sinkhorn (lines 64–76)

```python
def sinkhorn_log(logits, num_iters=10, tau=0.05):
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))
```

`tau=0.05` is a temperature applied before Sinkhorn. `num_iters=10` (paper says 20; mHC-lite
arXiv:2601.05732 argues fewer suffice). `log_marginal = zeros(n)` targets uniform marginals
(doubly stochastic). Handles both plain `[n, n]` and batched `[B, L, n, n]` inputs via
`logits.shape[:-1]` broadcasting.

### Parameter initialization (lines 187–232 in reference code)

```python
# H_res: near-identity doubly stochastic start
init_h_res = torch.full((num_residual_streams, num_residual_streams), -8.0)
init_h_res.fill_diagonal_(0.0)
self.H_res_logits = nn.Parameter(init_h_res)
# After sinkhorn_log: diagonal ≈ 1.0, off-diagonal ≈ 0.0  (near identity)

# H_pre: near-one-hot start (picks one stream per view)
init_h_pre = torch.full((num_input_views, num_residual_streams), -8.0)
init_h_pre[:, init_residual_index] = 0.0
self.H_pre_logits = nn.Parameter(init_h_pre)
# After softmax: one entry ≈ 1.0, rest ≈ 0.0  (hard selection of one stream)

# H_post: uniform start
self.H_post_logits = nn.Parameter(
    torch.zeros(num_input_views, num_residual_streams)
)
# After softmax: all entries = 1/num_streams  (uniform distribution to all streams)
```

### width_connection / depth_connection (lines 213–308)

`width_connection` applies H_res (Sinkhorn on `H_res_logits`) and H_pre (softmax on
`H_pre_logits`) to produce the mixed residual streams and the branch input:

```python
S = sinkhorn_log(self.H_res_logits, num_iters=self.mhc_num_iters, tau=self.mhc_tau)
residuals_out = einsum(h_res, maybe_transformed_residuals, "s t, ... s d -> ... t d")
h_pre = self.H_pre_logits.softmax(dim=-1)
branch_input = einsum(h_pre, residuals, "v s, ... s d -> ... v d")
```

`depth_connection` applies H_post (softmax on `H_post_logits`) to distribute the branch
output back across all streams, then adds to the mixed residual:

```python
output = einsum(branch_output, beta, "b ... d, s -> b ... s d")
residuals = self.depth_residual_fn(output, residuals)
```

---

## Our implementation

**File:** `/home/brandon/Projects/toy-validation/phase1/model.py`

| Symbol | Location |
|--------|----------|
| `KromHCResidual` | replaces `sinkhorn_log` |
| `HyperConnection.__init__` | uses `KromHCResidual` for H_res |
| `HyperConnection.forward` | unchanged structure |
| `TransformerBlock._forward_mhc` | unchanged |
| Stream expansion (embed → [B,L,n,d]) | unchanged |
| Stream collapse (learned weighted sum) | unchanged |

**Note:** Line numbers shift when the model is edited. Use `grep -n "class KromHCResidual"` etc. to locate current positions.

### Intentional deviations from the paper (matching reference code or later papers)

1. **softmax instead of sigmoid for H_pre / H_post.**
   Paper Eq. 8 uses `σ(H̃^pre)` and `2σ(H̃^post)`. The reference code and our implementation
   use `F.softmax(dim=-1)` for both. Softmax enforces Σ = 1 and non-negativity; sigmoid does
   not enforce Σ = 1.

2. **H_pre / H_post are 1-D vectors (shape `[n]`), not matrices.**
   The paper uses `H^pre ∈ ℝ^(1×n)` and `H^post ∈ ℝ^(1×n)` for a single-view case
   (`num_input_views=1`). Our `HyperConnection` always uses single-view, so these are stored
   as `[n]` tensors. Functionally equivalent.

3. **KromHC (arXiv:2601.21579) replaces Sinkhorn-Knopp for H_res.**
   Paper Eq. 9 uses iterative SK (t_max=20). mHC-lite (arXiv:2601.05732) and KromHC both
   demonstrate that SK leaves up to 100% column-sum deviation per layer. Our implementation
   uses **KromHC** with n=4=2×2 Kronecker factorization:
   - Two 2×2 factor matrices: `U_k = a_k·I + (1-a_k)·Swap`, `a_k = softmax(logits)[0]`
   - `H_res = U1 ⊗ U2` via `torch.kron` — exactly doubly stochastic by Theorem 4.2
   - Initialization: `factor_logits = [0, -8]` → `a≈1` → `U≈I` → `H_res ≈ I_4`
   - No iterations, no approximation error

4. **Dynamic variant (DHC) removed.**
   The original implementation supported `dynamic=True` (input-conditional H_res via Sinkhorn).
   KromHC is static-only. The `mhc_dynamic` CLI flag is accepted but ignored.

5. **Stream collapse uses learned softmax weights, not a fixed operation.**
   After all blocks, streams are collapsed via:
   ```python
   w = F.softmax(self.stream_collapse_logits, dim=0)
   h = torch.einsum("blnd, n -> bld", h, w)
   ```
   `stream_collapse_logits` is initialized to zeros (uniform at start).
   The paper does not specify a collapse operation; this is our design choice.

6. **n_streams=4 (paper's validated default).**
   Original Phase 1 used n=2 (below validated minimum). Re-run uses n=4, matching:
   - The original HC paper's validated default (OLMo-1B-DHC with n=4)
   - The KromHC paper's experimental configuration (n=4=2×2)
   - The Phase 3 target configuration

---

## Scale limitations and known failure modes

### No published results below 1B parameters

The original HC paper (ByteDance, arXiv:2409.19606) and the mHC paper both report
experiments at 1B parameters minimum. The smallest HC result is OLMo-1B-DHC with n=4.
**No sub-1B results exist in any of the three mHC-family papers.** Behavior at ~28M params
is extrapolation. The Phase 1 result (mHC: 3.5736 vs baseline: 3.5875) — mHC losing to
baseline — is consistent with HC gains being primarily a depth/scale phenomenon.

### n=2 is below the validated minimum

The original HC paper found **n=1 hurts vs baseline** and validates n=4 as the recommended
default. n=2 (used in this project for parameter-budget reasons) is the floor of potentially
helpful territory but is untested in the literature. At n=2, stream divergence is limited —
H_post can only route between two streams, giving less gradient diversity than n=4.

### The Sinkhorn approximation gap (from mHC-lite, arXiv:2601.05732)

mHC-lite demonstrates that 10–20 fixed SK iterations do not guarantee exact doubly
stochasticity. In 24-layer networks, column sums deviate from 1.0 by up to 100% per-layer,
and the composite product deviates up to 220%. At 8 layers the compounding is ~1/3 as
severe, but the constraint violation is still present.

**This is the most likely explanation for the rising grad norm observed in Phase 1 mHC
training (gnorm: 0.77 early → 1.395 at step 100k).** The doubly stochastic constraint is
not being fully enforced, allowing H_res to drift and spectral norm to slowly exceed 1.0.

### The rising grad norm pattern

Rising grad norm in mHC (vs. flat grad norm in baseline) is consistent with the HC
instability described in arXiv:2512.24880 Section 3.1: unconstrained or imperfectly
constrained H_res matrices compound across layers. At 8 layers with 10 SK iterations
the effect is slow rather than catastrophic (no loss surge), but it accumulates over 100k
steps.

---

## Follow-up work

### mHC-lite (arXiv:2601.05732, January 2026)

Replaces iterative SK with exact Birkhoff-von Neumann construction:

    H_l^{res} = Σ_{k=1}^{n!} a_{l,k} P_k,   a_l = softmax(α · x̂ · W + b)

Exactly doubly stochastic by construction. Native PyTorch (no custom kernels).
Outperforms mHC on val loss (3.185 vs 3.204) and throughput (+2% vs -6.7%).

**Critical limitation:** parameter count scales as O(nC · n!). Practical only for n ≤ 4–5.

**For n=2 (this project's n_streams=2):** mHC-lite is trivially exact. Only 2! = 2
permutation matrices exist (identity I and swap Swap):

    H_l^{res} = a · I + (1-a) · Swap,   a = softmax([logit_0, logit_1])[0]

A single scalar a ∈ (0, 1) parameterizes the full Birkhoff polytope at n=2.
Implementation cost: negligible. If re-running mHC, this is the correct H_res for n=2.

### KromHC (arXiv:2601.21579, January 2026)

Kronecker-product factorization: decomposes n×n H_res into smaller factor matrices
whose Kronecker product is exactly doubly stochastic (Theorem 4.2).

    H_l^{res} = U_l^1 ⊗ U_l^2,   each U_l^k constructed via B-vN on smaller dimension

Parameter complexity: O(n²C) — 48% fewer params than mHC, 61% fewer than mHC-lite at n=4.
Practical for large n (n=16 adds only 11.37M params vs mHC-lite's intractable n!).
Achieves best downstream accuracy and lowest gradient norms among all HC variants.

**Requirement:** n must be composite (n=4=2×2 ✓, n=6 ✓, n=8 ✓; n=5 prime ✗).

---

## Verification checklist

1. **Doubly stochastic output:** After `sinkhorn_log`, verify `H_res.sum(dim=-1)` ≈ 1.0 and
   `H_res.sum(dim=-2)` ≈ 1.0 for all entries; verify all entries ≥ 0.

2. **Near-identity initialization:** At init (before any gradient steps), confirm `H_res`
   diagonal ≈ 1.0 and off-diagonal entries are near 0 (should be < 0.01 with -8 logits and
   τ=0.05).

3. **H_pre near-one-hot at init:** At init, confirm `pre_weights = softmax(pre_logits)` has
   one entry near 1.0 and the rest near 0.0 (from -8 off-diagonal init).

4. **H_post uniform at init:** At init, confirm `post_weights = softmax(post_logits)` ≈
   `[1/n, ..., 1/n]` (from zeros init).

5. **Stream divergence:** Verify that after several training steps with n_streams=2, the two
   stream vectors are no longer identical. Without H_post broadcasting different weights,
   streams cannot diverge (see model.py comment at lines 109–112).

6. **mHC forward shapes:** Confirm `_forward_mhc` receives `[B, L, n, d]` and returns
   `[B, L, n, d]`; confirm `HyperConnection.forward` receives `[B, L, n, d]` and returns
   `[B, L, n, d]`.

7. **Stream expansion:** Confirm the `.clone()` at line 456 is present — without it, the
   expand creates a view and all streams share memory, which would break the divergence
   mechanism.

8. **Stream collapse weights:** Confirm `stream_collapse_logits` is initialized to zeros and
   that `softmax` is applied (not raw logits) before the einsum.

9. **Pre-norm inside branch_fn:** Confirm the lambda at lines 387–388 applies `norm1`/`norm2`
   inside the branch callable, so that the normalized input goes to the sublayer while the
   un-normalized stream participates in H_res mixing. This matches the paper's structure where
   F() operates on the pre-processed branch input, not the raw stream.

10. **Sinkhorn τ and n_iters match reference:** Confirm `sinkhorn_log` in our code uses
    `tau=0.05` and `n_iters=10`, matching the reference code defaults (not the paper's
    t_max=20).
