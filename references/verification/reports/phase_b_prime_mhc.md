# Phase B' Verification Report: mHC Component

**Date:** 2026-04-02 (re-run)
**Supersedes:** prior report dated 2026-04-02 (earlier same-day run)
**Component spec:** `references/components/mhc.md`
**Sources checked:**
- `references/sources/papers/mhc_2512.24880.md`
- `references/sources/code/mhc_hyper_connections.py`
- `references/sources/papers/kromhc_2601.21579.md` (KromHC — was missing in prior report; now present and verified)

**Implementation checked:** `phase1/model.py` — `KromHCResidual` (lines 353–392), `HyperConnection` (lines 395–459), `TransformerBlock._forward_mhc` (lines 691–704), `ToyTransformer` (lines 763–805)

---

## Overall Verdict: PASS

All equations, code snippets, implementation claims, and intentional deviations verified against primary sources and implementation. All checklist items pass. The one outstanding gap from the prior report (KromHC paper arXiv:2601.21579 not in references/sources/) is now resolved — the paper is present at `references/sources/papers/kromhc_2601.21579.md`. Two stale line numbers from the prior report remain in the spec (checklist items 7 and 9); they are documentation issues only, with no correctness impact.

---

## Authoritative Equations

### Eq. 1 — Standard residual
**Spec:** `x_{l+1} = x_l + F(x_l, W_l)`
**Paper source (mhc_2512.24880.md, Section 3):** identical
**Status: VERIFIED**

### Eq. 3 — Core mHC layer update
**Spec:** `x_{l+1} = H^res_l · x_l + (H^post_l)^T · F(H^pre_l · x_l, W_l)`
**Paper source (Section 4):** `x_{l+1} = H^res_l · x_l + H^post_l^T · F(H^pre_l · x_l, W_l)` — identical
**Status: VERIFIED**

### Eq. 4 — Multi-layer unrolled
**Spec:** matches paper Section 4 exactly
**Status: VERIFIED**

### Eq. 6 — Birkhoff polytope / doubly stochastic constraint
**Spec:** `P_M^res(H^res_l) := { H^res_l ∈ ℝ^(n×n) | H^res_l · 1_n = 1_n, 1_n^T · H^res_l = 1_n^T, H^res_l ≥ 0 }`
**Paper source (Section 4.1):** identical
**Status: VERIFIED**

### Eq. 8 — H_pre / H_post parameterization
**Spec:** Paper says `σ(H̃^pre)` / `2σ(H̃^post)`, with documented deviation that reference code and our implementation use softmax.
**Paper source (Section 4.2):** `H^pre_l = σ(H̃^pre_l)`, `H^post_l = 2σ(H̃^post_l)` — matches spec's description of paper formula
**Deviation correctly documented**
**Status: VERIFIED**

### Eq. 9 — Sinkhorn-Knopp projection
**Spec:** `M^(0) = exp(H̃^res_l)`, `M^(t) = T_r(T_c(M^(t-1)))`, `t_max=20` per paper.
**Paper source (Section 4.2):** identical
**Deviation (KromHC replaces SK) correctly documented**
**Status: VERIFIED**

---

## Reference Code Snippets

### sinkhorn_log (spec lines 86–100 vs. mhc_hyper_connections.py lines 64–76)
Spec snippet matches source verbatim. (Note: Our implementation does not use sinkhorn_log — KromHC replaces it. The snippet is preserved in the spec as historical context documenting what the reference code does.)
**Status: VERIFIED**

### Parameter initialization (spec lines 109–127 vs. mhc_hyper_connections.py lines 187–199)
- `H_res_logits`: `full(-8.0)`, `fill_diagonal_(0.0)` — verbatim match
- `H_pre_logits`: `full(-8.0)`, `[:, init_residual_index] = 0.0` — verbatim match
- `H_post_logits`: `zeros(num_input_views, num_residual_streams)` — verbatim match

**Status: VERIFIED**

### width_connection / depth_connection (spec lines 134–147)
- `sinkhorn_log(self.H_res_logits, ...)` and einsum for H_res: verbatim match (mhc_hyper_connections.py lines 232–244)
- `H_pre_logits.softmax(dim=-1)` and einsum for H_pre: verbatim match (lines 247–248)
- depth_connection einsum and `depth_residual_fn`: verbatim match (lines 298–306)

**Status: VERIFIED**

---

## Our Implementation Claims

### KromHCResidual replaces sinkhorn_log for H_res
**Code:** `self.krom_res = KromHCResidual()` in `HyperConnection.__init__` (line 419); `H_res = self.krom_res()` in `forward` (line 442). No `sinkhorn_log` call in model.py.
**Status: VERIFIED**

### KromHC factorization: U_k = a_k·I + (1-a_k)·Swap, a_k = softmax(logits)[0], H_res = U1 ⊗ U2
**Code (lines 384–392):**
```python
a1 = F.softmax(self.factor1_logits, dim=0)[0]
a2 = F.softmax(self.factor2_logits, dim=0)[0]
U1 = a1 * I2 + (1 - a1) * S2
U2 = a2 * I2 + (1 - a2) * S2
return torch.kron(U1, U2)
```
Matches spec Deviation 3 exactly.
**Status: VERIFIED**

### KromHC init: factor_logits = [0, -8] → a≈1 → U≈I → H_res ≈ I_4
**Code (lines 374–375):** `torch.tensor([0.0, -8.0])` for both factor1 and factor2 logits.
**Status: VERIFIED**

### H_pre init: near-one-hot (one entry=0, rest=-8)
**Code (lines 424–426):** `torch.full((n,), -8.0)`, one random entry set to `0.0`. Matches reference code pattern (random index vs. layer_index % n — functionally equivalent).
**Status: VERIFIED**

### H_post init: zeros → uniform softmax
**Code (line 431):** `torch.zeros(n)`
**Status: VERIFIED**

### H_pre / H_post use softmax (not sigmoid)
**Code (lines 446, 454):** `F.softmax(self.pre_logits, dim=0)`, `F.softmax(self.post_logits, dim=0)`
**Status: VERIFIED**

### H_pre / H_post stored as 1-D vectors (shape [n])
**Code:** `torch.full((n,), -8.0)` and `torch.zeros(n)` — both 1-D. Paper uses `ℝ^(1×n)` for single-view.
**Status: VERIFIED**

### HyperConnection forward: [B,L,n,d] → [B,L,n,d]
**Code (lines 433–459):** `mixed` from einsum `"ij, bljd -> blid"` is `[B,L,n,d]`; `distributed` from `branch_output.unsqueeze(2) * post_weights.view(1,1,n,1)` is `[B,L,n,d]`; return is `mixed + distributed`.
**Status: VERIFIED**

### _forward_mhc: pre-norm inside branch lambda
**Code (lines 702–703):**
```python
x = self.hc_attn(x, lambda inp: self.attn(self.norm1(inp)))
x = self.hc_ffn(x, lambda inp: self.ffn(self.norm2(inp)))
```
Norm applied inside lambda — un-normalized stream participates in H_res mixing; normalized input goes to sublayer.
**Status: VERIFIED**

### Stream expansion uses .clone()
**Code (line 798):** `h = h.unsqueeze(2).expand(-1, -1, self.n_streams, -1).clone()`
Prevents all streams from sharing memory via the expand view.
**Status: VERIFIED**

### Stream collapse uses softmax-normalized learned weights initialized to zeros
**Code (lines 765, 804–805):**
- `self.stream_collapse_logits = nn.Parameter(torch.zeros(self.n_streams))`
- `w = F.softmax(self.stream_collapse_logits, dim=0)`
- `h = torch.einsum("blnd, n -> bld", h, w)`

**Status: VERIFIED**

### n_streams=4 in mhc/compose configs
**Code (lines 716, 719):** `"mhc": dict(..., n_streams=4)`, `"compose": dict(..., n_streams=4)`
**Status: VERIFIED**

### mhc_dynamic removed (DHC static-only)
**Code:** KromHCResidual has no dynamic branch; HyperConnection has no dynamic path. Dynamic extension not implemented.
**Status: VERIFIED**

---

## Intentional Deviations

| Deviation | Description | Verified in Code |
|-----------|-------------|-----------------|
| 1 | softmax not sigmoid for H_pre/H_post | VERIFIED — `F.softmax` at lines 446, 454 |
| 2 | H_pre/H_post as 1-D vectors (single-view) | VERIFIED — `(n,)` shape at lines 424, 431 |
| 3 | KromHC replaces Sinkhorn-Knopp for H_res | VERIFIED — KromHCResidual at lines 353–392 |
| 4 | DHC removed, mhc_dynamic ignored | VERIFIED — no dynamic path in model.py |
| 5 | Stream collapse = learned softmax weights | VERIFIED — lines 765, 804–805 |
| 6 | n_streams=4 (paper's validated default) | VERIFIED — CONFIGS lines 716, 719 |

---

## Verification Checklist

| Item | Description | Status |
|------|-------------|--------|
| 1 | Doubly stochastic output: H_res row sums ≈ 1, col sums ≈ 1, entries ≥ 0 | VERIFIED — KromHC gives exact doubly stochastic by construction (Theorem 4.2, arXiv:2601.21579); U=a·I+(1-a)·Swap has rows/cols summing to 1; Kronecker product preserves this |
| 2 | Near-identity initialization | VERIFIED — factor_logits=[0,-8] → a≈1 → U≈I → H_res≈I_4 |
| 3 | H_pre near-one-hot at init | VERIFIED — one entry=0, rest=-8, softmax gives near-one-hot |
| 4 | H_post uniform at init | VERIFIED — zeros init, softmax gives 1/n per stream |
| 5 | Stream divergence after training | VERIFIED — structural property: non-uniform post_weights after training gives different scaled branch_output per stream |
| 6 | mHC forward shapes [B,L,n,d] | VERIFIED — both HyperConnection.forward and _forward_mhc verified |
| 7 | Stream expansion .clone() | VERIFIED — present at line 798 (spec says "line 456" — stale line number, not a code defect) |
| 8 | Stream collapse logits init to zeros, softmax applied | VERIFIED — lines 765, 804 |
| 9 | Pre-norm inside branch_fn lambda | VERIFIED — lines 702–703 (spec says "lines 387–388" — stale line numbers) |
| 10 | sinkhorn_log τ=0.05, n_iters=10 | NOT APPLICABLE — sinkhorn_log is not used in our implementation; KromHC replaces it (Deviation 3, documented) |

---

## Issues from Prior Report — Resolution Status

| Prior Issue | Resolution |
|-------------|------------|
| KromHC paper (arXiv:2601.21579) not in references/sources/ | RESOLVED — `references/sources/papers/kromhc_2601.21579.md` is now present |
| Stale line ref: stream_collapse_logits (spec says 456, was actually 532) | Still stale — actual line is now 765. Documentation issue only; no correctness impact. |
| Stale line ref: HyperConnection in TransformerBlock (spec says 387–388, was 446–447) | Still stale — actual lines are now 702–703. Documentation issue only; no correctness impact. |

---

## Summary

The mHC implementation in `phase1/model.py` is correct and consistent with the spec. All six intentional deviations are accurately described and confirmed in code. The KromHC factorization (Theorem 4.2 of arXiv:2601.21579) is now fully locally traceable with the paper present in `references/sources/papers/`. Two stale line number references remain in `mhc.md` checklist items 7 and 9 — these are minor documentation issues with no correctness impact.
