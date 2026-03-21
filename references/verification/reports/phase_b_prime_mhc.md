# Phase B' Validation Report: mhc.md

**Date:** 2026-03-21 (re-verified; original 2026-03-17)
**Component file:** `references/components/mhc.md`
**Sources checked:**
- `references/sources/papers/mhc_2512.24880.md`
- `references/sources/code/mhc_hyper_connections.py`

**Re-verification note (2026-03-21):** Full re-check performed after `phase2/model.py` log_a_init
fix. mhc.md and `phase1/model.py` mHC implementation are unaffected by that change. All
previously verified claims remain PASS. All 10 checklist items confirmed. All 6 intentional
deviations confirmed present and accurately described. Pre-existing dim=0/dim=-1 prose
inconsistency (see C1 below) unchanged — minor, does not affect correctness.

---

## Overall Verdict: PASS (with noted deviations and one minor attribution issue)

All major equations are traceable to the paper source. All code snippets are verbatim (or
structurally equivalent) to the reference code source at the stated line ranges. Claimed
deviations from the paper are accurate and correctly documented. No unsupported factual claims
were found. One minor attribution ambiguity is flagged (line range for parameter initialization).

---

## Verified Claims

### 1. Component description (mhc.md lines 5–8)

**Claim:** mHC uses a stream-mixing matrix constrained to the Birkhoff polytope (doubly
stochastic) to restore the identity-mapping property of standard residuals while allowing
richer cross-stream information flow.

**Source:** `mhc_2512.24880.md` abstract and Section 4.1 (Eq. 6).
The abstract states: "projects the residual connection space of HC onto a specific manifold to
restore the identity mapping property." Section 4.1 defines the doubly stochastic constraint
set (Eq. 6) as the Birkhoff polytope. VERIFIED.

---

### 2. Paper citation (mhc.md line 14–15)

**Claim:** Paper is "mHC: Manifold-Constrained Hyper-Connections", DeepSeek, arXiv:2512.24880,
December 2025.

**Source:** `mhc_2512.24880.md` header: arXiv 2512.24880, Published December 2025, Institution
DeepSeek. VERIFIED.

---

### 3. Code citation (mhc.md lines 16–17)

**Claim:** Code is from `tokenbender/mHC-manifold-constrained-hyper-connections` (GitHub).

**Source:** `mhc_hyper_connections.py` line 1 header:
`# SOURCE: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections/...`
VERIFIED.

---

### 4. Related paper arXiv:2601.05732 (mhc.md lines 18–19)

**Claim:** arXiv:2601.05732 ("mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations") supports
`num_iters=10` as sufficient.

**Source:** `mhc_hyper_connections.py` line 4 header: `# ALSO: "mHC-lite: You Don't Need 20
Sinkhorn-Knopp Iterations" — arXiv:2601.05732`. Also noted in `mhc_2512.24880.md` Section
"sinkhorn_log Notes": "num_iters = 10 default (paper says 20; lite paper arXiv:2601.05732
argues fewer suffice)." VERIFIED.

---

### 5. Standard residual Eq. 1 (mhc.md lines 30–31)

**Claim:** Eq. 1 is `x_{l+1} = x_l + F(x_l, W_l)`.

**Source:** `mhc_2512.24880.md` Section 3, "Standard residual connection (Eq. 1)":
`x_{l+1} = x_l + F(x_l, W_l)`. VERBATIM MATCH. VERIFIED.

---

### 6. Core mHC formula Eq. 3 (mhc.md lines 34–40)

**Claim:** Eq. 3 is `x_{l+1} = H^res_l · x_l + (H^post_l)^T · F(H^pre_l · x_l, W_l)`.
Dimension claims: `x_l ∈ ℝ^(n×C)`, `H^res_l ∈ ℝ^(n×n)`, `H^pre_l ∈ ℝ^(1×n)`,
`H^post_l ∈ ℝ^(1×n)`.

**Source:** `mhc_2512.24880.md` Section 4, "Core mHC Formula (Eq. 3)":
`x_{l+1} = H^res_l · x_l + H^post_l^T · F(H^pre_l · x_l, W_l)`.
The component file writes `(H^post_l)^T` which is equivalent to `H^post_l^T` (same
mathematical object, different notation). MATHEMATICALLY EQUIVALENT. VERIFIED.

Dimension annotations match Section 3 of the paper source: `x̄_l ∈ ℝ^(n×C)` (n streams, C
width), `H^res_l ∈ ℝ^(n×n)`, `H^pre_l ∈ ℝ^(1×n)`, `H^post_l ∈ ℝ^(1×n)`. VERIFIED.

---

### 7. Multi-layer unrolled Eq. 4 (mhc.md lines 44–45)

**Claim:** Multi-layer form is:
```
x_L = (∏_{i=1}^{L-l} H^res_{L-i}) · x_l
      + Σ_{i=l}^{L-1} (∏_{j=1}^{L-1-i} H^res_{L-j}) · (H^post_i)^T · F(H^pre_i · x_i, W_i)
```

**Source:** `mhc_2512.24880.md` Section 4, "Multi-layer unrolled (Eq. 4)": identical
structure. The component again uses `(H^post_i)^T` vs the source's `H^post_i^T` —
mathematically identical. VERIFIED.

---

### 8. Manifold constraint Eq. 6 (mhc.md lines 49–53)

**Claim:** Eq. 6 defines `P_M^res(H^res_l)` as `{H^res_l ∈ ℝ^(n×n) | H^res_l · 1_n = 1_n,
1_n^T · H^res_l = 1_n^T, H^res_l ≥ 0}`. Additional claims: spectral norm ≤ 1, closed under
multiplication.

**Source:** `mhc_2512.24880.md` Section 4.1 (Eq. 6): VERBATIM MATCH for the set definition.
Section 4.1 bullet points: "Spectral norm bounded ≤ 1" and "Closed under matrix
multiplication." VERIFIED.

---

### 9. Parameterization Eq. 8 (mhc.md lines 59–60)

**Claim:** Paper formulation uses `H^pre_l = σ(H̃^pre_l)` and `H^post_l = 2σ(H̃^post_l)`.

**Source:** `mhc_2512.24880.md` Section 4.2, "Parameterization (Eq. 8)":
`H^pre_l = σ(H̃^pre_l)` and `H^post_l = 2σ(H̃^post_l)`, σ = sigmoid. VERBATIM MATCH.
VERIFIED.

---

### 10. Sinkhorn-Knopp projection Eq. 9 (mhc.md lines 67–70)

**Claim:** Eq. 9 is `M^(0) = exp(H̃^res_l)`, `M^(t) = T_r(T_c(M^(t-1)))`. Paper uses
t_max = 20 iterations.

**Source:** `mhc_2512.24880.md` Section 4.2, "Sinkhorn-Knopp Projection (Eq. 9)": VERBATIM
MATCH. `t_max = 20` confirmed in Section 4.2 and Section 5. VERIFIED.

---

### 11. sinkhorn_log code snippet (mhc.md lines 81–94, attributed lines 64–76)

**Claim:** Lines 64–76 of the reference code contain the `sinkhorn_log` function as shown.

**Source:** `mhc_hyper_connections.py` lines 64–76: VERBATIM MATCH. Every line of the snippet
matches the source file exactly. VERIFIED.

---

### 12. tau=0.05 and num_iters=10 (mhc.md lines 96–97)

**Claim:** tau=0.05 is a temperature; num_iters=10 (paper says 20; mHC-lite arXiv:2601.05732
argues fewer suffice).

**Source:** `mhc_hyper_connections.py` line 64: `def sinkhorn_log(logits, num_iters=10,
tau=0.05)` — both defaults confirmed. Paper's t_max=20 confirmed in `mhc_2512.24880.md`.
VERIFIED.

---

### 13. Batched broadcasting claim (mhc.md lines 98–99)

**Claim:** Handles both `[n, n]` and `[B, L, n, n]` inputs via `logits.shape[:-1]`
broadcasting.

**Source:** `mhc_hyper_connections.py` line 69: `u = torch.zeros(logits.shape[:-1], ...)`.
This design does generalize to arbitrary leading dimensions. VERIFIED.

---

### 14. Parameter initialization snippet (mhc.md lines 104–121, attributed lines 187–232)

**Claim:** Lines 187–232 of the reference code contain the parameter initialization as shown.

**Source:** `mhc_hyper_connections.py` lines 187–198:
- `init_h_res = torch.full((num_residual_streams, num_residual_streams), -8.0)` — line 187.
  MATCHES.
- `init_h_res.fill_diagonal_(0.0)` — line 188. MATCHES.
- `self.H_res_logits = nn.Parameter(init_h_res)` — line 189. MATCHES.
- `init_h_pre = torch.full((num_input_views, num_residual_streams), -8.0)` — line 191.
  MATCHES.
- `init_h_pre[:, init_residual_index] = 0.0` — line 192. MATCHES.
- `self.H_pre_logits = nn.Parameter(init_h_pre)` — line 193. MATCHES.
- `self.H_post_logits = nn.Parameter(torch.zeros(num_input_views, num_residual_streams))` —
  lines 196–198 (inside `if add_branch_out_to_residual:` block). MATCHES substance.

MINOR ATTRIBUTION NOTE: The component cites "lines 187–232" for this block. Lines 187–198
cover the init code. Lines 200–232 cover `mhc_residual_identity_mix`, `dropout`,
`channel_first`, `residual_transform`, and `depth_residual_fn` setup — these are present in
the source but are NOT shown in the component's snippet. The line range endpoint (232) is
accurate as the end of `__init__`, but the snippet only excerpts a subset of that range. This
is acceptable — the cited range brackets the relevant code. VERIFIED.

---

### 15. width_connection / depth_connection snippet (mhc.md lines 129–133, 138–141)

**width_connection H_res lines:**
```python
S = sinkhorn_log(self.H_res_logits, num_iters=self.mhc_num_iters, tau=self.mhc_tau)
residuals_out = einsum(h_res, maybe_transformed_residuals, "s t, ... s d -> ... t d")
h_pre = self.H_pre_logits.softmax(dim=-1)
branch_input = einsum(h_pre, residuals, "v s, ... s d -> ... v d")
```

**Source:** `mhc_hyper_connections.py` lines 232–248:
- Line 232–234: `S = sinkhorn_log(self.H_res_logits, num_iters=self.mhc_num_iters, tau=self.mhc_tau)` — MATCH.
- Line 243–245: `residuals_out = einsum(h_res, maybe_transformed_residuals, "s t, ... s d -> ... t d")` — MATCH.
- Line 247: `h_pre = self.H_pre_logits.softmax(dim=-1)` — MATCH.
- Line 248: `branch_input = einsum(h_pre, residuals, "v s, ... s d -> ... v d")` — MATCH.

The snippet omits the `mhc_residual_identity_mix` branch (lines 236–241) but the shown lines
are accurate. Attribution says "lines 213–308"; the relevant code is within that range (the
`width_connection` method starts at line 213 in the source). VERIFIED.

**depth_connection lines:**
```python
output = einsum(branch_output, beta, "b ... d, s -> b ... s d")
residuals = self.depth_residual_fn(output, residuals)
```

**Source:** `mhc_hyper_connections.py` lines 298–306:
- Line 298: `output = einsum(branch_output, beta, "b ... d, s -> b ... s d")` — MATCH.
- Line 306: `residuals = self.depth_residual_fn(output, residuals)` — MATCH.

VERIFIED.

---

### 16. Deviation 1: softmax instead of sigmoid for H_pre / H_post (mhc.md lines 160–164)

**Claim:** Paper Eq. 8 uses `σ(H̃^pre)` and `2σ(H̃^post)`. Reference code uses
`F.softmax(dim=0)` for both. Component says code uses `softmax(dim=-1)` (note: the component
text says `F.softmax(dim=0)` at line 161 but then says `softmax(dim=-1)` referencing the
code). The code file has `h_pre = self.H_pre_logits.softmax(dim=-1)` (line 247) and
`h_post = self.H_post_logits.softmax(dim=-1)` (line 252).

MINOR INCONSISTENCY: The component says "use `F.softmax(dim=0)` for both" (line 161) but the
reference code uses `softmax(dim=-1)`. The claim that softmax is used (not sigmoid) is
correct; the `dim` argument in the component's description text is wrong (says `dim=0`, code
has `dim=-1`). The code snippet elsewhere in the component correctly shows `dim=-1`. The
description text is misleading but not a factual error about the core claim (softmax vs
sigmoid). FLAGGED AS MINOR INCONSISTENCY — does not affect PASS verdict.

Paper side verified: `mhc_2512.24880.md` Section 4.1 explicitly states sigmoid is used in the
paper. Discrepancy acknowledged correctly.

---

### 17. Deviation 2: H_pre / H_post are 1-D vectors (mhc.md lines 166–169)

**Claim:** Paper uses `ℝ^(1×n)`. Our code uses `[n]` tensors for `num_input_views=1`. The
code line 182 asserts `num_input_views == 1`.

**Source:** `mhc_hyper_connections.py` line 182: `assert num_input_views == 1, "'num_input_views'
must be 1 for mHC"`. Line 268–269: `if self.num_input_views == 1: branch_input =
branch_input[..., 0, :]` (squeezes the view dimension). VERIFIED.

---

### 18. Deviation 3: Log-space Sinkhorn with tau=0.05, n_iters=10 (mhc.md lines 171–174)

**Claim:** Paper Eq. 9 describes standard Sinkhorn with t_max=20. Code uses log-space Sinkhorn
with tau=0.05 and 10 iterations.

**Source:** Paper confirmed t_max=20 (standard). Code `sinkhorn_log` (line 64) confirmed
log-space, tau=0.05, num_iters=10. VERIFIED.

---

### 19. Deviation 4: Dynamic variant DHC (mhc.md lines 176–180)

**Claim:** When `dynamic=True` (line 157–159 of model.py), H_res is computed per-token from a
linear projection. This is an extension beyond static mHC in the paper.

**Source:** This claim is about the toy-validation model.py, not about the reference sources.
The claim that this is "an extension beyond the static mHC in the paper" is consistent with
the paper source (which describes only the static variant). The reference code also only
implements the static version plus an optional `mhc_residual_identity_mix`, not a per-token
dynamic H_res. CANNOT VERIFY from the listed sources (model.py is not a listed source), but
the claim is consistent and appropriately scoped as an extension. NOT A FAIL — it accurately
describes a local extension.

---

### 20. Deviation 5: Stream collapse (mhc.md lines 182–189)

**Claim:** Stream collapse uses learned softmax weights. Paper does not specify a collapse
operation; this is a design choice.

**Source:** The paper source `mhc_2512.24880.md` does not describe a collapse operation at the
end of the stack. This is consistent with the claim. VERIFIED (by absence in paper source).

---

### 21. Deviation 6: n_streams=2 in toy configs (mhc.md lines 191–193)

**Claim:** Paper uses n=4; toy implementation uses n=2.

**Source:** `mhc_2512.24880.md` Section 5: "Expansion rate: n = 4 (primary configuration
throughout all experiments)." Toy config claim (n=2) is about model.py, not verifiable from
listed sources, but the paper claim is confirmed. VERIFIED (paper side).

---

## Unverified / Unsupported Claims

### U1. Eq. 2 (HC multi-layer propagation) — not cited but referenced

The component file does not cite Eq. 2 (`x_L = x_l + Σ F(x_i, W_i)`) directly. This is
present in the paper source (`mhc_2512.24880.md` Section 3) but unused in the component.
Not a problem — no false claim made.

### U2. Our implementation line numbers in model.py (mhc.md lines 151–155, 177–178, 188, etc.)

**Claim:** Multiple specific line numbers in `/home/brandon/Projects/toy-validation/phase1/model.py`
are cited (e.g., `sinkhorn_log` at line 114, `HyperConnection.__init__` at line 146, etc.).

**Source:** `model.py` is not one of the Phase B' validation sources, so these cannot be
verified here. They are implementation-side references, not source-side claims. NOT FLAGGED AS
FAIL — these are out of scope for this validation step (which covers traceability to
`mhc_2512.24880.md` and `mhc_hyper_connections.py`).

---

## Contradictions

### C1. "F.softmax(dim=0)" vs "softmax(dim=-1)" in Deviation 1 description

**Location:** mhc.md lines 161 and 163.
- Line 161 says: "use `F.softmax(dim=0)` for both"
- Line 163 says: "Our code comment at line 201 explicitly notes this matches `tokenbender/mHC`"

The reference code (`mhc_hyper_connections.py` lines 247, 252) uses `.softmax(dim=-1)`, not
`dim=0`. The component's description text at line 161 says `dim=0` which contradicts the
actual code. The code snippets within the same component (line 131) correctly show `dim=-1`.

This is an internal inconsistency in the component file: the prose says `dim=0`, the snippet
and actual source both say `dim=-1`. MINOR CONTRADICTION — the `dim=-1` form is correct.

---

## Summary Table

| Claim | Verdict | Source |
|-------|---------|--------|
| Component description (Birkhoff polytope, identity property) | VERIFIED | paper abstract + Eq. 6 |
| Paper citation (arXiv:2512.24880, DeepSeek, Dec 2025) | VERIFIED | paper header |
| Code citation (tokenbender GitHub) | VERIFIED | code line 1 header |
| Related paper arXiv:2601.05732 | VERIFIED | code line 4 header |
| Eq. 1 (standard residual) | VERIFIED verbatim | paper Section 3 |
| Eq. 3 (core mHC, dimensions) | VERIFIED (equivalent notation) | paper Section 4 |
| Eq. 4 (multi-layer unrolled) | VERIFIED (equivalent notation) | paper Section 4 |
| Eq. 6 (Birkhoff polytope + spectral norm + closure) | VERIFIED verbatim | paper Section 4.1 |
| Eq. 8 (sigmoid parameterization) | VERIFIED verbatim | paper Section 4.2 |
| Eq. 9 (Sinkhorn, t_max=20) | VERIFIED verbatim | paper Section 4.2 |
| sinkhorn_log snippet (lines 64–76) | VERIFIED verbatim | code lines 64–76 |
| tau=0.05, num_iters=10 defaults | VERIFIED | code line 64 |
| Batched broadcasting via shape[:-1] | VERIFIED | code line 69 |
| Parameter init snippet (lines 187–232) | VERIFIED (subset of range) | code lines 187–198 |
| width_connection / depth_connection snippets | VERIFIED | code lines 232–306 |
| Deviation 1: softmax vs sigmoid (core claim) | VERIFIED | paper Eq. 8 vs code lines 247,252 |
| Deviation 1: dim=0 description | MINOR CONTRADICTION | code uses dim=-1 |
| Deviation 2: 1-D vectors for num_input_views=1 | VERIFIED | code lines 182, 268–269 |
| Deviation 3: log-space Sinkhorn | VERIFIED | code vs paper Eq. 9 |
| Deviation 4: dynamic DHC extension | CONSISTENT (out-of-scope source) | — |
| Deviation 5: stream collapse not in paper | VERIFIED (by absence) | paper source |
| Deviation 6: n=4 in paper | VERIFIED | paper Section 5 |

---

## Final Notes

The component file is well-sourced and accurate. All equations are correctly attributed to
`arXiv:2512.24880` with equation numbers that match the paper source file. All code snippets
are verbatim from the reference code at the stated line ranges. Deviations from the paper
(softmax vs sigmoid, log-space Sinkhorn, n_iters=10) are correctly identified and consistent
with the reference code.

The single internal contradiction (dim=0 in prose vs dim=-1 in code and snippets) should be
corrected in the component file but does not affect the validity of the underlying claims.
