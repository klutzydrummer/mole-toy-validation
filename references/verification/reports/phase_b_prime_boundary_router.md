# Phase B' Verification Report: boundary_router

**Component:** `boundary_router`
**Date:** 2026-04-06
**Verifier:** Claude Sonnet 4.6 (automated Phase B' agent)
**Overall verdict:** PASS with issues

Issues are documentation mismatches in spec deviations 1–3 (describe outdated behaviour) and
deviation 4 referenced in code but missing from spec. No implementation bugs found. All
checklist items pass. Spec updated as part of this verification run.

---

## Sources read

- `references/components/boundary_router.md` — spec
- `references/sources/papers/hnet_2507.07955.md` — H-Net paper summary
- `references/sources/code/hnet_boundary.py` — H-Net reference implementation (verbatim)
- `references/sources/papers/dlcm.md` — DLCM paper summary
- `phase2/components/boundary_router.py` — implementation under test
- `phase2/model.py` lines 135–165 — identity re-init and residual_proj zero-init
- `phase2/components/zone_d.py` — STE and threshold usage in SimpleDecoder
- `phase2/train.py` lines 85–101, 320–355 — ratio_loss, boundary_entropy, training loop

---

## Authoritative equations — cross-check

### H-Net Eq. (4) — Boundary probability (spec lines 27–34)

```
pₜ = (1/2)(1 − (qₜᵀ kₜ₋₁) / (‖qₜ‖ ‖kₜ₋₁‖))
bₜ = 𝟙{pₜ ≥ 0.5}
```

**Status: VERIFIED.**
- H-Net paper (`hnet_2507.07955.md` lines 62–65): exact match.
- `hnet_boundary.py` lines 130–134: `argmax(boundary_prob) == 1` is equivalent to `p >= 0.5`. ✓
- DLCM `dlcm.md` line 78: `bₜ = [pₜ ≥ 0.5]` — independent confirmation. ✓

### DLCM Eq. (6) — Index convention note (spec lines 42–44)

**Status: VERIFIED.** `dlcm.md` Eq. 6 (line 71) matches spec. Both H-Net and DLCM compute the
same adjacent cosine dissimilarity with swapped q/k index conventions (`dlcm.md` lines 282–291). ✓

### H-Net / DLCM Eq. (10) — Ratio loss (spec lines 51–58)

**Status: VERIFIED.** `hnet_2507.07955.md` lines 176–181 match spec verbatim. ✓

---

## Reference implementation — cross-check

### Identity initialization (spec claims `hnet_boundary.py` lines 79–83)

**Status: VERIFIED.** Lines 79–83 match spec verbatim. ✓

### Boundary probability, parallel mode (spec claims `hnet_boundary.py` lines 113–123)

**Status: VERIFIED.** Lines 113–123 match spec verbatim. ✓

### Boundary mask, threshold at 0.5 (spec claims `hnet_boundary.py` lines 130–134)

**Status: VERIFIED.** Lines 130–134 match spec verbatim. ✓

### Step (autoregressive) (spec claims `hnet_boundary.py` lines 169–192)

**Status: VERIFIED with stale line number.** Content matches; step function starts at line 168
not 169 (off by 1). Corrected in spec update.

---

## Our implementation — cross-check

### File pointer: `phase2/components/boundary_router.py:15`

**Status: VERIFIED.** `class BoundaryRouter(nn.Module):` is at line 15. ✓

### Router location: `phase2/model.py:142`

**Status: VERIFIED.** `self.router = BoundaryRouter(...)` is at lines 142–144. ✓

### Identity init in `boundary_router.py` (spec claims lines 51–52)

**Status: VERIFIED.** Lines 51–52: `nn.init.eye_(self.W_q.weight)` and `nn.init.eye_(self.W_k.weight)`. ✓

### Re-application of eye init in `model.py` (spec claims lines 156–158)

**Status: VERIFIED with stale line numbers.** The `if hasattr(self.router, "W_q"):` block is at
lines 157–159, not 156–158. Off by 1. Corrected in spec update.

### p₀ = 1.0 padded (spec claims `boundary_router.py:89, 97`)

**Status: VERIFIED with stale line number.** `cosine_rule` pad is at line 89 (correct). `learned_e2e`
pad is at line 104, not line 97 (stale). Corrected in spec update.

### Threshold selection

**Status: VERIFIED.** `boundary_router.py` line 111: `boundary_mask = (boundary_probs >= 0.5)`.
Matches H-Net `argmax == 1` convention. Consistent with `zone_d.py` line 91. ✓

### `boundary_entropy`

**Status: VERIFIED.** `train.py` lines 85–92: function body and clamp at `1e-6` match spec verbatim. ✓

---

## Intentional deviations — cross-check

### Deviation 1: "Top-M hard selection instead of soft argmax threshold" (spec lines 162–163)

**Status: INCORRECT — spec describes outdated behaviour.**

The spec still says: "Our `BoundaryRouter` always selects exactly `M = L // R` tokens via
`boundary_probs.topk(M)`."

The current implementation uses `boundary_mask = (boundary_probs >= 0.5)` with variable M per
sequence (`boundary_router.py` lines 111–114). There is no `topk(M)`. **Spec updated.**

### Deviation 2: "Simplified compression loss" (spec lines 164–168)

**Status: INCORRECT — spec describes outdated behaviour.**

The spec says we use a quadratic proxy `loss_comp = (boundary_probs.mean() - 1/R)²`.

The current `train.py` uses `ratio_loss()` (lines 95–101), which implements the full H-Net
Eq. 10: `ℒ_ratio = (N / (N-1)) * ((N-1) * F * G + (1-F) * (1-G))`, called at `train.py:333`.
The simple quadratic proxy no longer exists. Deviation 2 is no longer a deviation from H-Net;
the full ratio loss is used. **Spec updated.**

### Deviation 3: "No STE" (spec lines 170–171)

**Status: INCORRECT — spec describes outdated behaviour.**

The spec says we do not apply STE.

`zone_d.py` SimpleDecoder DOES apply STE (`use_ste=True` default, lines 105–106):
```python
if self.use_ste:
    ste_c = c_t + (1.0 - c_t).detach()
```
The ablation config `outer_crl_learned_noste` sets `use_ste=False`. **Spec updated.**

### Deviation 4: q/k index convention (referenced in code at line 101, NOT in spec)

**Status: MISSING FROM SPEC — added.**

`boundary_router.py` lines 94–101 document an intentional deviation:
- Our code: `dot(q_t, k_{t-1})` — W_q on current token, W_k on prior token.
- H-Net reference: `dot(q_{t-1}, k_t)` — W_q on prior token, W_k on current token.

At init (W_q=W_k=I) both are identical (dot product commutative for normalized vectors).
After learning, the specialization of W_q vs W_k reverses. The cosine dissimilarity semantics
are preserved; only the role of the two projections differs. This has not been ablated.

Both H-Net and DLCM use q from the earlier token and k from the later token. Our convention
is transposed relative to both references. **Deviation 4 added to spec.**

### Deviation (formerly 4, now 5 in spec): `cosine_rule` uses raw encoder outputs

**Status: VERIFIED.** `boundary_router.py` lines 83–89: cosine_rule normalizes `enc` directly
without W_q/W_k. ✓

### Deviation (formerly 6, now 6 in spec): `_no_reinit` omitted

**Status: VERIFIED.** `model.py` lines 157–159 confirm post-init re-application. ✓

### `learned_isolated` mode

**Status: VERIFIED removed.** `boundary_router.py` lines 41–52: only `cosine_rule`, `learned_e2e`,
`fixed_stride` exist. Stale references to `learned_isolated` removed from spec. ✓

---

## Verification checklist

1. **Identity init preserved.** PASS. `boundary_router.py` lines 51–52; `model.py` lines 157–159. ✓
2. **p₀ = 1.0 always set.** PASS. Both modes pad position 0 with 1.0 (lines 89, 104). ✓
3. **Threshold selection `p >= 0.5`, variable M.** PASS. Line 111: `boundary_mask = (boundary_probs >= 0.5)`. M varies; M_max is batch max. ✓
4. **`fixed_stride` has no parameters.** PASS. W_q/W_k only created for `learned_e2e`. ✓
5. **`learned_isolated` removed.** PASS. Only three modes exist. ✓
6. **`cosine_rule` and `learned_e2e` identical at init.** PASS. Both compute `normalize(enc_t) · normalize(enc_{t-1})` at init (both use q_t·k_{t-1} convention). ✓
7. **Training loss shape and value.** PASS (with deviation 2 correction). `ratio_loss()` returns a scalar. Minimum is 1.0 at F=G=1/N. ✓
8. **`boundary_entropy` NaN guard.** PASS. Clamp at `1e-6` prevents log(0). ✓
9. **Gradient flow in `learned_e2e`.** PASS. No `.detach()` on boundary_probs in learned_e2e branch. W_q, W_k receive gradients. ✓
10. **Ratio loss.** PASS (with deviation 2 correction). Full H-Net Eq. 10 ratio loss is used. Checklist item updated in spec. ✓

---

## Summary of spec changes made

1. **Deviation 1:** Replaced "Top-M hard selection via topk(M)" with "Threshold selection via `p >= 0.5`, variable M per sequence."
2. **Deviation 2:** Updated from "simplified quadratic proxy" to note that the full H-Net Eq. 10 ratio loss IS used; this is no longer a deviation.
3. **Deviation 3:** Updated from "No STE" to "STE IS applied in SimpleDecoder (`use_ste=True` default); ablation config `outer_crl_learned_noste` uses `use_ste=False`."
4. **Deviation 4:** Added — documents q/k index convention difference (our: `dot(q_t, k_{t-1})`; H-Net: `dot(q_{t-1}, k_t)`). Renumbered subsequent deviations.
5. **`learned_isolated` references:** Removed from routing modes table and deviation 5 description.
6. **Stale line numbers corrected:** `hnet_boundary.py` step (168 not 169); `model.py` eye re-init (157–159 not 156–158); `boundary_router.py` learned_e2e pad (104 not 97).
7. **Checklist item 10:** Updated to reflect ratio_loss (H-Net Eq. 10) is used, not quadratic proxy.
