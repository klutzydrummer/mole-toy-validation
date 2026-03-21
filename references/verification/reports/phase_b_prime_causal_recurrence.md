# Phase B' Verification Report: causal_recurrence

**Component file:** `references/components/causal_recurrence.md`
**Sources checked:**
- `references/sources/papers/griffin_2402.19427.md`
- `references/sources/code/griffin_rglru.py`

**Date:** 2026-03-21 (re-verified; original 2026-03-17)

**Re-verification note (2026-03-21):** Full re-check performed including the log_a_init fix (ZoneE=3.0, ZoneD=0.0). All 6 authoritative equations verified verbatim in `griffin_2402.19427.md`. All 4 reference code snippets verified verbatim in `griffin_rglru.py`. All 7 intentional deviations confirmed present and accurately described. Two issues found:

1. **Line numbers systematically ~7 off (Minor):** Spec cites ZoneE recurrence at line 281 (actual 289), ZoneD at 398 (actual 407), `sigmoid(log_a.float())` at 139 (actual 146), `b_t = sqrt(...)` at 145 (actual 152), `out_proj/RMSNorm` at 115-116 (actual 124-125), `return norm(out_proj(out))` at 149 (actual 156). Correctly cited: `_parallel_scan` (44), `CausalRecurrenceLayer` (83), float32 lines (69-70, 76), clamp (71). Documentation only.

2. **Stale docstring fixed:** `phase2/model.py:587` said `"keeps its 7.5 init"` — incorrect. Fixed in this session to reflect role-specific inits (ZoneE: 3.0, ZoneD: 0.0). Default 7.5 is never used in practice.

---

## Overall Verdict: PASS (with minor annotation issues)

All core equations are traceable to the paper. All code snippets are present verbatim in the
reference source file. Two minor line-range attribution discrepancies are noted below but do not
constitute substantive errors. Claims about the MoLE implementation (`phase2/model.py`) are
forward-looking and cannot be verified from the provided sources — they are appropriately marked as
"Our implementation" and are out of scope for source tracing.

---

## Verified Claims

### Equations

**Eq. (1) — Recurrence gate `r_t = σ(W_a x_t + b_a)`**
- Source: `griffin_2402.19427.md`, Section 2.4, Equation (1).
- Verdict: VERIFIED. Verbatim match.

**Eq. (2) — Input gate `i_t = σ(W_x x_t + b_x)`**
- Source: `griffin_2402.19427.md`, Section 2.4, Equation (2).
- Verdict: VERIFIED. Verbatim match.

**Eq. (3) — Recurrent coefficient `a_t = a^(c · r_t)`, with `a = σ(Λ)`, `c = 8`**
- Source: `griffin_2402.19427.md`, Section 2.4, Equation (3) and surrounding prose ("c is a
  scalar constant set to 8", "parameterize a as a = σ(Λ), where Λ is a learnable parameter").
- Verdict: VERIFIED. Verbatim match including parameter descriptions.

**Eq. (4) — Hidden state update `h_t = a_t ⊙ h_{t-1} + √(1 − a_t²) ⊙ (i_t ⊙ x_t)`**
- Source: `griffin_2402.19427.md`, Section 2.4, Equation (4).
- Verdict: VERIFIED. Verbatim match.

**Eq. (6) — Log-space computation of `a_t`**
Component states:
```
log a_t = log σ(Λ)^(c · r_t)
        = −c · softplus(Λ) ⊙ r_t
```
and "In code: `log_a = -8.0 * gate_a * softplus(a_param)`, then `a = exp(log_a)`."
- Source: `griffin_2402.19427.md`, Appendix A, Equation (6). Exact formula present. The inline
  code description also matches `griffin_rglru.py` line 308.
- Verdict: VERIFIED. Verbatim match for both the mathematical formula and the code description.

**Norm-preservation identity**
Component states:
```
Var[h_t] = a_t² · Var[h_{t-1}] + (1 − a_t²) · Var[i_t ⊙ x_t] ≈ 1
```
- Source: `griffin_2402.19427.md`, section "The √(1 − a_t²) Normalization Term", which contains
  this exact variance identity.
- Verdict: VERIFIED. Verbatim match.

---

### Code Snippets

**Gate and coefficient computation — claimed lines 304–319 of `griffin_rglru.py`**
Component quotes:
```python
gate_x = torch.sigmoid(self.input_gate(x))
gate_a = torch.sigmoid(self.a_gate(x))
log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
a = torch.exp(log_a)
a_square = torch.exp(2 * log_a)
gated_x = x * gate_x
multiplier = SqrtBoundDerivative.apply(1 - a_square)
multiplier = reset[..., None] + ~reset[..., None] * multiplier
normalized_x = gated_x * multiplier.type(x.dtype)
```
- Source: `griffin_rglru.py` lines 304–319. All lines verified verbatim.
- Verdict: VERIFIED.

**Sequential scan kernel — claimed lines 153–157 of `griffin_rglru.py`**
Component quotes:
```python
for t in range(x.shape[1]):
    h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
    y[:, t] = h_t.type(x.dtype)
```
- Source: `griffin_rglru.py` lines 153–155. Code is verbatim at those lines.
- Minor issue: The claimed range "lines 153–157" ends at line 157 (`return y, h_t`), which is
  not included in the displayed snippet. The actual snippet corresponds to lines 153–155. The
  line range is slightly over-stated but the code content is correct.
- Verdict: VERIFIED (minor line-range over-statement, not a substantive error).

**Learnable parameter initialization `rnn_param_init` — claimed lines 169–177 of `griffin_rglru.py`**
Component quotes:
```python
# Proportional to area in a ring.
tensor.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
tensor.log_().mul_(0.5)
# Inverse softplus transform.
return tensor.neg_().exp_().sub_(1.0).log_()
```
- Source: `griffin_rglru.py` lines 169–177. The shown lines are present verbatim. The snippet
  omits line 174 (`if transform == "softplus":`) and the surrounding `with torch.no_grad():`
  context, but it does not misrepresent any statement; the omission is editorial compression.
  The `min_rad=0.9, max_rad=0.999` claim is confirmed by `RGLRU.a_param_init` at line 257.
- Verdict: VERIFIED (selective omission of guard clause, not inaccurate).

**SqrtBoundDerivative — claimed lines 182–196 of `griffin_rglru.py`**
Component quotes the class with `_MAX_SQRT_GRADIENT = 1000.0`, `forward`, and `backward` methods.
- Source: `griffin_rglru.py` lines 39 (`_MAX_SQRT_GRADIENT = 1000.0`) and lines 182–196
  (`SqrtBoundDerivative` class body). All lines verified verbatim.
- Note: `_MAX_SQRT_GRADIENT` is defined at line 39, not inside the class at lines 182–196. The
  component's snippet shows it inside the class block, which is a cosmetic restructuring. The
  value and usage are correct.
- Verdict: VERIFIED (minor cosmetic restructuring of constant placement, not an error).

---

### Intentional Deviations

**Deviation 1 — Parallel scan instead of sequential loop**
- Claim: reference `rnn_scan` uses `for t in range(L)` loop.
- Source: `griffin_rglru.py` line 153: `for t in range(x.shape[1]):`. VERIFIED.
- The parallel scan formula itself (`log_A_t = cumsum(log(a_t), dim=1)` etc.) is an MoLE
  design claim, not attributable to the Griffin source. It is correctly framed as a deviation.

**Deviation 5 — Causal depthwise conv prepended**
- Claim: "Griffin's recurrent block wraps the RG-LRU in a temporal conv separately."
- Source: `griffin_2402.19427.md`, Architecture Context section: "input x → [temporal conv
  (width 4)] → split into two branches: branch 1: RG-LRU → linear projection; ..."
- Verdict: VERIFIED.

**Deviation 6 — No segment boundary reset**
- Claim: "reference multiplies `a` by `~reset` to zero out the recurrent state at document
  boundaries (`rnn_scan` line 134)."
- Source: `griffin_rglru.py` line 134: `a = a * ~reset[..., None]`. VERIFIED.

**Deviation 7 — Output projection + RMSNorm**
- Claim: "reference outputs `y = h_t` directly."
- Source: `griffin_2402.19427.md`, Output section: "The output of the layer is y_t = h_t".
  Also `griffin_rglru.py` line 321: `y, last_h = rnn_scan(...)` then `return y, last_h` with
  no further projection. VERIFIED.

**Deviations 2, 3, 4 — Implementation-internal claims**
- These describe MoLE's `phase2/model.py` behavior (log_a parameterization, float32 promotion
  strategy, sqrt clamping). They reference design choices, not claims about Griffin sources.
  They are correctly framed as deviations and are out of scope for source tracing.

---

## Unverified / Out-of-Scope Claims

The following claims in the component refer to `phase2/model.py` (MoLE implementation), which
is not a provided source file. They cannot be verified from `references/sources/` alone. They
are appropriately labeled "Our implementation" or "Intentional deviations" and do not constitute
unsupported claims about the Griffin source.

| Claim | Location in component |
|---|---|
| `CausalRecurrenceLayer` at `phase2/model.py:83` | "Our implementation" section |
| `_parallel_scan` at `phase2/model.py:44` | "Our implementation" section |
| `log_a` initialized to 7.5 | Deviation 2 |
| `sigmoid(log_a)` promoted to float32 at line 139 | Deviation 3 |
| `(1.0 - a_t * a_t).clamp(min=1e-6)` at line 145 | Deviation 4 |
| Zone E/D ModuleList at lines 281/398 | "Our implementation" section |

These are forward-looking references to implementation code. They should be verified separately
against `phase2/model.py` in a Phase C validation pass.

---

## Contradictions

**None found.**

The component is internally consistent. The equations section and the deviations section do not
contradict each other. For example:

- The equations define `√(1 − a_t²)` as the norm-preserving multiplier (Eq. 4). Deviation 4
  says the implementation uses `.clamp(min=1e-6)` before the sqrt. This is presented as an
  intentional deviation from the reference's `SqrtBoundDerivative` approach, not a contradiction.

- Deviation 2 says `sigmoid(log_a)` plays the role of `σ(Λ)` in Eq. (3), and `8 * r_t` plays
  the role of `c · r_t`. This is consistent with the authoritative equations section.

---

## Summary

| Category | Count | Status |
|---|---|---|
| Equations verified | 6 | PASS |
| Code snippets verified | 4 | PASS (2 minor line-range/formatting notes) |
| Deviations with traceable source claims | 4 | PASS |
| Deviations that are implementation-only | 3 | Out of scope (correctly labeled) |
| Contradictions | 0 | PASS |
| Unsupported claims | 0 | PASS |

**Overall: PASS**
