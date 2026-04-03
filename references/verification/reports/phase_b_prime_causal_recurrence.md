# Phase B' Verification Report: causal_recurrence

**Date:** 2026-04-02
**Verifier:** Claude Sonnet 4.6 (automated Phase B' agent)
**Component spec:** `references/components/causal_recurrence.md`
**Sources checked:**
- `references/sources/papers/griffin_2402.19427.md`
- `references/sources/code/griffin_rglru.py`
**Implementation checked:** `phase2/model.py` — `CausalRecurrenceLayer` (line 82), `_parallel_scan` (line 43)

---

## Overall Verdict: PASS

All authoritative equations match their cited sources. All reference code snippets match the source file. All implementation claims verified at the cited line numbers. All intentional deviations are accurately described and present in the code. All checklist items pass.

---

## Authoritative Equations — Cross-check against griffin_2402.19427.md

| Claim | Status | Notes |
|-------|--------|-------|
| Eq. (1): `r_t = σ(W_a x_t + b_a)` | VERIFIED | Matches paper Section 2.4 exactly |
| Eq. (2): `i_t = σ(W_x x_t + b_x)` | VERIFIED | Matches paper Section 2.4 exactly |
| Eq. (3): `a_t = a^(c · r_t)`, `a = σ(Λ)`, `c = 8` | VERIFIED | Matches paper Section 2.4 exactly, c=8 confirmed |
| Eq. (4): `h_t = a_t ⊙ h_{t-1} + √(1 − a_t²) ⊙ (i_t ⊙ x_t)` | VERIFIED | Matches paper Section 2.4 exactly |
| Eq. (6) log-space: `log a_t = −c · softplus(Λ) ⊙ r_t` | VERIFIED | Matches paper Appendix A exactly |
| Norm-preservation identity: `Var[h_t] = a_t² · Var[h_{t-1}] + (1 − a_t²) · Var[i_t ⊙ x_t] ≈ 1` | VERIFIED | Matches paper's gamma normalization section exactly |

---

## Reference Code Snippets — Cross-check against griffin_rglru.py

| Spec snippet | Status | Notes |
|--------------|--------|-------|
| Gate and coefficient computation (RGLRU.forward, lines 304–319) | VERIFIED | griffin_rglru.py lines 304–319 contain `gate_x`, `gate_a`, `log_a`, `a`, `a_square`, `gated_x`, `multiplier`, `normalized_x`. Snippet matches verbatim. |
| Sequential scan kernel (rnn_scan, lines 153–157) | VERIFIED | griffin_rglru.py lines 153–157: `for t in range(x.shape[1]): h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype); y[:, t] = h_t.type(x.dtype)`. Matches spec verbatim. |
| rnn_param_init (lines 169–177): `tensor.uniform_(...)`, `tensor.log_().mul_(0.5)`, `return tensor.neg_().exp_().sub_(1.0).log_()` | VERIFIED | griffin_rglru.py lines 169–177 match verbatim. |
| SqrtBoundDerivative (lines 182–196): `_MAX_SQRT_GRADIENT = 1000.0`, custom backward | VERIFIED | griffin_rglru.py lines 182–196 match verbatim including `clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))`. |

---

## Implementation Claims — Cross-check against phase2/model.py

| Claim (spec) | Cited line | Actual line | Status | Notes |
|--------------|-----------|------------|--------|-------|
| Primary class `CausalRecurrenceLayer` | 83 | 82 | VERIFIED | 1-line drift; class definition confirmed at line 82 |
| `_parallel_scan` helper | 44 | 43 | VERIFIED | 1-line drift; function definition at line 43 |
| ZoneE `ModuleList` of 3 `CausalRecurrenceLayer(d_outer, log_a_init=3.0)` | ~281 | 288 | VERIFIED | Present at line 288 |
| ZoneD `ModuleList` of 3 `CausalRecurrenceLayer(d_outer, log_a_init=0.0)` | ~398 | 406 | VERIFIED | Present at line 406 |

---

## Intentional Deviations — Verified Present in Code

| Deviation | Status | Notes |
|-----------|--------|-------|
| 1. Parallel scan instead of sequential loop | VERIFIED | `_parallel_scan` at line 43 uses `cumsum(log(a_t))` log-space approach. No sequential for-loop in CausalRecurrenceLayer. |
| 2. `log_a` parameterization instead of `softplus(a_param)`; role-specific init | VERIFIED | `self.log_a = nn.Parameter(torch.full((d,), log_a_init))` at line 120. `a_base = torch.sigmoid(self.log_a.float()).to(x.dtype)` at line 145. ZoneE: `log_a_init=3.0` (line 288), ZoneD: `log_a_init=0.0` (line 406). |
| 3. float32 promotion for sigmoid(log_a) only | VERIFIED | `torch.sigmoid(self.log_a.float()).to(x.dtype)` at line 145. Parallel scan runs in float32 via `a_t.float()` / `b_t.float()` at lines 68–69. |
| 4. sqrt clamped to 1e-6 instead of custom autograd function | VERIFIED | `(1.0 - a_t * a_t).clamp(min=1e-6)` at line 151. No `SqrtBoundDerivative` in model.py. |
| 5. Causal depthwise conv (kernel=4) prepended | VERIFIED | Lines 111–135: `self.conv_weight`, `self.conv_bias`, left-pad of 3, `F.conv1d(..., groups=d)`. Gates applied to post-conv `x_conv`. |
| 6. No segment boundary reset | VERIFIED | No reset mask or segment position tracking anywhere in `CausalRecurrenceLayer` or `_parallel_scan`. |
| 7. Output projection + RMSNorm | VERIFIED | `self.out_proj = nn.Linear(d, d, bias=False)` at line 123; `self.norm = RMSNorm(d)` at line 124; applied at line 155: `self.norm(self.out_proj(out))`. |

---

## Verification Checklist

| Item | Status | Notes |
|------|--------|-------|
| 1. Gate computation: `r_t = sigmoid(W_r(x_conv))`, `i_t = sigmoid(W_i(x_conv))` at lines 130–131 | VERIFIED | Lines 138–139: both sigmoid calls confirmed. Values in (0,1). |
| 2. `a_t = sigmoid(log_a)^(8 * r_t)` in (0,1); log_a_init=3.0 ZoneE, 0.0 ZoneD | VERIFIED | Line 146: `a_t = a_base.pow(8.0 * r_t)`. Init values confirmed at lines 288, 406. |
| 3. float32 promotion for sigmoid(log_a) | VERIFIED | Line 145: `torch.sigmoid(self.log_a.float()).to(x.dtype)`. Spec cites line 139 — 6-line drift, code correct. |
| 4. Norm-preserving input: `b_t = sqrt((1-a_t²).clamp(min=1e-6)) * (i_t * x_conv)` | VERIFIED | Line 151. Spec cites line 145 — 6-line drift, code correct. |
| 5. Parallel scan correctness for L=2, h0=0 | VERIFIED | Math verified: `h_1 = A_1*(b_1/A_1) = b_1`; `h_2 = A_2*(b_1/A_1 + b_2/A_2) = a_2*b_1 + b_2`. Correct. |
| 6. float32 inside parallel scan: `a = a_t.float()`, `b = b_t.float()` at lines 69–70 | VERIFIED | Lines 68–69. Cast back at line 75: `.to(a_t.dtype)`. |
| 7. `log_A clamping: cumsum(log(a)).clamp(min=-80.0)` at line 71 | VERIFIED | Line 70: `.clamp(min=-80.0)` present. No upper clamp needed (log of values ≤ 1 is always ≤ 0). |
| 8. Zone E 3-layer stack at line ~281 | VERIFIED | Line 288. |
| 9. Zone D 3-layer stack at line ~398 | VERIFIED | Line 406. |
| 10. h_state float32 accumulation, cast back to input dtype | VERIFIED | Lines 68–69: `.float()`. Line 75: `.to(a_t.dtype)`. |

---

## Minor Discrepancies (Non-blocking, Documentation Only)

1. **Line number drift:** Spec cites line 83 for `CausalRecurrenceLayer` (actual: 82), line 44 for `_parallel_scan` (actual: 43), checklist item 3 cites line 139 (actual: 145), checklist item 4 cites line 145 (actual: 151), checklist item 6 cites lines 69–70 (actual: 68–69). All are 1–6 line offsets from prior edits. The code is correct at each location.

2. **log_A upper clamp:** The spec docstring (line 58) says "clamp to [-80, 0]" but code only clamps the minimum. Mathematically, `log(a_t)` where `a_t ∈ (0,1]` is always ≤ 0, so an upper clamp at 0 is a no-op. Not a bug.

Neither discrepancy affects correctness.

---

## Summary

The `CausalRecurrenceLayer` and `_parallel_scan` implementation correctly implements the Griffin RG-LRU equations. All six intentional deviations are accurately documented and verified in code. The parallel scan math is correct (verified algebraically). Float32 promotion is present in both the sigmoid and the scan itself. The log_a_init values are set correctly per role (3.0 for Zone E, 0.0 for Zone D), preventing the 9.7 BPC collapse that plagued the original implementation. The norm-preserving sqrt term with 1e-6 clamp is present. No correctness issues found.
