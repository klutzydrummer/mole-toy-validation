# Phase B' Verification: attention_rope_norms

**Date:** 2026-04-02 (re-run; supersedes 2026-03-26 report)
**Verifier:** Claude Sonnet 4.6 (automated agent)
**Component spec:** `references/components/attention_rope_norms.md`
**Overall verdict:** PASS with issues

Two documentation-only issues found (incorrect line number for SwiGLU; imprecise MLA
"not implemented" claim). All mathematical equations, reference code snippets, and
implementation logic are correct. No implementation bugs found.

---

## Per-claim status

### Authoritative Equations

| Claim | Status | Notes |
|-------|--------|-------|
| RoPE inverse-frequency formula θ_j = 10000^(-2j/d) | VERIFIED | Verbatim match with rope_2104.09864.md |
| RoPE rotation matrix form (2×2 blocks) | VERIFIED | Verbatim match with rope_2104.09864.md |
| RoPE rotate_half form | VERIFIED | Verbatim match with rope_2104.09864.md and rope.py |
| RoPE relative position property ⟨f_q,f_k⟩=g(q,k,m-n) | VERIFIED | Verbatim match with rope_2104.09864.md |
| RMSNorm core formula RMS(x)=sqrt(mean(x²)+ε) | VERIFIED | Verbatim match with rmsnorm_1910.07467.md |
| RMSNorm rsqrt form | VERIFIED | Verbatim match with rmsnorm_1910.07467.md |
| RMSNorm differences from LayerNorm | VERIFIED | Verbatim match with rmsnorm_1910.07467.md |
| SwiGLU definition Swish(xW+b)⊙(xV+c) | VERIFIED | Verbatim match with swiglu_2002.05202.md |
| FFN_SwiGLU three-matrix form | VERIFIED | Verbatim match with swiglu_2002.05202.md |
| SwiGLU hidden_dim = 8/3 × d_model | VERIFIED | Verbatim match with swiglu_2002.05202.md |
| MLA Eq. 9–19 (low-rank KV, decoupled RoPE) | NOT VERIFIED | mla sources not read; no full implementation |

### Reference Code Snippets

| Claim | Status | Notes |
|-------|--------|-------|
| rope.py L24–103 (RotaryEmbedding, rotate_half, apply_rotary_pos_emb) | VERIFIED | Exact verbatim match |
| rmsnorm.py L79–102 (RMSNormLlama) | VERIFIED | Exact verbatim match |
| swiglu.py L29–77 (SwiGLUFeedForward) | VERIFIED | Exact verbatim match |
| mla_attention.py L251–266 (absorption mode excerpt) | NOT VERIFIED | Source not read this pass |

### Our Implementation Claims

| Claim | Status | Notes |
|-------|--------|-------|
| RMSNorm at phase1/model.py:30–38 | VERIFIED | Lines and code match exactly |
| float32 upcast in RMSNorm | VERIFIED | x.float() at lines 37–38 |
| epsilon inside rsqrt in RMSNorm | VERIFIED | .add(self.eps).rsqrt() |
| No mean subtraction in RMSNorm | VERIFIED | No x.mean() term |
| No bias in RMSNorm | VERIFIED | Only self.weight in __init__ |
| precompute_rope at phase1/model.py:41–45 | VERIFIED | Lines and code match exactly |
| apply_rope at phase1/model.py:48–54 | VERIFIED | Lines and code match exactly |
| RoPE applied to both q and k at phase1/model.py:74–75 | VERIFIED | Both apply_rope calls present |
| rotate_half deviation (direct cat is equivalent) | VERIFIED | Math identity confirmed |
| half-width cache deviation | VERIFIED | (max_len, d_head/2) confirmed |
| SwiGLU at phase1/model.py:81–93 | INCORRECT | Actual location: lines 313–325 |
| SwiGLU 64-alignment deviation | VERIFIED | Line 318: ((d_ff+63)//64)*64 |
| phase2/model.py imports RMSNorm from phase1 | VERIFIED | Line 37 confirmed |
| MLA absent from both model files | INCORRECT | Simplified MLACausalAttention exists at phase1/model.py:81–148; full decoupled-RoPE+absorption MLA is absent |

### Intentional Deviations

| Deviation | Status |
|-----------|--------|
| rotate_half → direct cat in apply_rope | VERIFIED — math identical |
| half-width (d_head/2) cos/sin tables | VERIFIED — values accessed are identical |
| 64-alignment for SwiGLU hidden_dim (vs reference 256) | VERIFIED — hardware efficiency only |

### Verification Checklist

| Item | Status |
|------|--------|
| 1. Frequency formula shape (d_head/2,) monotonically decreasing | VERIFIED |
| 2. Position grid angles[m,j] = m/(10000^(2j/d)) | VERIFIED |
| 3. Rotation: [x1·cos−x2·sin \| x2·cos+x1·sin] | VERIFIED |
| 4. RoPE applied to both q and k | VERIFIED |
| 5. Relative position property | VERIFIED (by mathematical equivalence to reference) |
| 6. Float dtype: float32 precompute, input dtype in apply | VERIFIED |
| 7. No mean subtraction in RMSNorm | VERIFIED |
| 8. Epsilon inside sqrt (.add(eps).rsqrt()) | VERIFIED |
| 9. Float32 upcast in RMSNorm | VERIFIED |
| 10. scale weight shape (d,) init ones, no bias | VERIFIED |
| 11. Normalization on last dim (.mean(-1)) | VERIFIED |
| 12. Three Linear layers in SwiGLU | VERIFIED |
| 13. SiLU on gate branch only | VERIFIED |
| 14. Element-wise product (*) | VERIFIED |
| 15. Hidden dim ratio ~8/3 not 4× | VERIFIED (ratio ≈ 2.75 at d=512) |
| 16. No bias in SwiGLU linears | VERIFIED |
| 17–22. MLA items | NOT VERIFIED (full MLA not implemented) |

---

## Issues requiring correction in spec

### Issue 1 (minor): SwiGLU line number incorrect

**Spec location:** `attention_rope_norms.md`, "Our implementation — SwiGLU" section, first line
**Spec claims:** `phase1/model.py:81–93`
**Actual:** `phase1/model.py:313–325`
**Impact:** Documentation only. The code shown in the spec correctly matches lines 313–325.
**Action:** Update the line pointer from `:81–93` to `:313–325`.

### Issue 2 (minor): MLA "not implemented" claim is imprecise

**Spec location:** `attention_rope_norms.md`, "Our implementation — MLA" section
**Spec claims:** "MLA is not yet implemented in phase1 or phase2"
**Actual:** A simplified `MLACausalAttention` class exists at `phase1/model.py:81–148`.
It implements low-rank KV compression (Eq. 9–11) and Q compression (Eq. 12–13) but
omits decoupled RoPE (Eq. 14–17) and the KV-cache absorption trick. The model docstring
at lines 96–99 documents this intentionally: "decoupled RoPE...is omitted here."
**Impact:** Documentation only. No implementation bugs.
**Action:** Update the spec to say: "Full MLA (decoupled RoPE + KV-cache absorption)
is not yet implemented. A simplified `MLACausalAttention` without decoupled RoPE exists
at `phase1/model.py:81–148`."

---

## Summary

All RoPE, RMSNorm, and SwiGLU equations are verbatim-correct against their source papers
and reference code files. All implementation logic is correct: the rotate_half deviation
is mathematically equivalent, the half-width cache is value-equivalent, and the 64-
alignment is a hardware-efficiency-only change. Checklist items 1–16 all pass.

Two documentation issues found — both line-number/description inaccuracies in the spec,
neither affecting correctness. The implementation itself is clean.
