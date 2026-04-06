# Phase B' Verification Report: attention_rope_norms

**Date:** 2026-04-05
**Supersedes:** prior report dated 2026-04-02
**Component:** `attention_rope_norms` — RoPE, RMSNorm, SwiGLU shared primitives
**Spec:** `references/components/attention_rope_norms.md`
**Implementation:** `phase1/components/attention_rope_norms.py`, `phase1/components/_shared.py`
**Sources checked:** `references/sources/papers/rope_2104.09864.md`, `references/sources/papers/rmsnorm_1910.07467.md`, `references/sources/papers/swiglu_2002.05202.md`, `references/sources/papers/mla_deepseek_v2_2405.04434.md`, `references/sources/code/rope.py`, `references/sources/code/rmsnorm.py`, `references/sources/code/swiglu.py`, `references/sources/code/mla_attention.py`

---

## Overall verdict: PASS

All equations trace verbatim to cited papers. All reference code snippets match exactly. All implementation line citations are current and accurate. All intentional deviations are present and correctly described. No stale pointers — the line number issues from the prior 2026-04-02 report are fully resolved.

---

## Per-claim status

### Authoritative equations

| Claim | Status | Notes |
|-------|--------|-------|
| RoPE rotation: pairs (m, n) rotated by θ=10000^(-2i/d) | VERIFIED | rope_2104.09864.md §3.1 |
| RMSNorm: x / RMS(x) · g, RMS = sqrt(mean(x²) + ε) | VERIFIED | rmsnorm_1910.07467.md Eq. 4 |
| SwiGLU: FFN(x) = (xW₁ ⊙ SiLU(xV)) W₂ | VERIFIED | swiglu_2002.05202.md Eq. 6 |
| MLA RoPE on decoupled keys: q^R and k^R (Eq. 9–19) | VERIFIED | mla_deepseek_v2_2405.04434.md |
| All 16 equation claims in spec | VERIFIED | All match cited sources verbatim |

### Reference code snippets

| Claim | Status | Notes |
|-------|--------|-------|
| rope.py:24–103 | VERIFIED | Exact match |
| rmsnorm.py:79–102 | VERIFIED | Exact match |
| swiglu.py:29–77 | VERIFIED | Exact match |
| mla_attention.py:251–266 (verbatim block) | VERIFIED | Exact match |

### Our implementation

| Claim | Status | Notes |
|-------|--------|-------|
| `_shared.py:15–23` — RMSNorm | VERIFIED | Correct file and line range |
| `_shared.py:26–39` — precompute_rope + apply_rope | VERIFIED | Correct file and line range |
| `_shared.py:42–54` — SwiGLU | VERIFIED | Correct file and line range |
| `attention_rope_norms.py:33–34` — RoPE applied to q and k | VERIFIED | Correct file and line range |

### Intentional deviations

| Deviation | Status | Notes |
|-----------|--------|-------|
| rotate_half inlined as direct cat (not function call) | VERIFIED | Mathematically identical |
| Half-width cos/sin tables `(max_len, d_head/2)` instead of full | VERIFIED | Values equivalent; more memory-efficient |
| SwiGLU multiple_of=64 (vs reference 256) | VERIFIED | Hardware alignment only, no functional difference |
| MLA omits decoupled RoPE / absorption trick | VERIFIED | Correctly documented as intentional deviation |

### Verification checklist

| Item | Status | Notes |
|------|--------|-------|
| Items 1–16 (RoPE, RMSNorm, SwiGLU) | VERIFIED | All pass |
| Items 17–22 (MLA decoupled RoPE) | N/A for this component | MLA-specific items verified in mla_attention component |

---

## Summary

Clean pass. All citations are accurate against the new component file layout. The two stale `phase1/model.py` references from the prior report are fully corrected in the current spec — all citations now correctly point to `phase1/components/_shared.py` and `phase1/components/attention_rope_norms.py`. No correctness issues found.
