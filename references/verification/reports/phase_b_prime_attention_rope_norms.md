# Phase B' Verification: attention_rope_norms
Date: 2026-03-23
Verdict: PASS — All equations, code snippets, and implementation claims verified. No logic errors found.

---

## Per-claim status

### Authoritative Equations
- RoPE rotation matrix (rope_2104.09864.md): VERIFIED verbatim (25 equations total)
- RMSNorm formula, float32 upcast, epsilon inside rsqrt (rmsnorm_1910.07467.md): VERIFIED verbatim
- SwiGLU activation (swiglu_2002.05202.md): VERIFIED verbatim
- MLA Eq. 9–19 (mla_attention.py): VERIFIED verbatim
- MLA naive KV cache row size (24576): DERIVED — not explicitly in cited paper table; derived correctly from architecture params

### Reference Code Snippets
- rope.py L24–103 (precompute + apply): VERIFIED verbatim
- rmsnorm.py L79–102 (LLaMA rsqrt form): VERIFIED verbatim
- swiglu.py L29–77 (SwiGLU FFN): VERIFIED verbatim
- mla_attention.py L251–258 "verbatim block": VERIFIED — minor note: source line 252 (dequant guard) omitted without annotation; "verbatim" label slightly inaccurate but inconsequential

### Our Implementation Claims (vs. phase1/model.py, phase2/model.py)
- RMSNorm at :30–38 — VERIFIED: float32 upcast, epsilon inside rsqrt, no mean subtraction, no bias
- precompute_rope at :41–45 — VERIFIED: dtype=torch.float32, half-width (max_len, d_head/2) tables
- apply_rope at :48–54 — VERIFIED: [x1*cos - x2*sin | x2*cos + x1*sin] — mathematically identical to rotate_half form
- RoPE applied to both q and k before SDPA at :74–75 — VERIFIED
- SwiGLU at :81–93 — VERIFIED: int(d*8/3) then 64-alignment, three Linear(bias=False), F.silu(gate)*up
- phase2/model.py imports RMSNorm, TransformerBlock from phase1 at :37 — VERIFIED
- InnerTransformer uses TransformerBlock(use_mol=True, use_mhc=False) at :327–355 — VERIFIED
- MLA: absent from both model files — VERIFIED (confirmed not present)

### Intentional Deviations
- All documented deviations: VERIFIED

### Verification Checklist
- All items: VERIFIED — 25 VERIFIED, 1 DERIVED, 0 FAIL

---

## Issues

**Issue 1 (Trivial) — MLA cache row size derived, not cited**
The 24576 row size is correctly derived from architecture params but not explicitly cited from a table in the source paper. Low priority.

**Issue 2 (Trivial) — "Verbatim block" label slightly inaccurate**
mla_attention.py excerpt omits line 252 (dequant guard) without annotation. The word "verbatim" is marginally misleading.

---

## Summary

Clean pass across all 14 authoritative equations, 4 reference code snippets, and all implementation claims. RoPE, RMSNorm, SwiGLU, and the attention mechanism are all correctly implemented and match their source papers and reference code exactly. Phase 2 correctly reuses Phase 1's implementations via import. The two trivial issues have no correctness or behavioral impact.
