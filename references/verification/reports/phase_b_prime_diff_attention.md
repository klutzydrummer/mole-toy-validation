# Phase B' Verification Report: diff_attention.md

**Date:** 2026-03-31 (re-verified 2026-04-02)
**Verifier:** Claude Sonnet 4.6 (Phase B' agent, full re-verification 2026-04-02)
**Verdict:** PASS

---

## Pseudocode verification

Checked against `references/sources/papers/diff_attn_v2_2026_01.md` lines 66–83.

| Claim | Source | Status | Notes |
|-------|--------|--------|-------|
| q shape: (N, 2h, d) | diff_attn_v2_2026_01.md line 71 | VERIFIED | Verbatim match |
| k shape: (N, h_kv, d) | diff_attn_v2_2026_01.md line 72 | VERIFIED | Verbatim match |
| v shape: (N, h_kv, d) | diff_attn_v2_2026_01.md line 73 | VERIFIED | Verbatim match |
| lam shape: (N, h, 1) | diff_attn_v2_2026_01.md line 74 | VERIFIED | Verbatim match |
| Head split: `attn[:, 0::2]` and `attn[:, 1::2]` | diff_attn_v2_2026_01.md lines 77–78 | VERIFIED | Verbatim match; interleaved not blocked |
| `sigmoid(lam)` applied before subtraction | diff_attn_v2_2026_01.md line 80 | VERIFIED | `lam_val = sigmoid(lam)` before `attn1 - lam_val * attn2` |

---

## V2 changes from V1 table

Checked against `diff_attn_v2_2026_01.md` lines 37–44 and `diff_attn_v1_2410.05258.md`.

| Row | V1 value in spec | V2 value in spec | Status | Notes |
|-----|-----------------|-----------------|--------|-------|
| Q heads | h/2 pairs (separate q1, q2) | 2h heads (contiguous pairs) | VERIFIED | Source line 39: "h/2 pairs (q1, q2 separate)" → "2h heads (doubled; contiguous pairs)" |
| K/V width | h/2 heads, V double-wide (2·d_h) | h heads, V standard (d_h) | VERIFIED | Source line 40: "h/2 (with double-wide V)" → "h (standard V, same as baseline)" |
| λ | static exp reparameterization + λ_init | token-specific sigmoid projection | VERIFIED | Source line 42: "static exp reparameterization" → "token-specific sigmoid(W_λ · x) per head" |
| Per-head RMSNorm | yes | removed | VERIFIED | Source line 43: "yes" → "no" |
| FlashAttention | requires custom kernel | standard (no modification needed) | VERIFIED | Source lines 24–26 confirm custom kernel required for V1; V2 uses standard FlashAttention |
| Context RMS range | [1/√n, 1) after RMSNorm | (0, √2) | VERIFIED | Source lines 125–126, 134–135 confirm both bounds |

---

## DifferentialCausalAttention implementation claims

All lines verified against `phase1/model.py` (lines 151–225 read directly).

| Claim | Stated Line | Status | Notes |
|-------|-------------|--------|-------|
| `self.W_Q = nn.Linear(d, 2 * d, bias=False)` | 183 | VERIFIED | Exact match at line 183 |
| `self.W_K = nn.Linear(d, d, bias=False)` | 184 | VERIFIED | Exact match at line 184 |
| `self.W_V = nn.Linear(d, d, bias=False)` | 185 | VERIFIED | Exact match at line 185 |
| `self.W_lambda = nn.Linear(d, n_heads, bias=True)` | 189 | VERIFIED | Exact match at line 189 |
| `self.out = nn.Linear(d, d, bias=False)` | 191 | VERIFIED | Exact match at line 191 |
| Q reshape to `[B, L, 2 * nh, dh]` | 202 | VERIFIED | `self.W_Q(x).reshape(B, L, 2 * nh, dh).transpose(1, 2)` at line 202 |
| K reshape to `[B, L, nh, dh]` | 203 | VERIFIED | `self.W_K(x).reshape(B, L, nh, dh).transpose(1, 2)` at line 203 |
| `apply_rope` on both q and k | 206–207 | VERIFIED | Lines 206–207 apply `apply_rope` to q and k respectively |
| `k.repeat_interleave(2, dim=1)` | 211 | VERIFIED | Exact match at line 211 |
| `v.repeat_interleave(2, dim=1)` | 212 | VERIFIED | Exact match at line 212 |
| SDPA call | 215 | VERIFIED | `F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)` at line 215 |
| `attn[:, 0::2]` | 218 | VERIFIED | Exact match at line 218 |
| `attn[:, 1::2]` | 219 | VERIFIED | Exact match at line 219 |
| `torch.sigmoid(self.W_lambda(x))` | 222 | VERIFIED | Exact match at line 222 |
| Differential subtraction | 224 | VERIFIED | `(attn1 - lam * attn2).transpose(1, 2).reshape(B, L, D)` at line 224 |
| Class spans lines 151–225 | 151–225 | VERIFIED | Class definition at line 151; `return self.out(out)` at line 225 |

---

## DiffMLAAttention implementation claims

All lines verified against `phase1/model.py` (lines 228–310 read directly).

| Claim | Stated Line | Status | Notes |
|-------|-------------|--------|-------|
| `self.W_DKV = nn.Linear(d, d_c, bias=False)` | 265 | VERIFIED | Exact match at line 265 |
| `self.W_UK = nn.Linear(d_c, nh * dh, bias=False)` | 266 | VERIFIED | Exact match at line 266 |
| `self.W_UV = nn.Linear(d_c, nh * dh, bias=False)` | 267 | VERIFIED | Exact match at line 267 |
| `self.W_DQ = nn.Linear(d, d_c_q, bias=False)` | 270 | VERIFIED | Exact match at line 270 |
| `self.W_UQ = nn.Linear(d_c_q, 2 * nh * dh, bias=False)` | 271 | VERIFIED | Exact match at line 271 |
| `self.W_lambda = nn.Linear(d, nh, bias=True)` | 274 | VERIFIED | Exact match at line 274 |
| KV compression forward: `c_kv = self.W_DKV(x)` | 287 | VERIFIED | Exact match at line 287 |
| `k = self.W_UK(c_kv)...` and `v = self.W_UV(c_kv)...` | 288–289 | VERIFIED | Lines 288–289; both operate on the same `c_kv` |
| Q compression: `c_q = self.W_DQ(x)` | 292 | VERIFIED | Exact match at line 292 |
| `q = self.W_UQ(c_q)...` | 293 | VERIFIED | Exact match at line 293 |
| `k.repeat_interleave(2, dim=1)` | 299 | VERIFIED | Exact match at line 299 |
| `attn[:, 0::2]` | 304 | VERIFIED | Exact match at line 304 |
| `attn[:, 1::2]` | 305 | VERIFIED | Exact match at line 305 |
| lam computation | 307 | VERIFIED | `torch.sigmoid(self.W_lambda(x)).transpose(1, 2).unsqueeze(-1)` at line 307 |
| Differential subtraction | 309 | VERIFIED | `(attn1 - lam * attn2).transpose(1, 2).reshape(B, L, D)` at line 309 |
| Class spans lines 228–310 | 228–310 | VERIFIED | Class definition at line 228; `return self.out(out)` at line 310 |

---

## Intentional deviations

| Deviation | Status | Notes |
|-----------|--------|-------|
| `repeat_interleave` + SDPA instead of `flash_attn_func` | VERIFIED | Lines 211–215 (DifferentialCausalAttention) and 299–302 (DiffMLAAttention) use `repeat_interleave` + `F.scaled_dot_product_attention`; no `flash_attn_func` present |
| `layer_idx` unused in V2 | VERIFIED | `layer_idx` is in `__init__` signatures (line 176 and line 249) but does not appear in either `forward` method; not used in any computation |
| V is standard width d_h (not double-wide) in DiffMLAAttention | VERIFIED | `self.W_UV = nn.Linear(d_c, nh * dh, bias=False)` at line 267; output dim is `nh * dh`, not `2 * nh * dh` |

---

## Verification checklist

| Item | Status | Notes |
|------|--------|-------|
| 1. Doubled Q, standard KV: `W_Q = Linear(d, 2*d)` at 183, `W_K = Linear(d, d)` at 184 | VERIFIED | Both confirmed |
| 2. GQA pairing via repeat_interleave: `k.repeat_interleave(2, dim=1)` at 211, `v.repeat_interleave(2, dim=1)` at 212 | VERIFIED | Both confirmed; produces interleaved repetition |
| 3. Correct head split (interleaved): `attn[:, 0::2]` at 218, `attn[:, 1::2]` at 219 | VERIFIED | Interleaved split used; blocked split not present |
| 4. Sigmoid λ, not exp: `torch.sigmoid(self.W_lambda(x))` at 222 | VERIFIED | No `torch.exp()` in λ computation |
| 5. W_lambda bias=True: `nn.Linear(d, n_heads, bias=True)` at 189 | VERIFIED | Exact match |
| 6. No per-head RMSNorm on attn/attn1/attn2 | VERIFIED | No RMSNorm, LayerNorm, or any norm applied to attention outputs before or after differential; only `self.out` projection after reshape |
| 7. λ shape: `[B, L, n_heads]` → `.transpose(1,2)` → `[B, n_heads, L]` → `.unsqueeze(-1)` → `[B, n_heads, L, 1]` at 222 | VERIFIED | Exact sequence at line 222 |
| 8. Output shape: `(attn1 - lam * attn2)` → `.transpose(1,2).reshape(B, L, D)` at 224 | VERIFIED | Correct collapse to `[B, L, D]` where `D = n_heads * d_head` |
| 9. KV shared latent in DiffMLAAttention: `W_DKV` at 265, `W_UK`/`W_UV` at 266–267, both using `c_kv` at 288–289 | VERIFIED | Both K and V produced from the same `c_kv = self.W_DKV(x)` |
| 10. Doubled Q via MLA: `W_UQ = Linear(d_c_q, 2 * nh * dh)` at 271 | VERIFIED | Exact match |
| 11. K and V standard width in DiffMLAAttention: `W_UK = Linear(d_c, nh * dh)` at 266 | VERIFIED | Output dim is `nh * dh`, not doubled |
| 12. Same GQA pairing in DiffMLAAttention: `k.repeat_interleave(2, dim=1)` at 299, `attn[:, 0::2]` at 304 | VERIFIED | Both confirmed |
| 13. W_lambda same spec in DiffMLAAttention: `nn.Linear(d, n_heads, bias=True)` at 274 | VERIFIED | Identical specification to DifferentialCausalAttention line 189 |

---

## Summary

All 13 checklist items pass. All stated line numbers match the actual code exactly. The V2 pseudocode in the spec is verbatim from `diff_attn_v2_2026_01.md`. The V2 changes table accurately reflects the differences between V1 and V2 as documented in the source. All three intentional deviations are accurately described and present in the code.

No discrepancies found between the spec, the primary sources, and the implementation.
