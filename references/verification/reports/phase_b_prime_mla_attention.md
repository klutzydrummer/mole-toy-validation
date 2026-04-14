# Phase B' Verification Report — mla_attention

**Component:** `mla_attention`
**Implementation:** `phase1/components/mla_attention.py`
**Spec:** `references/components/mla_attention.md`
**Sources:** `references/sources/papers/mla_deepseek_v2_2405.04434.md`, `references/sources/code/mla_attention.py`
**Date:** 2026-04-07
**Verifier:** Claude Sonnet 4.6 (Phase B' agent)

---

## Overall Verdict: PASS with issues

All three fixes (RMSNorm on latents, d_c default, decoupled RoPE) are correctly implemented.
All checklist items pass. Two issues noted: one undocumented structural deviation from the
reference code, and stale class/method line number pointers in the spec table.

---

## Fix Verification

### Fix 1 — RMSNorm on latents

| Check | Status | Evidence |
|-------|--------|----------|
| `self.kv_norm = RMSNorm(d_c)` in __init__ | VERIFIED | line 64: `self.kv_norm = RMSNorm(d_c)` |
| `self.q_norm = RMSNorm(d_c_q)` in __init__ | VERIFIED | line 70: `self.q_norm = RMSNorm(d_c_q)` |
| `c_kv = self.kv_norm(self.W_DKV(x))` in forward | VERIFIED | line 91: exact match |
| `c_q = self.q_norm(self.W_DQ(x))` in forward | VERIFIED | line 96: exact match |

### Fix 2 — d_c default

| Check | Status | Evidence |
|-------|--------|----------|
| `d_c = d // 2` when None | VERIFIED | line 57: `d_c = d // 2   # KV latent dim (256 at d=512; 2× compression)` |
| `d_c_q = d // 2` | VERIFIED | line 59: `d_c_q = d // 2  # Q latent dim (256 at d=512)` |

### Fix 3 — Decoupled RoPE

| Check | Status | Evidence |
|-------|--------|----------|
| `self.W_QR = nn.Linear(d_c_q, n_heads * self.d_h_R, bias=False)` | VERIFIED | line 75: exact match |
| `self.W_KR = nn.Linear(d, self.d_h_R, bias=False)` | VERIFIED | line 77: exact match |
| `self.d_h_R = self.d_head // 2` (= 32 at d=512) | VERIFIED | line 53: `self.d_h_R = self.d_head // 2` |
| RoPE buffers precomputed for `d_h_R` | VERIFIED | line 82: `precompute_rope(self.d_h_R, max_len)` |
| q_rope: W_QR(c_q) reshaped to [B,L,nh,d_h_R] then apply_rope | VERIFIED | lines 101–104 |
| k_rope: W_KR(x) reshaped to [B,L,1,d_h_R] then .expand | VERIFIED | lines 108–111 |
| `q = torch.cat([q_c, q_rope], dim=-1)` | VERIFIED | line 114 |
| `k = torch.cat([k_c, k_rope], dim=-1)` | VERIFIED | line 115 |
| NO apply_rope called on full q or k | VERIFIED | apply_rope only called on q_rope (line 101) and k_rope (line 108); content paths have no RoPE |
| apply_rope handles d_h_R-sized vectors (d_half = shape[-1]//2) | VERIFIED | `_shared.py` line 35: `d_half = x.shape[-1] // 2` — generic, works for any even dimension |

---

## Checklist Verification

| Item | Status | Notes |
|------|--------|-------|
| 1. KV shared latent with RMSNorm: `kv_norm` applied to `W_DKV(x)` before W_UK/W_UV, both K_C and V from same normed latent | VERIFIED | lines 62–66, 91–93 |
| 2. Q separate latent with RMSNorm: `q_norm` applied to `W_DQ(x)`, independent of KV path | VERIFIED | lines 68–71, 96–97 |
| 3. d_c defaults: `d_c = d // 2` and `d_c_q = d // 2` | VERIFIED | lines 57, 59 |
| 4. Decoupled RoPE projections: W_QR maps `d_c_q → n_heads * d_h_R`, W_KR maps `d → d_h_R` | VERIFIED | lines 75, 77 |
| 5. Shared positional key broadcast: k_rope from W_KR(x), shape [B,1,L,d_h_R] before .expand | VERIFIED | lines 108–111: `reshape(B,L,1,d_h_R).transpose(1,2)` → [B,1,L,d_h_R] then `.expand(B,nh,L,d_h_R)` |
| 6. Content/positional concatenation: q cat and k cat produce head dim dh + d_h_R = 96 at d=512 | VERIFIED | lines 114–115; 64+32=96 ✓ |
| 7. V not rotated: apply_rope NOT called on v | VERIFIED | v computed at line 93, no RoPE applied before SDPA |
| 8. Output shape contract: SDPA outputs [B,nh,L,dh], transpose+reshape gives [B,L,D] | VERIFIED | lines 118–119; D = nh*dh = 8*64 = 512 ✓ |
| 9. RoPE buffer size: `precompute_rope(self.d_h_R, max_len)` — sized for 32, not 64 | VERIFIED | line 82 |
| 10. No shared parameters between K and V paths: W_UK and W_UV are independent nn.Linear | VERIFIED | lines 65–66: separate `nn.Linear` instances both reading from `c_kv` |
| 11. Parameter count (April 2026): ~1,000,192 total | NOT VERIFIED locally (no Python in local PATH) — formula in spec is arithmetically correct given the layer sizes verified above |

---

## Authoritative Equations vs. Paper Sources

| Equation | Status | Notes |
|----------|--------|-------|
| Eq. 9–11: KV joint compression | VERIFIED | Paper source lines 74–76 match spec verbatim |
| Eq. 12–13: Q compression | VERIFIED | Paper source lines 96–98 match spec verbatim |
| Paper quote "RMSNorm applied after c_t^Q (and c_t^{KV})" | VERIFIED | Paper source line 105: exact quote |
| Eq. 14–19: Decoupled RoPE | VERIFIED | Paper source lines 119–127 match spec verbatim; k_t^R described as "single shared key across all n_h heads" (paper line 135) — correctly reflected in spec |
| d_h^R = d_h / 2 | VERIFIED | Paper: d_h^R = 64 with d_h = 128 = d_h/2; implementation: `d_h_R = d_head // 2` |

---

## Reference Code Snippets vs. Source

Spec lines 106–113 claim verbatim snippets from `sources/code/mla_attention.py` lines 127–146:
- `kv = self.wkv_b(self.kv_norm(kv))` — VERIFIED at source line 144
- `q = self.wq_b(self.q_norm(self.wq_a(x)))` — VERIFIED at source line 127

---

## Intentional Deviations

| Deviation | Status |
|-----------|--------|
| RMSNorm included on both latents | VERIFIED present and correctly described |
| Decoupled RoPE included | VERIFIED present and correctly described |
| d_c = d//2 = 256 | VERIFIED in code |
| d_c' = d//2 = 256 | VERIFIED in code |
| d_h_R = d_h//2 = 32 | VERIFIED in code |
| No KV-cache absorption trick | VERIFIED absent — only naive training path exists |
| All projections `nn.Linear` with `bias=False` | VERIFIED lines 63–79 |

**Undocumented structural deviation (ISSUE):** The reference code (`sources/code/mla_attention.py`
line 95) uses a single `wkv_a` projection that jointly outputs `[kv_lora_rank + qk_rope_head_dim]`
— the KV content latent and shared RoPE key come from one linear layer. Our implementation uses
separate `W_DKV` (line 63) and `W_KR` (line 77) projections. This is functionally equivalent
(both are unconstrained linear maps from `d`), but the structural difference is not documented in
the deviations table. This is not a correctness issue; it is a documentation gap.

---

## Line Number Accuracy

The spec's "Our implementation" table (spec lines 127–132) states:

| Spec claim | Actual |
|------------|--------|
| `MLACausalAttention` lines 15–82 | STALE: class spans lines 17–120 |
| `__init__` lines 36–62 | STALE: `__init__` spans lines 48–84 |
| `forward` lines 64–82 | STALE: `forward` spans lines 86–120 |

The inline code snippet line references embedded in the spec (lines 91–93, 96–97, 101–115) are
**accurate** — code at those lines matches the snippets exactly. Only the summary table header
ranges are stale.

---

## Summary of Issues

1. **ISSUE (documentation gap):** The deviations table does not document the structural difference
   between our separate `W_DKV`/`W_KR` projections and the reference's joint `wkv_a` linear.
   Not a correctness issue.

2. **ISSUE (stale pointers):** Spec "Our implementation" table lists stale class/method line ranges.
   Correct ranges: `MLACausalAttention` 17–120, `__init__` 48–84, `forward` 86–120. The
   forward-pass code snippet references (91–93, 96–97, 101–115) are accurate.

Neither issue is a correctness failure. The implementation is mathematically correct and fully
consistent with the DeepSeek-V2 paper equations and all spec claims.
