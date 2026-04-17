# Phase B' Verification Report: diff_attention

**Date:** 2026-04-16
**Verifier:** Claude Sonnet 4.6 (Phase B' agent)
**Verdict:** PASS with issues (stale line number pointers only — non-blocking)

---

## Component

`DifferentialCausalAttention` and `DiffMLAAttention` in `phase1/components/diff_attention.py`

Spec: `references/components/diff_attention.md`

---

## Sources verified

- `references/sources/papers/diff_attn_v1_2410.05258.md` — V1 baseline
- `references/sources/papers/diff_attn_v2_2026_01.md` — V2 authoritative pseudocode
- `references/sources/papers/mla_deepseek_v2_2405.04434.md` — MLA Eq. 9–19
- `references/sources/code/mla_attention.py` — DeepSeek-V3 naive forward

---

## DifferentialCausalAttention — checklist items 1–8

All mathematically correct. Stale line number pointers in spec checklist (from pre-extraction refactor 2026-04-05):

| Checklist item | Spec line ref | Correct current line |
|----------------|--------------|---------------------|
| W_Q, W_K | 183, 184 | 50, 51 |
| W_lambda bias=True | 189 | 56 |
| repeat_interleave | 211, 212 | 78, 79 |
| attn split 0::2 / 1::2 | 218, 219 | 85, 86 |
| sigmoid + unsqueeze | 222 | 89 |
| out reshape | 224 | 91–92 |

**All checks VERIFIED at correct locations.**

1. **Doubled Q, standard KV** — `W_Q = Linear(d, 2*d)` at line 50, `W_K = Linear(d, d)` at line 51. VERIFIED.
2. **GQA pairing via repeat_interleave** — `k.repeat_interleave(2, dim=1)` at line 78. VERIFIED.
3. **Correct interleaved split** — `attn[:, 0::2]` / `attn[:, 1::2]` at lines 85–86. NOT `[:nh//2]`. VERIFIED.
4. **Sigmoid λ** — `torch.sigmoid(self.W_lambda(x))` at line 89. No `torch.exp`. VERIFIED.
5. **W_lambda bias=True** — `nn.Linear(d, n_heads, bias=True)` at line 56. VERIFIED.
6. **No per-head RMSNorm** — no RMSNorm on attn1/attn2. VERIFIED.
7. **λ shape** — `[B, L, nh]` → transpose → `[B, nh, L]` → unsqueeze → `[B, nh, L, 1]`. VERIFIED.
8. **Output shape** — `(attn1 - lam * attn2)` is `[B, nh, L, dh]`; reshape to `[B, L, D]`. VERIFIED.

---

## DiffMLAAttention — checklist items 9–15 (updated April 2026)

All updated changes verified against the updated spec and MLA paper Eq. 9–19.

9. **KV shared latent with RMSNorm** — `self.kv_norm = RMSNorm(d_c)` at line 144; `c_kv = self.kv_norm(self.W_DKV(x))` at line 182. d_c = d//2 = 256 at d=512 (line 135). VERIFIED.

10. **Q latent with RMSNorm, doubled heads** — `self.q_norm = RMSNorm(d_c_q)` at line 150; `self.W_UQ = Linear(d_c_q, 2*nh*dh)` at line 153. VERIFIED.

11. **K and V standard width** — `self.W_UK = Linear(d_c, nh*dh)` at line 146. V not double-wide. VERIFIED.

12. **Decoupled RoPE projections** — `self.W_QR = Linear(d_c_q, 2*nh*d_h_R)` at line 155 (2×nh — matches doubled Q, novel design point); `self.W_KR = Linear(d, d_h_R)` at line 157; `precompute_rope(d_h_R, max_len)` at line 165. VERIFIED.

13. **Decoupled RoPE forward** — `q = torch.cat([q_c, q_rope], dim=-1)` and `k_full = torch.cat([k_c, k_rope], dim=-1)` at lines 195–196. V not rotated. SDPA output has V's head dim (dh, not dh+d_h_R); `reshape(B, L, D)` valid since D = nh*dh. VERIFIED.

14. **GQA pairing on full K** — `k_full.repeat_interleave(2, dim=1)` at line 199 (full concatenated K, not k_c alone). `attn[:, 0::2]` / `attn[:, 1::2]` at lines 203–204. VERIFIED.

15. **W_lambda same spec** — `Linear(d, n_heads, bias=True)` at line 159. Token-specific, head-specific, sigmoid-bounded. VERIFIED.

---

## Source cross-check

### V2 differential operation (diff_attn_v2_2026_01.md pseudocode)

Blog: `attn1, attn2 = attn[:, 0::2], attn[:, 1::2]`; `lam_val = sigmoid(lam)`; `attn = attn1 - lam_val * attn2`

Implementation: matches exactly at lines 85–92 (`DifferentialCausalAttention`) and 203–208 (`DiffMLAAttention`). VERIFIED.

### MLA Eq. 9–11 KV compression (mla_deepseek_v2_2405.04434.md)

`c_KV = W_DKV · h`, `K_C = W_UK · c_KV`, `V = W_UV · c_KV`

Implementation: lines 182–184 (`DiffMLAAttention`). k_c and v both read from same c_kv. VERIFIED.

### MLA Eq. 14–19 decoupled RoPE

`Q_R = RoPE(W_QR · c_Q)`, `K_R = RoPE(W_KR · h)`, `Q = [Q_C; Q_R]`, `K = [K_C; K_R]`

Implementation: lines 187–199 (`DiffMLAAttention`). **Novel extension**: W_QR maps `d_c_q → 2*nh*d_h_R` instead of `d_c_q → nh*d_h_R` — required because Q is doubled to 2h heads for the Diff mechanism. Not covered by the MLA paper; documented as intentional deviation. VERIFIED as internally consistent.

---

## Issues (non-blocking)

**Stale line number pointers in verification checklist items 1–8.** Correct locations documented in table above. These are pointer staleness only — no mathematical errors. The spec's equations, pseudocode references, and deviation table all remain accurate.

**Recommendation:** Update checklist items 1–8 line numbers in `diff_attention.md` on next spec edit pass.

---

## Verdict: PASS with issues

All mathematical claims verified. Implementation matches spec. Deviations accurately documented. Only stale line pointers (non-blocking).
